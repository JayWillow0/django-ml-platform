# mysite 生产部署手册（阿里云/腾讯云 Linux + 宝塔，禁 Docker）

## 0. 架构

```
公网:80 ──► 宝塔 Nginx（反代 + 静态/媒体直供）
                │
                ▼
        Gunicorn (127.0.0.1:8000) ──► MySQL 8.0
                │  投递任务
                ▼
             Redis 7.2 ──► Celery Worker (prefork) ──► 训练/预处理
```

常驻两个进程：① **gunicorn**（web）② **celery worker**（训练）。由 systemd 或宝塔「Python 项目管理器」守护。
代码对平台零感知，所有环境差异由 `.env` 决定（见第 3 节对照表）。

---

## 1. 服务器前置：宝塔装组件

宝塔面板 → 软件商店，安装：
- **Nginx**（宝塔自带）
- **MySQL 8.0**
- **Redis 7.2**（⚠️ 注意：7.4+ 改 AGPL 协议，商用合规环境务必选 7.2，或改用 Valkey）
- **Python 项目管理器**（可选；不用则用本目录的 systemd 模板）

云控制台安全组：放行 **80**（HTTP）、**443**（HTTPS）、22（SSH）。
**不要**对公网开放 3306 / 6379 / 8000。

---

## 2. 部署步骤

### 2.1 上传代码
把项目（含 `manage.py` 的 `mysite/` 目录）放到 `/www/wwwroot/mysite`。
（宝塔文件管理上传 zip 解压，或 `git clone`）

### 2.2 建虚拟环境 + 装依赖
```bash
cd /www/wwwroot/mysite
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> 依赖已用 PyMySQL 替代 mysqlclient，纯 Python，**无需 mysql_config / libmysqlclient-dev**，跨机装得干净。

### 2.3 配置 .env
```bash
cp .env.production.example .env
vi .env        # 填公网IP、数据库口令、新生成的 SECRET_KEY
```

### 2.4 MySQL 建库建用户
先编辑 `deploy/setup_mysql8.sql`，把两处 `<强口令>` 换成真实密码（与 `.env` 的 `DB_PASSWORD` 一致）：
```bash
mysql -u root -p < deploy/setup_mysql8.sql
```

### 2.5 初始化数据库 + 静态 + 管理员
```bash
cd /www/wwwroot/mysite
source venv/bin/activate
mkdir -p deploy/logs staticfiles media

python manage.py migrate
python manage.py collectstatic --noinput
python manage.py initadmin
```

### 2.6 配置 Nginx
宝塔 → 网站 → 添加站点（域名填公网IP，纯静态）→ 站点设置 → 配置文件，
把 `server { ... }` 整段替换为 `deploy/nginx_mysite.conf` 内容，按文件内注释改路径与 `server_name`。

### 2.7 启动 gunicorn + celery（二选一）

**方案 A：systemd（推荐，通用稳定）**
```bash
sudo cp deploy/mysite-gunicorn.service /etc/systemd/system/
sudo cp deploy/mysite-celery.service   /etc/systemd/system/
# 先编辑两个 service 确认路径/User 正确
sudo systemctl daemon-reload
sudo systemctl enable --now mysite-gunicorn
sudo systemctl enable --now mysite-celery
```

**方案 B：宝塔「Python 项目管理器」**
- gunicorn：添加项目 → 路径 `/www/wwwroot/mysite` → 启动命令填
  `gunicorn mysite.wsgi:application -c deploy/gunicorn_conf.py`
- celery：该插件对 celery 支持有限，**celery 仍建议用方案 A 的 systemd 守护**。

### 2.8 验证
浏览器访问 `http://你的公网IP`：
- [ ] 登录页**验证码正常显示**
- [ ] 登录成功
- [ ] 静态样式（CSS/字体）加载正常
- [ ] 上传数据集 → 训练 → 进度轮询 → 完成/取消 全流程

命令行自检：
```bash
curl -I http://127.0.0.1:8000       # gunicorn，应 200/302
celery -A mysite inspect ping        # worker 在线应回 pong
redis-cli LLEN celery                # 提交训练任务后队列应有值
```

---

## 3. 跨平台 .env 对照（一份代码两边跑）

| 变量 | Windows 开发 | Linux 生产 |
|---|---|---|
| `DEBUG` | True | False |
| `ALLOWED_HOSTS` | localhost,127.0.0.1 | 公网IP,域名 |
| `DB_HOST` | 127.0.0.1（本地 MySQL 5.7） | 127.0.0.1（MySQL 8.0） |
| `CELERY_BROKER_URL` | 不填 → sqlite 文件 | redis://127.0.0.1:6379/0 |
| `CELERY_WORKER_POOL` | 不填 → solo（单进程） | prefork（多进程并行） |
| `CSRF_TRUSTED_ORIGINS` | 不填 | http://公网IP |

**Windows 开发 .env 保持现状即可**（不填 CELERY_* 变量，自动回退 sqlite + solo，免装 Redis）。

---

## 4. 常见问题（FAQ）

**Q：登录验证码不显示 / 报错找不到字体**
A：`Monaco.ttf` 必须随项目部署到项目根（已在仓库内）。`code.py` 已改用 `BASE_DIR` 绝对路径加载，确认 systemd 的 `WorkingDirectory` 指向项目根。

**Q：提交训练任务一直 pending**
A：celery worker 没起来或连不上 Redis。`sudo systemctl status mysite-celery`、`redis-cli ping`。

**Q：POST 表单报 CSRF 校验失败**
A：`.env` 的 `CSRF_TRUSTED_ORIGINS` 没填或漏了 scheme（必须 `http://` 或 `https://` 开头）。

**Q：matplotlib 报 SimHei 字体 warning**
A：正常现象。PNG 图表已废弃、改用前端 ECharts，Linux 缺该字体仅打印警告，不影响功能。

**Q：`mysql_native_password` 不可用错误**
A：服务器若是 MySQL 8.4+（默认禁用该插件）。编辑 `setup_mysql8.sql`，按注释改为默认 `caching_sha2_password`（去掉 `IDENTIFIED WITH mysql_native_password`，用 `IDENTIFIED BY '<口令>'`），PyMySQL 1.2 支持。

---

## 5. 后续：迁移到其他 Windows 电脑
1. copy 整个项目目录到新 Windows 电脑
2. 装 Python（≥3.10）+ 本地 MySQL
3. `python -m venv venv` → `venv\Scripts\activate` → `pip install -r requirements.txt`
4. 用**开发 .env**（不填 CELERY_* 变量 → 自动走 sqlite + solo，免装 Redis）
5. `python manage.py migrate` → `python manage.py runserver`
> Windows 上 Celery 自动用 solo 池（单进程串行），训练吞吐不如 Linux prefork，这是 Windows 固有限制，非配置问题。
