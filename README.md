# MLPlatform — 基于 Django 的机器学习平台

一个使用 Django + Celery 构建的机器学习平台，覆盖从数据上传到在线预测的完整流程：
数据集管理、数据预处理、模型训练、任务对比与在线预测，内置一份面向使用者的帮助文档。

## 功能特性

- **数据集管理**：CSV / Excel 上传、数据浏览与基本统计
- **数据预处理**：缺失值处理、特征编码、特征选择等常用操作
- **模型训练**：基于 scikit-learn 的批量训练，统一指标排序，快速建立候选基线
- **任务对比**：多次训练任务的指标横向对比，支持分页浏览
- **在线预测**：上传新数据即可用已训练模型预测，自动校验列名与顺序
- **异步任务**：Celery 承载训练任务，训练完成自动保存模型与评估图表
- **帮助文档**：内置从数据到建模的概念说明，降低使用门槛

## 技术栈

- **Web 框架**：Django 4.2.16（LTS）
- **数据库**：MySQL（库名 `ml_db`；开发 5.7、生产 8.0+）
- **数据库驱动**：PyMySQL（纯 Python，替代 mysqlclient，Windows/Linux 均免编译）
- **异步任务**：Celery——broker 走环境变量：开发默认 SQLAlchemy + SQLite（免装 Redis），生产 Redis 7.2；池 Windows=`solo` / Linux=`prefork`
- **生产服务器**：Gunicorn + Nginx（部署模板见 `deploy/`）
- **机器学习**：pandas / numpy / scipy / scikit-learn / joblib

## 前置要求

- Python 3.10+（项目测试版本为 3.11.7）
- MySQL 服务（开发 5.7 / 生产 8.0+；本地 5.7 需额外配置，见下文「数据库准备」）
- 生产部署额外需要：Nginx、Redis 7.2（详见 [`deploy/DEPLOY.md`](deploy/DEPLOY.md)）

---

## ⚠️ 运行前必须配置 `.env`

**本项目不附带 `.env` 文件**——其中包含数据库密码、密钥等敏感信息，已被 `.gitignore` 忽略，
不会上传 GitHub。首次运行前必须自行创建，否则项目无法启动。

### 1. 复制示例文件

```bash
cp .env.example .env          # Git Bash / Linux / macOS
# 或 Windows CMD:
copy .env.example .env
```

### 2. 生成 Django SECRET_KEY 并填入 `.env`

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

把输出粘到 `.env` 的 `DJANGO_SECRET_KEY=` 之后。
**值含 `$` 等特殊字符，务必保留外层单引号**，否则会被 dotenv 当作变量插值而损坏。

### 3. 环境变量说明

| 变量 | 说明 | 示例 |
|---|---|---|
| `DJANGO_SECRET_KEY` | Django 密钥（会话/CSRF/密码哈希加盐），必须保密 | `'随机生成的 50 位字符串'` |
| `DEBUG` | 调试模式，开发 `True`，**生产务必 `False`** | `True` |
| `ALLOWED_HOSTS` | 允许访问的域名/IP，逗号分隔 | `localhost,127.0.0.1` |
| `DB_NAME` / `DB_USER` / `DB_PASSWORD` | MySQL 库名 / 用户 / 密码 | `ml_db` / `root` / `your_password` |
| `DB_HOST` / `DB_PORT` | MySQL 主机 / 端口 | `127.0.0.1` / `3306` |
| `INITADMIN_USERNAME` / `INITADMIN_PASSWORD` | 初始化超级管理员 账号 / 密码 | `admin` / `change_me` |
| `SKIP_DB_VERSION_CHECK` | 本地 MySQL 5.7 + Django 4.2 时设 `true` 跳过版本检查 | `true` |
| `CELERY_BROKER_URL` | Celery broker；**留空**用 SQLite（开发），生产填 redis 地址 | *(开发留空)* |
| `CELERY_WORKER_POOL` | Celery 池；**留空**默认 solo（Win），生产填 `prefork` | *(开发留空)* |
| `CSRF_TRUSTED_ORIGINS` | 生产 HTTPS 反代信任来源（带 `http(s)://`） | `http://公网IP` |

> 开发环境用 [`.env.example`](.env.example)（最小集）；生产环境的完整变量（含 Celery / CSRF / Redis）
> 见 [`.env.production.example`](.env.production.example)。

---

## 安装

```bash
# 建议新建虚拟环境
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
```

> `requirements.txt` 锁定了全部依赖及精确版本，便于环境复现。
> MySQL 驱动使用 `PyMySQL`（纯 Python，免编译）；生产部署额外需要 `gunicorn`。如需精简，可只保留：
> `Django`、`celery`、`PyMySQL`、`redis`、`gunicorn`、`pandas`、`numpy`、`scipy`、
> `scikit-learn`、`joblib`、`Pillow`、`matplotlib`、`python-dotenv`。

## 数据库准备

在 MySQL 中创建库（库名需与 `.env` 的 `DB_NAME` 一致）：

```sql
CREATE DATABASE ml_db CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
```

执行迁移：

```bash
python manage.py migrate
```

> **本地 MySQL 5.7 用户**：Django 4.2 强制要求 MySQL 8.0+，迁移/runserver 前需在 `.env` 设
> `SKIP_DB_VERSION_CHECK=true` 跳过版本检查（根治方案是升级本地 MySQL 到 8.0）。
> 生产环境（MySQL 8.0）的完整建库 SQL 见 [`deploy/setup_mysql8.sql`](deploy/setup_mysql8.sql)。

## 初始化超级管理员

```bash
python manage.py initadmin
```

账号 / 密码取自 `.env` 的 `INITADMIN_USERNAME` / `INITADMIN_PASSWORD`。
若已存在同名用户会自动跳过，不会覆盖。


## 运行

启动 Web 服务（开发）：

```bash
python manage.py runserver
```

启动 Celery worker（池由 `.env` 的 `CELERY_WORKER_POOL` 决定；不填则 Windows 用 solo）：

```bash
celery -A mysite worker --loglevel=info
```

生产部署（静态文件收集、gunicorn + celery 守护、Nginx 反代）见下方「生产部署」章节。

---

## 安全说明

- `.env` 已被 `.gitignore` 忽略，**切勿手动提交或外传**。
- 生产部署务必将 `DEBUG=False`，并设置正确的 `ALLOWED_HOSTS`。
- 请勿复用示例中的弱口令（如 `admin123`），生产环境请使用强密码。
- **维护提醒**：`SECRET_KEY` 被 `ml_platform/utils/encrypt.py` 用于 Admin 密码的 MD5 加盐。
  一旦更换 `SECRET_KEY`，库中已有的 Admin 密码哈希会失效、无法登录，需重新设置密码
  （可删除该用户后重跑 `python manage.py initadmin`，或用 Django shell 重设
  `password = md5('新密码')`）。

---

## 生产部署（Linux + 宝塔）

本项目支持同一份代码在 Windows 开发机和 Linux 服务器部署，无需 Docker。

- **完整手册**：见 [`deploy/DEPLOY.md`](deploy/DEPLOY.md)（步骤 + 跨平台 .env 对照 + FAQ）
- **架构**：Nginx（反代 + 静态/媒体直供）→ Gunicorn → MySQL 8.0；Celery worker ← Redis 7.2
- **生产环境变量模板**：[`.env.production.example`](.env.production.example)
- **进程守护**：gunicorn + celery worker，由 systemd 或宝塔「Python 项目管理器」管理
- **部署模板**（`deploy/`）：`gunicorn_conf.py` / `nginx_mysite.conf` / `mysite-gunicorn.service` / `mysite-celery.service` / `setup_mysql8.sql`
- **Redis 选 7.2**（7.4+ 改 AGPL 协议，商用合规风险）；如需规避可用 Valkey

快速入口：上传代码 → 建 venv + `pip install -r requirements.txt` → `cp .env.production.example .env` 并填写 →
`deploy/setup_mysql8.sql` 建库 → `migrate` + `collectstatic` + `initadmin` → 配 Nginx → 启动 gunicorn + celery。
