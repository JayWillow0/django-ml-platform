"""
Gunicorn 配置（生产 WSGI 服务器）
============================================================
启动命令：
    gunicorn mysite.wsgi:application -c deploy/gunicorn_conf.py

说明：
- 只监听 127.0.0.1:8000，对外由 Nginx 反代，公网不直接暴露 gunicorn
- 训练任务在 Celery 里跑，gunicorn 只处理 web 请求，timeout 设 120s 足够
- 日志输出到 deploy/logs/（首次部署前需 mkdir -p deploy/logs）
"""
import multiprocessing
import os

# 绑定地址：仅本地，Nginx 反代
bind = '127.0.0.1:8000'

# worker 数：默认 2*CPU+1（内存吃紧可改小，纯 web 并发不需太多）
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'sync'

# 请求超时（秒）。训练不在这里跑，web 请求应很快完成
timeout = 120
keepalive = 5

# 防内存泄漏：每个 worker 处理 max_requests 后自动重启
max_requests = 1000
max_requests_jitter = 50

# 日志
loglevel = 'info'
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_log_dir, exist_ok=True)   # 自动建日志目录，避免首次启动因目录不存在而失败
accesslog = os.path.join(_log_dir, 'gunicorn_access.log')
errorlog = os.path.join(_log_dir, 'gunicorn_error.log')
