"""Celery 配置入口"""
import os
from celery import Celery

# 设置 Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

app = Celery('mysite')

# 从 Django settings 中读取 CELERY_ 前缀的配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动发现各 app 下的 tasks.py
app.autodiscover_tasks()
