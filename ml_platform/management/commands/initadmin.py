"""
初始化超级管理员账户
====================
用法:
    python manage.py initadmin
    python manage.py initadmin --username admin --password admin123

适用于 Docker 部署等全新环境，数据库中无任何管理员时使用。
如果同名用户已存在则跳过，不会覆盖。
"""
import os

from django.core.management.base import BaseCommand, CommandError
from ml_platform.models import Admin
from ml_platform.utils.encrypt import md5


class Command(BaseCommand):
    help = '初始化超级管理员账户（username=admin）。如已存在则跳过。'

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            default=os.getenv('INITADMIN_USERNAME', 'admin'),
            help='管理员用户名（默认: admin）',
        )
        parser.add_argument(
            '--password',
            type=str,
            default=os.getenv('INITADMIN_PASSWORD', 'admin123'),
            help='管理员密码（默认: admin123）',
        )

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']

        if not username:
            raise CommandError('用户名不能为空')
        if not password:
            raise CommandError('密码不能为空')

        # 检查是否已存在
        if Admin.objects.filter(username=username).exists():
            self.stdout.write(
                self.style.WARNING(f'管理员 "{username}" 已存在，跳过创建。')
            )
            return

        Admin.objects.create(
            username=username,
            password=md5(password),
        )
        self.stdout.write(
            self.style.SUCCESS(f'超级管理员 "{username}" 创建成功。')
        )
