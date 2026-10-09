"""mysite URL Configuration
"""
from django.urls import path, re_path, include
from django.views.static import serve
from django.conf import settings
from ml_platform import views
urlpatterns = [
    # 开发环境由 Django 直接 serve 媒体文件；生产由 Nginx 的 `location /media/` 优先拦截，本路由不会被命中
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}, name='media'),

    # 登录
    path('login/', views.login),
    path('logout/', views.logout),
    path('image/code/', views.image_code),

    # 管理员管理
    path('admin/list/', views.admin_list),
    path('admin/add/', views.admin_add),
    path('admin/<int:nid>/edit/', views.admin_edit),
    path('admin/<int:nid>/delete/', views.admin_delete),
    path('admin/<int:nid>/reset/', views.admin_reset),
    path('api/admin/list/', views.api_admin_list),

    # ML平台模块
    path('', include('ml_platform.urls')),
]
