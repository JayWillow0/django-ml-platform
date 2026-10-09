from django.shortcuts import redirect
from functools import wraps


def login_required(view_func):
    """登录验证装饰器：检查session中的用户信息"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('info'):
            return redirect('/login/')
        return view_func(request, *args, **kwargs)
    return wrapper
