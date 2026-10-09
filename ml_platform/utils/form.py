import re
from django.shortcuts import render, redirect
from django.http import HttpResponse
from ml_platform import models
from django import forms
from django.core.validators import RegexValidator

from ml_platform .utils.pagination import Pagination
from ml_platform .utils.bootstrap import BootstrapModelForm
from ml_platform .utils.encrypt import md5

class AdminModelForm(BootstrapModelForm):
    confirm_password = forms.CharField(label="确认密码", widget=forms.PasswordInput)
    class Meta:
        model = models.Admin
        fields = ["username", "password", "confirm_password"]
        widgets = {
            "password": forms.PasswordInput(render_value=True) 
        }


    # 加密
    def clean_password(self):
        password = self.cleaned_data.get("password")
        return md5(password)
    
    # 钩子方法，校验两次输入的密码是否一致
    def clean_confirm_password(self):
        password = self.cleaned_data.get("password")
        confirm_password = md5(self.cleaned_data.get("confirm_password"))
        if password != confirm_password:
            raise forms.ValidationError("两次输入的密码不一致")
        
        # 返回什么，此字段就保存到数据库中什么
        return confirm_password

class AdminEditModelForm(BootstrapModelForm):
    class Meta:
        model = models.Admin
        fields = ['username']

class AdminResetModelForm(BootstrapModelForm):
    confirm_password = forms.CharField(label="确认密码", widget=forms.PasswordInput)
    class Meta:
        model = models.Admin
        fields = ["password", "confirm_password"]
        widgets = {
            "password": forms.PasswordInput(render_value=True) 
        }

    # 加密
    def clean_password(self):
        password = self.cleaned_data.get("password")

        # 去数据库校验当前密码和新输入密码是否一致
        exists = models.Admin.objects.filter(id=self.instance.pk, password=md5(password)).exists()
        if exists:
            raise forms.ValidationError('密码不能与之前的一致')       

        return md5(password)
    
    # 钩子方法，校验两次输入的密码是否一致
    def clean_confirm_password(self):
        password = self.cleaned_data.get("password")
        confirm_password = md5(self.cleaned_data.get("confirm_password"))
        if password != confirm_password:
            raise forms.ValidationError("两次输入的密码不一致")
        
        # 返回什么，此字段就保存到数据库中什么
        return confirm_password
   
# 登录界面
class LoginForm(forms.Form):
    username = forms.CharField(
        label='用户名', 
        widget=forms.TextInput,
        required=True
        )
    password = forms.CharField(
        label='密码', 
        widget=forms.PasswordInput,
        required=True
        )
    
    code = forms.CharField(
        label='验证码', 
        widget=forms.TextInput,
        required=True
        )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():

            if field.widget.attrs:
                field.widget.attrs['class'] = ' form-control' # 如果已经有class属性，则在原有的基础上添加form-control类
                field.widget.attrs['placeholder'] = field.label # 添加placeholder属性，值为字段的标签

            else:
                field.widget.attrs = {
                    'class': 'form-control', 
                    'placeholder': field.label
                } 

    def clean_password(self):
        pwd = self.cleaned_data.get("password")

        return md5(pwd)


# ============================================================
# ML平台表单
# ============================================================

class MLTaskCreateForm(BootstrapModelForm):
    """训练任务创建表单"""
    class Meta:
        model = models.MLTask
        fields = ['task_name', 'task_type']


class PredictionUploadForm(forms.Form):
    """在线预测上传表单"""
    prediction_file = forms.FileField(
        label='预测数据文件',
        widget=forms.ClearableFileInput(attrs={'accept': '.csv,.xlsx,.xls'}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs = {
                'class': 'form-control',
                'placeholder': field.label,
            }


