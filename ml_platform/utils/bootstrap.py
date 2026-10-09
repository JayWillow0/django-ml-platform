from django import forms

class BootstrapModelForm(forms.ModelForm):
    """Bootstrap样式的ModelForm"""
    bootstrap_exclude_fields= []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():

            # 设置某些标签无form样式
            if name in self.bootstrap_exclude_fields:
                continue

            if field.widget.attrs:
                field.widget.attrs['class'] = ' form-control' # 如果已经有class属性，则在原有的基础上添加form-control类
                field.widget.attrs['placeholder'] = field.label # 添加placeholder属性，值为字段的标签

            else:
                field.widget.attrs = {
                    'class': 'form-control', 
                    'placeholder': field.label
                } 



                