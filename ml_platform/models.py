from django.db import models


class Admin(models.Model):
    """管理员表"""
    username = models.CharField(verbose_name='用户名', max_length=32)
    password = models.CharField(verbose_name='密码', max_length=64)

    def __str__(self):
        return self.username


class DataSet(models.Model):
    """上传数据集"""
    STATUS_CHOICES = (
        ('uploaded', '已上传'),
        ('preprocessed', '已配置'),
        ('completed', '已完成'),
    )

    admin = models.ForeignKey(Admin, verbose_name='所属用户',
                              on_delete=models.CASCADE, null=True, blank=True)
    file_name = models.CharField(verbose_name='文件名', max_length=256)
    file_path = models.CharField(verbose_name='文件相对路径', max_length=512)
    row_count = models.IntegerField(verbose_name='行数', default=0)
    col_count = models.IntegerField(verbose_name='列数', default=0)
    columns = models.JSONField(verbose_name='列名列表', default=list)
    dtypes = models.JSONField(verbose_name='列类型', default=dict)
    missing_values = models.JSONField(verbose_name='缺失值统计', default=dict)
    preview = models.JSONField(verbose_name='前5行预览', default=list)
    status = models.CharField(verbose_name='状态', max_length=32,
                              choices=STATUS_CHOICES, default='uploaded')
    # 预处理配置（模块2填写）
    feature_cols = models.JSONField(verbose_name='选择特征列', default=list)
    target_col = models.CharField(verbose_name='目标列', max_length=128,
                                  null=True, blank=True)
    fill_strategy = models.CharField(verbose_name='填充策略', max_length=32,
                                     default='median')
    scale_numeric = models.BooleanField(verbose_name='是否缩放', default=True)
    task_type = models.CharField(verbose_name='任务类型', max_length=32,
                                 null=True, blank=True)
    statistics = models.JSONField(verbose_name='描述性统计', default=dict)
    create_time = models.DateTimeField(verbose_name='创建时间', auto_now_add=True)

    def __str__(self):
        return self.file_name


class MLTask(models.Model):
    """训练任务"""
    STATUS_CHOICES = (
        ('pending', '等待中'),
        ('running', '运行中'),
        ('completed', '已完成'),
        ('failed', '失败'),
        ('cancelled', '已取消'),
    )
    TASK_TYPE_CHOICES = (
        ('supervised', '监督学习'),
        ('unsupervised.clustering', '无监督-聚类分析'),
        ('unsupervised.anomaly', '无监督-异常检测'),
        ('unsupervised.dim_reduction', '无监督-降维可视化'),
    )

    admin = models.ForeignKey(Admin, verbose_name='所属用户',
                              on_delete=models.CASCADE, null=True, blank=True)
    dataset = models.ForeignKey(DataSet, verbose_name='关联数据集',
                                on_delete=models.CASCADE)
    task_name = models.CharField(verbose_name='任务名称', max_length=256)
    task_type = models.CharField(verbose_name='任务类型', max_length=32,
                                 choices=TASK_TYPE_CHOICES, default='supervised')
    status = models.CharField(verbose_name='状态', max_length=32,
                              choices=STATUS_CHOICES, default='pending')
    # 文件路径（相对路径）
    model_path = models.CharField(verbose_name='模型文件路径', max_length=512,
                                  null=True, blank=True)
    celery_task_id = models.CharField(verbose_name='Celery任务ID', max_length=256,
                                      null=True, blank=True)
    # Pipeline 结果
    results_json = models.JSONField(verbose_name='训练结果', default=dict)
    charts_json = models.JSONField(verbose_name='图表配置', default=dict)
    # 错误信息
    error_msg = models.TextField(verbose_name='错误信息', null=True, blank=True)
    error_detail = models.JSONField(verbose_name='错误详情', default=dict)
    # 训练进度
    progress_info = models.JSONField(verbose_name='进度信息', default=dict)

    create_time = models.DateTimeField(verbose_name='创建时间', auto_now_add=True)
    update_time = models.DateTimeField(verbose_name='更新时间', auto_now=True)

    def __str__(self):
        return self.task_name
