from django.urls import path
from ml_platform import views

app_name = 'ml_platform'

urlpatterns = [
    # === 页面 View（渲染 HTML shell）===
    path('data/', views.data_page, name='data_page'),
    path('preprocess/<int:dataset_id>/', views.preprocess_page, name='preprocess_page'),
    path('dashboard/', views.dashboard_page, name='dashboard_page'),
    path('predict/', views.predict_page, name='predict_page'),
    path('tasks/', views.task_page, name='task_page'),
    path('help/', views.help_page, name='help_page'),

    # === 数据管理 API ===
    path('api/datasets/', views.api_dataset_list, name='api_dataset_list'),
    path('api/datasets/upload/', views.api_dataset_upload, name='api_dataset_upload'),
    path('api/datasets/<int:dataset_id>/preview/', views.api_dataset_preview, name='api_dataset_preview'),
    path('api/datasets/<int:dataset_id>/statistics/', views.api_dataset_statistics, name='api_dataset_statistics'),
    path('api/datasets/<int:dataset_id>/delete/', views.api_dataset_delete, name='api_dataset_delete'),

    # === 特征配置 API ===
    path('api/datasets/<int:dataset_id>/columns/', views.api_dataset_columns, name='api_dataset_columns'),
    path('api/datasets/<int:dataset_id>/preprocess/', views.api_preprocess_save, name='api_preprocess_save'),

    # === 模型看板 API ===
    path('api/tasks/train/', views.api_train_start, name='api_train_start'),
    path('api/tasks/<int:task_id>/status/', views.api_train_status, name='api_train_status'),
    path('api/tasks/<int:task_id>/results/', views.api_train_results, name='api_train_results'),
    path('api/tasks/<int:task_id>/chart/<str:chart_type>/', views.api_train_chart, name='api_train_chart'),

    # === 在线预测 API ===
    path('api/predict/<int:task_id>/', views.api_predict_submit, name='api_predict_submit'),
    path('api/predict/<int:task_id>/download/', views.api_predict_download, name='api_predict_download'),

    # === 任务追踪 API ===
    path('api/tasks/', views.api_task_list, name='api_task_list'),
    path('api/tasks/<int:task_id>/', views.api_task_detail, name='api_task_detail'),
    path('api/tasks/<int:task_id>/cancel/', views.api_task_cancel, name='api_task_cancel'),
    path('api/tasks/compare/', views.api_task_compare, name='api_task_compare'),

    # === 任务对比页面 ===
    path('compare/', views.task_compare_page, name='task_compare_page'),

    # === 后台任务查看（仅超级管理员）===
    path('backend/', views.backend_data_page, name='backend_data_page'),
    path('api/backend/tasks/', views.api_backend_tasks, name='api_backend_tasks'),
    path('api/backend/datasets/', views.api_backend_datasets, name='api_backend_datasets'),
]