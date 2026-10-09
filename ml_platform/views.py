import os
import json
import re
import random
from datetime import datetime
from functools import wraps

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django import forms

from ml_platform import models
from ml_platform.utils.pagination import Pagination
from ml_platform.utils.bootstrap import BootstrapModelForm
from ml_platform.utils.form import (
    AdminModelForm, AdminEditModelForm, AdminResetModelForm,
    LoginForm, MLTaskCreateForm, PredictionUploadForm,
)
from ml_platform.utils.encrypt import md5
from ml_platform.utils.code import check_code
from ml_platform.utils.auth import login_required
from io import BytesIO


# ============================================================
# 工具函数
# ============================================================

def _get_admin_id(request):
    """从session获取当前登录用户ID"""
    info = request.session.get('info')
    return info['id'] if info else None


def _get_admin_name(request):
    """从session获取当前登录用户名"""
    info = request.session.get('info')
    return info['name'] if info else None


def _is_superadmin(request):
    """判断当前登录用户是否为超级管理员（username=admin）"""
    info = request.session.get('info')
    return info is not None and info.get('name') == 'admin'


def superadmin_required(view_func):
    """超级管理员权限装饰器：仅 username=admin 可访问"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('info'):
            return redirect('/login/')
        if not _is_superadmin(request):
            return redirect('/data/')
        return view_func(request, *args, **kwargs)
    return wrapper


def _json_error(msg, code=1):
    return JsonResponse({'code': code, 'msg': msg})


def _json_success(data=None, msg=''):
    resp = {'code': 0, 'msg': msg}
    if data is not None:
        resp['data'] = data
    return JsonResponse(resp)


def _abs_path(relative_path):
    """相对路径转绝对路径"""
    return os.path.join(settings.BASE_DIR, relative_path)


def _status_display(status):
    """状态码转中文"""
    mapping = {
        'uploaded': '已上传', 'preprocessed': '已配置', 'completed': '已完成',
        'pending': '等待中', 'running': '运行中', 'failed': '失败',
        'cancelled': '已取消',
        'supervised': '监督学习',
        'unsupervised': '无监督-聚类分析',
        'unsupervised.clustering': '无监督-聚类分析',
        'unsupervised.anomaly': '无监督-异常检测',
        'unsupervised.dim_reduction': '无监督-降维可视化',
    }
    return mapping.get(status, status)


def _page_size(request, default=10, max_size=100):
    """从 GET 读取每页条数（前端 page_size 参数），带默认值与上限保护，防异常输入"""
    try:
        size = int(request.GET.get('page_size', default))
    except (TypeError, ValueError):
        size = default
    return max(1, min(size, max_size))


# ============================================================
# 原有：管理员管理（仅超级管理员）
# ============================================================

@superadmin_required
def admin_list(request):
    """管理员列表页（数据由 /api/admin/list/ 异步加载）"""
    return render(request, 'admin_list.html')


@superadmin_required
def api_admin_list(request):
    """管理员列表 API — 用户名搜索（数据量小，不分页，一次返回全部）"""
    keyword = request.GET.get('keyword', '').strip()
    qs = models.Admin.objects.all()
    if keyword:
        qs = qs.filter(username__contains=keyword)
    qs = qs.order_by('id')
    data = [
        {'id': a.id, 'username': a.username, 'is_super': a.username == 'admin'}
        for a in qs
    ]
    return _json_success({'list': data})


@superadmin_required
def admin_add(request):
    """添加管理员"""
    title = '添加管理员'
    if request.method == 'GET':
        form = AdminModelForm()
        return render(request, 'change.html', {'form': form, 'title': title})
    form = AdminModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect('/admin/list/')
    return render(request, 'change.html', {'form': form, 'title': title})


@superadmin_required
def admin_edit(request, nid):
    """编辑管理员"""
    row_object = models.Admin.objects.filter(id=nid).first()
    if not row_object:
        return redirect('/admin/list/')
    # 保护超级管理员：username=admin 的账户不可编辑（防止改用户名导致超管身份丢失）
    if row_object.username == 'admin':
        return redirect('/admin/list/')
    title = '编辑管理员'
    if request.method == 'GET':
        form = AdminEditModelForm(instance=row_object)
        return render(request, 'change.html', {'form': form, 'title': title})
    form = AdminEditModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        form.save()
        return redirect('/admin/list/')
    return render(request, 'change.html', {'form': form, 'title': title})


@superadmin_required
def admin_delete(request, nid):
    """删除管理员（仅接受POST，防GET误触发）"""
    if request.method != 'POST':
        return redirect('/admin/list/')
    row_object = models.Admin.objects.filter(id=nid).first()
    # 保护超级管理员：username=admin 的账户不可删除（防止系统失去唯一超管）
    if row_object and row_object.username == 'admin':
        return redirect('/admin/list/')
    models.Admin.objects.filter(id=nid).delete()
    return redirect('/admin/list/')


@superadmin_required
def admin_reset(request, nid):
    """重置密码"""
    row_object = models.Admin.objects.filter(id=nid).first()
    if not row_object:
        return redirect('/admin/list/')
    title = '重置密码 - {}'.format(row_object.username)
    if request.method == 'GET':
        form = AdminResetModelForm()
        return render(request, 'change.html', {'form': form, 'title': title})
    form = AdminResetModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        form.save()
        return redirect('/admin/list/')
    return render(request, 'change.html', {'form': form, 'title': title})


# ============================================================
# 原有：登录/注销/验证码
# ============================================================

def login(request):
    """登录界面"""
    if request.method == 'GET':
        form = LoginForm()
        return render(request, 'login.html', {'form': form})
    form = LoginForm(data=request.POST)
    if form.is_valid():
        user_input_code = form.cleaned_data.pop('code')
        code = request.session.get('image_code', '')
        if code.upper() != user_input_code.upper():
            form.add_error('code', '验证码错误')
            return render(request, 'login.html', {'form': form})
        admin_object = models.Admin.objects.filter(**form.cleaned_data).first()
        if not admin_object:
            form.add_error('password', '用户名或密码错误')
            return render(request, 'login.html', {'form': form})
        request.session['info'] = {'id': admin_object.id, 'name': admin_object.username}
        request.session.set_expiry(60 * 60 * 24 * 7)
        return redirect('/data/')
    return render(request, 'login.html', {'form': form})


def image_code(request):
    """生成图片验证码"""
    img, code_string = check_code()
    request.session['image_code'] = code_string
    request.session.set_expiry(60)
    stream = BytesIO()
    img.save(stream, 'png')
    return HttpResponse(stream.getvalue())


def logout(request):
    """注销"""
    request.session.clear()
    return redirect('/login/')


# ============================================================
# 模块1：数据管理 — 页面 View
# ============================================================

@login_required
def data_page(request):
    """数据管理页面"""
    return render(request, 'data_list.html')


@login_required
def help_page(request):
    """帮助文档页面"""
    return render(request, 'help.html')


# ============================================================
# 模块1：数据管理 — API
# ============================================================

@login_required
def api_dataset_list(request):
    """数据集列表 API"""
    admin_id = _get_admin_id(request)
    datasets = models.DataSet.objects.filter(admin_id=admin_id).order_by('-create_time')
    data = []
    for ds in datasets:
        data.append({
            'id': ds.id,
            'file_name': ds.file_name,
            'row_count': ds.row_count,
            'col_count': ds.col_count,
            'status': ds.status,
            'status_display': _status_display(ds.status),
            'create_time': ds.create_time.strftime('%Y-%m-%d %H:%M') if ds.create_time else '',
        })
    return _json_success(data)


@login_required
def api_dataset_upload(request):
    """上传数据集 API"""
    if request.method != 'POST':
        return _json_error('仅支持POST')

    upload_file = request.FILES.get('file')
    if not upload_file:
        return _json_error('未找到上传文件')

    # 检查文件格式
    ext = os.path.splitext(upload_file.name)[1].lower()
    if ext not in ('.csv', '.xlsx', '.xls'):
        return _json_error('仅支持 CSV/XLSX/XLS 格式')

    # 保存到 media/upload_data/<username>/
    admin_name = _get_admin_name(request)
    save_dir = os.path.join('media', 'upload_data', admin_name)
    abs_save_dir = _abs_path(save_dir)
    os.makedirs(abs_save_dir, exist_ok=True)

    # 避免文件名冲突：加时间戳
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    safe_name = '{}_{}{}'.format(timestamp, os.path.splitext(upload_file.name)[0], ext)
    abs_file_path = os.path.join(abs_save_dir, safe_name)
    relative_path = os.path.join(save_dir, safe_name)

    with open(abs_file_path, 'wb') as f:
        for chunk in upload_file.chunks():
            f.write(chunk)

    # 调用 DataImporter 解析
    from ml_platform.utils.ml_utils import DataImporter
    try:
        importer = DataImporter()
        result = importer.load_data(abs_file_path)
    except Exception as e:
        # 解析失败，删除已保存文件
        if os.path.exists(abs_file_path):
            os.remove(abs_file_path)
        return _json_error('文件解析失败: {}'.format(str(e)))

    meta = result['metadata']
    # 序列化 preview（确保JSON可序列化）
    from ml_platform.utils.ml_utils import _to_serializable, compute_dataset_statistics
    preview_data = _to_serializable(result['preview'])

    # 计算描述性统计
    statistics = {}
    try:
        statistics = compute_dataset_statistics(result['data'])
    except Exception:
        pass  # 统计失败不影响上传

    # 创建 DataSet 记录
    ds = models.DataSet.objects.create(
        admin_id=_get_admin_id(request),
        file_name=upload_file.name,
        file_path=relative_path.replace('\\', '/'),
        row_count=meta['row_count'],
        col_count=meta['column_count'],
        columns=meta['columns'],
        dtypes=meta['dtypes'],
        missing_values=meta['missing_values'],
        preview=preview_data,
        statistics=statistics,
        status='uploaded',
    )

    return _json_success({
        'id': ds.id,
        'file_name': ds.file_name,
        'row_count': ds.row_count,
        'col_count': ds.col_count,
        'columns': ds.columns,
        'preview': ds.preview,
    }, msg='上传成功')
    # 保存上下文：最近上传的数据集 ID
    request.session['last_dataset_id'] = ds.id


@login_required
def api_dataset_preview(request, dataset_id):
    """数据预览 API"""
    admin_id = _get_admin_id(request)
    ds = models.DataSet.objects.filter(id=dataset_id, admin_id=admin_id).first()
    if not ds:
        return _json_error('数据集不存在')

    return _json_success({
        'columns': ds.columns,
        'dtypes': ds.dtypes,
        'missing_values': ds.missing_values,
        'preview': ds.preview,
        'row_count': ds.row_count,
    })


@login_required
def api_dataset_statistics(request, dataset_id):
    """数据集统计信息 API（支持旧数据懒计算）"""
    admin_id = _get_admin_id(request)
    ds = models.DataSet.objects.filter(id=dataset_id, admin_id=admin_id).first()
    if not ds:
        return _json_error('数据集不存在')

    # 已有统计信息直接返回
    if ds.statistics and isinstance(ds.statistics, dict) and ds.statistics.get('numeric'):
        return _json_success(ds.statistics)

    # 旧数据懒计算：读取文件重新生成统计
    from ml_platform.utils.ml_utils import DataImporter, compute_dataset_statistics
    try:
        abs_path = _abs_path(ds.file_path)
        importer = DataImporter()
        result = importer.load_data(abs_path)
        stats = compute_dataset_statistics(result['data'])
        ds.statistics = stats
        ds.save()
        return _json_success(stats)
    except Exception as e:
        return _json_error('统计计算失败: {}'.format(str(e)))


@login_required
def api_dataset_delete(request, dataset_id):
    """删除数据集 API"""
    admin_id = _get_admin_id(request)
    ds = models.DataSet.objects.filter(id=dataset_id, admin_id=admin_id).first()
    if not ds:
        return _json_error('数据集不存在')

    # 删除物理文件
    abs_path = _abs_path(ds.file_path)
    if os.path.exists(abs_path):
        os.remove(abs_path)

    ds.delete()
    return _json_success(msg='删除成功')


# ============================================================
# 模块2：特征配置 — API
# ============================================================

@login_required
def api_dataset_columns(request, dataset_id):
    """获取数据集列信息 API"""
    admin_id = _get_admin_id(request)
    ds = models.DataSet.objects.filter(id=dataset_id, admin_id=admin_id).first()
    if not ds:
        return _json_error('数据集不存在')

    return _json_success({
        'columns': ds.columns,
        'dtypes': ds.dtypes,
        'missing_values': ds.missing_values,
        'row_count': ds.row_count,
        'current_config': {
            'feature_cols': ds.feature_cols,
            'target_col': ds.target_col,
            'fill_strategy': ds.fill_strategy,
            'scale_numeric': ds.scale_numeric,
            'task_type': ds.task_type,
        },
    })


@login_required
def api_preprocess_save(request, dataset_id):
    """保存预处理配置 API"""
    if request.method != 'POST':
        return _json_error('仅支持POST')

    admin_id = _get_admin_id(request)
    ds = models.DataSet.objects.filter(id=dataset_id, admin_id=admin_id).first()
    if not ds:
        return _json_error('数据集不存在')

    if ds.status not in ('uploaded', 'preprocessed', 'completed'):
        return _json_error('数据集状态不允许修改配置（当前: {}）'.format(_status_display(ds.status)))

    try:
        body = json.loads(request.body)
    except Exception:
        return _json_error('请求格式错误')

    feature_cols = body.get('feature_cols', [])
    target_col = body.get('target_col')
    fill_strategy = body.get('fill_strategy', 'median')
    scale_numeric = body.get('scale_numeric', True)
    task_type = body.get('task_type', 'supervised')

    # 校验特征列
    for col in feature_cols:
        if col not in ds.columns:
            return _json_error('特征列不存在: {}'.format(col))

    # 监督学习必须有目标列
    if task_type == 'supervised':
        if not target_col or target_col not in ds.columns:
            return _json_error('监督学习需要指定目标列')
        if target_col in feature_cols:
            return _json_error('目标列不能同时作为特征列')

    # 自动检测任务子类型
    detected_type = task_type
    if task_type == 'supervised' and target_col:
        from ml_platform.utils.ml_utils import _detect_task_type
        import pandas as pd
        # 从文件读取目标列数据
        abs_path = _abs_path(ds.file_path)
        try:
            if abs_path.endswith('.csv'):
                df = pd.read_csv(abs_path)
            else:
                df = pd.read_excel(abs_path)
            detected_type = _detect_task_type(df[target_col])
        except Exception:
            pass

    ds.feature_cols = feature_cols
    ds.target_col = target_col
    ds.fill_strategy = fill_strategy
    ds.scale_numeric = scale_numeric
    ds.task_type = detected_type
    ds.status = 'preprocessed'
    ds.save()

    return _json_success({'task_type': detected_type, 'dataset_id': ds.id}, msg='配置保存成功')


# ============================================================
# 模块2：特征配置 — 页面 View
# ============================================================

@login_required
def preprocess_page(request, dataset_id):
    """特征配置页面"""
    return render(request, 'preprocess.html', {'dataset_id': dataset_id})


# ============================================================
# 模块3：模型看板 — 页面 View
# ============================================================

@login_required
def dashboard_page(request):
    """模型看板页面"""
    task_id = request.GET.get('task_id', '')
    return render(request, 'dashboard.html', {'task_id': task_id})


# ============================================================
# 模块3：模型看板 — API
# ============================================================

@login_required
def api_train_start(request):
    """启动训练 API"""
    if request.method != 'POST':
        return _json_error('仅支持POST')

    try:
        body = json.loads(request.body)
    except Exception:
        return _json_error('请求格式错误')

    dataset_id = body.get('dataset_id')
    task_name = body.get('task_name', '').strip()
    task_type = body.get('task_type', 'supervised')
    params = body.get('params', {})

    if not dataset_id or not task_name:
        return _json_error('参数不完整')

    valid_types = ['supervised', 'unsupervised.clustering', 'unsupervised.anomaly', 'unsupervised.dim_reduction']
    if task_type not in valid_types:
        return _json_error('无效的任务类型: {}'.format(task_type))

    admin_id = _get_admin_id(request)
    ds = models.DataSet.objects.filter(id=dataset_id, admin_id=admin_id).first()
    if not ds:
        return _json_error('数据集不存在')

    if ds.status != 'preprocessed':
        return _json_error('数据集未完成配置，请先进行特征配置')

    # 创建任务
    task = models.MLTask.objects.create(
        admin_id=admin_id,
        dataset_id=dataset_id,
        task_name=task_name,
        task_type=task_type,
        status='running',
    )

    # Celery 异步执行训练
    from ml_platform.tasks import run_ml_pipeline_task
    celery_result = run_ml_pipeline_task.delay(task_id=task.id, params=params)
    task.celery_task_id = celery_result.id
    task.save()

    return _json_success({
        'task_id': task.id,
        'status': 'running',
    })


@login_required
def api_train_status(request, task_id):
    """轮询训练状态 API"""
    admin_id = _get_admin_id(request)
    task = models.MLTask.objects.filter(id=task_id, admin_id=admin_id).first()
    if not task:
        return _json_error('任务不存在')

    # 队列信息（solo pool 下帮助用户了解排队情况）
    queue_info = None
    if task.status == 'pending':
        ahead = models.MLTask.objects.filter(
            status='pending', create_time__lt=task.create_time
        ).count()
        running = models.MLTask.objects.filter(status='running').first()
        queue_info = {
            'position': ahead + 1,
            'blocking_task': running.task_name if running else None,
        }
    elif task.status == 'running':
        waiting = models.MLTask.objects.filter(status='pending').count()
        queue_info = {'tasks_waiting': waiting}

    return _json_success({
        'task_id': task.id,
        'status': task.status,
        'status_display': _status_display(task.status),
        'error_msg': task.error_msg,
        'error_detail': task.error_detail if task.status == 'failed' else None,
        'progress': task.progress_info if task.status == 'running' else None,
        'queue_info': queue_info,
    })


@login_required
def api_task_cancel(request, task_id):
    """取消训练任务 API"""
    if request.method != 'POST':
        return _json_error('仅支持POST')

    admin_id = _get_admin_id(request)
    task = models.MLTask.objects.filter(id=task_id, admin_id=admin_id).first()
    if not task:
        return _json_error('任务不存在')

    if task.status != 'running':
        return _json_error('只能取消运行中的任务（当前状态: {}）'.format(
            _status_display(task.status)))

    # 设置任务状态为已取消，Celery 任务会在下一个检查点自行退出
    task.status = 'cancelled'
    task.save()

    # 回退数据集状态，允许重新训练
    ds = task.dataset
    if ds and ds.status != 'preprocessed':
        ds.status = 'preprocessed'
        ds.save()

    return _json_success({
        'task_id': task.id,
        'status': 'cancelled',
        'status_display': _status_display('cancelled'),
    })


@login_required
def api_train_results(request, task_id):
    """获取训练结果 API"""
    admin_id = _get_admin_id(request)
    task = models.MLTask.objects.filter(id=task_id, admin_id=admin_id).first()
    if not task:
        return _json_error('任务不存在')
    if task.status != 'completed':
        return _json_error('任务未完成')

    r = task.results_json

    # 聚类分析摘要
    if task.task_type in ('unsupervised', 'unsupervised.clustering'):
        cluster_results = r.get('cluster_results', [])
        best_algo = {}
        if cluster_results:
            best_algo = max(cluster_results,
                            key=lambda x: x.get('silhouette_score') or -1)

        dim_reduction_info = r.get('dim_reduction') or {}
        dim_methods = list(dim_reduction_info.keys())
        pca_cumulative = None
        if 'PCA' in dim_reduction_info:
            cum = dim_reduction_info['PCA'].get('cumulative_variance', [])
            if cum:
                pca_cumulative = round(cum[-1] * 100, 1)

        best_score = best_algo.get('silhouette_score')
        return _json_success({
            'task_name': task.task_name,
            'task_type': task.task_type,
            'task_type_detected': r.get('task_type', ''),
            'cluster_best_algo': best_algo.get('algorithm', '-'),
            'cluster_best_score': round(best_score, 4) if best_score is not None else None,
            'cluster_n_best': best_algo.get('n_clusters', '-') if best_algo else '-',
            'cluster_algorithms': [
                {
                    'name': c.get('algorithm', ''),
                    'score': round(s, 4) if (s := c.get('silhouette_score')) is not None else None,
                    'n_clusters': c.get('n_clusters', 0),
                }
                for c in cluster_results
            ],
            'dim_reduction_methods': dim_methods,
            'pca_variance_explained': pca_cumulative,
        })

    # 异常检测摘要
    if task.task_type == 'unsupervised.anomaly':
        anomaly = r.get('anomaly_results', {})
        dim_reduction_info = r.get('dim_reduction') or {}
        dim_methods = list(dim_reduction_info.keys())
        pca_cumulative = None
        if 'PCA' in dim_reduction_info:
            cum = dim_reduction_info['PCA'].get('cumulative_variance', [])
            if cum:
                pca_cumulative = round(cum[-1] * 100, 1)

        return _json_success({
            'task_name': task.task_name,
            'task_type': task.task_type,
            'n_anomalies': anomaly.get('n_anomalies', 0),
            'n_normal': anomaly.get('n_normal', 0),
            'anomaly_ratio': anomaly.get('anomaly_ratio'),
            'contamination': anomaly.get('contamination'),
            'dim_reduction_methods': dim_methods,
            'pca_variance_explained': pca_cumulative,
        })

    # 降维可视化摘要
    if task.task_type == 'unsupervised.dim_reduction':
        dim_reduction_info = r.get('dim_reduction') or {}
        dim_methods = list(dim_reduction_info.keys())
        pca_cumulative = None
        if 'PCA' in dim_reduction_info:
            cum = dim_reduction_info['PCA'].get('cumulative_variance', [])
            if cum:
                pca_cumulative = round(cum[-1] * 100, 1)

        return _json_success({
            'task_name': task.task_name,
            'task_type': task.task_type,
            'dim_reduction_methods': dim_methods,
            'pca_variance_explained': pca_cumulative,
        })

    # 监督学习摘要
    return _json_success({
        'task_name': task.task_name,
        'task_type': task.task_type,
        'task_type_detected': r.get('task_type', ''),
        'model_name': r.get('training_results', {}).get('best_model_name', ''),
        'best_score': r.get('tuning_results', {}).get('best_score'),
        'n_models_trained': r.get('training_results', {}).get('n_models_trained', 0),
        'selected_features': r.get('feature_selection', {}).get('selected_features', []),
        'feature_importance': r.get('feature_importance', []),
        'training_models': r.get('training_results', {}).get('models', []),
    })


@login_required
def api_train_chart(request, task_id, chart_type):
    """获取指定图表 ECharts JSON API"""
    admin_id = _get_admin_id(request)
    task = models.MLTask.objects.filter(id=task_id, admin_id=admin_id).first()
    if not task:
        return _json_error('任务不存在')
    if task.status != 'completed':
        return _json_error('任务未完成')

    chart_data = task.charts_json.get(chart_type)
    if not chart_data:
        return _json_error('图表不存在: {}'.format(chart_type))

    return JsonResponse({'code': 0, 'echarts': chart_data.get('echarts', {})})


# ============================================================
# 模块4：在线预测 — 页面 View
# ============================================================

@login_required
def predict_page(request):
    """在线预测页面"""
    return render(request, 'predict.html')


# ============================================================
# 模块4：在线预测 — API
# ============================================================

@login_required
def api_predict_submit(request, task_id):
    """提交预测 API"""
    if request.method != 'POST':
        return _json_error('仅支持POST')

    admin_id = _get_admin_id(request)
    task = models.MLTask.objects.filter(id=task_id, admin_id=admin_id).first()
    if not task:
        return _json_error('任务不存在')
    if task.status != 'completed':
        return _json_error('任务未完成，无法预测')
    if task.task_type != 'supervised':
        return _json_error('仅支持监督学习模型进行预测')
    if not task.model_path:
        return _json_error('模型文件不存在')

    upload_file = request.FILES.get('file')
    if not upload_file:
        return _json_error('未找到上传文件')

    # 保存新数据
    admin_name = _get_admin_name(request)
    save_dir = os.path.join('media', 'upload_data', admin_name)
    abs_save_dir = _abs_path(save_dir)
    os.makedirs(abs_save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    new_data_name = 'predict_{}_{}'.format(timestamp, upload_file.name)
    abs_new_path = os.path.join(abs_save_dir, new_data_name)

    with open(abs_new_path, 'wb') as f:
        for chunk in upload_file.chunks():
            f.write(chunk)

    # 预测输出路径
    predict_dir = os.path.join('media', 'predictions', admin_name)
    abs_predict_dir = _abs_path(predict_dir)
    os.makedirs(abs_predict_dir, exist_ok=True)
    output_name = 'prediction_{}.csv'.format(timestamp)
    abs_output_path = os.path.join(abs_predict_dir, output_name)
    relative_output = os.path.join(predict_dir, output_name).replace('\\', '/')

    try:
        from ml_platform.utils.ml_utils import MLPipeline, _to_serializable
        pipeline = MLPipeline()
        abs_model_path = _abs_path(task.model_path)
        pred_result = pipeline.predict_new(
            model_path=abs_model_path,
            new_data_path=abs_new_path,
            output_path=abs_output_path,
        )
    except Exception as e:
        return _json_error('预测失败: {}'.format(str(e)))

    # 预览（前20行）
    from ml_platform.utils.ml_utils import _to_serializable
    preview_df = pred_result['predictions_df'].head(20)
    preview = _to_serializable(preview_df.to_dict(orient='records'))

    return _json_success({
        'prediction_count': pred_result['prediction_count'],
        'preview': preview,
        'download_url': '/api/predict/{}/download/?file={}'.format(task_id, relative_output),
    })


@login_required
def api_predict_download(request, task_id):
    """下载预测结果 API"""
    admin_id = _get_admin_id(request)
    task = models.MLTask.objects.filter(id=task_id, admin_id=admin_id).first()
    if not task:
        return _json_error('任务不存在')

    file_relative = request.GET.get('file', '')
    if not file_relative:
        return _json_error('未指定文件')

    abs_path = _abs_path(file_relative)
    if not os.path.exists(abs_path):
        return _json_error('文件不存在')

    return FileResponse(
        open(abs_path, 'rb'),
        as_attachment=True,
        filename=os.path.basename(abs_path),
    )


# ============================================================
# 模块5：任务追踪 — 页面 View
# ============================================================

@login_required
def task_page(request):
    """任务追踪页面"""
    return render(request, 'task_list.html')


# ============================================================
# 模块5：任务追踪 — API
# ============================================================

@login_required
def api_task_list(request):
    """任务列表 API（仅当前用户的任务）
    - 带 page 参数：分页 + 关键字搜索，返回 {list, page, page_size, total_page, total}（任务追踪列表浏览）
    - 不带 page 参数：返回全量数组（任务对比 / 在线预测等“选择”场景，需看到全部可选任务）
    """
    admin_id = _get_admin_id(request)
    keyword = request.GET.get('keyword', '').strip()
    tasks = models.MLTask.objects.filter(admin_id=admin_id).order_by('-create_time')
    if keyword:
        tasks = tasks.filter(task_name__icontains=keyword)

    def serialize(qs):
        data = []
        for t in qs:
            data.append({
                'id': t.id,
                'task_name': t.task_name,
                'task_type': t.task_type,
                'task_type_display': _status_display(t.task_type),
                'status': t.status,
                'status_display': _status_display(t.status),
                'dataset_name': t.dataset.file_name if t.dataset else '-',
                'create_time': t.create_time.strftime('%Y-%m-%d %H:%M') if t.create_time else '',
                'update_time': t.update_time.strftime('%Y-%m-%d %H:%M') if t.update_time else '',
                'error_msg': t.error_msg,
            })
        return data

    # 选择场景（任务对比 / 在线预测）：不带 page → 全量数组，保持旧行为
    if 'page' not in request.GET:
        return _json_success(serialize(tasks))

    # 浏览场景（任务追踪）：带 page → 分页对象
    page_size = _page_size(request)
    page_object = Pagination(request, queryset=tasks, page_size=page_size, page_param='page')
    return _json_success({
        'list': serialize(page_object.page_queryset),
        'page': page_object.page,
        'page_size': page_size,
        'total_page': page_object.total_page_count,
        'total': tasks.count(),
    })


@login_required
def api_task_detail(request, task_id):
    """任务详情 API"""
    admin_id = _get_admin_id(request)
    task = models.MLTask.objects.filter(id=task_id, admin_id=admin_id).first()
    if not task:
        return _json_error('任务不存在')

    results_summary = {}
    if task.task_type == 'supervised' and task.results_json:
        r = task.results_json
        results_summary = {
            'best_model': r.get('training_results', {}).get('best_model_name', '-'),
            'best_score': r.get('tuning_results', {}).get('best_score'),
            'feature_count': len(r.get('feature_selection', {}).get('selected_features', [])),
        }

    available_charts = list(task.charts_json.keys()) if task.charts_json else []

    return _json_success({
        'id': task.id,
        'task_name': task.task_name,
        'task_type': task.task_type,
        'status': task.status,
        'status_display': _status_display(task.status),
        'create_time': task.create_time.strftime('%Y-%m-%d %H:%M') if task.create_time else '',
        'update_time': task.update_time.strftime('%Y-%m-%d %H:%M') if task.update_time else '',
        'error_msg': task.error_msg,
        'error_detail': task.error_detail if task.status == 'failed' else None,
        'results_summary': results_summary,
        'available_charts': available_charts,
    })


# ============================================================
# 模块5.5：跨任务模型对比
# ============================================================

@login_required
def task_compare_page(request):
    """任务对比页面"""
    return render(request, 'task_compare.html')


@login_required
def api_task_compare(request):
    """任务对比 API — 接受多个 task_id，返回对比数据"""
    admin_id = _get_admin_id(request)
    ids_str = request.GET.get('ids', '')
    try:
        task_ids = [int(tid.strip()) for tid in ids_str.split(',') if tid.strip().isdigit()]
    except (ValueError, TypeError):
        return _json_error('参数格式错误')

    if len(task_ids) < 2 or len(task_ids) > 5:
        return _json_error('请选择 2-5 个任务进行对比')

    task_queryset = models.MLTask.objects.filter(
        id__in=task_ids, admin_id=admin_id, status='completed'
    ).select_related('dataset')

    tasks_by_id = {t.id: t for t in task_queryset}
    tasks = [tasks_by_id[task_id] for task_id in task_ids if task_id in tasks_by_id]

    if len(tasks) < 2:
        return _json_error('至少需要 2 个已完成的任务')

    # 检查类型一致性
    task_types = set(t.task_type for t in tasks)
    if len(task_types) > 1:
        return _json_error('只能对比相同类型的任务')

    comparison_data = []
    for t in tasks:
        r = t.results_json or {}
        entry = {
            'task_id': t.id,
            'task_name': t.task_name,
            'task_type': t.task_type,
            'create_time': t.create_time.strftime('%Y-%m-%d %H:%M') if t.create_time else '',
            'dataset_name': t.dataset.file_name if t.dataset else '-',
        }

        if t.task_type == 'supervised':
            entry['best_model'] = r.get('training_results', {}).get('best_model_name', '-')
            entry['best_score'] = r.get('tuning_results', {}).get('best_score')
            entry['n_features'] = len(r.get('feature_selection', {}).get('selected_features', []))
            entry['feature_list'] = r.get('feature_selection', {}).get('selected_features', [])
            entry['models'] = r.get('training_results', {}).get('models', [])
            entry['tuning_params'] = r.get('tuning_results', {}).get('best_params', {})
            entry['n_trials'] = r.get('tuning_results', {}).get('n_trials', 0)
        elif t.task_type in ('unsupervised', 'unsupervised.clustering'):
            cluster_results = r.get('cluster_results', [])
            best = max(cluster_results, key=lambda x: x.get('silhouette_score') or -1) if cluster_results else {}
            entry['best_algo'] = best.get('algorithm', '-')
            entry['best_score'] = best.get('silhouette_score')
            entry['n_clusters'] = best.get('n_clusters')
            entry['all_algorithms'] = cluster_results
        elif t.task_type == 'unsupervised.anomaly':
            anomaly = r.get('anomaly_results', {})
            entry['n_anomalies'] = anomaly.get('n_anomalies')
            entry['n_normal'] = anomaly.get('n_normal')
            entry['anomaly_ratio'] = anomaly.get('anomaly_ratio')
            entry['contamination'] = anomaly.get('contamination')
        elif t.task_type == 'unsupervised.dim_reduction':
            dim = r.get('dim_reduction', {})
            methods = list(dim.keys()) if dim else []
            entry['dim_methods'] = methods
            if 'PCA' in dim:
                cum = dim['PCA'].get('cumulative_variance', [])
                entry['pca_variance'] = cum[-1] if cum else None

        comparison_data.append(entry)

    return _json_success({
        'tasks': comparison_data,
        'task_type': comparison_data[0]['task_type'] if comparison_data else '',
    })


# ============================================================
# 模块6：后台任务查看（仅超级管理员）
# ============================================================

@superadmin_required
def backend_data_page(request):
    """后台任务查看页面"""
    return render(request, 'backend_data.html')


@superadmin_required
def api_backend_tasks(request):
    """后台任务列表 API — 分页返回所有用户的所有 MLTask"""
    tasks = models.MLTask.objects.select_related('admin', 'dataset').order_by('-create_time')
    page_size = _page_size(request)
    page_object = Pagination(request, queryset=tasks, page_size=page_size, page_param='page')
    data = []
    for t in page_object.page_queryset:
        data.append({
            'id': t.id,
            'task_name': t.task_name,
            'task_type': t.task_type,
            'task_type_display': _status_display(t.task_type),
            'status': t.status,
            'status_display': _status_display(t.status),
            'admin_name': t.admin.username if t.admin else '-',
            'dataset_name': t.dataset.file_name if t.dataset else '-',
            'create_time': t.create_time.strftime('%Y-%m-%d %H:%M') if t.create_time else '',
            'update_time': t.update_time.strftime('%Y-%m-%d %H:%M') if t.update_time else '',
            'error_msg': t.error_msg,
        })
    return _json_success({
        'list': data,
        'page': page_object.page,
        'page_size': page_size,
        'total_page': page_object.total_page_count,
        'total': tasks.count(),
    })


@superadmin_required
def api_backend_datasets(request):
    """后台数据集列表 API — 分页返回所有用户的所有 DataSet"""
    datasets = models.DataSet.objects.select_related('admin').order_by('-create_time')
    page_size = _page_size(request)
    page_object = Pagination(request, queryset=datasets, page_size=page_size, page_param='page')
    data = []
    for ds in page_object.page_queryset:
        data.append({
            'id': ds.id,
            'file_name': ds.file_name,
            'row_count': ds.row_count,
            'col_count': ds.col_count,
            'status': ds.status,
            'status_display': _status_display(ds.status),
            'admin_name': ds.admin.username if ds.admin else '-',
            'task_type': _status_display(ds.task_type) if ds.task_type else '-',
            'create_time': ds.create_time.strftime('%Y-%m-%d %H:%M') if ds.create_time else '',
        })
    return _json_success({
        'list': data,
        'page': page_object.page,
        'page_size': page_size,
        'total_page': page_object.total_page_count,
        'total': datasets.count(),
    })
