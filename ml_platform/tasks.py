"""
Celery 异步任务定义
====================
训练任务在此执行，view 层只负责创建任务记录和返回 task_id。
"""
import os
import traceback
import logging

from celery import shared_task
from django.conf import settings

logger = logging.getLogger('ml_platform.tasks')


class TaskCancelledException(Exception):
    """用户取消了正在运行的任务"""
    pass


def _check_cancelled(task_id):
    """检查任务是否已被用户取消，若已取消则抛出异常"""
    from ml_platform import models
    task = models.MLTask.objects.filter(id=task_id).first()
    if task and task.status == 'cancelled':
        raise TaskCancelledException(f'Task {task_id} was cancelled by user')


def _abs_path(relative_path):
    """相对路径转绝对路径"""
    return os.path.join(settings.BASE_DIR, relative_path)


def _to_serializable(obj):
    """将 numpy/pandas 类型转为严格 JSON 可接受的 Python 原生类型。"""
    import math
    import numpy as np
    import pandas as pd

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    elif isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    elif isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    elif isinstance(obj, pd.Series):
        return _to_serializable(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return _to_serializable(obj.to_dict(orient='records'))
    elif isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@shared_task(bind=True)
def run_ml_pipeline_task(self, task_id, params=None):
    """
    异步执行 ML 训练流水线。

    Parameters:
        task_id: MLTask 记录的 ID
        params: 训练参数字典（来自前端配置面板）
    """
    from ml_platform import models
    from ml_platform.utils.ml_utils import MLPipeline, Visualizer

    if params is None:
        params = {}

    logger.info(f"[Celery] 开始执行训练任务 task_id={task_id}, params={params}")

    task = models.MLTask.objects.filter(id=task_id).first()
    if not task:
        logger.error(f"[Celery] 任务不存在 task_id={task_id}")
        return

    ds = task.dataset
    task_type = task.task_type

    try:
        # 更新 Celery task ID 以便追踪
        task.celery_task_id = self.request.id
        task.status = 'running'
        task.save()

        _check_cancelled(task_id)

        pipeline = MLPipeline()
        abs_data_path = _abs_path(ds.file_path)

        # 创建进度回调
        from ml_platform.utils.progress import ProgressCallback
        progress_cb = ProgressCallback(task_id=task_id, pipeline_type=task_type)

        # 模型保存路径
        admin = task.admin
        admin_name = admin.username if admin else 'unknown'
        model_dir = os.path.join('media', 'models', admin_name)
        abs_model_dir = _abs_path(model_dir)
        os.makedirs(abs_model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f'task_{task_id}_model')

        _check_cancelled(task_id)

        if task_type == 'supervised':
            result = pipeline.run_supervised(
                file_path=abs_data_path,
                target_col=ds.target_col,
                feature_cols=ds.feature_cols or None,
                task_type=ds.task_type or 'auto',
                test_size=params.get('test_size', 0.2),
                feature_selection_method=params.get('feature_selection_method', 'f_test'),
                correlation_threshold=params.get('correlation_threshold', params.get('mi_threshold', 0.05)),
                n_features=params.get('n_features'),
                tune=True,
                n_trials=params.get('n_trials', 10),
                cv_folds=params.get('cv_folds', 3),
                random_state=params.get('random_state', 42),
                save_path=_abs_path(model_save_path),
                progress_callback=progress_cb,
                numeric_strategy=ds.fill_strategy or 'median',
                scale_numeric=ds.scale_numeric,
                scaler_type=(
                    params.get('scaler_type')
                    if params.get('scaler_type') in ('standard', 'minmax')
                    else 'standard'
                ),
            )
        elif task_type in ('unsupervised', 'unsupervised.clustering'):
            k_range = params.get('n_clusters_range', [2, 10])
            result = pipeline.run_unsupervised(
                file_path=abs_data_path,
                feature_cols=ds.feature_cols or None,
                n_clusters=params.get('n_clusters', 3),
                n_clusters_range=(k_range[0], k_range[1]) if len(k_range) == 2 else (2, 10),
                dim_reduction=params.get('dim_reduction', True),
                random_state=params.get('random_state', 42),
                progress_callback=progress_cb,
            )
            model_save_path = None
        elif task_type == 'unsupervised.anomaly':
            result = pipeline.run_anomaly(
                file_path=abs_data_path,
                feature_cols=ds.feature_cols or None,
                contamination=params.get('contamination', 0.1),
                random_state=params.get('random_state', 42),
                progress_callback=progress_cb,
            )
            model_save_path = None
        elif task_type == 'unsupervised.dim_reduction':
            result = pipeline.run_dim_reduction(
                file_path=abs_data_path,
                feature_cols=ds.feature_cols or None,
                random_state=params.get('random_state', 42),
                progress_callback=progress_cb,
            )
            model_save_path = None
        else:
            raise ValueError(f"未知的任务类型: {task_type}")

        _check_cancelled(task_id)

        # 生成可视化图表（PNG 保存到 media/charts/<task_id>/）
        charts_dir = os.path.join('media', 'charts', str(task_id))
        viz = Visualizer(output_dir=_abs_path(charts_dir))
        if task_type == 'supervised' and result.get('success'):
            charts = viz.generate_supervised_charts(result)
        elif task_type in ('unsupervised', 'unsupervised.clustering') and result.get('success'):
            charts = viz.generate_unsupervised_charts(result)
        elif task_type == 'unsupervised.anomaly' and result.get('success'):
            charts = viz.generate_anomaly_charts(result)
        elif task_type == 'unsupervised.dim_reduction' and result.get('success'):
            charts = viz.generate_dim_reduction_charts(result)
        else:
            charts = {}

        # 序列化结果
        serializable_result = _to_serializable(result)
        serializable_charts = {}
        for chart_name, chart_data in charts.items():
            serializable_charts[chart_name] = {
                'echarts': _to_serializable(chart_data.get('echarts', {})),
                'png_path': chart_data.get('png_path'),
            }

        # 更新任务状态
        task.results_json = serializable_result
        task.charts_json = serializable_charts
        task.status = 'completed'
        if model_save_path:
            task.model_path = model_save_path.replace('\\', '/') + '.joblib'
        task.save()

        # 更新数据集状态
        ds.status = 'completed'
        ds.save()

        logger.info(f"[Celery] 训练完成 task_id={task_id}")

    except TaskCancelledException:
        logger.info(f"[Celery] 任务被用户取消 task_id={task_id}")
        # status 已由 cancel API 设为 'cancelled'，此处仅回退数据集状态
        ds.status = 'preprocessed'
        ds.save()

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[Celery] 训练失败 task_id={task_id}: {error_msg}")
        traceback.print_exc()

        from ml_platform.utils.error_classifier import classify_error
        error_detail = classify_error(e)
        raw_error = error_detail.pop('raw_error')

        task.status = 'failed'
        task.error_msg = raw_error
        task.error_detail = error_detail
        # 结果保存失败时，task 内存中可能仍保留无法写入 MySQL JSON 的值。
        # 仅更新错误字段，避免把同一份非法 results_json 再次写入数据库。
        task.save(update_fields=['status', 'error_msg', 'error_detail'])

        # 训练失败时将数据集状态回退到 preprocessed，允许重新训练
        ds.status = 'preprocessed'
        ds.save()
