"""
训练进度回调
============
Celery 任务通过此回调将训练阶段信息实时写入 MLTask.progress_info，
前端轮询 status API 即可获得进度反馈。

用法:
    from ml_platform.utils.progress import ProgressCallback
    cb = ProgressCallback(task_id=42, pipeline_type='supervised')
    cb('data_import', message='加载CSV文件...')
    cb('preprocessing', message='预处理 15 个特征')
"""

from datetime import datetime


# ============================================================
# 各流水线的阶段定义
# ============================================================

STAGE_DEFINITIONS = {
    'supervised': [
        ('data_import', '数据导入'),
        ('preprocessing', '数据预处理'),
        ('correlation', '相关性分析'),
        ('feature_selection', '特征选择'),
        ('model_training', '模型训练'),
        ('hyperparameter_tuning', '超参数调优'),
        ('model_saving', '模型保存'),
    ],
    'unsupervised.clustering': [
        ('data_import', '数据导入'),
        ('preprocessing', '数据预处理'),
        ('clustering', '聚类分析'),
        ('dim_reduction', '降维可视化'),
    ],
    'unsupervised.anomaly': [
        ('data_import', '数据导入'),
        ('preprocessing', '数据预处理'),
        ('anomaly_detection', '异常检测'),
        ('dim_reduction', '降维可视化'),
    ],
    'unsupervised.dim_reduction': [
        ('data_import', '数据导入'),
        ('preprocessing', '数据预处理'),
        ('dim_reduction', '降维计算'),
    ],
    # 历史兼容：旧数据中可能存在 task_type='unsupervised'
    'unsupervised': [
        ('data_import', '数据导入'),
        ('preprocessing', '数据预处理'),
        ('clustering', '聚类分析'),
        ('dim_reduction', '降维可视化'),
    ],
}

MAX_LOG_LINES = 20


class ProgressCallback:
    """
    可调用对象，ML 流水线在每个阶段调用以报告进度。

    进度信息直接写入 MLTask.progress_info（JSONField），
    使用 objects.filter().update() 避免覆盖其他字段。
    """

    def __init__(self, task_id, pipeline_type, max_log_lines=MAX_LOG_LINES):
        """
        Parameters:
            task_id: MLTask 主键
            pipeline_type: 流水线类型（对应 task_type）
            max_log_lines: 保留的最大日志条数
        """
        self.task_id = task_id
        self.stages = STAGE_DEFINITIONS.get(pipeline_type, [])
        self.max_log_lines = max_log_lines
        self.log_lines = []
        self._current_stage = None
        self._current_index = 0

    def __call__(self, stage, message=''):
        """
        报告当前阶段进度。

        Parameters:
            stage: 阶段标识（如 'data_import'）
            message: 人类可读的日志消息
        """
        # 查找阶段索引
        stage_index = 0
        total_stages = len(self.stages)
        for i, (key, _) in enumerate(self.stages):
            if key == stage:
                stage_index = i + 1
                break

        if stage_index == 0:
            # 未注册的阶段，仍记录日志
            stage_index = self._current_index
        else:
            self._current_index = stage_index

        self._current_stage = stage

        # 计算百分比
        if total_stages > 0:
            percent = int((stage_index / total_stages) * 100)
        else:
            percent = 0

        # 追加日志
        now = datetime.now().strftime('%H:%M:%S')
        stage_display = ''
        for key, display in self.stages:
            if key == stage:
                stage_display = display
                break

        self.log_lines.append({
            'time': now,
            'msg': message or stage_display or stage,
        })
        # 保留最近 N 条
        if len(self.log_lines) > self.max_log_lines:
            self.log_lines = self.log_lines[-self.max_log_lines:]

        # 持久化到数据库
        self._persist(stage, stage_display, percent, stage_index, total_stages)

    def _persist(self, stage, stage_display, percent, stage_index, total_stages):
        """将进度信息写入 MLTask.progress_info。"""
        from ml_platform.models import MLTask

        progress_info = {
            'stage': stage,
            'stage_display': stage_display,
            'percent': percent,
            'stage_index': stage_index,
            'total_stages': total_stages,
            'log_lines': list(self.log_lines),
        }

        MLTask.objects.filter(id=self.task_id).update(
            progress_info=progress_info,
        )
