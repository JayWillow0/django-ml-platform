"""
错误分类器
==========
将 ML 训练异常映射为用户友好的分类、摘要和修复建议。

用法:
    from ml_platform.utils.error_classifier import classify_error
    detail = classify_error(exc)
    # detail = {
    #     'category': 'data_quality',
    #     'category_display': '数据质量问题',
    #     'summary': '...',
    #     'suggestion': '...',
    #     'raw_error': 'MLDataError: ...',
    # }
"""

import traceback


# ============================================================
# 错误类别定义
# ============================================================

ERROR_CATEGORIES = {
    'data_quality': {
        'category_display': '数据质量问题',
        'icon': '📄',
        'color': '#e74c3c',
    },
    'feature_error': {
        'category_display': '特征配置问题',
        'icon': '🔧',
        'color': '#e67e22',
    },
    'training_error': {
        'category_display': '模型训练问题',
        'icon': '⚙️',
        'color': '#9b59b6',
    },
    'tuning_error': {
        'category_display': '超参数调优问题',
        'icon': '🎯',
        'color': '#3498db',
    },
    'system_error': {
        'category_display': '系统错误',
        'icon': '⚠️',
        'color': '#95a5a6',
    },
}


def _build(category, summary, suggestion, raw_error):
    """构造标准化的错误详情字典。"""
    cat_info = ERROR_CATEGORIES.get(category, ERROR_CATEGORIES['system_error'])
    return {
        'category': category,
        'category_display': cat_info['category_display'],
        'icon': cat_info['icon'],
        'summary': summary,
        'suggestion': suggestion,
        'raw_error': raw_error,
    }


# ============================================================
# 各类错误的摘要与建议生成
# ============================================================

def _summarize_data_error(exc, msg):
    """数据相关错误的摘要和建议。"""
    summary = '数据文件存在问题，无法正常加载或处理'
    suggestion = '请检查数据文件格式是否正确，确保为有效的 CSV 或 Excel 文件'

    msg_lower = msg.lower()

    if '编码' in msg or 'encoding' in msg_lower or 'decode' in msg_lower:
        summary = '数据文件编码格式无法自动识别'
        suggestion = '请将文件另存为 UTF-8 编码的 CSV 后重新上传'
    elif '缺失' in msg or 'missing' in msg_lower or 'nan' in msg_lower or '空值' in msg:
        summary = '数据中存在大量缺失值，影响处理流程'
        suggestion = '建议在预处理阶段移除缺失率过高的列，或选择合适的填充策略（中位数/均值）'
    elif '列' in msg or 'column' in msg_lower:
        summary = '数据列存在问题（可能存在空列或格式异常的列）'
        suggestion = '请检查数据中是否有完全为空的列，删除后重新上传'
    elif '空' in msg or 'empty' in msg_lower:
        summary = '数据文件为空或没有有效数据'
        suggestion = '请检查上传的文件是否包含有效数据行'
    elif '类型' in msg or 'dtype' in msg_lower or 'type' in msg_lower:
        summary = '数据中包含无法处理的类型'
        suggestion = '请确保数值列中不包含文字内容，分类列的内容为合法文本'

    return summary, suggestion


def _summarize_feature_error(exc, msg):
    """特征相关错误的摘要和建议。"""
    summary = '特征选择过程中出现问题'
    suggestion = '请检查特征配置是否合理'

    if '无特征' in msg or '没有' in msg or 'no feature' in msg.lower() or '至少' in msg:
        summary = '特征选择后没有可用的特征列'
        suggestion = '请确保选择了足够的特征列，并检查相关性阈值是否过于严格'
    elif '目标' in msg or 'target' in msg.lower():
        summary = '目标列配置有误'
        suggestion = '请确保目标列存在且包含有效数据，目标列不能同时作为特征列'
    elif '相关' in msg or 'correlation' in msg.lower() or 'f_score' in msg.lower() or 'f检验' in msg:
        summary = '特征选择算法无法计算特征得分'
        suggestion = '可能是特征与目标之间没有显著相关性，尝试更换特征选择方法或增加特征数量'

    return summary, suggestion


def _summarize_training_error(exc, msg):
    """训练相关错误的摘要和建议。"""
    summary = '模型训练过程中出现错误'
    suggestion = '请检查数据和参数配置是否合理'

    msg_lower = msg.lower()

    if '收敛' in msg or 'convergence' in msg_lower or 'converge' in msg_lower:
        summary = '模型训练未能收敛'
        suggestion = '建议尝试：增加迭代次数、对数据进行标准化、或降低模型复杂度'
    elif '内存' in msg or 'memory' in msg_lower:
        summary = '训练过程中内存不足'
        suggestion = '建议：减少数据量（采样）、减少特征数量、或关闭其他占用内存的程序'
    elif '维度' in msg or 'dimension' in msg_lower or 'shape' in msg_lower:
        summary = '数据维度不匹配，可能存在空特征或类型冲突'
        suggestion = '请检查特征列中是否有全为空的列，或数据中是否存在混合类型'
    elif 'lazypredict' in msg_lower or 'lazy' in msg_lower:
        summary = '批量模型训练（lazypredict）执行失败'
        suggestion = '可能是数据量过小或特征异常，建议检查数据质量后重试'
    elif 'svc' in msg_lower or 'svr' in msg_lower or '支持向量' in msg:
        summary = 'SVM 模型训练失败（可能由于数据规模或核函数问题）'
        suggestion = 'SVM 对大数据集较敏感，建议减少数据量或跳过 SVM 模型'

    return summary, suggestion


def _summarize_tuning_error(exc, msg):
    """调参相关错误的摘要和建议。"""
    summary = '超参数调优（Optuna）过程中出现错误'
    suggestion = '训练将继续使用默认参数，如需调优请检查参数范围'

    msg_lower = msg.lower()

    if 'trial' in msg_lower or '试验' in msg:
        summary = '调优试验执行失败'
        suggestion = '建议减少调优次数（n_trials）或检查数据是否存在异常值'
    elif 'pruned' in msg_lower or '剪枝' in msg:
        summary = '调优试验被提前终止（性能不佳）'
        suggestion = '这是正常现象，系统会自动跳过表现差的参数组合'

    return summary, suggestion


def _summarize_system_error(exc, msg):
    """系统级错误的摘要和建议。"""
    exc_type = type(exc).__name__

    if exc_type == 'FileNotFoundError' or 'No such file' in msg:
        return '训练所需的数据文件未找到', '数据文件可能已被移动或删除，请重新上传数据集'
    elif exc_type == 'PermissionError' or 'permission' in msg.lower():
        return '文件权限不足，无法读取或写入', '请检查 media 目录的读写权限'
    elif exc_type == 'MemoryError':
        return '系统内存不足', '建议减少数据量或特征数量，关闭其他占用内存的程序后重试'
    elif exc_type == 'TimeoutError' or 'timeout' in msg.lower():
        return '训练任务执行超时', '数据量可能过大，建议采样减少数据量后重试'
    elif exc_type == 'ValueError' and '任务类型' in msg:
        return '任务类型配置无效', '请选择有效的任务类型（监督学习/聚类/异常检测/降维）'

    return '系统发生了未知错误', '请稍后重试，如果问题持续请联系管理员并提供以下技术信息'


# ============================================================
# 主入口
# ============================================================

def classify_error(exc):
    """
    将异常分类为用户友好的错误详情。

    Parameters:
        exc (Exception): 捕获到的异常对象

    Returns:
        dict: {
            'category': str,
            'category_display': str,
            'icon': str,
            'summary': str,
            'suggestion': str,
            'raw_error': str,
        }
    """
    # 延迟导入，避免循环依赖
    from ml_platform.utils.ml_utils import (
        MLError, MLDataError, MLFeatureError,
        MLTrainingError, MLTuningError, MLModelError,
    )

    exc_type = type(exc).__name__
    msg = str(exc)
    raw_error = '{}: {}'.format(exc_type, msg)

    # 也可以附上简短 traceback（最后 3 帧）
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    # 只保留最后几行有意义的信息
    if len(tb_lines) > 4:
        raw_error += '\n' + ''.join(tb_lines[-4:])

    # 1. ML 自定义异常体系
    if isinstance(exc, MLDataError):
        summary, suggestion = _summarize_data_error(exc, msg)
        return _build('data_quality', summary, suggestion, raw_error)

    if isinstance(exc, MLFeatureError):
        summary, suggestion = _summarize_feature_error(exc, msg)
        return _build('feature_error', summary, suggestion, raw_error)

    if isinstance(exc, MLTrainingError):
        summary, suggestion = _summarize_training_error(exc, msg)
        return _build('training_error', summary, suggestion, raw_error)

    if isinstance(exc, MLTuningError):
        summary, suggestion = _summarize_tuning_error(exc, msg)
        return _build('tuning_error', summary, suggestion, raw_error)

    if isinstance(exc, MLModelError):
        return _build('system_error', '模型保存或加载失败',
                      '请检查存储空间是否充足，或联系管理员', raw_error)

    # 2. Python 内置异常（按常见模式匹配）
    if isinstance(exc, MemoryError):
        return _build('system_error', *_summarize_system_error(exc, msg), raw_error)

    if isinstance(exc, FileNotFoundError):
        return _build('system_error', *_summarize_system_error(exc, msg), raw_error)

    if isinstance(exc, PermissionError):
        return _build('system_error', *_summarize_system_error(exc, msg), raw_error)

    # 3. 通用 MLError（兜底）
    if isinstance(exc, MLError):
        return _build('system_error', 'ML 流水线执行失败',
                      '请检查数据和配置后重试', raw_error)

    # 4. 完全未预期的异常 — 按消息关键词猜测
    msg_lower = msg.lower()
    if any(kw in msg_lower for kw in ['data', '数据', '文件', 'file', '编码', '列', 'column']):
        summary, suggestion = _summarize_data_error(exc, msg)
        return _build('data_quality', summary, suggestion, raw_error)

    if any(kw in msg_lower for kw in ['feature', '特征', '目标', 'target']):
        summary, suggestion = _summarize_feature_error(exc, msg)
        return _build('feature_error', summary, suggestion, raw_error)

    if any(kw in msg_lower for kw in ['memory', '内存', 'convergence', '收敛']):
        summary, suggestion = _summarize_training_error(exc, msg)
        return _build('training_error', summary, suggestion, raw_error)

    # 5. 最终兜底
    summary, suggestion = _summarize_system_error(exc, msg)
    return _build('system_error', summary, suggestion, raw_error)
