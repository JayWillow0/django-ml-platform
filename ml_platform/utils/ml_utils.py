"""
ML算法平台核心工具模块
======================
提供数据导入、预处理、特征选择、监督/无监督学习、模型调参、模型管理的完整流水线。
不依赖 Django，可独立运行和测试。

使用方式:
    from ml_platform.utils.ml_utils import MLPipeline

    pipeline = MLPipeline()
    result = pipeline.run_supervised('data.csv', target_col='target')
    result = pipeline.run_unsupervised('data.csv')
    pipeline.predict_new('model.joblib', 'new_data.csv', 'output.csv')
"""

import os
import json
import logging
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from scipy import stats as scipy_stats

from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    f_regression, f_classif, mutual_info_regression, mutual_info_classif,
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    silhouette_score, calinski_harabasz_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import all_estimators
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 可选依赖：lazypredict
try:
    from lazypredict.Supervised import LazyClassifier, LazyRegressor
except ImportError:
    LazyClassifier = None
    LazyRegressor = None

# 可选依赖：optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

logger = logging.getLogger('ml_utils')

__all__ = [
    'DataImporter', 'DataPreprocessor', 'FeatureSelector',
    'StatisticalFeatureSelector',
    'SupervisedTrainer', 'UnsupervisedTrainer',
    'ModelTuner', 'ModelManager', 'MLPipeline', 'Visualizer',
    'MLError', 'MLDataError', 'MLFeatureError',
    'MLTrainingError', 'MLTuningError', 'MLModelError',
]


# ============================================================
# 异常体系
# ============================================================

class MLError(Exception):
    """ML模块基础异常"""
    pass


class MLDataError(MLError):
    """数据导入/预处理错误"""
    pass


class MLFeatureError(MLError):
    """特征选择错误"""
    pass


class MLTrainingError(MLError):
    """模型训练错误"""
    pass


class MLTuningError(MLError):
    """Optuna调参错误"""
    pass


class MLModelError(MLError):
    """模型保存/加载/预测错误"""
    pass


# ============================================================
# 常量
# ============================================================

VALID_TASK_TYPES = ('regression', 'classification', 'unsupervised')

SUPPORTED_FILE_EXTENSIONS = ('.csv', '.xlsx', '.xls')

CHART_BLUE = '#1A8BFB'
CHART_GREEN = '#36BC8A'
CHART_DARK = '#141818'
CHART_MID_GRAY = '#C3C8D0'
CHART_LIGHT_GRAY = '#E1E4E7'
CHART_SOFT_GRAY = '#D8DCE0'
CHART_BRAND_PALETTE = [CHART_BLUE, CHART_GREEN, CHART_MID_GRAY, CHART_LIGHT_GRAY, CHART_SOFT_GRAY]
CHART_OVERFLOW = CHART_LIGHT_GRAY
CHART_BLUE_AREA = 'rgba(26,139,251,0.15)'


def _chart_bar_colors(count: int) -> List[str]:
    return [
        CHART_BRAND_PALETTE[i] if i < len(CHART_BRAND_PALETTE) else CHART_OVERFLOW
        for i in range(count)
    ]


def _ranked_bar_colors(values: List[float]) -> List[str]:
    ranked = sorted(
        enumerate(values),
        key=lambda item: item[1] if item[1] is not None else float('-inf'),
        reverse=True,
    )
    colors = [CHART_OVERFLOW] * len(values)
    for rank, (index, _) in enumerate(ranked):
        colors[index] = CHART_BRAND_PALETTE[rank] if rank < len(CHART_BRAND_PALETTE) else CHART_OVERFLOW
    return colors

# 构建sklearn估计器名称→类的映射（惰性初始化）
_ESTIMATOR_MAP = None


def _get_estimator_map() -> Dict[str, type]:
    """获取sklearn所有估计器的名称→类映射（惰性初始化）"""
    global _ESTIMATOR_MAP
    if _ESTIMATOR_MAP is None:
        _ESTIMATOR_MAP = {}
        for name, cls in all_estimators():
            _ESTIMATOR_MAP[name] = cls
    return _ESTIMATOR_MAP


# ============================================================
# 工具函数
# ============================================================

def _detect_task_type(y: pd.Series) -> str:
    """
    根据目标列自动判断任务类型。

    规则：
    - 字符串/布尔型 → 分类
    - 整数且唯一值 ≤ 20 或唯一值占比 < 5% → 分类
    - 其他 → 回归
    """
    if y.dtype == 'object' or pd.api.types.is_bool_dtype(y):
        return 'classification'
    if pd.api.types.is_string_dtype(y):
        return 'classification'

    n_unique = y.nunique()
    n_total = len(y)

    if pd.api.types.is_integer_dtype(y):
        if n_unique <= 20 or (n_total > 0 and n_unique / n_total < 0.05):
            return 'classification'

    if n_total > 0 and n_unique / n_total > 0.1:
        return 'regression'

    if n_unique <= 20:
        return 'classification'

    return 'regression'


def _to_serializable(obj: Any) -> Any:
    """将numpy/pandas类型转为Python原生类型，便于JSON序列化。

    NaN/Inf 会被转成 None：MySQL 的 JSON 列严格遵循 JSON 标准，不接受
    NaN/Infinity，直接插入会报 (3140, 'Invalid JSON text')。pandas to_dict
    后的缺失值是 Python float('nan')，必须在此兜底。
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return _safe_float(float(obj))
    elif isinstance(obj, float):
        # Python 原生 float 也可能含 NaN/Inf（pandas to_dict 的缺失值即此情况）
        return _safe_float(obj)
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


def _safe_float(val: Any) -> Optional[float]:
    """安全地将值转为float，NaN/Inf/转换失败均返回None。"""
    try:
        if val is None:
            return None
        f = float(val)
        if not np.isfinite(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def compute_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    计算数据集的综合统计信息，用于前端数据探索展示。

    Parameters:
        df: pandas DataFrame

    Returns:
        dict: {
            'numeric': {col: {mean, std, min, 25%, 50%, 75%, max, nunique}},
            'categorical': {col: {nunique, top, top_freq, values}},
            'correlation_matrix': {columns: [...], matrix: [[...]]},
            'missing_pattern': {total_cells, total_missing, missing_ratio, by_column},
        }
    """
    stats = {
        'numeric': {},
        'categorical': {},
        'correlation_matrix': {},
        'missing_pattern': {},
    }

    # 数值列描述性统计
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        for col in numeric_cols:
            col_desc = desc.get(col, {})
            stats['numeric'][col] = {
                'mean': _safe_float(col_desc.get('mean')),
                'std': _safe_float(col_desc.get('std')),
                'min': _safe_float(col_desc.get('min')),
                '25%': _safe_float(col_desc.get('25%')),
                '50%': _safe_float(col_desc.get('50%')),
                '75%': _safe_float(col_desc.get('75%')),
                'max': _safe_float(col_desc.get('max')),
                'nunique': int(df[col].nunique()),
            }

    # 分类列统计
    cat_cols = list(df.select_dtypes(include=['object', 'string', 'bool']).columns)
    for col in cat_cols:
        try:
            vc = df[col].value_counts().head(10)
            stats['categorical'][col] = {
                'nunique': int(df[col].nunique()),
                'top': str(vc.index[0]) if len(vc) > 0 else '',
                'top_freq': int(vc.iloc[0]) if len(vc) > 0 else 0,
                'values': {str(k): int(v) for k, v in vc.items()},
            }
        except Exception:
            stats['categorical'][col] = {
                'nunique': int(df[col].nunique()),
                'top': '', 'top_freq': 0, 'values': {},
            }

    # 相关矩阵（数值列，≤50 列时计算）
    if len(numeric_cols) > 1 and len(numeric_cols) <= 50:
        try:
            corr = df[numeric_cols].corr()
            # 将 NaN 替换为 0（全常量列会产出 NaN）
            corr = corr.fillna(0)
            stats['correlation_matrix'] = {
                'columns': numeric_cols,
                'matrix': _to_serializable(corr.values.tolist()),
            }
        except Exception:
            pass

    # 缺失值模式
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum()
    stats['missing_pattern'] = {
        'total_cells': total_cells,
        'total_missing': total_missing,
        'missing_ratio': round(total_missing / max(total_cells, 1), 4),
        'by_column': {
            col: int(count)
            for col, count in missing_by_col.items() if count > 0
        },
    }

    return stats


# ============================================================
# DataImporter — 数据导入
# ============================================================

class DataImporter:
    """
    数据导入器，支持CSV和XLSX格式。

    用法:
        importer = DataImporter()
        result = importer.load_data('data.csv')
        df = result['data']
    """

    def load_data(self, file_path: str, **kwargs) -> Dict:
        """
        加载CSV或XLSX文件为DataFrame。

        Parameters:
            file_path: 文件路径（.csv / .xlsx / .xls）
            **kwargs: 传递给pd.read_csv或pd.read_excel的额外参数

        Returns:
            {
                'success': True,
                'data': pd.DataFrame,
                'metadata': {row_count, column_count, columns, dtypes, missing_values, memory_mb},
                'preview': list[dict],  # 前5行
            }

        Raises:
            MLDataError: 文件不存在、格式不支持、文件为空等
        """
        if not file_path or not isinstance(file_path, str):
            raise MLDataError("文件路径不能为空")

        if not os.path.exists(file_path):
            raise MLDataError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            raise MLDataError(
                f"不支持的文件格式: {ext}，仅支持 {SUPPORTED_FILE_EXTENSIONS}"
            )

        try:
            if ext == '.csv':
                # 尝试自动检测编码
                encoding = kwargs.pop('encoding', None)
                if encoding is None:
                    encoding = self._detect_encoding(file_path)
                data = pd.read_csv(file_path, encoding=encoding, **kwargs)
            else:
                sheet_name = kwargs.pop('sheet_name', 0)
                data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        except UnicodeDecodeError:
            # 回退到其他编码
            try:
                data = pd.read_csv(file_path, encoding='gbk', **kwargs)
            except Exception as e:
                raise MLDataError(f"文件编码读取失败: {e}")
        except Exception as e:
            raise MLDataError(f"文件读取失败: {e}")

        if data.empty:
            raise MLDataError("文件内容为空")

        if data.shape[1] == 0:
            raise MLDataError("文件没有任何列")

        metadata = {
            'row_count': int(data.shape[0]),
            'column_count': int(data.shape[1]),
            'columns': data.columns.tolist(),
            'dtypes': {col: str(data[col].dtype) for col in data.columns},
            'missing_values': {col: int(data[col].isna().sum()) for col in data.columns},
            'memory_mb': round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

        preview = data.head(5).to_dict(orient='records')

        logger.info(
            f"数据加载成功: {metadata['row_count']}行 x {metadata['column_count']}列"
        )

        return {
            'success': True,
            'data': data,
            'metadata': metadata,
            'preview': preview,
        }

    @staticmethod
    def _detect_encoding(file_path: str) -> str:
        """尝试检测CSV文件编码。"""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(10000)
            # 尝试常见编码
            for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']:
                try:
                    raw.decode(enc)
                    return enc
                except (UnicodeDecodeError, LookupError):
                    continue
        except Exception:
            pass
        return 'utf-8'


# ============================================================
# DataPreprocessor — 数据预处理
# ============================================================

def _infer_feature_types(
    X: pd.DataFrame,
    max_cat_unique: int = 10,
) -> Tuple[List[str], List[str]]:
    """仅根据特征数据识别数值列和分类列。"""
    numeric_features = []
    categorical_features = []
    for col in X.columns:
        if (
            X[col].dtype == 'object'
            or pd.api.types.is_string_dtype(X[col])
            or pd.api.types.is_bool_dtype(X[col])
        ):
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            if pd.api.types.is_integer_dtype(X[col]) and X[col].nunique() <= max_cat_unique:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            categorical_features.append(col)
    return numeric_features, categorical_features


def _build_preprocessing_transformer(
    X: pd.DataFrame,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent',
    scale_numeric: bool = True,
    scaler_type: str = 'standard',
    max_cat_unique: int = 10,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """构建可被 sklearn clone 的预处理器；统计量在 fit 时才学习。"""
    if scaler_type not in ('standard', 'minmax'):
        raise MLDataError(f"不支持的缩放类型: {scaler_type}")

    numeric_features, categorical_features = _infer_feature_types(
        X, max_cat_unique=max_cat_unique,
    )
    transformers = []

    if numeric_features:
        num_steps = [('imputer', SimpleImputer(strategy=numeric_strategy))]
        if scale_numeric:
            scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
            num_steps.append(('scaler', scaler))
        transformers.append(('num', Pipeline(num_steps), numeric_features))

    if categorical_features:
        cat_steps = [
            ('imputer', SimpleImputer(
                strategy=categorical_strategy,
                fill_value='missing',
            )),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
            )),
        ]
        transformers.append(('cat', Pipeline(cat_steps), categorical_features))

    if not transformers:
        raise MLDataError("没有可处理的特征列")

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform='pandas')
    return transformer, numeric_features, categorical_features


class DataPreprocessor:
    """
    数据预处理器：缺失值填充、分类编码、数值缩放。

    用法:
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(df, target_col='target', task_type='regression')
        X = result['X']
        y = result['y']

        # 新数据预测时
        X_new = preprocessor.transform_new(new_df)
    """

    def __init__(self):
        self._pipeline = None               # 拟合后的ColumnTransformer
        self._feature_names_in = None       # 原始特征列名
        self._numeric_features = None       # 数值特征列名
        self._categorical_features = None   # 分类特征列名
        self._target_name = None            # 目标列名
        self._label_encoder = None          # 分类任务的LabelEncoder
        self._task_type = None              # 任务类型
        self._scaler_type = None            # 缩放器类型

    @property
    def pipeline(self):
        """获取拟合后的预处理pipeline（用于保存）"""
        return self._pipeline

    @property
    def label_encoder(self):
        """获取LabelEncoder（分类任务用于解码预测结果）"""
        return self._label_encoder

    @property
    def feature_names_in(self):
        """获取原始输入特征列名"""
        return self._feature_names_in

    @property
    def target_name(self):
        """获取目标列名"""
        return self._target_name

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_col: Optional[str] = None,
        task_type: str = 'regression',
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        scale_numeric: bool = True,
        scaler_type: str = 'standard',
        max_cat_unique: int = 10,
    ) -> Dict:
        """
        预处理数据：缺失值填充 → 分类编码 → 数值缩放。

        Parameters:
            data: 原始DataFrame
            target_col: 目标列名（无监督学习时为None）
            task_type: 'regression' / 'classification' / 'unsupervised'
            numeric_strategy: 数值缺失值填充策略 ('mean'/'median'/'constant')
            categorical_strategy: 分类缺失值填充策略 ('most_frequent'/'constant')
            scale_numeric: 是否对数值特征缩放
            scaler_type: 'standard'(StandardScaler) 或 'minmax'(MinMaxScaler)
            max_cat_unique: 唯一值 ≤ 此数的非对象类型列视为分类

        Returns:
            {
                'success': True,
                'X': pd.DataFrame,
                'y': pd.Series or None,
                'feature_names': list[str],
                'numeric_features': list[str],
                'categorical_features': list[str],
                'preprocessing_info': dict,
            }
        """
        if data is None or data.empty:
            raise MLDataError("输入数据为空")

        # 分离特征和目标
        self._target_name = target_col
        self._task_type = task_type

        if target_col and target_col in data.columns:
            y = data[target_col].copy()
            X = data.drop(columns=[target_col])
        else:
            y = None
            X = data.copy()

        if X.shape[1] == 0:
            raise MLDataError("没有特征列可用（排除目标列后为空）")

        self._feature_names_in = X.columns.tolist()
        self._scaler_type = scaler_type

        # 记录哪些列有缺失值
        imputation_needed = X.columns[X.isna().any()].tolist()

        (
            self._pipeline,
            self._numeric_features,
            self._categorical_features,
        ) = _build_preprocessing_transformer(
            X,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            scale_numeric=scale_numeric,
            scaler_type=scaler_type,
            max_cat_unique=max_cat_unique,
        )

        # 拟合并转换
        X_transformed = self._pipeline.fit_transform(X)

        # 确保列名是字符串
        X_transformed.columns = [str(c) for c in X_transformed.columns]
        feature_names = X_transformed.columns.tolist()

        # 处理目标变量编码
        target_encoded = False
        if y is not None and task_type == 'classification':
            if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
                self._label_encoder = LabelEncoder()
                y = pd.Series(
                    self._label_encoder.fit_transform(y),
                    index=y.index,
                    name=target_col,
                )
                target_encoded = True

        preprocessing_info = {
            'imputation_applied': imputation_needed,
            'encoding_applied': self._categorical_features,
            'scaling_applied': scale_numeric and len(self._numeric_features) > 0,
            'scaler_type': scaler_type,
            'target_encoded': target_encoded,
            'numeric_strategy': numeric_strategy,
            'categorical_strategy': categorical_strategy,
        }

        logger.info(
            f"预处理完成: {len(self._numeric_features)}个数值特征, "
            f"{len(self._categorical_features)}个分类特征 → {len(feature_names)}个特征"
        )

        return {
            'success': True,
            'X': X_transformed,
            'y': y,
            'feature_names': feature_names,
            'numeric_features': self._numeric_features,
            'categorical_features': self._categorical_features,
            'preprocessing_info': preprocessing_info,
        }

    def transform_new(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对新数据应用已拟合的预处理pipeline。

        Parameters:
            data: 新数据的DataFrame

        Returns:
            转换后的DataFrame

        Raises:
            MLModelError: pipeline未拟合
        """
        if self._pipeline is None:
            raise MLModelError("预处理器未拟合，请先调用fit_transform()")

        # 只选择训练时的特征列
        available_cols = [c for c in self._feature_names_in if c in data.columns]
        missing_cols = [c for c in self._feature_names_in if c not in data.columns]
        if missing_cols:
            logger.warning(f"新数据缺少列: {missing_cols}，将填充默认值")
            for col in missing_cols:
                data[col] = np.nan

        X = data[self._feature_names_in]
        X_transformed = self._pipeline.transform(X)
        X_transformed.columns = [str(c) for c in X_transformed.columns]
        return X_transformed

    def get_bundle(self) -> Dict:
        """获取预处理器完整信息，用于保存。"""
        return {
            'pipeline': self._pipeline,
            'feature_names_in': self._feature_names_in,
            'numeric_features': self._numeric_features,
            'categorical_features': self._categorical_features,
            'target_name': self._target_name,
            'label_encoder': self._label_encoder,
            'task_type': self._task_type,
            'scaler_type': self._scaler_type,
        }

    def load_bundle(self, bundle: Dict):
        """从保存的信息恢复预处理器状态。"""
        self._pipeline = bundle.get('pipeline')
        self._feature_names_in = bundle.get('feature_names_in')
        self._numeric_features = bundle.get('numeric_features')
        self._categorical_features = bundle.get('categorical_features')
        self._target_name = bundle.get('target_name')
        self._label_encoder = bundle.get('label_encoder')
        self._task_type = bundle.get('task_type')
        self._scaler_type = bundle.get('scaler_type')

    def set_fitted_pipeline(
        self,
        pipeline: ColumnTransformer,
        feature_names_in: List[str],
        numeric_features: List[str],
        categorical_features: List[str],
        target_name: Optional[str],
        task_type: str,
        scaler_type: str,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        """接收监督学习 Pipeline 中已经拟合的预处理器，用于保存与预测。"""
        self._pipeline = pipeline
        self._feature_names_in = list(feature_names_in)
        self._numeric_features = list(numeric_features)
        self._categorical_features = list(categorical_features)
        self._target_name = target_name
        self._task_type = task_type
        self._scaler_type = scaler_type
        self._label_encoder = label_encoder


# ============================================================
# FeatureSelector — 相关性分析与特征选择
# ============================================================

class FeatureSelector:
    """
    特征选择器：相关性分析、特征筛选。

    用法:
        selector = FeatureSelector()
        corr = selector.analyze_correlation(X, y, 'regression')
        sel = selector.select_features(X, y, 'regression', threshold=0.1)
    """

    def analyze_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
    ) -> Dict:
        """
        计算特征与目标之间的相关性/统计检验。

        Parameters:
            X: 特征DataFrame（预处理后的数值型）
            y: 目标Series
            task_type: 'regression' 或 'classification'

        Returns:
            {
                'success': True,
                'correlations': [{'feature', 'score', 'p_value', 'method'}, ...],
                'heatmap_data': {'features': [...], 'scores': [...]},
            }
        """
        if X.shape[0] != len(y):
            raise MLFeatureError("特征和目标的样本数不一致")
        if X.shape[1] == 0:
            raise MLFeatureError("没有特征列可分析")

        correlations = []

        if task_type == 'regression':
            correlations = self._regression_correlation(X, y)
        else:
            correlations = self._classification_correlation(X, y)

        # 按绝对值降序排列
        correlations.sort(key=lambda x: abs(x['score']) if x['score'] is not None else 0, reverse=True)

        heatmap_data = {
            'features': [c['feature'] for c in correlations],
            'scores': [_safe_float(c['score']) for c in correlations],
        }

        logger.info(f"相关性分析完成: {len(correlations)}个特征")

        return {
            'success': True,
            'correlations': correlations,
            'heatmap_data': heatmap_data,
        }

    def _regression_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """回归任务的相关性分析：Pearson + mutual_info"""
        results = []

        # Pearson相关系数 + F检验
        f_scores, f_pvalues = f_regression(X.fillna(0), y)

        # 互信息
        mi_scores = mutual_info_regression(X.fillna(0), y, random_state=42)

        for i, col in enumerate(X.columns):
            # Pearson
            try:
                pearson_r, pearson_p = scipy_stats.pearsonr(X[col].fillna(0), y)
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan

            results.append({
                'feature': col,
                'pearson_r': _safe_float(pearson_r),
                'f_score': _safe_float(f_scores[i]),
                'p_value': _safe_float(f_pvalues[i]),
                'mutual_info': _safe_float(mi_scores[i]),
                'score': _safe_float(f_scores[i]),
                'method': 'f_regression',
            })

        return results

    def _classification_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """分类任务的相关性分析：ANOVA F + mutual_info"""
        results = []

        # ANOVA F检验
        f_scores, f_pvalues = f_classif(X.fillna(0), y)

        # 互信息
        mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)

        for i, col in enumerate(X.columns):
            results.append({
                'feature': col,
                'f_score': _safe_float(f_scores[i]),
                'p_value': _safe_float(f_pvalues[i]),
                'mutual_info': _safe_float(mi_scores[i]),
                'score': _safe_float(f_scores[i]),
                'method': 'f_classif',
            })

        return results

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        method: str = 'f_test',
        threshold: float = 0.05,
        k: Optional[int] = None,
    ) -> Dict:
        """
        基于统计检验选择特征子集。

        Parameters:
            X: 特征DataFrame
            y: 目标Series
            task_type: 'regression' / 'classification'
            method: 'f_test'（F检验p值阈值）/ 'kbest'（Top-K）/ 'mutual_info'
            threshold: p值阈值（method='f_test'时）或得分阈值
            k: 保留的特征数量（method='kbest'时）

        Returns:
            {
                'success': True,
                'selected_features': list[str],
                'removed_features': list[str],
                'selection_scores': dict,
                'X_selected': pd.DataFrame,
            }
        """
        if X.shape[0] != len(y):
            raise MLFeatureError("特征和目标的样本数不一致")

        # 计算统计量
        if task_type == 'regression':
            scores, pvalues = f_regression(X.fillna(0), y)
            mi = mutual_info_regression(X.fillna(0), y, random_state=42)
        else:
            scores, pvalues = f_classif(X.fillna(0), y)
            mi = mutual_info_classif(X.fillna(0), y, random_state=42)

        selection_scores = {}
        for i, col in enumerate(X.columns):
            selection_scores[col] = {
                'f_score': _safe_float(scores[i]),
                'p_value': _safe_float(pvalues[i]),
                'mutual_info': _safe_float(mi[i]),
            }

        # 根据方法选择特征
        if method == 'kbest' and k is not None:
            # 按F得分排序，取Top-K
            sorted_features = sorted(
                X.columns,
                key=lambda c: scores[list(X.columns).index(c)] if not np.isnan(scores[list(X.columns).index(c)]) else 0,
                reverse=True,
            )
            selected = sorted_features[:min(k, len(sorted_features))]
        elif method == 'mutual_info':
            # 互信息得分阈值
            selected = [
                col for col in X.columns
                if mi[list(X.columns).index(col)] >= threshold
            ]
            if not selected:
                # 如果没有满足条件的，取互信息最高的
                sorted_features = sorted(
                    X.columns,
                    key=lambda c: mi[list(X.columns).index(c)] if not np.isnan(mi[list(X.columns).index(c)]) else 0,
                    reverse=True,
                )
                selected = sorted_features[:max(1, len(sorted_features) // 2)]
                logger.warning(f"没有特征满足互信息阈值{threshold}，已自动选择前{len(selected)}个特征")
        else:
            # F检验p值阈值（默认）
            selected = [
                col for col in X.columns
                if pvalues[list(X.columns).index(col)] is not None
                and not np.isnan(pvalues[list(X.columns).index(col)])
                and pvalues[list(X.columns).index(col)] <= threshold
            ]
            if not selected:
                # 如果所有p值都不满足，取F得分最高的几个
                sorted_features = sorted(
                    X.columns,
                    key=lambda c: scores[list(X.columns).index(c)] if not np.isnan(scores[list(X.columns).index(c)]) else 0,
                    reverse=True,
                )
                selected = sorted_features[:max(1, len(sorted_features) // 2)]
                logger.warning(f"没有特征满足p值阈值{threshold}，已自动选择前{len(selected)}个特征")

        removed = [col for col in X.columns if col not in selected]
        X_selected = X[selected].copy()

        logger.info(f"特征选择完成: 保留{len(selected)}个，移除{len(removed)}个")

        return {
            'success': True,
            'selected_features': selected,
            'removed_features': removed,
            'selection_scores': selection_scores,
            'X_selected': X_selected,
        }


class StatisticalFeatureSelector(BaseEstimator, TransformerMixin):
    """可放入 sklearn Pipeline 的无泄露统计特征选择器。"""

    def __init__(
        self,
        task_type: str,
        method: str = 'f_test',
        threshold: float = 0.05,
        k: Optional[int] = None,
    ):
        self.task_type = task_type
        self.method = method
        self.threshold = threshold
        self.k = k

    def fit(self, X, y):
        X_df = self._as_dataframe(X)
        result = FeatureSelector().select_features(
            X_df,
            pd.Series(y, index=X_df.index),
            self.task_type,
            method=self.method,
            threshold=self.threshold,
            k=self.k,
        )
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.selected_features_ = list(result['selected_features'])
        self.removed_features_ = list(result['removed_features'])
        self.selection_scores_ = result['selection_scores']
        return self

    def transform(self, X):
        if not hasattr(self, 'selected_features_'):
            raise MLFeatureError("特征选择器未拟合")
        X_df = self._as_dataframe(X)
        return X_df.reindex(columns=self.selected_features_, fill_value=0)

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, 'selected_features_'):
            raise MLFeatureError("特征选择器未拟合")
        return np.asarray(self.selected_features_, dtype=object)

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            result = X.copy()
            result.columns = [str(c) for c in result.columns]
            return result
        return pd.DataFrame(X, columns=[str(i) for i in range(np.asarray(X).shape[1])])


# ============================================================
# SupervisedTrainer — 监督学习（lazypredict）
# ============================================================

class SupervisedTrainer:
    """
    监督学习训练器：使用lazypredict批量训练和比较模型。

    用法:
        trainer = SupervisedTrainer()
        result = trainer.train_all(X_train, X_test, y_train, y_test, 'regression')
    """

    def train_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        task_type: str,
        random_state: int = 42,
    ) -> Dict:
        """
        使用lazypredict训练所有可用模型并比较。

        Parameters:
            X_train, X_test: 特征DataFrame
            y_train, y_test: 目标Series
            task_type: 'regression' / 'classification'
            random_state: 随机种子

        Returns:
            {
                'success': True,
                'task_type': str,
                'models': [{'rank', 'model_name', metrics...}, ...],
                'best_model_name': str,
                'best_model_class': type,
                'n_models_trained': int,
                'n_models_failed': int,
            }
        """
        if LazyClassifier is None or LazyRegressor is None:
            raise MLTrainingError(
                "lazypredict未安装，请执行: pip install lazypredict"
            )

        if task_type not in ('regression', 'classification'):
            raise MLTrainingError(f"不支持的任务类型: {task_type}")

        # lazypredict需要numpy数组或DataFrame
        X_tr = X_train.values if hasattr(X_train, 'values') else X_train
        X_te = X_test.values if hasattr(X_test, 'values') else X_test
        y_tr = y_train.values if hasattr(y_train, 'values') else y_train
        y_te = y_test.values if hasattr(y_test, 'values') else y_test

        try:
            if task_type == 'classification':
                lazy_model = LazyClassifier(
                    verbose=0, ignore_warnings=True, random_state=random_state,
                )
            else:
                lazy_model = LazyRegressor(
                    verbose=0, ignore_warnings=True, random_state=random_state,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models_df, predictions_df = lazy_model.fit(X_tr, X_te, y_tr, y_te)

        except Exception as e:
            raise MLTrainingError(f"lazypredict训练失败: {e}")

        if models_df is None or models_df.empty:
            raise MLTrainingError("lazypredict未返回任何模型结果")

        # 解析模型结果
        models_list = self._parse_models_df(models_df, task_type)

        # 获取最佳模型
        if not models_list:
            raise MLTrainingError("所有模型训练失败")

        best = models_list[0]
        best_model_class = self.get_model_class(best['model_name'])

        logger.info(
            f"训练完成: {len(models_list)}个模型成功, "
            f"最佳: {best['model_name']}"
        )

        return {
            'success': True,
            'task_type': task_type,
            'models': models_list,
            'best_model_name': best['model_name'],
            'best_model_class': best_model_class,
            'n_models_trained': len(models_list),
            'n_models_failed': int(models_df.shape[0] - len(models_list)),
        }

    def _parse_models_df(self, models_df: pd.DataFrame, task_type: str) -> List[Dict]:
        """解析lazypredict返回的模型DataFrame。"""
        models = []

        # lazypredict返回的DataFrame，模型名可能在index或'Model'列
        if 'Model' in models_df.columns:
            model_names = models_df['Model'].values
            metric_cols = [c for c in models_df.columns if c != 'Model']
        else:
            model_names = models_df.index.tolist()
            metric_cols = models_df.columns.tolist()

        for i, name in enumerate(model_names):
            row = models_df.iloc[i]
            entry = {
                'rank': 0,  # 后面赋值
                'model_name': str(name),
                'time_taken': _safe_float(row.get('Time Taken', None)),
            }

            if task_type == 'regression':
                entry['adjusted_r_squared'] = _safe_float(row.get('Adjusted R-Squared', None))
                entry['r_squared'] = _safe_float(row.get('R-Squared', None))
                entry['rmse'] = _safe_float(row.get('RMSE', None))
            else:
                entry['accuracy'] = _safe_float(row.get('Accuracy', None))
                entry['balanced_accuracy'] = _safe_float(row.get('Balanced Accuracy', None))
                entry['roc_auc'] = _safe_float(row.get('ROC AUC', None))
                entry['f1_score'] = _safe_float(row.get('F1 Score', None))

            models.append(entry)

        # 排序
        if task_type == 'regression':
            models.sort(
                key=lambda x: (
                    x.get('r_squared')
                    if x.get('r_squared') is not None
                    else float('-inf')
                ),
                reverse=True,
            )
        else:
            models.sort(
                key=lambda x: (
                    x.get('balanced_accuracy')
                    if x.get('balanced_accuracy') is not None
                    else float('-inf')
                ),
                reverse=True,
            )

        # 赋排名
        for rank, model in enumerate(models, 1):
            model['rank'] = rank

        return models

    @staticmethod
    def get_model_class(model_name: str) -> type:
        """
        将模型名称解析为sklearn类。

        Parameters:
            model_name: 模型名称字符串（如 'RandomForestRegressor'）

        Returns:
            对应的sklearn估计器类

        Raises:
            MLTrainingError: 找不到对应的模型类
        """
        estimator_map = _get_estimator_map()

        # 精确匹配
        if model_name in estimator_map:
            return estimator_map[model_name]

        # 大小写不敏感匹配
        name_lower = model_name.lower()
        for name, cls in estimator_map.items():
            if name.lower() == name_lower:
                return cls

        raise MLTrainingError(f"未找到模型类: {model_name}")


# ============================================================
# UnsupervisedTrainer — 无监督学习
# ============================================================

class UnsupervisedTrainer:
    """
    无监督学习训练器：聚类 + 降维。

    用法:
        trainer = UnsupervisedTrainer()
        cluster_result = trainer.cluster(X)
        dim_result = trainer.reduce_dimensions(X)
    """

    DEFAULT_CLUSTER_ALGORITHMS = ['KMeans', 'DBSCAN', 'AgglomerativeClustering', 'SpectralClustering', 'GaussianMixture']

    def cluster(
        self,
        X: pd.DataFrame,
        algorithms: Optional[List[str]] = None,
        n_clusters: int = 3,
        n_clusters_range: Tuple[int, int] = (2, 10),
        random_state: int = 42,
    ) -> Dict:
        """
        运行多种聚类算法。

        Parameters:
            X: 预处理后的特征DataFrame
            algorithms: 聚类算法列表，默认 ['KMeans', 'DBSCAN', 'AgglomerativeClustering', 'GaussianMixture']
            n_clusters: 默认聚类数
            n_clusters_range: KMeans肘部法则的k搜索范围
            random_state: 随机种子

        Returns:
            {
                'success': True,
                'results': {algorithm_name: {labels, n_clusters, metrics...}},
                'comparison': [{'algorithm', 'n_clusters', 'silhouette_score'}],
                'elbow_data': [{'k', 'inertia'}],
            }
        """
        if X is None or X.empty:
            raise MLTrainingError("输入数据为空")

        if algorithms is None:
            algorithms = self.DEFAULT_CLUSTER_ALGORITHMS

        X_array = X.values if hasattr(X, 'values') else np.array(X)
        results = {}
        comparison = []
        elbow_data = []

        for algo_name in algorithms:
            try:
                if algo_name == 'KMeans':
                    res = self._cluster_kmeans(X_array, n_clusters, n_clusters_range, random_state)
                    elbow_data = res.pop('elbow_data', [])
                elif algo_name == 'DBSCAN':
                    res = self._cluster_dbscan(X_array)
                elif algo_name == 'AgglomerativeClustering':
                    res = self._cluster_agglomerative(X_array, n_clusters)
                elif algo_name == 'GaussianMixture':
                    res = self._cluster_gmm(X_array, n_clusters, random_state)
                elif algo_name == 'SpectralClustering':
                    res = self._cluster_spectral(X_array, n_clusters, random_state)
                else:
                    logger.warning(f"未知聚类算法: {algo_name}，跳过")
                    continue

                results[algo_name] = res
                comparison.append({
                    'algorithm': algo_name,
                    'n_clusters': res.get('n_clusters', 0),
                    'silhouette_score': res.get('silhouette_score'),
                })

            except Exception as e:
                logger.warning(f"聚类算法 {algo_name} 执行失败: {e}")
                results[algo_name] = {'error': str(e)}

        logger.info(f"聚类完成: {len(results)}个算法")

        return {
            'success': True,
            'results': results,
            'comparison': comparison,
            'elbow_data': elbow_data,
        }

    def _cluster_kmeans(self, X, n_clusters, n_clusters_range, random_state):
        """KMeans聚类 + 肘部法则"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        # 肘部法则数据
        elbow_data = []
        for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km.fit(X)
            elbow_data.append({'k': k, 'inertia': float(km.inertia_)})

        # 轮廓系数
        sil_score = None
        if n_clusters >= 2 and len(set(labels)) >= 2:
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                pass

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'inertia': float(kmeans.inertia_),
            'silhouette_score': sil_score,
            'cluster_centers': kmeans.cluster_centers_,
            'elbow_data': elbow_data,
        }

    def _cluster_dbscan(self, X):
        """DBSCAN聚类"""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        sil_score = None
        if n_clusters >= 2:
            try:
                # 排除噪声点计算轮廓系数
                mask = labels != -1
                if mask.sum() > 1:
                    sil_score = float(silhouette_score(X[mask], labels[mask]))
            except Exception:
                pass

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': sil_score,
            'eps': 0.5,
            'min_samples': 5,
        }

    def _cluster_agglomerative(self, X, n_clusters):
        """层次聚类"""
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X)

        sil_score = None
        if n_clusters >= 2:
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                pass

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
        }

    def _cluster_gmm(self, X, n_clusters, random_state):
        """高斯混合模型"""
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = gmm.fit_predict(X)

        sil_score = None
        if n_clusters >= 2:
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                pass

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'bic': float(gmm.bic(X)),
            'aic': float(gmm.aic(X)),
        }

    def _cluster_spectral(self, X, n_clusters, random_state):
        """谱聚类"""
        spectral = SpectralClustering(
            n_clusters=n_clusters, random_state=random_state, affinity='nearest_neighbors',
        )
        labels = spectral.fit_predict(X)

        sil_score = None
        if n_clusters >= 2:
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                pass

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
        }

    def reduce_dimensions(
        self,
        X: pd.DataFrame,
        methods: Optional[List[str]] = None,
        n_components: int = 2,
        random_state: int = 42,
    ) -> Dict:
        """
        降维可视化。

        Parameters:
            X: 特征DataFrame
            methods: 降维方法列表，默认 ['PCA', 'TSNE']
            n_components: 降维目标维度（2或3）
            random_state: 随机种子

        Returns:
            {
                'success': True,
                'results': {
                    'PCA': {'coordinates', 'explained_variance_ratio', ...},
                    'TSNE': {'coordinates', ...},
                },
            }
        """
        if X is None or X.empty:
            raise MLTrainingError("输入数据为空")

        if methods is None:
            methods = ['PCA', 'TSNE']

        X_array = X.values if hasattr(X, 'values') else np.array(X)
        results = {}

        for method in methods:
            try:
                if method == 'PCA':
                    pca = PCA(n_components=n_components)
                    coords = pca.fit_transform(X_array)
                    results['PCA'] = {
                        'coordinates': coords,
                        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                        'n_components': n_components,
                    }
                elif method == 'TSNE':
                    # t-SNE对大数据集较慢，给警告
                    if X_array.shape[0] > 10000:
                        logger.warning(
                            "数据量>10000，t-SNE可能较慢。建议先用PCA降维。"
                        )
                    perplexity = min(30, X_array.shape[0] - 1)
                    tsne = TSNE(
                        n_components=n_components,
                        random_state=random_state,
                        perplexity=perplexity,
                    )
                    coords = tsne.fit_transform(X_array)
                    results['TSNE'] = {
                        'coordinates': coords,
                        'perplexity': perplexity,
                        'n_components': n_components,
                    }
                else:
                    logger.warning(f"未知降维方法: {method}，跳过")

            except Exception as e:
                logger.warning(f"降维方法 {method} 执行失败: {e}")
                results[method] = {'error': str(e)}

        logger.info(f"降维完成: {list(results.keys())}")

        return {
            'success': True,
            'results': results,
        }

    def detect_anomalies(
        self,
        X: pd.DataFrame,
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> Dict:
        """
        异常检测 (IsolationForest)。

        Parameters:
            X: 预处理后的特征DataFrame
            contamination: 预期异常点比例 (0.01 ~ 0.5)
            random_state: 随机种子

        Returns:
            {
                'success': True,
                'labels': np.ndarray (-1=异常, 1=正常),
                'anomaly_scores': np.ndarray (决策函数值),
                'n_anomalies': int,
                'n_normal': int,
                'anomaly_ratio': float,
                'contamination': float,
            }
        """
        if X is None or X.empty:
            raise MLTrainingError("输入数据为空")

        X_array = X.values if hasattr(X, 'values') else np.array(X)

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        labels = iso_forest.fit_predict(X_array)
        anomaly_scores = iso_forest.decision_function(X_array)

        n_anomalies = int(np.sum(labels == -1))
        n_normal = int(np.sum(labels == 1))
        anomaly_ratio = round(n_anomalies / len(labels), 4) if len(labels) > 0 else 0.0

        logger.info(f"异常检测完成: {n_anomalies}个异常, {n_normal}个正常, 比例={anomaly_ratio:.4f}")

        return {
            'success': True,
            'labels': labels,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'anomaly_ratio': anomaly_ratio,
            'contamination': contamination,
        }


# ============================================================
# Optuna搜索空间注册表
# ============================================================

_SEARCH_SPACES: Dict[str, callable] = {}


def register_search_space(model_name: str):
    """装饰器：为指定模型注册Optuna超参数搜索空间。"""
    def decorator(func):
        _SEARCH_SPACES[model_name] = func
        return func
    return decorator


# --- 线性模型 ---

@register_search_space('Ridge')
def _ridge_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']),
    }

@register_search_space('RidgeCV')
def _ridgecv_space(trial):
    alpha_center = trial.suggest_float('alpha_center', 1e-3, 100.0, log=True)
    alphas = sorted({
        float(np.clip(alpha_center * multiplier, 1e-3, 100.0))
        for multiplier in (0.01, 0.1, 1.0, 10.0, 100.0)
    })
    return {
        'alphas': alphas,
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }

@register_search_space('Lasso')
def _lasso_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
    }

@register_search_space('ElasticNet')
def _elasticnet_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
    }

@register_search_space('LinearRegression')
def _linear_regression_space(trial):
    return {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }

@register_search_space('LogisticRegression')
def _logistic_regression_space(trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    params = {
        'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
        'penalty': penalty,
        'solver': 'saga',
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
    }
    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
    return params

@register_search_space('SGDRegressor')
def _sgd_regressor_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'eta0': trial.suggest_float('eta0', 1e-4, 0.1, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
    }

@register_search_space('SGDClassifier')
def _sgd_classifier_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'eta0': trial.suggest_float('eta0', 1e-4, 0.1, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
    }

@register_search_space('BayesianRidge')
def _bayesian_ridge_space(trial):
    return {
        'alpha_1': trial.suggest_float('alpha_1', 1e-7, 1e-1, log=True),
        'alpha_2': trial.suggest_float('alpha_2', 1e-7, 1e-1, log=True),
        'lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-1, log=True),
        'lambda_2': trial.suggest_float('lambda_2', 1e-7, 1e-1, log=True),
    }

@register_search_space('ARDRegression')
def _ard_regression_space(trial):
    return {
        'alpha_1': trial.suggest_float('alpha_1', 1e-7, 1e-1, log=True),
        'alpha_2': trial.suggest_float('alpha_2', 1e-7, 1e-1, log=True),
        'lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-1, log=True),
        'lambda_2': trial.suggest_float('lambda_2', 1e-7, 1e-1, log=True),
    }

@register_search_space('HuberRegressor')
def _huber_space(trial):
    return {
        'epsilon': trial.suggest_float('epsilon', 1.0, 2.0),
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
    }


# --- 树模型 ---

@register_search_space('DecisionTreeRegressor')
@register_search_space('DecisionTreeClassifier')
def _decision_tree_space(trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }

@register_search_space('RandomForestRegressor')
@register_search_space('RandomForestClassifier')
def _random_forest_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }

@register_search_space('ExtraTreesRegressor')
@register_search_space('ExtraTreesClassifier')
def _extra_trees_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }


# --- Boosting模型 ---

@register_search_space('GradientBoostingRegressor')
@register_search_space('GradientBoostingClassifier')
def _gradient_boosting_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }

@register_search_space('AdaBoostRegressor')
@register_search_space('AdaBoostClassifier')
def _adaboost_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
    }

@register_search_space('HistGradientBoostingRegressor')
@register_search_space('HistGradientBoostingClassifier')
def _hist_gradient_boosting_space(trial):
    return {
        'max_iter': trial.suggest_int('max_iter', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
    }


# --- SVM ---

@register_search_space('SVR')
@register_search_space('SVC')
def _svm_space(trial):
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-2, 100.0, log=True),
        'kernel': kernel,
    }
    if kernel in ('rbf', 'poly', 'sigmoid'):
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    return params

@register_search_space('LinearSVC')
def _linear_svc_space(trial):
    return {
        'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
        'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
        'max_iter': trial.suggest_int('max_iter', 1000, 10000),
    }


@register_search_space('LinearSVR')
def _linear_svr_space(trial):
    return {
        'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
        'loss': trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
        'max_iter': trial.suggest_int('max_iter', 1000, 10000),
    }

@register_search_space('NuSVC')
def _nu_svc_space(trial):
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    params = {
        'nu': trial.suggest_float('nu', 0.1, 0.9),
        'kernel': kernel,
    }
    if kernel in ('rbf', 'poly', 'sigmoid'):
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    return params


@register_search_space('NuSVR')
def _nu_svr_space(trial):
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    params = {
        'nu': trial.suggest_float('nu', 0.1, 0.9),
        'C': trial.suggest_float('C', 1e-2, 100.0, log=True),
        'kernel': kernel,
    }
    if kernel in ('rbf', 'poly', 'sigmoid'):
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    return params


# --- KNN ---

@register_search_space('KNeighborsRegressor')
@register_search_space('KNeighborsClassifier')
def _knn_space(trial):
    return {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
        'p': trial.suggest_int('p', 1, 3),
    }


# --- Bagging ---

@register_search_space('BaggingRegressor')
@register_search_space('BaggingClassifier')
def _bagging_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
    }


# --- 其他 ---

@register_search_space('PassiveAggressiveClassifier')
def _passive_aggressive_classifier_space(trial):
    return {
        'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
    }


@register_search_space('PassiveAggressiveRegressor')
def _passive_aggressive_regressor_space(trial):
    return {
        'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'loss': trial.suggest_categorical(
            'loss',
            ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        ),
    }

@register_search_space('Perceptron')
def _perceptron_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet', None]),
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'eta0': trial.suggest_float('eta0', 0.001, 1.0, log=True),
    }

@register_search_space('Lars')
@register_search_space('LassoLars')
def _lars_space(trial):
    return {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }

@register_search_space('OrthogonalMatchingPursuit')
def _omp_space(trial):
    return {
        'n_nonzero_coefs': trial.suggest_int('n_nonzero_coefs', 1, 50),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }


# --- 通用回退搜索空间 ---

_GENERIC_PARAM_SUGGESTIONS = {
    'alpha': lambda t: t.suggest_float('alpha', 1e-4, 100.0, log=True),
    'C': lambda t: t.suggest_float('C', 1e-3, 100.0, log=True),
    'n_estimators': lambda t: t.suggest_int('n_estimators', 50, 500),
    'max_depth': lambda t: t.suggest_int('max_depth', 3, 30),
    'learning_rate': lambda t: t.suggest_float('learning_rate', 0.001, 0.3, log=True),
    'min_samples_split': lambda t: t.suggest_int('min_samples_split', 2, 20),
    'min_samples_leaf': lambda t: t.suggest_int('min_samples_leaf', 1, 10),
    'max_iter': lambda t: t.suggest_int('max_iter', 100, 1000),
    'n_neighbors': lambda t: t.suggest_int('n_neighbors', 3, 30),
    'subsample': lambda t: t.suggest_float('subsample', 0.5, 1.0),
    'fit_intercept': lambda t: t.suggest_categorical('fit_intercept', [True, False]),
    'l1_ratio': lambda t: t.suggest_float('l1_ratio', 0.0, 1.0),
}


def _get_generic_search_space(model_class: type):
    """
    为未注册的模型生成通用搜索空间。
    扫描模型默认参数，匹配常见参数名。
    """
    def space_func(trial):
        params = {}
        try:
            default_params = model_class().get_params()
        except Exception:
            return params

        for param_name in default_params:
            if param_name in _GENERIC_PARAM_SUGGESTIONS:
                try:
                    params[param_name] = _GENERIC_PARAM_SUGGESTIONS[param_name](trial)
                except Exception:
                    pass
        return params

    return space_func


# ============================================================
# ModelTuner — Optuna自动调参
# ============================================================

class ModelTuner:
    """
    模型调参器：使用Optuna自动搜索最优超参数。

    用法:
        tuner = ModelTuner()
        result = tuner.tune(RandomForestRegressor, X, y, 'regression', n_trials=50)
    """

    def __init__(self):
        self._study = None

    def tune(
        self,
        model_class: type,
        X_train,
        y_train,
        task_type: str,
        n_trials: int = 10,
        cv_folds: int = 3,
        scoring: Optional[str] = None,
        timeout: int = 600,
        random_state: int = 42,
        n_jobs: int = -1,
        pipeline_factory: Optional[Callable[[Any], Any]] = None,
    ) -> Dict:
        """
        使用Optuna对指定模型进行超参数调优。

        Parameters:
            model_class: sklearn估计器类
            X_train: 训练特征
            y_train: 训练目标
            task_type: 'regression' / 'classification'
            n_trials: Optuna试验次数
            cv_folds: 交叉验证折数
            scoring: 评估指标（默认：回归'r2'，分类'balanced_accuracy'）
            timeout: 最大优化时间（秒）
            random_state: 随机种子
            n_jobs: 并行数
            pipeline_factory: 可选，接收模型并返回含预处理/特征选择的Pipeline

        Returns:
            {
                'success': True,
                'model_name': str,
                'best_params': dict,
                'best_score': float,
                'best_model': sklearn estimator（已拟合）,
                'optimization_history': list[float],
                'n_trials_completed': int,
            }
        """
        if optuna is None:
            raise MLTuningError("optuna未安装，请执行: pip install optuna")

        if task_type not in ('regression', 'classification'):
            raise MLTuningError(f"不支持的任务类型: {task_type}")

        model_name = model_class.__name__

        # 确定评估指标
        if scoring is None:
            scoring = 'r2' if task_type == 'regression' else 'balanced_accuracy'

        # 确定交叉验证策略
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # 获取搜索空间
        model_name_str = model_class.__name__
        if model_name_str in _SEARCH_SPACES:
            search_space_func = _SEARCH_SPACES[model_name_str]
        else:
            logger.info(f"模型 {model_name_str} 未注册搜索空间，使用通用回退")
            search_space_func = _get_generic_search_space(model_class)

        X_arr = (
            X_train
            if pipeline_factory is not None
            else np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        )
        y_arr = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        # 检查模型是否支持random_state
        has_random_state = False
        try:
            default_params = model_class().get_params()
            has_random_state = 'random_state' in default_params
        except Exception:
            pass

        # 创建Optuna目标函数
        def objective(trial):
            params = search_space_func(trial)
            if has_random_state:
                params['random_state'] = random_state
            try:
                model = model_class(**params)
                estimator = pipeline_factory(model) if pipeline_factory else model
                scores = cross_val_score(
                    estimator, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=1,
                )
                score = float(scores.mean())
                return score if np.isfinite(score) else float('-inf')
            except Exception as e:
                logger.debug(f"Trial {trial.number} 失败: {e}")
                return float('-inf')

        # 运行Optuna优化
        try:
            sampler = optuna.samplers.TPESampler(seed=random_state)
            self._study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
            )
            self._study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False,
                catch=(Exception,),
            )
        except Exception as e:
            raise MLTuningError(f"Optuna优化失败: {e}")

        valid_trials = [
            trial for trial in self._study.trials
            if trial.value is not None and np.isfinite(float(trial.value))
        ]
        if not valid_trials:
            raise MLTuningError("Optuna未完成任何有效试验")

        best_trial = max(valid_trials, key=lambda trial: float(trial.value))
        # 重新执行搜索空间函数，恢复未写入 trial.params 的固定参数和条件参数。
        best_params = search_space_func(best_trial)
        if has_random_state:
            best_params['random_state'] = random_state

        # 用最佳参数在全部训练数据上拟合模型
        try:
            best_model = model_class(**best_params)
            best_estimator = (
                pipeline_factory(best_model) if pipeline_factory else best_model
            )
            best_estimator.fit(X_arr, y_arr)
            if pipeline_factory and hasattr(best_estimator, 'named_steps'):
                best_model = best_estimator.named_steps['model']
        except Exception as e:
            raise MLTuningError(f"最佳模型拟合失败: {e}")

        # 优化历史
        opt_history = []
        for trial in self._study.trials:
            if trial.value is not None and np.isfinite(float(trial.value)):
                opt_history.append(float(trial.value))

        logger.info(
            f"调参完成: {model_name}, "
            f"最佳得分={best_trial.value:.4f}, "
            f"试验次数={len(self._study.trials)}"
        )

        return {
            'success': True,
            'model_name': model_name,
            'best_params': best_params,
            'best_score': float(best_trial.value),
            'best_model': best_model,
            'best_estimator': best_estimator,
            'cv_metric': scoring,
            'optimization_history': opt_history,
            'n_trials_completed': len(self._study.trials),
        }


# ============================================================
# ModelManager — 模型保存/加载/预测
# ============================================================

class ModelManager:
    """
    模型管理器：保存、加载、预测、导出。

    用法:
        manager = ModelManager()
        manager.save_model(model, 'my_model', metadata={...})
        loaded = manager.load_model('my_model')
        pred = manager.predict(loaded['model'], new_data)
        manager.export_predictions(pred['predictions_df'], 'output.csv')
    """

    def save_model(
        self,
        model,
        file_path: str,
        metadata: Optional[Dict] = None,
        preprocessor_bundle: Optional[Dict] = None,
    ) -> Dict:
        """
        保存模型和元数据。

        Parameters:
            model: 训练好的sklearn模型
            file_path: 保存路径（不含扩展名）
            metadata: 元数据字典
            preprocessor_bundle: DataPreprocessor.get_bundle()的返回值

        Returns:
            {'success': True, 'model_path': str, 'metadata_path': str}

        Creates:
            <file_path>.joblib  — 模型 + 预处理器
            <file_path>_meta.json — 元数据
        """
        if model is None:
            raise MLModelError("模型不能为空")

        # 确保目录存在
        save_dir = os.path.dirname(file_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # 保存模型bundle（模型 + 预处理器）
        bundle = {
            'model': model,
            'preprocessor_bundle': preprocessor_bundle,
        }

        model_path = file_path if file_path.endswith('.joblib') else f"{file_path}.joblib"
        try:
            joblib.dump(bundle, model_path, compress=3)
        except Exception as e:
            raise MLModelError(f"模型保存失败: {e}")

        # 保存元数据JSON
        meta = metadata or {}
        meta['saved_at'] = datetime.now().isoformat()
        try:
            import sklearn
            meta['sklearn_version'] = sklearn.__version__
        except Exception:
            pass

        meta_path = (
            file_path.replace('.joblib', '_meta.json')
            if file_path.endswith('.joblib')
            else f"{file_path}_meta.json"
        )

        try:
            serializable_meta = _to_serializable(meta)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"元数据保存失败（不影响模型）: {e}")

        logger.info(f"模型已保存: {model_path}")

        return {
            'success': True,
            'model_path': model_path,
            'metadata_path': meta_path,
        }

    def load_model(self, file_path: str) -> Dict:
        """
        加载模型和元数据。

        Parameters:
            file_path: 模型文件路径（.joblib）

        Returns:
            {
                'success': True,
                'model': sklearn estimator,
                'preprocessor_bundle': dict or None,
                'metadata': dict,
            }
        """
        model_path = file_path if file_path.endswith('.joblib') else f"{file_path}.joblib"

        if not os.path.exists(model_path):
            raise MLModelError(f"模型文件不存在: {model_path}")

        # 加载模型bundle
        try:
            bundle = joblib.load(model_path)
        except Exception as e:
            raise MLModelError(f"模型加载失败: {e}")

        if isinstance(bundle, dict) and 'model' in bundle:
            model = bundle['model']
            preprocessor_bundle = bundle.get('preprocessor_bundle')
        else:
            # 兼容直接保存的模型
            model = bundle
            preprocessor_bundle = None

        # 加载元数据
        meta_path = model_path.replace('.joblib', '_meta.json')
        metadata = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"元数据加载失败: {e}")

        # 版本检查
        try:
            import sklearn
            saved_version = metadata.get('sklearn_version', '')
            if saved_version and saved_version != sklearn.__version__:
                logger.warning(
                    f"sklearn版本不匹配: 保存时={saved_version}, 当前={sklearn.__version__}"
                )
        except Exception:
            pass

        logger.info(f"模型已加载: {model_path}")

        return {
            'success': True,
            'model': model,
            'preprocessor_bundle': preprocessor_bundle,
            'metadata': metadata,
        }

    def predict(
        self,
        model,
        X_new: pd.DataFrame,
        preprocessor: Optional[DataPreprocessor] = None,
    ) -> Dict:
        """
        使用模型对新数据进行预测。

        Parameters:
            model: 训练好的模型
            X_new: 新数据DataFrame（原始格式）
            preprocessor: 已拟合的DataPreprocessor（用于预处理新数据）

        Returns:
            {
                'success': True,
                'predictions': np.ndarray,
                'predictions_df': pd.DataFrame,  # 原始数据 + 预测列
                'prediction_count': int,
            }
        """
        if model is None:
            raise MLModelError("模型不能为空")
        if X_new is None or X_new.empty:
            raise MLModelError("预测数据不能为空")

        # 预处理新数据
        if preprocessor is not None:
            X_processed = preprocessor.transform_new(X_new)
        else:
            X_processed = X_new

        # 预测
        try:
            predictions = model.predict(X_processed)
        except Exception as e:
            raise MLModelError(f"预测失败: {e}")

        # 构建结果DataFrame
        result_df = X_new.copy()

        # 如果有LabelEncoder，解码预测结果
        if preprocessor is not None and preprocessor.label_encoder is not None:
            try:
                predictions_decoded = preprocessor.label_encoder.inverse_transform(
                    predictions.astype(int)
                )
                result_df['prediction'] = predictions_decoded
            except Exception:
                result_df['prediction'] = predictions
        else:
            result_df['prediction'] = predictions

        return {
            'success': True,
            'predictions': predictions,
            'predictions_df': result_df,
            'prediction_count': len(predictions),
        }

    def export_predictions(
        self,
        predictions_df: pd.DataFrame,
        file_path: str,
        format: str = 'csv',
    ) -> Dict:
        """
        导出预测结果到文件。

        Parameters:
            predictions_df: 包含预测结果的DataFrame
            file_path: 输出文件路径
            format: 'csv' 或 'xlsx'

        Returns:
            {'success': True, 'file_path': str, 'row_count': int}
        """
        if predictions_df is None or predictions_df.empty:
            raise MLModelError("预测结果为空")

        # 确保目录存在
        save_dir = os.path.dirname(file_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        try:
            if format == 'xlsx':
                try:
                    import openpyxl  # noqa: F401
                except ImportError:
                    raise MLModelError(
                        "XLSX导出需要openpyxl，请执行: pip install openpyxl"
                    )
                predictions_df.to_excel(file_path, index=False)
            else:
                predictions_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        except MLModelError:
            raise
        except Exception as e:
            raise MLModelError(f"导出失败: {e}")

        logger.info(f"预测结果已导出: {file_path}, {len(predictions_df)}行")

        return {
            'success': True,
            'file_path': file_path,
            'row_count': len(predictions_df),
        }


# ============================================================
# MLPipeline — 端到端编排器
# ============================================================

class MLPipeline:
    """
    ML流水线编排器：串联所有步骤，提供端到端API。

    用法:
        pipeline = MLPipeline()

        # 监督学习
        result = pipeline.run_supervised('data.csv', target_col='target', task_type='auto')

        # 无监督学习
        result = pipeline.run_unsupervised('data.csv')

        # 新数据预测
        pipeline.predict_new('saved_model', 'new_data.csv', 'output.csv')
    """

    def __init__(self):
        self.importer = DataImporter()
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector()
        self.supervised_trainer = SupervisedTrainer()
        self.unsupervised_trainer = UnsupervisedTrainer()
        self.tuner = ModelTuner()
        self.manager = ModelManager()

        # 内部状态
        self._raw_data = None
        self._data_metadata = None
        self._preprocessed = None
        self._correlation_results = None
        self._selected_features = None
        self._X_selected = None
        self._y = None
        self._training_results = None
        self._tuned_result = None
        self._saved_path = None

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data_metadata(self):
        return self._data_metadata

    @property
    def preprocessed(self):
        return self._preprocessed

    @property
    def correlation_results(self):
        return self._correlation_results

    @property
    def selected_features(self):
        return self._selected_features

    @property
    def training_results(self):
        return self._training_results

    @property
    def tuned_result(self):
        return self._tuned_result

    def _run_supervised_legacy(
        self,
        file_path: str,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        task_type: str = 'auto',
        test_size: float = 0.2,
        correlation_threshold: float = 0.05,
        feature_selection_method: str = 'f_test',
        n_features: Optional[int] = None,
        tune: bool = True,
        n_trials: int = 10,
        cv_folds: int = 3,
        random_state: int = 42,
        save_path: Optional[str] = None,
        progress_callback=None,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        scale_numeric: bool = True,
        scaler_type: str = 'standard',
        **kwargs,
    ) -> Dict:
        """
        端到端监督学习流水线。

        Parameters:
            file_path: 数据文件路径
            target_col: 目标列名
            feature_cols: 指定特征列（None则使用所有非目标列）
            task_type: 'auto'/'regression'/'classification'
            test_size: 测试集比例
            correlation_threshold: 特征选择p值阈值
            feature_selection_method: 'f_test'/'kbest'/'mutual_info'
            n_features: 保留特征数（仅method='kbest'时有效）
            tune: 是否用Optuna调参
            n_trials: Optuna试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子
            save_path: 模型保存路径（None则不保存）
        numeric_strategy: 数值缺失填充策略 ('mean'/'median'/'constant')
        scale_numeric: 是否对数值特征缩放
        scaler_type: 缩放类型 'standard'(默认) / 'minmax'

        Returns:
            包含全流程结果的字典
        """
        logger.info("=" * 60)
        logger.info("开始监督学习流水线")
        logger.info("=" * 60)

        # Step 1: 数据导入
        logger.info("[Step 1] 数据导入...")
        if progress_callback:
            progress_callback('data_import', message='加载数据文件...')
        import_result = self.importer.load_data(file_path, **kwargs)
        self._raw_data = import_result['data']
        self._data_metadata = import_result['metadata']

        df = import_result['data']

        if target_col not in df.columns:
            raise MLDataError(f"目标列 '{target_col}' 不存在于数据中")

        # 筛选指定特征列
        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise MLDataError(f"指定特征列不存在: {missing}")
            df = df[feature_cols + [target_col]]

        # 自动检测任务类型
        if task_type == 'auto':
            task_type = _detect_task_type(df[target_col])
            logger.info(f"自动检测任务类型: {task_type}")

        # Step 2: 数据预处理
        logger.info("[Step 2] 数据预处理...")
        if progress_callback:
            progress_callback('preprocessing', message=f'预处理 {df.shape[1]} 列数据...')
        prep_result = self.preprocessor.fit_transform(
            df, target_col=target_col, task_type=task_type,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            scale_numeric=scale_numeric,
            scaler_type=scaler_type,
        )
        self._preprocessed = prep_result
        X = prep_result['X']
        y = prep_result['y']

        # Step 3: 相关性分析
        logger.info("[Step 3] 相关性分析...")
        if progress_callback:
            progress_callback('correlation', message='分析特征与目标的相关性...')
        corr_result = self.feature_selector.analyze_correlation(X, y, task_type)
        self._correlation_results = corr_result

        # Step 4: 特征选择
        logger.info("[Step 4] 特征选择...")
        if progress_callback:
            progress_callback('feature_selection', message=f'使用 {feature_selection_method} 筛选特征...')
        sel_result = self.feature_selector.select_features(
            X, y, task_type,
            method=feature_selection_method,
            threshold=correlation_threshold,
            k=n_features,
        )
        self._selected_features = sel_result['selected_features']
        self._X_selected = sel_result['X_selected']
        self._y = y

        X_final = sel_result['X_selected']

        if X_final.shape[1] == 0:
            raise MLFeatureError("特征选择后没有可用特征")

        # Step 5: 数据分割 + lazypredict训练
        logger.info("[Step 5] 模型训练（lazypredict）...")
        if progress_callback:
            progress_callback('model_training', message=f'批量训练模型（{X_final.shape[1]} 个特征）...')
        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y, test_size=test_size, random_state=random_state, stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y, test_size=test_size, random_state=random_state,
            )

        train_result = self.supervised_trainer.train_all(
            X_train, X_test, y_train, y_test, task_type, random_state,
        )
        self._training_results = train_result

        best_model_name = train_result['best_model_name']
        best_model_class = train_result['best_model_class']
        best_rank_info = train_result['models'][0]

        logger.info(f"最佳模型: {best_model_name}")

        # Step 6: Optuna调参
        tuned_result = None
        if tune:
            logger.info(f"[Step 6] Optuna调参: {best_model_name}...")
            if progress_callback:
                progress_callback('hyperparameter_tuning', message=f'Optuna 调参: {best_model_name}（{n_trials} 次试验）...')
            try:
                tuned_result = self.tuner.tune(
                    best_model_class, X_final, y, task_type,
                    n_trials=n_trials, cv_folds=cv_folds,
                    random_state=random_state,
                )
                self._tuned_result = tuned_result
                logger.info(
                    f"调参完成: 最佳得分={tuned_result['best_score']:.4f}"
                )
            except Exception as e:
                logger.warning(f"调参失败（使用默认参数）: {e}")
                # 使用lazypredict默认参数的模型
                try:
                    default_model = best_model_class()
                    if 'random_state' in default_model.get_params():
                        default_model.set_params(random_state=random_state)
                    default_model.fit(X_final, y)
                    tuned_result = {
                        'success': True,
                        'model_name': best_model_name,
                        'best_params': default_model.get_params(),
                        'best_score': None,
                        'best_model': default_model,
                        'optimization_history': [],
                        'n_trials_completed': 0,
                    }
                    self._tuned_result = tuned_result
                except Exception as e2:
                    raise MLTrainingError(f"默认模型训练也失败: {e2}")

        # Step 7: 保存模型
        saved_info = None
        if save_path:
            logger.info("[Step 7] 保存模型...")
            if progress_callback:
                progress_callback('model_saving', message='保存训练模型...')
            metadata = {
                'model_class': best_model_name,
                'task_type': task_type,
                'feature_names': self._selected_features,
                'target_name': target_col,
                'training_metrics': _to_serializable(best_rank_info),
                'tuning_params': _to_serializable(tuned_result['best_params']) if tuned_result else None,
                'tuning_score': _safe_float(tuned_result['best_score']) if tuned_result else None,
                'sample_count': int(len(y)),
                'feature_count': int(X_final.shape[1]),
                'test_size': test_size,
                'random_state': random_state,
            }
            preprocessor_bundle = self.preprocessor.get_bundle()
            # 记录特征选择后的列名，预测时需要筛选
            preprocessor_bundle['selected_features'] = self._selected_features
            saved_info = self.manager.save_model(
                tuned_result['best_model'] if tuned_result else best_model_class(),
                save_path,
                metadata=metadata,
                preprocessor_bundle=preprocessor_bundle,
            )
            self._saved_path = saved_info['model_path']

        # 提取特征重要性
        feature_importance = self._extract_feature_importance(
            tuned_result['best_model'] if tuned_result else None,
            self._selected_features,
        )

        logger.info("=" * 60)
        logger.info("监督学习流水线完成")
        logger.info("=" * 60)

        return {
            'success': True,
            'task_type': task_type,
            'data_metadata': import_result['metadata'],
            'preprocessing_info': prep_result['preprocessing_info'],
            'correlation_results': corr_result,
            'feature_selection': {
                'selected_features': self._selected_features,
                'removed_features': sel_result['removed_features'],
                'selection_scores': _to_serializable(sel_result['selection_scores']),
            },
            'training_results': {
                'models': _to_serializable(train_result['models']),
                'best_model_name': best_model_name,
                'n_models_trained': train_result['n_models_trained'],
            },
            'tuning_results': {
                'best_params': _to_serializable(tuned_result['best_params']) if tuned_result else None,
                'best_score': _safe_float(tuned_result['best_score']) if tuned_result else None,
                'n_trials': tuned_result['n_trials_completed'] if tuned_result else 0,
                'optimization_history': tuned_result['optimization_history'] if tuned_result else [],
            } if tuned_result else None,
            'feature_importance': feature_importance,
            'saved_info': saved_info,
        }

    def run_supervised(
        self,
        file_path: str,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        task_type: str = 'auto',
        test_size: float = 0.2,
        correlation_threshold: float = 0.05,
        feature_selection_method: str = 'f_test',
        n_features: Optional[int] = None,
        tune: bool = True,
        n_trials: int = 10,
        cv_folds: int = 3,
        random_state: int = 42,
        save_path: Optional[str] = None,
        progress_callback=None,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        scale_numeric: bool = True,
        scaler_type: str = 'standard',
        **kwargs,
    ) -> Dict:
        """端到端监督学习流水线；测试集只用于最终一次评估。"""
        logger.info("=" * 60)
        logger.info("开始监督学习流水线")
        logger.info("=" * 60)

        if progress_callback:
            progress_callback('data_import', message='加载数据文件...')
        import_result = self.importer.load_data(file_path, **kwargs)
        self._raw_data = import_result['data']
        self._data_metadata = import_result['metadata']
        df = import_result['data'].copy()

        if target_col not in df.columns:
            raise MLDataError(f"目标列 '{target_col}' 不存在于数据中")
        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise MLDataError(f"指定特征列不存在: {missing}")
            df = df[feature_cols + [target_col]]
        if task_type == 'auto':
            task_type = _detect_task_type(df[target_col])
            logger.info(f"自动检测任务类型: {task_type}")

        X_raw = df.drop(columns=[target_col])
        y_raw = df[target_col].copy()
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw if task_type == 'classification' else None,
        )

        label_encoder = None
        if task_type == 'classification' and (
            y_train_raw.dtype == 'object'
            or pd.api.types.is_string_dtype(y_train_raw)
        ):
            label_encoder = LabelEncoder().fit(y_train_raw)
            unknown_labels = sorted(set(y_test_raw) - set(label_encoder.classes_))
            if unknown_labels:
                raise MLDataError(f"测试集包含训练集中不存在的类别: {unknown_labels}")
            y_train = pd.Series(
                label_encoder.transform(y_train_raw),
                index=y_train_raw.index,
                name=target_col,
            )
            y_test = pd.Series(
                label_encoder.transform(y_test_raw),
                index=y_test_raw.index,
                name=target_col,
            )
        else:
            y_train = y_train_raw.copy()
            y_test = y_test_raw.copy()

        # 报告统计只基于外层训练集。
        if progress_callback:
            progress_callback(
                'preprocessing',
                message=f'预处理 {X_train_raw.shape[1]} 列训练数据...',
            )
        report_df = X_train_raw.copy()
        report_df[target_col] = y_train
        report_preprocessor = DataPreprocessor()
        prep_result = report_preprocessor.fit_transform(
            report_df,
            target_col=target_col,
            task_type=task_type,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            scale_numeric=scale_numeric,
            scaler_type=scaler_type,
        )
        prep_result['preprocessing_info']['target_encoded'] = label_encoder is not None
        self._preprocessed = prep_result
        X_train_report = prep_result['X']

        if progress_callback:
            progress_callback('correlation', message='分析训练集特征与目标的相关性...')
        corr_result = self.feature_selector.analyze_correlation(
            X_train_report, y_train, task_type,
        )
        self._correlation_results = corr_result

        if progress_callback:
            progress_callback(
                'feature_selection',
                message=f'使用 {feature_selection_method} 筛选训练集特征...',
            )
        self.feature_selector.select_features(
            X_train_report,
            y_train,
            task_type,
            method=feature_selection_method,
            threshold=correlation_threshold,
            k=n_features,
        )

        # LazyPredict只看外层训练集中的内部训练/验证划分。
        if progress_callback:
            progress_callback('model_training', message='在训练集内部批量比较模型...')
        X_lazy_train_raw, X_lazy_val_raw, y_lazy_train, y_lazy_val = train_test_split(
            X_train_raw,
            y_train,
            test_size=0.2,
            random_state=random_state,
            stratify=y_train if task_type == 'classification' else None,
        )
        lazy_df = X_lazy_train_raw.copy()
        lazy_df[target_col] = y_lazy_train
        lazy_preprocessor = DataPreprocessor()
        lazy_prep_result = lazy_preprocessor.fit_transform(
            lazy_df,
            target_col=target_col,
            task_type=task_type,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            scale_numeric=scale_numeric,
            scaler_type=scaler_type,
        )
        X_lazy_train = lazy_prep_result['X']
        X_lazy_val = lazy_preprocessor.transform_new(X_lazy_val_raw)
        lazy_sel_result = self.feature_selector.select_features(
            X_lazy_train,
            y_lazy_train,
            task_type,
            method=feature_selection_method,
            threshold=correlation_threshold,
            k=n_features,
        )
        lazy_features = lazy_sel_result['selected_features']
        train_result = self.supervised_trainer.train_all(
            lazy_sel_result['X_selected'],
            X_lazy_val.reindex(columns=lazy_features, fill_value=0),
            y_lazy_train,
            y_lazy_val,
            task_type,
            random_state,
        )
        self._training_results = train_result
        best_model_name = train_result['best_model_name']
        best_model_class = train_result['best_model_class']
        best_rank_info = train_result['models'][0]

        (
            preprocessor_template,
            numeric_features,
            categorical_features,
        ) = _build_preprocessing_transformer(
            X_train_raw,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            scale_numeric=scale_numeric,
            scaler_type=scaler_type,
        )

        def pipeline_factory(model):
            return Pipeline([
                ('preprocessor', preprocessor_template),
                ('selector', StatisticalFeatureSelector(
                    task_type=task_type,
                    method=feature_selection_method,
                    threshold=correlation_threshold,
                    k=n_features,
                )),
                ('model', model),
            ])

        metric_name = 'r2' if task_type == 'regression' else 'balanced_accuracy'
        tuned_result = None
        final_estimator = None
        if tune:
            if progress_callback:
                progress_callback(
                    'hyperparameter_tuning',
                    message=f'Optuna 调参: {best_model_name}（{n_trials} 次试验）...',
                )
            try:
                tuned_result = self.tuner.tune(
                    best_model_class,
                    X_train_raw,
                    y_train,
                    task_type,
                    n_trials=n_trials,
                    cv_folds=cv_folds,
                    scoring=metric_name,
                    random_state=random_state,
                    pipeline_factory=pipeline_factory,
                )
                final_estimator = tuned_result['best_estimator']
            except Exception as e:
                logger.warning(f"调参失败（使用默认参数）: {e}")

        if final_estimator is None:
            try:
                default_model = best_model_class()
                if 'random_state' in default_model.get_params():
                    default_model.set_params(random_state=random_state)
                final_estimator = pipeline_factory(default_model)
                final_estimator.fit(X_train_raw, y_train)
            except Exception as e:
                raise MLTrainingError(f"默认模型训练失败: {e}")
            if tune:
                tuned_result = {
                    'success': True,
                    'model_name': best_model_name,
                    'best_params': default_model.get_params(),
                    'best_score': None,
                    'best_model': final_estimator.named_steps['model'],
                    'best_estimator': final_estimator,
                    'cv_metric': metric_name,
                    'optimization_history': [],
                    'n_trials_completed': 0,
                }
        self._tuned_result = tuned_result

        # 此处是测试集唯一一次进入模型流程的位置。
        test_predictions = final_estimator.predict(X_test_raw)
        test_score = float(
            r2_score(y_test, test_predictions)
            if task_type == 'regression'
            else balanced_accuracy_score(y_test, test_predictions)
        )

        final_preprocessor = final_estimator.named_steps['preprocessor']
        final_selector = final_estimator.named_steps['selector']
        final_model = final_estimator.named_steps['model']
        selected_features = list(final_selector.get_feature_names_out())
        X_train_selected = final_selector.transform(
            final_preprocessor.transform(X_train_raw),
        )
        self.preprocessor.set_fitted_pipeline(
            final_preprocessor,
            feature_names_in=X_train_raw.columns.tolist(),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_name=target_col,
            task_type=task_type,
            scaler_type=scaler_type,
            label_encoder=label_encoder,
        )
        self._selected_features = selected_features
        self._X_selected = X_train_selected
        self._y = y_train

        if tuned_result:
            tuned_result['test_metric'] = metric_name
            tuned_result['test_score'] = test_score

        saved_info = None
        if save_path:
            if progress_callback:
                progress_callback('model_saving', message='保存训练集模型...')
            metadata = {
                'model_class': best_model_name,
                'task_type': task_type,
                'feature_names': selected_features,
                'target_name': target_col,
                'training_metrics': _to_serializable(best_rank_info),
                'tuning_params': _to_serializable(tuned_result['best_params']) if tuned_result else None,
                'tuning_score': _safe_float(tuned_result['best_score']) if tuned_result else None,
                'cv_metric': metric_name,
                'test_metric': metric_name,
                'test_score': test_score,
                'sample_count': int(len(y_train)),
                'total_sample_count': int(len(y_raw)),
                'test_sample_count': int(len(y_test)),
                'feature_count': int(len(selected_features)),
                'test_size': test_size,
                'random_state': random_state,
            }
            preprocessor_bundle = self.preprocessor.get_bundle()
            preprocessor_bundle['selected_features'] = selected_features
            saved_info = self.manager.save_model(
                final_model,
                save_path,
                metadata=metadata,
                preprocessor_bundle=preprocessor_bundle,
            )
            self._saved_path = saved_info['model_path']

        feature_importance = self._extract_feature_importance(
            final_model, selected_features,
        )
        final_selection_scores = final_selector.selection_scores_
        logger.info(
            f"监督学习流水线完成: {best_model_name}, "
            f"{metric_name}测试得分={test_score:.4f}"
        )

        return {
            'success': True,
            'task_type': task_type,
            'data_metadata': import_result['metadata'],
            'preprocessing_info': prep_result['preprocessing_info'],
            'correlation_results': corr_result,
            'feature_selection': {
                'selected_features': selected_features,
                'removed_features': list(final_selector.removed_features_),
                'selection_scores': _to_serializable(final_selection_scores),
            },
            'training_results': {
                'models': _to_serializable(train_result['models']),
                'best_model_name': best_model_name,
                'n_models_trained': train_result['n_models_trained'],
            },
            'tuning_results': {
                'best_params': _to_serializable(tuned_result['best_params']),
                'best_score': _safe_float(tuned_result['best_score']),
                'n_trials': tuned_result['n_trials_completed'],
                'optimization_history': tuned_result['optimization_history'],
                'cv_metric': metric_name,
                'test_metric': metric_name,
                'test_score': test_score,
            } if tuned_result else None,
            'feature_importance': feature_importance,
            'saved_info': saved_info,
        }

    def run_unsupervised(
        self,
        file_path: str,
        feature_cols: Optional[List[str]] = None,
        n_clusters: int = 3,
        algorithms: Optional[List[str]] = None,
        n_clusters_range: Tuple[int, int] = (2, 10),
        dim_reduction: bool = True,
        random_state: int = 42,
        save_path: Optional[str] = None,
        progress_callback=None,
        **kwargs,
    ) -> Dict:
        """
        端到端无监督学习流水线。

        Parameters:
            file_path: 数据文件路径
            feature_cols: 指定特征列
            n_clusters: 聚类数
            algorithms: 聚类算法列表
            n_clusters_range: KMeans肘部法则k范围
            dim_reduction: 是否进行降维
            random_state: 随机种子
            save_path: 结果保存路径

        Returns:
            包含全流程结果的字典
        """
        logger.info("=" * 60)
        logger.info("开始无监督学习流水线")
        logger.info("=" * 60)

        # Step 1: 数据导入
        logger.info("[Step 1] 数据导入...")
        if progress_callback:
            progress_callback('data_import', message='加载数据文件...')
        import_result = self.importer.load_data(file_path, **kwargs)
        self._raw_data = import_result['data']
        self._data_metadata = import_result['metadata']

        df = import_result['data']
        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise MLDataError(f"指定特征列不存在: {missing}")
            df = df[feature_cols]

        # Step 2: 数据预处理
        logger.info("[Step 2] 数据预处理...")
        if progress_callback:
            progress_callback('preprocessing', message=f'预处理 {df.shape[1]} 列数据...')
        prep_result = self.preprocessor.fit_transform(
            df, target_col=None, task_type='unsupervised',
        )
        self._preprocessed = prep_result
        X = prep_result['X']

        # Step 3: 聚类
        logger.info("[Step 3] 聚类分析...")
        if progress_callback:
            progress_callback('clustering', message=f'聚类分析（K={n_clusters}）...')
        cluster_result = self.unsupervised_trainer.cluster(
            X, algorithms=algorithms, n_clusters=n_clusters,
            n_clusters_range=n_clusters_range, random_state=random_state,
        )

        # Step 4: 降维
        dim_result = None
        if dim_reduction:
            logger.info("[Step 4] 降维可视化...")
            if progress_callback:
                progress_callback('dim_reduction', message='PCA + t-SNE 降维...')
            dim_result = self.unsupervised_trainer.reduce_dimensions(
                X, random_state=random_state,
            )

        # 合并聚类标签到原始数据
        labeled_df = self._raw_data.copy()
        cluster_results = cluster_result.get('results', {})
        for algo_name, algo_result in cluster_results.items():
            if isinstance(algo_result, dict) and 'labels' in algo_result:
                labels = algo_result['labels']
                if len(labels) == len(labeled_df):
                    labeled_df[f'cluster_{algo_name}'] = labels

        logger.info("=" * 60)
        logger.info("无监督学习流水线完成")
        logger.info("=" * 60)

        # 收集聚类标签原始数据（用于可视化）
        cluster_labels = {}
        for algo_name, algo_result in cluster_results.items():
            if isinstance(algo_result, dict) and 'labels' in algo_result:
                cluster_labels[algo_name] = algo_result['labels']       

        result = {
            'success': True,
            'task_type': 'unsupervised',
            'data_metadata': import_result['metadata'],
            'preprocessing_info': prep_result['preprocessing_info'],
            'cluster_results': _to_serializable(cluster_result.get('comparison', [])),
            'cluster_labels': cluster_labels,
            'elbow_data': _to_serializable(cluster_result.get('elbow_data', [])),
            'dim_reduction': {
                method: {
                    k: _to_serializable(v) for k, v in data.items()
                } for method, data in (dim_result.get('results', {}) if dim_result else {}).items()
            } if dim_result else None,
            'labeled_data': labeled_df,
        }

        result = _to_serializable(result)

        return result

    def run_anomaly(
        self,
        file_path: str,
        feature_cols: Optional[List[str]] = None,
        contamination: float = 0.1,
        random_state: int = 42,
        progress_callback=None,
        **kwargs,
    ) -> Dict:
        """
        端到端异常检测流水线 (IsolationForest)。

        Parameters:
            file_path: 数据文件路径
            feature_cols: 指定特征列
            contamination: 预期异常比例 (0.01 ~ 0.5)
            random_state: 随机种子

        Returns:
            包含异常检测结果和降维可视化的字典
        """
        logger.info("=" * 60)
        logger.info("开始异常检测流水线")
        logger.info("=" * 60)

        # Step 1: 数据导入
        logger.info("[Step 1] 数据导入...")
        if progress_callback:
            progress_callback('data_import', message='加载数据文件...')
        import_result = self.importer.load_data(file_path, **kwargs)
        self._raw_data = import_result['data']
        self._data_metadata = import_result['metadata']

        df = import_result['data']
        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise MLDataError(f"指定特征列不存在: {missing}")
            df = df[feature_cols]

        # Step 2: 数据预处理
        logger.info("[Step 2] 数据预处理...")
        if progress_callback:
            progress_callback('preprocessing', message=f'预处理 {df.shape[1]} 列数据...')
        prep_result = self.preprocessor.fit_transform(
            df, target_col=None, task_type='unsupervised',
        )
        self._preprocessed = prep_result
        X = prep_result['X']

        # Step 3: 异常检测
        logger.info("[Step 3] 异常检测 (IsolationForest)...")
        if progress_callback:
            progress_callback('anomaly_detection', message=f'IsolationForest（污染度={contamination}）...')
        anomaly_result = self.unsupervised_trainer.detect_anomalies(
            X, contamination=contamination, random_state=random_state,
        )

        # Step 4: 降维可视化
        logger.info("[Step 4] 降维可视化...")
        if progress_callback:
            progress_callback('dim_reduction', message='PCA + t-SNE 降维...')
        dim_result = self.unsupervised_trainer.reduce_dimensions(
            X, random_state=random_state,
        )

        # 合并异常标签到原始数据
        labeled_df = self._raw_data.copy()
        anomaly_labels = anomaly_result['labels']
        if len(anomaly_labels) == len(labeled_df):
            labeled_df['anomaly_label'] = anomaly_labels
            labeled_df['anomaly_score'] = anomaly_result['anomaly_scores']

        logger.info("=" * 60)
        logger.info("异常检测流水线完成")
        logger.info("=" * 60)

        result = {
            'success': True,
            'task_type': 'unsupervised.anomaly',
            'data_metadata': import_result['metadata'],
            'preprocessing_info': prep_result['preprocessing_info'],
            'anomaly_results': {
                'n_anomalies': anomaly_result['n_anomalies'],
                'n_normal': anomaly_result['n_normal'],
                'anomaly_ratio': anomaly_result['anomaly_ratio'],
                'contamination': anomaly_result['contamination'],
            },
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_result['anomaly_scores'],
            'dim_reduction': {
                method: {
                    k: _to_serializable(v) for k, v in data.items()
                } for method, data in dim_result.get('results', {}).items()
            } if dim_result else None,
            'labeled_data': labeled_df,
        }

        result = _to_serializable(result)

        return result

    def run_dim_reduction(
        self,
        file_path: str,
        feature_cols: Optional[List[str]] = None,
        random_state: int = 42,
        progress_callback=None,
        **kwargs,
    ) -> Dict:
        """
        端到端降维可视化流水线 (PCA + t-SNE)。

        Parameters:
            file_path: 数据文件路径
            feature_cols: 指定特征列
            random_state: 随机种子

        Returns:
            包含降维结果的字典
        """
        logger.info("=" * 60)
        logger.info("开始降维可视化流水线")
        logger.info("=" * 60)

        # Step 1: 数据导入
        logger.info("[Step 1] 数据导入...")
        if progress_callback:
            progress_callback('data_import', message='加载数据文件...')
        import_result = self.importer.load_data(file_path, **kwargs)
        self._raw_data = import_result['data']
        self._data_metadata = import_result['metadata']

        df = import_result['data']
        if feature_cols:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise MLDataError(f"指定特征列不存在: {missing}")
            df = df[feature_cols]

        # Step 2: 数据预处理
        logger.info("[Step 2] 数据预处理...")
        if progress_callback:
            progress_callback('preprocessing', message=f'预处理 {df.shape[1]} 列数据...')
        prep_result = self.preprocessor.fit_transform(
            df, target_col=None, task_type='unsupervised',
        )
        self._preprocessed = prep_result
        X = prep_result['X']

        # Step 3: 降维
        logger.info("[Step 3] 降维可视化...")
        if progress_callback:
            progress_callback('dim_reduction', message='PCA + t-SNE 降维...')
        dim_result = self.unsupervised_trainer.reduce_dimensions(
            X, random_state=random_state,
        )

        logger.info("=" * 60)
        logger.info("降维可视化流水线完成")
        logger.info("=" * 60)

        result = {
            'success': True,
            'task_type': 'unsupervised.dim_reduction',
            'data_metadata': import_result['metadata'],
            'preprocessing_info': prep_result['preprocessing_info'],
            'dim_reduction': {
                method: {
                    k: _to_serializable(v) for k, v in data.items()
                } for method, data in dim_result.get('results', {}).items()
            } if dim_result else None,
        }

        result = _to_serializable(result)

        return result

    def predict_new(
        self,
        model_path: str,
        new_data_path: str,
        output_path: Optional[str] = None,
        output_format: str = 'csv',
        **kwargs,
    ) -> Dict:
        """
        使用已保存的模型对新数据进行预测。

        Parameters:
            model_path: 已保存模型的路径（.joblib）
            new_data_path: 新数据文件路径
            output_path: 预测结果导出路径
            output_format: 'csv' / 'xlsx'

        Returns:
            {
                'success': True,
                'predictions': np.ndarray,
                'predictions_df': pd.DataFrame,
                'metadata': dict,
            }
        """
        logger.info("=" * 60)
        logger.info("开始预测流水线")
        logger.info("=" * 60)

        # Step 1: 加载模型
        logger.info("[Step 1] 加载模型...")
        loaded = self.manager.load_model(model_path)
        model = loaded['model']
        metadata = loaded['metadata']
        preprocessor_bundle = loaded.get('preprocessor_bundle')

        # 恢复预处理器
        # 提取特征选择后的列名（如果有）
        selected_features = None
        if preprocessor_bundle and 'selected_features' in preprocessor_bundle:
            selected_features = preprocessor_bundle['selected_features']

        preprocessor = None
        if preprocessor_bundle:
            preprocessor = DataPreprocessor()
            preprocessor.load_bundle(preprocessor_bundle)

        # Step 2: 加载新数据
        logger.info("[Step 2] 加载新数据...")
        import_result = self.importer.load_data(new_data_path, **kwargs)
        new_data = import_result['data']

        # 严格列名校验：训练所需的原始特征列必须全部出现在预测数据中，
        # 否则拒绝预测——避免列名不一致(如 width/WD、大小写、首尾空格)时
        # 被静默填默认值，导致预测失真却不报错。
        if preprocessor_bundle and 'feature_names_in' in preprocessor_bundle:
            expected_cols = list(preprocessor_bundle['feature_names_in'])
            user_cols = set(new_data.columns)
            missing_cols = [c for c in expected_cols if c not in user_cols]
            extra_cols = [c for c in new_data.columns if c not in set(expected_cols)]
            if missing_cols:
                msg = "预测数据列名与训练时不一致，缺少训练所需特征列: {}".format(missing_cols)
                if extra_cols:
                    msg += "；同时存在未识别的列: {}".format(extra_cols)
                raise MLModelError(msg)
            if extra_cols:
                logger.warning("预测数据含未识别的列（将被忽略）: {}".format(extra_cols))

        # Step 3: 预处理 + 特征筛选
        logger.info("[Step 3] 预测...")
        if preprocessor is not None:
            X_processed = preprocessor.transform_new(new_data)
            if selected_features:
                # 按训练时选定的特征列强对齐：保证列顺序与训练一致，
                # 缺失列补0（OneHot未见类别本应为0），多余列丢弃
                X_processed = X_processed.reindex(
                    columns=selected_features, fill_value=0,
                )
        else:
            X_processed = new_data

        # 直接用处理好的特征预测
        try:
            predictions = model.predict(X_processed)
        except Exception as e:
            raise MLModelError(f"预测失败: {e}")

        # 构建结果DataFrame
        result_df = new_data.copy()
        if preprocessor is not None and preprocessor.label_encoder is not None:
            try:
                predictions_decoded = preprocessor.label_encoder.inverse_transform(
                    predictions.astype(int)
                )
                result_df['prediction'] = predictions_decoded
            except Exception:
                result_df['prediction'] = predictions
        else:
            result_df['prediction'] = predictions

        pred_result = {
            'success': True,
            'predictions': predictions,
            'predictions_df': result_df,
            'prediction_count': len(predictions),
        }

        # Step 4: 导出
        exported = None
        if output_path:
            logger.info("[Step 4] 导出结果...")
            exported = self.manager.export_predictions(
                pred_result['predictions_df'], output_path, output_format,
            )

        logger.info("=" * 60)
        logger.info(f"预测完成: {pred_result['prediction_count']}条")
        logger.info("=" * 60)

        return {
            'success': True,
            'predictions': pred_result['predictions'],
            'predictions_df': pred_result['predictions_df'],
            'prediction_count': pred_result['prediction_count'],
            'metadata': metadata,
            'exported': exported,
        }

    @staticmethod
    def _extract_feature_importance(model, feature_names: List[str]) -> Optional[List[Dict]]:
        """从模型中提取特征重要性。"""
        if model is None or feature_names is None:
            return None

        importance = None
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    importance = np.mean(np.abs(coef), axis=0)
                else:
                    importance = np.abs(coef)
        except Exception:
            pass

        if importance is None or len(importance) != len(feature_names):
            return None

        result = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(feature_names, importance)
        ]
        result.sort(key=lambda x: x['importance'], reverse=True)
        return result


# ============================================================
# Visualizer — 可视化（ECharts JSON + Matplotlib PNG）
# ============================================================

class Visualizer:
    """
    ML可视化器：从各模块返回的数据生成图表。
    每个方法同时返回 ECharts JSON 配置和 Matplotlib PNG 图片路径。

    用法:
        viz = Visualizer(output_dir='charts/')
        chart = viz.plot_correlation_heatmap(correlations)
        # chart['echarts'] → 前端ECharts配置
        # chart['png_path'] → PNG图片路径

        # 一键生成
        charts = viz.generate_supervised_charts(pipeline_result)
    """

    # 中英文标签映射
    _LABELS = {
        'zh': {
            'correlation_title': '特征相关性分析',
            'correlation_x': 'F统计量',
            'importance_title': '特征重要性',
            'importance_x': '重要性',
            'selection_title': '特征选择得分',
            'model_comparison_reg': '模型回归性能对比',
            'model_comparison_clf': '模型分类性能对比',
            'model_score': '得分',
            'opt_history_title': 'Optuna调参优化过程',
            'opt_history_x': '试验轮次',
            'opt_history_y': '交叉验证得分',
            'elbow_title': 'KMeans肘部法则',
            'elbow_x': '聚类数K',
            'elbow_y': 'SSE (Inertia)',
            'cluster_comp_title': '聚类算法轮廓系数对比',
            'silhouette': '轮廓系数',
            'scree_title': 'PCA方差解释比例',
            'scree_bar': '方差占比',
            'scree_line': '累积方差',
            'variance_ratio': '方差占比',
            'scatter_pca': 'PCA降维可视化',
            'scatter_tsne': 't-SNE降维可视化',
            'missing_title': '数据缺失值统计',
            'missing_count': '缺失数量',
            'algorithm': '算法',
            'feature': '特征',
            'anomaly_scatter_title': '异常检测散点图',
            'anomaly_normal': '正常样本',
            'anomaly_anomaly': '异常样本',
            'anomaly_score_title': '异常评分分布',
            'anomaly_score_x': '评分',
            'anomaly_score_y': '样本数',
            'anomaly_summary_title': '异常/正常样本占比',
        },
        'en': {
            'correlation_title': 'Feature Correlation Analysis',
            'correlation_x': 'F-Score',
            'importance_title': 'Feature Importance',
            'importance_x': 'Importance',
            'selection_title': 'Feature Selection Scores',
            'model_comparison_reg': 'Model Regression Performance',
            'model_comparison_clf': 'Model Classification Performance',
            'model_score': 'Score',
            'opt_history_title': 'Optuna Optimization History',
            'opt_history_x': 'Trial',
            'opt_history_y': 'CV Score',
            'elbow_title': 'KMeans Elbow Method',
            'elbow_x': 'Number of Clusters (K)',
            'elbow_y': 'SSE (Inertia)',
            'cluster_comp_title': 'Clustering Silhouette Score Comparison',
            'silhouette': 'Silhouette Score',
            'scree_title': 'PCA Explained Variance Ratio',
            'scree_bar': 'Variance Ratio',
            'scree_line': 'Cumulative Variance',
            'variance_ratio': 'Variance Ratio',
            'scatter_pca': 'PCA Visualization',
            'scatter_tsne': 't-SNE Visualization',
            'missing_title': 'Missing Values',
            'missing_count': 'Missing Count',
            'algorithm': 'Algorithm',
            'feature': 'Feature',
            'anomaly_scatter_title': 'Anomaly Detection Scatter',
            'anomaly_normal': 'Normal',
            'anomaly_anomaly': 'Anomaly',
            'anomaly_score_title': 'Anomaly Score Distribution',
            'anomaly_score_x': 'Score',
            'anomaly_score_y': 'Count',
            'anomaly_summary_title': 'Normal vs Anomaly',
        },
    }

    def __init__(self, output_dir: str = '.', figsize=(10, 6), dpi: int = 150, lang: str = 'zh'):
        """
        Parameters:
            output_dir: PNG图片保存目录（None则不生成PNG）
            figsize: 图片尺寸 (宽, 高) 英寸
            dpi: 图片分辨率
            lang: 'zh' 或 'en'
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.lang = lang if lang in self._LABELS else 'zh'
        self._L = self._LABELS[self.lang]

    def _label(self, key: str) -> str:
        return self._L.get(key, key)

    def _save_png(self, fig, name: str) -> Optional[str]:
        """保存matplotlib图片并关闭figure。"""
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None
        # if self.output_dir is None:
        #     import matplotlib.pyplot as plt
        #     plt.close(fig)
        #     return None
        # os.makedirs(self.output_dir, exist_ok=True)
        # path = os.path.join(self.output_dir, f'{name}.png')
        # fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        # import matplotlib.pyplot as plt
        # plt.close(fig)
        # return path

    # ----------------------------------------------------------
    # 1. 相关性分析条形图
    # ----------------------------------------------------------

    def plot_correlation_heatmap(self, correlations: List[Dict]) -> Dict:
        """
        特征相关性/统计检验得分条形图。

        Parameters:
            correlations: FeatureSelector.analyze_correlation() 返回的 correlations 列表

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not correlations:
            return {'echarts': {}, 'png_path': None}

        features = [c['feature'] for c in correlations]
        scores = [_safe_float(c.get('score')) or 0 for c in correlations]
        p_values = [_safe_float(c.get('p_value')) for c in correlations]

        # ECharts
        echarts = {
            'mlTheme': {'barMode': 'rank', 'rankBy': 'abs'},
            'title': {'text': self._label('correlation_title')},
            'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
            'grid': {'left': '20%', 'right': '10%', 'bottom': '10%', 'top': '15%'},
            'xAxis': {'type': 'value', 'name': self._label('correlation_x')},
            'yAxis': {
                'type': 'category',
                'data': features[::-1],
                'axisLabel': {'fontSize': 10},
            },
            'series': [{
                'type': 'bar',
                'data': [round(v, 2) for v in scores[::-1]],
                'label': {'show': True, 'position': 'right', 'formatter': '{c}'},
            }],
        }

        # Matplotlib
        png_path = self._draw_correlation_png(features, scores, p_values)

        return {'echarts': echarts, 'png_path': png_path}

    def _draw_correlation_png(self, features, scores, p_values) -> Optional[str]:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt        
        plt.rcParams['font.sans-serif'] = ['SimHei']

        fig, ax = plt.subplots(figsize=self.figsize)
        colors = [CHART_BLUE if (p is not None and p < 0.05) else CHART_MID_GRAY for p in p_values]
        y_pos = range(len(features))
        bars = ax.barh(y_pos, scores, color=colors[::-1])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features[::-1], fontsize=9)
        ax.set_xlabel(self._label('correlation_x'))
        ax.set_title(self._label('correlation_title'))
        # 标注数值
        for bar, val in zip(bars, scores[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=8)
        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=CHART_BLUE, label='p < 0.05'),
            Patch(facecolor=CHART_MID_GRAY, label='p >= 0.05'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        fig.tight_layout()
        return self._save_png(fig, 'correlation_heatmap')

    # ----------------------------------------------------------
    # 2. 特征重要性条形图
    # ----------------------------------------------------------

    def plot_feature_importance(self, feature_importance: List[Dict]) -> Dict:
        """
        特征重要性横向条形图。

        Parameters:
            feature_importance: MLPipeline._extract_feature_importance() 返回值

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not feature_importance:
            return {'echarts': {}, 'png_path': None}

        features = [f['feature'] for f in feature_importance]
        values = [f['importance'] for f in feature_importance]

        # ECharts
        echarts = {
            'mlTheme': {'barMode': 'rank'},
            'title': {'text': self._label('importance_title')},
            'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
            'grid': {'left': '25%', 'right': '10%', 'bottom': '10%', 'top': '15%'},
            'xAxis': {'type': 'value', 'name': self._label('importance_x')},
            'yAxis': {'type': 'category', 'data': features[::-1], 'axisLabel': {'fontSize': 10}},
            'series': [{
                'type': 'bar',
                'data': [round(v, 4) for v in values[::-1]],
                'label': {'show': True, 'position': 'right', 'formatter': '{c}'},
            }],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            y_pos = range(len(features))
            ax.barh(y_pos, values, color=_chart_bar_colors(len(values)))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features[::-1], fontsize=9)
            ax.set_xlabel(self._label('importance_x'))
            ax.set_title(self._label('importance_title'))
            fig.tight_layout()
            png_path = self._save_png(fig, 'feature_importance')
        except Exception as e:
            logger.warning(f"特征重要性PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 3. 特征选择得分图
    # ----------------------------------------------------------

    def plot_feature_selection(self, selection_scores: Dict, selected_features: List[str]) -> Dict:
        """
        特征选择得分分组条形图。

        Parameters:
            selection_scores: FeatureSelector.select_features() 返回的 selection_scores
            selected_features: 被选中的特征列表

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not selection_scores:
            return {'echarts': {}, 'png_path': None}

        features = list(selection_scores.keys())
        f_scores = [_safe_float(selection_scores[f].get('f_score')) or 0 for f in features]
        mi_scores = [_safe_float(selection_scores[f].get('mutual_info')) or 0 for f in features]
        is_selected = [f in selected_features for f in features]

        # ECharts
        echarts = {
            'title': {'text': self._label('selection_title')},
            'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
            'legend': {'data': ['F-Score', 'Mutual Info']},
            'grid': {'left': '10%', 'right': '5%', 'bottom': '25%', 'top': '15%'},
            'xAxis': {'type': 'category', 'data': features, 'axisLabel': {'rotate': 45, 'fontSize': 9}},
            'yAxis': {'type': 'value'},
            'series': [
                {
                    'name': 'F-Score', 'type': 'bar',
                    'data': [{'value': v, 'itemStyle': {'color': CHART_BLUE if s else CHART_LIGHT_GRAY}}
                             for v, s in zip(f_scores, is_selected)],
                },
                {
                    'name': 'Mutual Info', 'type': 'bar',
                    'data': [{'value': v, 'itemStyle': {'color': CHART_GREEN if s else CHART_LIGHT_GRAY}}
                             for v, s in zip(mi_scores, is_selected)],
                },
            ],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            x = np.arange(len(features))
            width = 0.35
            bar1 = ax.bar(x - width / 2, f_scores, width, label='F-Score',
                          color=[CHART_BLUE if s else CHART_LIGHT_GRAY for s in is_selected])
            bar2 = ax.bar(x + width / 2, mi_scores, width, label='Mutual Info',
                          color=[CHART_GREEN if s else CHART_LIGHT_GRAY for s in is_selected])
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
            ax.set_title(self._label('selection_title'))
            ax.legend(fontsize=8)
            fig.tight_layout()
            png_path = self._save_png(fig, 'feature_selection')
        except Exception as e:
            logger.warning(f"特征选择PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 4. 模型性能对比条形图
    # ----------------------------------------------------------

    def plot_model_comparison(self, models: List[Dict], task_type: str) -> Dict:
        """
        lazypredict所有模型得分对比条形图。

        Parameters:
            models: SupervisedTrainer.train_all() 返回的 models 列表
            task_type: 'regression' / 'classification'

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not models:
            return {'echarts': {}, 'png_path': None}

        model_names = [m['model_name'] for m in models]
        if task_type == 'regression':
            scores = [_safe_float(m.get('adjusted_r_squared')) or 0 for m in models]
            score_label = 'Adjusted R²'
        else:
            scores = [_safe_float(m.get('balanced_accuracy')) or 0 for m in models]
            score_label = 'Balanced Accuracy'

        display_pairs = sorted(zip(model_names, scores), key=lambda item: item[1])
        display_model_names = [name for name, _ in display_pairs]
        display_scores = [score for _, score in display_pairs]

        title = self._label('model_comparison_reg' if task_type == 'regression'
                            else 'model_comparison_clf')

        # ECharts
        echarts = {
            'mlTheme': {'barMode': 'rank'},
            'title': {'text': title},
            'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
            'grid': {'left': '30%', 'right': '10%', 'bottom': '10%', 'top': '15%'},
            'xAxis': {'type': 'value', 'name': score_label},
            'yAxis': {
                'type': 'category',
                'data': display_model_names,
                'axisLabel': {'fontSize': 8},
            },
            'series': [{
                'type': 'bar',
                'data': [round(v, 4) for v in display_scores],
                'label': {'show': True, 'position': 'right', 'fontSize': 8,
                          'formatter': '{c}'},
            }],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(self.figsize[0], max(self.figsize[1], len(models) * 0.3)))
            y_pos = range(len(display_model_names))
            colors = _ranked_bar_colors(display_scores)
            ax.barh(y_pos, display_scores, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(display_model_names, fontsize=7)
            ax.set_xlabel(score_label)
            ax.set_title(title)
            fig.tight_layout()
            png_path = self._save_png(fig, 'model_comparison')
        except Exception as e:
            logger.warning(f"模型对比PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 5. Optuna调参优化历史折线图
    # ----------------------------------------------------------

    def plot_optimization_history(self, history: List[float],
                                  model_name: str = '') -> Dict:
        """
        Optuna优化历史折线图。

        Parameters:
            history: ModelTuner.tune() 返回的 optimization_history
            model_name: 模型名称（用于标题）

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not history:
            return {'echarts': {}, 'png_path': None}

        # 计算累积最佳得分
        best_so_far = []
        current_best = float('-inf')
        for s in history:
            current_best = max(current_best, s)
            best_so_far.append(current_best)

        title = self._label('opt_history_title')
        if model_name:
            title += f' - {model_name}'

        # ECharts
        echarts = {
            'mlTheme': {'chartType': 'optimization'},
            'title': {'text': title},
            'tooltip': {'trigger': 'axis'},
            'grid': {'left': '10%', 'right': '5%', 'bottom': '18%', 'top': '15%'},
            'xAxis': {'type': 'category', 'data': list(range(1, len(history) + 1)),
                      'name': self._label('opt_history_x')},
            'yAxis': {'type': 'value', 'name': self._label('opt_history_y')},
            'series': [
                {
                    'name': 'Trial Score', 'type': 'line', 'data': history,
                    'symbol': 'circle', 'symbolSize': 4,
                    'lineStyle': {'color': CHART_MID_GRAY, 'width': 1.8},
                    'itemStyle': {'color': CHART_MID_GRAY},
                },
                {
                    'name': 'Best Score', 'type': 'line', 'data': best_so_far,
                    'symbol': 'none',
                    'lineStyle': {'color': CHART_BLUE, 'width': 2.2},
                    'itemStyle': {'color': CHART_BLUE},
                },
            ],
            'legend': {'data': ['Trial Score', 'Best Score'], 'bottom': 0},
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            trials = list(range(1, len(history) + 1))
            ax.plot(trials, history, 'o-', color=CHART_MID_GRAY, markersize=3,
                    label='Trial Score', alpha=0.7)
            ax.plot(trials, best_so_far, '-', color=CHART_BLUE, linewidth=2,
                    label='Best Score')
            ax.set_xlabel(self._label('opt_history_x'))
            ax.set_ylabel(self._label('opt_history_y'))
            ax.set_title(title)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png_path = self._save_png(fig, 'optimization_history')
        except Exception as e:
            logger.warning(f"优化历史PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 6. KMeans肘部法则折线图
    # ----------------------------------------------------------

    def plot_elbow_curve(self, elbow_data: List[Dict]) -> Dict:
        """
        KMeans肘部法则折线图。

        Parameters:
            elbow_data: UnsupervisedTrainer.cluster() 返回的 elbow_data

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not elbow_data:
            return {'echarts': {}, 'png_path': None}

        ks = [d['k'] for d in elbow_data]
        inertias = [d['inertia'] for d in elbow_data]

        # ECharts
        echarts = {
            'title': {'text': self._label('elbow_title')},
            'tooltip': {'trigger': 'axis'},
            'grid': {'left': '12%', 'right': '5%', 'bottom': '15%', 'top': '15%'},
            'xAxis': {'type': 'category', 'data': ks, 'name': self._label('elbow_x')},
            'yAxis': {'type': 'value', 'name': self._label('elbow_y')},
            'series': [{
                'type': 'line', 'data': [int(v) for v in inertias],
                'symbol': 'circle', 'symbolSize': 8,
                'areaStyle': {'color': CHART_BLUE_AREA},
                'label': {'show': True, 'formatter': '{c}', 'fontSize': 8},
            }],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(ks, inertias, 'o-', color=CHART_BLUE, linewidth=2, markersize=8)
            ax.fill_between(ks, inertias, alpha=0.15, color=CHART_BLUE)
            ax.set_xlabel(self._label('elbow_x'))
            ax.set_ylabel(self._label('elbow_y'))
            ax.set_title(self._label('elbow_title'))
            ax.set_xticks(ks)
            ax.grid(True, alpha=0.3)
            for k, inertia in zip(ks, inertias):
                ax.annotate(f'{inertia:.0f}', (k, inertia), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8)
            fig.tight_layout()
            png_path = self._save_png(fig, 'elbow_curve')
        except Exception as e:
            logger.warning(f"肘部法则PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 7. 聚类算法轮廓系数对比
    # ----------------------------------------------------------

    def plot_cluster_comparison(self, comparison: List[Dict]) -> Dict:
        """
        各聚类算法轮廓系数对比条形图。

        Parameters:
            comparison: UnsupervisedTrainer.cluster() 返回的 comparison 列表

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not comparison:
            return {'echarts': {}, 'png_path': None}

        # 缩短算法名用于图表显示
        _ALGO_SHORT = {
            'KMeans': 'KMeans',
            'DBSCAN': 'DBSCAN',
            'AgglomerativeClustering': 'Agglo.',
            'SpectralClustering': 'Spectral',
            'GaussianMixture': 'GMM',
        }
        algos = [c['algorithm'] for c in comparison]
        algos_short = [_ALGO_SHORT.get(a, a) for a in algos]
        scores = [_safe_float(c.get('silhouette_score')) for c in comparison]
        n_clusters = [c.get('n_clusters', 0) for c in comparison]

        # ECharts
        echarts = {
            'title': {'text': self._label('cluster_comp_title')},
            'tooltip': {
                'trigger': 'axis',
                'formatter': '{b}<br/>{a}: {c}',
            },
            'grid': {'left': '10%', 'right': '5%', 'bottom': '15%', 'top': '15%'},
            'xAxis': {'type': 'category', 'data': algos_short, 'name': self._label('algorithm')},
            'yAxis': {'type': 'value', 'name': self._label('silhouette')},
            'series': [{
                'type': 'bar', 'data': [round(v, 4) if v is not None else 'N/A' for v in scores],
                'label': {'show': True, 'position': 'top', 'formatter': '{c}'},
            }],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            colors = _chart_bar_colors(len(algos_short))
            # None 值替换为 0 画柱，但标注 N/A
            plot_scores = [s if s is not None else 0 for s in scores]
            bars = ax.bar(algos_short, plot_scores, color=colors[:len(algos_short)])
            ax.set_ylabel(self._label('silhouette'))
            ax.set_title(self._label('cluster_comp_title'))
            for bar, score, nc in zip(bars, scores, n_clusters):
                score_text = f'{score:.3f}\n(k={nc})' if score is not None else f'N/A\n(k={nc})'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        score_text, ha='center', fontsize=9)
            valid_scores = [s for s in scores if s is not None]
            ax.set_ylim(0, max(valid_scores) * 1.2 if valid_scores else 1)
            fig.tight_layout()
            png_path = self._save_png(fig, 'cluster_comparison')
        except Exception as e:
            logger.warning(f"聚类对比PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 8. PCA碎石图
    # ----------------------------------------------------------

    def plot_scree(self, explained_variance_ratio: List[float],
                   cumulative_variance: List[float]) -> Dict:
        """
        PCA方差解释比例碎石图。

        Parameters:
            explained_variance_ratio: PCA 的 explained_variance_ratio 列表
            cumulative_variance: PCA 的 cumulative_variance 列表

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not explained_variance_ratio:
            return {'echarts': {}, 'png_path': None}

        components = [f'PC{i + 1}' for i in range(len(explained_variance_ratio))]
        ratios = [float(v) for v in explained_variance_ratio]
        cumul = [float(v) for v in cumulative_variance] if cumulative_variance else []

        # ECharts
        echarts = {
            'title': {'text': self._label('scree_title')},
            'tooltip': {'trigger': 'axis'},
            'legend': {'data': [self._label('scree_bar'), self._label('scree_line')]},
            'grid': {'left': '10%', 'right': '10%', 'bottom': '15%', 'top': '15%'},
            'xAxis': {'type': 'category', 'data': components},
            'yAxis': [
                {'type': 'value', 'name': self._label('variance_ratio'),
                 'axisLabel': {'formatter': '{value}%'}},
                {'type': 'value', 'name': self._label('scree_line'),
                 'axisLabel': {'formatter': '{value}%'}},
            ],
            'series': [
                {
                    'name': self._label('scree_bar'), 'type': 'bar',
                    'data': [round(r * 100, 1) for r in ratios],
                    'label': {'show': True, 'formatter': '{c}%', 'fontSize': 8},
                },
                {
                    'name': self._label('scree_line'), 'type': 'line',
                    'data': [round(c * 100, 1) for c in cumul] if cumul else [],
                    'yAxisIndex': 1,
                    'symbol': 'circle', 'symbolSize': 6,
                },
            ],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=self.figsize)
            x = range(len(components))
            ax1.bar(x, [r * 100 for r in ratios], color=CHART_BLUE, alpha=0.8,
                    label=self._label('scree_bar'))
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel(self._label('variance_ratio') + ' (%)', color=CHART_BLUE)
            ax1.set_xticks(x)
            ax1.set_xticklabels(components)
            ax1.tick_params(axis='y', labelcolor=CHART_BLUE)

            if cumul:
                ax2 = ax1.twinx()
                ax2.plot(x, [c * 100 for c in cumul], 'o-', color=CHART_GREEN,
                         linewidth=2, markersize=6, label=self._label('scree_line'))
                ax2.set_ylabel(self._label('scree_line') + ' (%)', color=CHART_GREEN)
                ax2.tick_params(axis='y', labelcolor=CHART_GREEN)

            ax1.set_title(self._label('scree_title'))
            fig.tight_layout()
            png_path = self._save_png(fig, 'scree_plot')
        except Exception as e:
            logger.warning(f"碎石图PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 9. 散点图（PCA / t-SNE / 聚类）
    # ----------------------------------------------------------

    def plot_scatter(self, coordinates, labels=None, title: str = '',
                    algorithm_name: str = '') -> Dict:
        """
        2D散点图（PCA/t-SNE降维可视化 或 聚类结果）。

        Parameters:
            coordinates: shape (n_samples, 2) 的坐标数组
            labels: 聚类/分类标签（可选）
            title: 图表标题
            algorithm_name: 算法名称（用于文件名）

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if coordinates is None:
            return {'echarts': {}, 'png_path': None}

        coords = np.array(coordinates)
        if coords.ndim != 2 or coords.shape[1] < 2:
            return {'echarts': {}, 'png_path': None}

        x_vals = coords[:, 0].tolist()
        y_vals = coords[:, 1].tolist()

        if labels is not None:
            labels = np.array(labels).tolist()
        else:
            labels = [0] * len(x_vals)

        unique_labels = sorted(set(labels))
        # 为每个标签生成一个系列
        series_data = {}
        for ul in unique_labels:
            series_data[ul] = {'x': [], 'y': []}
        for xi, yi, li in zip(x_vals, y_vals, labels):
            series_data[li]['x'].append(xi)
            series_data[li]['y'].append(yi)

        # 颜色列表
        palette = CHART_BRAND_PALETTE + [CHART_OVERFLOW]

        if not title:
            title = self._label('scatter_pca')

        # ECharts 与 PNG 共用附件采样色板。
        echarts_palette = CHART_BRAND_PALETTE + [CHART_OVERFLOW]
        echarts_series = []
        for i, ul in enumerate(unique_labels):
            color = echarts_palette[i % len(echarts_palette)]
            echarts_series.append({
                'name': f'Cluster {ul}' if ul != -1 else 'Noise',
                'type': 'scatter',
                'data': [[round(x, 2), round(y, 2)]
                         for x, y in zip(series_data[ul]['x'], series_data[ul]['y'])],
                'symbolSize': 6,
                'itemStyle': {'color': color},
            })

        echarts = {
            'title': {'text': title},
            'tooltip': {
                'trigger': 'item',
            },
            'grid': {'left': '10%', 'right': '5%', 'bottom': '15%', 'top': '15%'},
            'xAxis': {'type': 'value', 'name': 'Component 1'},
            'yAxis': {'type': 'value', 'name': 'Component 2'},
            'series': echarts_series,
            'legend': {},
        }

        # Matplotlib
        fname = f'scatter_{algorithm_name}' if algorithm_name else 'scatter'
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            for i, ul in enumerate(unique_labels):
                color = palette[i % len(palette)]
                label_name = f'Cluster {ul}' if ul != -1 else 'Noise'
                mask = [li == ul for li in labels]
                ax.scatter(
                    [xi for xi, m in zip(x_vals, mask) if m],
                    [yi for yi, m in zip(y_vals, mask) if m],
                    c=color, label=label_name, s=20, alpha=0.7, edgecolors='white', linewidths=0.3,
                )
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            png_path = self._save_png(fig, fname)
        except Exception as e:
            logger.warning(f"散点图PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 10. 数据缺失值统计图
    # ----------------------------------------------------------

    def plot_data_preview(self, data_metadata: Dict) -> Dict:
        """
        数据概览：缺失值统计条形图。

        Parameters:
            data_metadata: DataImporter.load_data() 返回的 metadata

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not data_metadata:
            return {'echarts': {}, 'png_path': None}

        missing = data_metadata.get('missing_values', {})
        # 只显示有缺失值的列
        cols_with_missing = {k: v for k, v in missing.items() if v > 0}

        if not cols_with_missing:
            # 无缺失值，展示全部列的数据类型
            dtypes = data_metadata.get('dtypes', {})
            echarts = {
                'title': {'text': self._label('missing_title') + ' (No Missing Values)'},
                'tooltip': {},
                'series': [{'type': 'pie', 'data': [
                    {'name': k, 'value': 1} for k in dtypes.keys()
                ], 'radius': '50%'}],
            }
            return {'echarts': echarts, 'png_path': None}

        features = list(cols_with_missing.keys())
        counts = list(cols_with_missing.values())

        # ECharts
        echarts = {
            'title': {'text': self._label('missing_title')},
            'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}},
            'grid': {'left': '10%', 'right': '5%', 'bottom': '25%', 'top': '15%'},
            'xAxis': {'type': 'category', 'data': features,
                      'axisLabel': {'rotate': 45, 'fontSize': 9}},
            'yAxis': {'type': 'value', 'name': self._label('missing_count')},
            'series': [{
                'type': 'bar', 'data': counts,
                'label': {'show': True, 'position': 'top'},
            }],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.bar(features, counts, color=_chart_bar_colors(len(features)))
            ax.set_ylabel(self._label('missing_count'))
            ax.set_title(self._label('missing_title'))
            ax.tick_params(axis='x', rotation=45)
            for i, (f, c) in enumerate(zip(features, counts)):
                ax.text(i, c + 0.5, str(c), ha='center', fontsize=9)
            fig.tight_layout()
            png_path = self._save_png(fig, 'data_preview')
        except Exception as e:
            logger.warning(f"数据概览PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    # ----------------------------------------------------------
    # 一键生成：监督学习
    # ----------------------------------------------------------

    def generate_supervised_charts(self, pipeline_result: Dict) -> Dict:
        """
        从 MLPipeline.run_supervised() 返回值一键生成所有监督学习图表。

        Returns:
            {
                'data_preview': {...},
                'correlation': {...},
                'feature_selection': {...},
                'model_comparison': {...},
                'optimization': {...},
                'feature_importance': {...},
            }
        """
        charts = {}

        # 数据概览
        if pipeline_result.get('data_metadata'):
            charts['data_preview'] = self.plot_data_preview(pipeline_result['data_metadata'])

        # 相关性分析
        if pipeline_result.get('correlation_results', {}).get('correlations'):
            charts['correlation'] = self.plot_correlation_heatmap(
                pipeline_result['correlation_results']['correlations']
            )

        # 特征选择
        if pipeline_result.get('feature_selection', {}).get('selection_scores'):
            charts['feature_selection'] = self.plot_feature_selection(
                pipeline_result['feature_selection']['selection_scores'],
                pipeline_result['feature_selection']['selected_features'],
            )

        # 模型对比
        if pipeline_result.get('training_results', {}).get('models'):
            charts['model_comparison'] = self.plot_model_comparison(
                pipeline_result['training_results']['models'],
                pipeline_result['task_type'],
            )

        # 调参历史
        tuning = pipeline_result.get('tuning_results')
        if tuning and tuning.get('optimization_history'):
            charts['optimization'] = self.plot_optimization_history(
                tuning['optimization_history'],
                pipeline_result.get('training_results', {}).get('best_model_name', ''),
            )

        # 特征重要性
        if pipeline_result.get('feature_importance'):
            charts['feature_importance'] = self.plot_feature_importance(
                pipeline_result['feature_importance']
            )

        return charts

    # ----------------------------------------------------------
    # 一键生成：无监督学习
    # ----------------------------------------------------------

    def generate_unsupervised_charts(self, pipeline_result: Dict) -> Dict:
        """
        从 MLPipeline.run_unsupervised() 返回值一键生成所有无监督学习图表。

        Returns:
            {
                'data_preview': {...},
                'elbow': {...},
                'cluster_comparison': {...},
                'scree': {...},
                'scatter_pca': {...},
                'scatter_tsne': {...},
            }
        """
        charts = {}

        # 数据概览
        if pipeline_result.get('data_metadata'):
            charts['data_preview'] = self.plot_data_preview(pipeline_result['data_metadata'])

        # 肘部法则
        if pipeline_result.get('elbow_data'):
            charts['elbow'] = self.plot_elbow_curve(pipeline_result['elbow_data'])

        # 聚类对比
        if pipeline_result.get('cluster_results'):
            charts['cluster_comparison'] = self.plot_cluster_comparison(
                pipeline_result['cluster_results']
            )

        # 降维可视化
        dim = pipeline_result.get('dim_reduction') or {}
        cluster_labels = pipeline_result.get('cluster_labels', {})
        # 优先使用KMeans标签
        default_labels = cluster_labels.get('KMeans', None)

        if dim.get('PCA'):
            pca = dim['PCA']
            charts['scree'] = self.plot_scree(
                pca.get('explained_variance_ratio', []),
                pca.get('cumulative_variance', []),
            )
            if pca.get('coordinates'):
                charts['scatter_pca'] = self.plot_scatter(
                    pca['coordinates'], labels=default_labels,
                    title=self._label('scatter_pca'), algorithm_name='pca',
                )

        if dim.get('TSNE'):
            tsne = dim['TSNE']
            if tsne.get('coordinates'):
                charts['scatter_tsne'] = self.plot_scatter(
                    tsne['coordinates'], labels=default_labels,
                    title=self._label('scatter_tsne'), algorithm_name='tsne',
                )

        return charts

    # ----------------------------------------------------------
    # 一键生成：异常检测
    # ----------------------------------------------------------

    def plot_anomaly_score_distribution(self, anomaly_scores, labels,
                                         title: str = '') -> Dict:
        """
        异常评分分布直方图。

        Parameters:
            anomaly_scores: IsolationForest decision_function 输出
            labels: 异常标签 (-1=异常, 1=正常)
            title: 图表标题

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if anomaly_scores is None or labels is None:
            return {'echarts': {}, 'png_path': None}

        scores = np.array(anomaly_scores)
        labels = np.array(labels)

        if not title:
            title = self._label('anomaly_score_title')

        # 分离正常/异常评分
        normal_scores = scores[labels == 1].tolist()
        anomaly_scores_filtered = scores[labels == -1].tolist()

        # 共享分桶边界
        all_scores = scores.tolist()
        n_bins = 25
        x_min = float(min(all_scores))
        x_max = float(max(all_scores))
        if x_min == x_max:
            x_max = x_min + 1
        bin_width = (x_max - x_min) / n_bins
        bins = [round(x_min + i * bin_width, 4) for i in range(n_bins + 1)]
        bin_labels = [round((bins[i] + bins[i + 1]) / 2, 4) for i in range(n_bins)]

        def histogram(values):
            counts = [0] * n_bins
            for v in values:
                idx = int((v - x_min) / bin_width)
                if idx >= n_bins:
                    idx = n_bins - 1
                counts[idx] += 1
            return counts

        normal_counts = histogram(normal_scores)
        anomaly_counts = histogram(anomaly_scores_filtered)

        echarts = {
            'title': {'text': title},
            'tooltip': {'trigger': 'axis'},
            'grid': {'left': '10%', 'right': '5%', 'bottom': '15%', 'top': '15%'},
            'xAxis': {
                'type': 'category',
                'data': [round(b, 3) for b in bin_labels],
                'name': self._label('anomaly_score_x'),
            },
            'yAxis': {
                'type': 'value',
                'name': self._label('anomaly_score_y'),
            },
            'series': [
                {
                    'name': self._label('anomaly_normal'),
                    'type': 'bar',
                    'stack': 'total',
                    'data': normal_counts,
                    'itemStyle': {'color': CHART_GREEN},
                },
                {
                    'name': self._label('anomaly_anomaly'),
                    'type': 'bar',
                    'stack': 'total',
                    'data': anomaly_counts,
                    'itemStyle': {'color': CHART_BLUE},
                },
            ],
            'legend': {},
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.hist(normal_scores, bins=bins, color=CHART_GREEN, alpha=0.7,
                    label=self._label('anomaly_normal'))
            ax.hist(anomaly_scores_filtered, bins=bins, color=CHART_BLUE, alpha=0.7,
                    label=self._label('anomaly_anomaly'))
            ax.set_xlabel(self._label('anomaly_score_x'))
            ax.set_ylabel(self._label('anomaly_score_y'))
            ax.set_title(title)
            ax.legend(fontsize=8)
            fig.tight_layout()
            png_path = self._save_png(fig, 'anomaly_score_distribution')
        except Exception as e:
            logger.warning(f"异常评分分布PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    def plot_anomaly_summary(self, n_normal: int, n_anomalies: int,
                              title: str = '') -> Dict:
        """
        异常/正常样本占比饼图。

        Parameters:
            n_normal: 正常样本数
            n_anomalies: 异常样本数
            title: 图表标题

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if not title:
            title = self._label('anomaly_summary_title')

        echarts = {
            'title': {'text': title, 'left': 'center'},
            'tooltip': {'trigger': 'item', 'formatter': '{b}: {c} ({d}%)'},
            'legend': {'orient': 'vertical', 'left': 'left'},
            'series': [{
                'type': 'pie',
                'radius': ['40%', '70%'],
                'avoidLabelOverlap': True,
                'data': [
                    {
                        'value': n_normal,
                        'name': self._label('anomaly_normal'),
                        'itemStyle': {'color': CHART_GREEN},
                    },
                    {
                        'value': n_anomalies,
                        'name': self._label('anomaly_anomaly'),
                        'itemStyle': {'color': CHART_BLUE},
                    },
                ],
                'label': {'show': True, 'formatter': '{b}\n{c} ({d}%)'},
            }],
        }

        # Matplotlib
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            sizes = [n_normal, n_anomalies]
            labels_list = [self._label('anomaly_normal'), self._label('anomaly_anomaly')]
            colors = [CHART_GREEN, CHART_BLUE]
            ax.pie(sizes, labels=labels_list, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax.set_title(title)
            fig.tight_layout()
            png_path = self._save_png(fig, 'anomaly_summary')
        except Exception as e:
            logger.warning(f"异常占比饼图PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    def plot_anomaly_scatter(self, coordinates, labels, title: str = '',
                             algorithm_name: str = '') -> Dict:
        """
        异常检测散点图：正常样本(绿) vs 异常样本(红)。

        Parameters:
            coordinates: shape (n_samples, 2) 的坐标数组
            labels: 异常标签 (-1=异常, 1=正常)
            title: 图表标题
            algorithm_name: 算法名称（用于文件名）

        Returns:
            {'echarts': dict, 'png_path': str or None}
        """
        if coordinates is None:
            return {'echarts': {}, 'png_path': None}

        coords = np.array(coordinates)
        if coords.ndim != 2 or coords.shape[1] < 2:
            return {'echarts': {}, 'png_path': None}

        x_vals = coords[:, 0].tolist()
        y_vals = coords[:, 1].tolist()

        if labels is not None:
            labels = np.array(labels).tolist()
        else:
            labels = [1] * len(x_vals)

        if not title:
            title = self._label('anomaly_scatter_title')

        # 按标签分组：1=正常, -1=异常
        normal_x, normal_y = [], []
        anomaly_x, anomaly_y = [], []
        for xi, yi, li in zip(x_vals, y_vals, labels):
            if li == -1:
                anomaly_x.append(xi)
                anomaly_y.append(yi)
            else:
                normal_x.append(xi)
                normal_y.append(yi)

        echarts_series = [
            {
                'name': self._label('anomaly_normal'),
                'type': 'scatter',
                'data': [[round(x, 2), round(y, 2)]
                         for x, y in zip(normal_x, normal_y)],
                'symbolSize': 6,
                'itemStyle': {'color': CHART_GREEN},
            },
            {
                'name': self._label('anomaly_anomaly'),
                'type': 'scatter',
                'data': [[round(x, 2), round(y, 2)]
                         for x, y in zip(anomaly_x, anomaly_y)],
                'symbolSize': 8,
                'itemStyle': {'color': CHART_BLUE},
            },
        ]

        echarts = {
            'title': {'text': title},
            'tooltip': {'trigger': 'item'},
            'grid': {'left': '10%', 'right': '5%', 'bottom': '15%', 'top': '15%'},
            'xAxis': {'type': 'value', 'name': 'Component 1'},
            'yAxis': {'type': 'value', 'name': 'Component 2'},
            'series': echarts_series,
            'legend': {},
        }

        # Matplotlib
        fname = f'scatter_anomaly_{algorithm_name}' if algorithm_name else 'scatter_anomaly'
        png_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.scatter(normal_x, normal_y, c=CHART_GREEN, label=self._label('anomaly_normal'),
                       s=20, alpha=0.7, edgecolors='white', linewidths=0.3)
            ax.scatter(anomaly_x, anomaly_y, c=CHART_BLUE, label=self._label('anomaly_anomaly'),
                       s=25, alpha=0.8, edgecolors='white', linewidths=0.3, marker='x')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            png_path = self._save_png(fig, fname)
        except Exception as e:
            logger.warning(f"异常检测散点图PNG生成失败: {e}")

        return {'echarts': echarts, 'png_path': png_path}

    def generate_anomaly_charts(self, pipeline_result: Dict) -> Dict:
        """
        从 MLPipeline.run_anomaly() 返回值一键生成异常检测图表。

        Returns:
            {
                'data_preview': {...},
                'anomaly_summary': {...},
                'anomaly_score_distribution': {...},
                'scatter_pca': {...},
                'scatter_tsne': {...},
            }
        """
        charts = {}

        # 数据概览
        if pipeline_result.get('data_metadata'):
            charts['data_preview'] = self.plot_data_preview(pipeline_result['data_metadata'])

        # 异常/正常占比饼图
        anomaly_results = pipeline_result.get('anomaly_results', {})
        n_normal = anomaly_results.get('n_normal', 0)
        n_anomalies = anomaly_results.get('n_anomalies', 0)
        charts['anomaly_summary'] = self.plot_anomaly_summary(n_normal, n_anomalies)

        # 异常评分分布直方图
        anomaly_labels = pipeline_result.get('anomaly_labels')
        anomaly_scores = pipeline_result.get('anomaly_scores')
        if anomaly_scores is not None and anomaly_labels is not None:
            charts['anomaly_score_distribution'] = self.plot_anomaly_score_distribution(
                anomaly_scores, anomaly_labels,
            )

        # 异常检测散点图
        dim = pipeline_result.get('dim_reduction') or {}

        if dim.get('PCA'):
            pca = dim['PCA']
            if pca.get('coordinates'):
                charts['scatter_pca'] = self.plot_anomaly_scatter(
                    pca['coordinates'], labels=anomaly_labels,
                    title=self._label('anomaly_scatter_title') + ' (PCA)',
                    algorithm_name='pca',
                )

        if dim.get('TSNE'):
            tsne = dim['TSNE']
            if tsne.get('coordinates'):
                charts['scatter_tsne'] = self.plot_anomaly_scatter(
                    tsne['coordinates'], labels=anomaly_labels,
                    title=self._label('anomaly_scatter_title') + ' (t-SNE)',
                    algorithm_name='tsne',
                )

        return charts

    # ----------------------------------------------------------
    # 一键生成：降维可视化
    # ----------------------------------------------------------

    def generate_dim_reduction_charts(self, pipeline_result: Dict) -> Dict:
        """
        从 MLPipeline.run_dim_reduction() 返回值一键生成降维图表。

        Returns:
            {
                'data_preview': {...},
                'scree': {...},
                'scatter_pca': {...},
                'scatter_tsne': {...},
            }
        """
        charts = {}

        # 数据概览
        if pipeline_result.get('data_metadata'):
            charts['data_preview'] = self.plot_data_preview(pipeline_result['data_metadata'])

        # 降维可视化（无聚类标签）
        dim = pipeline_result.get('dim_reduction') or {}

        if dim.get('PCA'):
            pca = dim['PCA']
            if pca.get('explained_variance_ratio'):
                charts['scree'] = self.plot_scree(
                    pca.get('explained_variance_ratio', []),
                    pca.get('cumulative_variance', []),
                )
            if pca.get('coordinates'):
                charts['scatter_pca'] = self.plot_scatter(
                    pca['coordinates'], labels=None,
                    title=self._label('scatter_pca'), algorithm_name='pca',
                )

        if dim.get('TSNE'):
            tsne = dim['TSNE']
            if tsne.get('coordinates'):
                charts['scatter_tsne'] = self.plot_scatter(
                    tsne['coordinates'], labels=None,
                    title=self._label('scatter_tsne'), algorithm_name='tsne',
                )

        return charts
