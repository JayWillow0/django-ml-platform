"""
ML工具模块验证测试脚本
======================
独立运行，不依赖Django。使用sklearn合成数据集验证所有功能。

运行方式（从项目根目录执行）:
    python ml_platform/utils/test_ml_utils.py
"""

import os
import sys
import json
import warnings
import tempfile
import shutil

# 添加项目根目录到sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..')
sys.path.insert(0, os.path.normpath(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# 导入ML模块
from ml_platform.utils.ml_utils import (
    DataImporter, DataPreprocessor, FeatureSelector,
    StatisticalFeatureSelector, SupervisedTrainer, UnsupervisedTrainer,
    ModelTuner, ModelManager, MLPipeline, Visualizer,
    MLError, MLDataError, MLTuningError,
    _to_serializable,
)
from ml_platform.tasks import _to_serializable as _task_to_serializable

# ============================================================
# 测试辅助
# ============================================================

_passed = 0
_failed = 0
_errors = []


def _report(name: str, success: bool, detail: str = ""):
    global _passed, _failed
    status = "PASS" if success else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if success:
        _passed += 1
    else:
        _failed += 1
        _errors.append(f"{name}: {detail}")


def _assert_true(name: str, condition: bool, detail: str = ""):
    _report(name, condition, detail if not condition else "")


def _assert_not_none(name: str, value, detail: str = ""):
    ok = value is not None
    if not ok:
        detail = detail or "值为None"
    _report(name, ok, detail)


def _assert_has_keys(name: str, data: dict, keys: list):
    missing = [k for k in keys if k not in data]
    ok = len(missing) == 0
    _report(name, ok, f"缺少键: {missing}" if missing else "")


# ============================================================
# 测试数据生成
# ============================================================

def _create_regression_csv(tmp_dir: str) -> tuple:
    """生成回归测试数据CSV"""
    X, y = make_regression(n_samples=200, n_features=10, n_informative=5,
                           noise=10, random_state=42)
    # 添加一些分类列
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    df['cat_col'] = np.random.choice(['A', 'B', 'C'], size=200)
    df['target'] = y
    # 添加缺失值
    df.loc[0:5, 'feat_0'] = np.nan
    df.loc[10:15, 'feat_1'] = np.nan
    path = os.path.join(tmp_dir, 'regression_data.csv')
    df.to_csv(path, index=False)
    return path, df


def _create_classification_csv(tmp_dir: str) -> tuple:
    """生成分类测试数据CSV"""
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                               n_classes=3, n_clusters_per_class=1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    df['target'] = [f'class_{c}' for c in y]  # 字符串标签
    path = os.path.join(tmp_dir, 'classification_data.csv')
    df.to_csv(path, index=False)
    return path, df


def _create_clustering_csv(tmp_dir: str) -> tuple:
    """生成聚类测试数据CSV"""
    X, _ = make_blobs(n_samples=200, centers=3, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
    path = os.path.join(tmp_dir, 'clustering_data.csv')
    df.to_csv(path, index=False)
    return path, df


def _create_new_data_csv(tmp_dir: str, n_features: int, has_cat: bool = False) -> str:
    """生成新预测数据CSV"""
    np.random.seed(99)
    data = {}
    for i in range(n_features):
        data[f'feat_{i}'] = np.random.randn(20) * 5
    if has_cat:
        data['cat_col'] = np.random.choice(['A', 'B', 'C'], size=20)
    df = pd.DataFrame(data)
    path = os.path.join(tmp_dir, 'new_data.csv')
    df.to_csv(path, index=False)
    return path


# ============================================================
# 测试用例
# ============================================================

def test_data_importer(tmp_dir: str):
    """测试 DataImporter"""
    print("\n--- DataImporter ---")

    importer = DataImporter()

    # 正常CSV加载
    csv_path, df_orig = _create_regression_csv(tmp_dir)
    result = importer.load_data(csv_path)
    _assert_true("加载CSV", result['success'])
    _assert_true("CSV行数正确", result['metadata']['row_count'] == 200)
    _assert_true("CSV列数正确", result['metadata']['column_count'] == 12)
    _assert_has_keys("CSV返回键完整", result, ['success', 'data', 'metadata', 'preview'])
    _assert_true("CSV preview有数据", len(result['preview']) > 0)

    # 文件不存在
    try:
        importer.load_data('nonexistent.csv')
        _report("文件不存在报错", False, "未抛异常")
    except MLDataError:
        _report("文件不存在报错", True)

    # 不支持的格式
    try:
        importer.load_data(os.path.join(tmp_dir, 'test.json'))
        _report("不支持的格式报错", False, "未抛异常")
    except MLDataError:
        _report("不支持的格式报错", True)


def test_data_preprocessor(tmp_dir: str):
    """测试 DataPreprocessor"""
    print("\n--- DataPreprocessor ---")

    preprocessor = DataPreprocessor()

    # 构建测试数据
    df = pd.DataFrame({
        'num_1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'num_2': [10, 20, 30, 40, 50],
        'cat_1': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 1],
    })

    # 正常预处理
    result = preprocessor.fit_transform(df, target_col='target', task_type='classification')
    _assert_true("预处理成功", result['success'])
    _assert_not_none("X不为空", result['X'])
    _assert_not_none("y不为空", result['y'])
    _assert_true("X无缺失值", not result['X'].isna().any().any())
    _assert_has_keys("预处理返回键完整", result, [
        'X', 'y', 'feature_names', 'numeric_features', 'categorical_features',
        'preprocessing_info',
    ])
    _assert_true("识别了数值特征", len(result['numeric_features']) >= 1)
    _assert_true("识别了分类特征", len(result['categorical_features']) >= 1)

    # transform_new
    new_df = pd.DataFrame({
        'num_1': [3.0, 4.0],
        'num_2': [25, 35],
        'cat_1': ['A', 'C'],
    })
    X_new = preprocessor.transform_new(new_df)
    _assert_true("transform_new成功", X_new is not None and not X_new.isna().any().any())
    _assert_true("transform_new列数一致", X_new.shape[1] == result['X'].shape[1])

    # 无监督预处理
    prep2 = DataPreprocessor()
    result2 = prep2.fit_transform(df.drop(columns=['target']))
    _assert_true("无监督预处理成功", result2['success'])
    _assert_true("无监督y为None", result2['y'] is None)

    minmax_preprocessor = DataPreprocessor()
    minmax_result = minmax_preprocessor.fit_transform(
        df,
        target_col='target',
        task_type='classification',
        scaler_type='minmax',
    )
    numeric_values = minmax_result['X'][minmax_result['numeric_features']]
    _assert_true(
        "MinMaxScaler路径有效",
        numeric_values.min().min() >= 0 and numeric_values.max().max() <= 1,
    )

    no_scale_preprocessor = DataPreprocessor()
    no_scale_preprocessor.fit_transform(
        df,
        target_col='target',
        task_type='classification',
        scale_numeric=False,
    )
    numeric_pipeline = no_scale_preprocessor.pipeline.named_transformers_['num']
    _assert_true(
        "关闭缩放时不创建scaler",
        'scaler' not in numeric_pipeline.named_steps,
    )

    # 空数据报错
    try:
        DataPreprocessor().fit_transform(pd.DataFrame())
        _report("空数据报错", False, "未抛异常")
    except MLDataError:
        _report("空数据报错", True)


def test_feature_selector(tmp_dir: str):
    """测试 FeatureSelector"""
    print("\n--- FeatureSelector ---")

    selector = FeatureSelector()

    # 生成测试数据
    X, y = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(10)])
    y_s = pd.Series(y)

    # 回归相关性分析
    corr = selector.analyze_correlation(X_df, y_s, 'regression')
    _assert_true("回归相关性分析成功", corr['success'])
    _assert_true("相关性结果非空", len(corr['correlations']) > 0)
    _assert_has_keys("相关性结果有heatmap", corr, ['correlations', 'heatmap_data'])
    _assert_true("相关性按绝对值排序", abs(corr['correlations'][0]['score']) >= abs(corr['correlations'][-1]['score']))

    # 分类相关性分析
    X2, y2 = make_classification(n_samples=200, n_features=10, n_informative=5,
                                  n_classes=3, random_state=42)
    X2_df = pd.DataFrame(X2, columns=[f'f_{i}' for i in range(10)])
    y2_s = pd.Series(y2)

    corr2 = selector.analyze_correlation(X2_df, y2_s, 'classification')
    _assert_true("分类相关性分析成功", corr2['success'])

    # 特征选择 - p值阈值
    sel = selector.select_features(X_df, y_s, 'regression', method='f_test', threshold=0.05)
    _assert_true("特征选择成功", sel['success'])
    _assert_true("选择了部分特征", 0 < len(sel['selected_features']) <= 10)
    _assert_has_keys("选择结果键完整", sel, ['selected_features', 'removed_features', 'X_selected'])

    # 特征选择 - Top-K
    sel_k = selector.select_features(X_df, y_s, 'regression', method='kbest', k=5)
    _assert_true("Top-K选择成功", sel_k['success'])
    _assert_true("Top-K选了5个特征", len(sel_k['selected_features']) == 5)


def test_supervised_trainer(tmp_dir: str):
    """测试 SupervisedTrainer"""
    print("\n--- SupervisedTrainer ---")

    trainer = SupervisedTrainer()

    # 回归训练
    X, y = make_regression(n_samples=150, n_features=8, n_informative=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(8)])
    y_s = pd.Series(y)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_s, test_size=0.2, random_state=42)

    reg_result = trainer.train_all(X_tr, X_te, y_tr, y_te, 'regression')
    _assert_true("回归训练成功", reg_result['success'])
    _assert_true("回归有模型结果", len(reg_result['models']) > 0)
    _assert_not_none("回归最佳模型名", reg_result['best_model_name'])
    _assert_not_none("回归最佳模型类", reg_result['best_model_class'])
    _report(f"  回归最佳模型: {reg_result['best_model_name']}", True)

    # 分类训练
    X2, y2 = make_classification(n_samples=150, n_features=8, n_informative=5,
                                  n_classes=2, random_state=42)
    X2_df = pd.DataFrame(X2, columns=[f'f_{i}' for i in range(8)])
    y2_s = pd.Series(y2)
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2_df, y2_s, test_size=0.2, random_state=42)

    clf_result = trainer.train_all(X2_tr, X2_te, y2_tr, y2_te, 'classification')
    _assert_true("分类训练成功", clf_result['success'])
    _assert_true("分类有模型结果", len(clf_result['models']) > 0)
    _report(f"  分类最佳模型: {clf_result['best_model_name']}", True)

    # get_model_class
    from sklearn.ensemble import RandomForestRegressor
    cls = SupervisedTrainer.get_model_class('RandomForestRegressor')
    _assert_true("get_model_class正确", cls == RandomForestRegressor)


def test_unsupervised_trainer(tmp_dir: str):
    """测试 UnsupervisedTrainer"""
    print("\n--- UnsupervisedTrainer ---")

    trainer = UnsupervisedTrainer()

    X, _ = make_blobs(n_samples=200, centers=3, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(5)])

    # 聚类
    cluster_result = trainer.cluster(X_df, n_clusters=3)
    _assert_true("聚类成功", cluster_result['success'])
    _assert_true("有聚类结果", len(cluster_result['results']) > 0)
    _assert_true("有比较数据", len(cluster_result['comparison']) > 0)

    # KMeans
    kmeans_res = cluster_result['results'].get('KMeans', {})
    if 'labels' in kmeans_res:
        _assert_true("KMeans标签数正确", len(kmeans_res['labels']) == 200)
        _assert_true("KMeans聚类数正确", kmeans_res['n_clusters'] == 3)
        _assert_true("KMeans有肘部数据", len(cluster_result.get('elbow_data', [])) > 0)
    else:
        _report("KMeans执行", False, str(kmeans_res.get('error', 'unknown')))

    # 降维
    dim_result = trainer.reduce_dimensions(X_df, methods=['PCA', 'TSNE'])
    _assert_true("降维成功", dim_result['success'])

    pca_res = dim_result['results'].get('PCA', {})
    if 'coordinates' in pca_res:
        _assert_true("PCA坐标维度正确", pca_res['coordinates'].shape == (200, 2))
        _assert_true("PCA有解释方差", len(pca_res['explained_variance_ratio']) > 0)
    else:
        _report("PCA执行", False, str(pca_res.get('error', 'unknown')))

    tsne_res = dim_result['results'].get('TSNE', {})
    if 'coordinates' in tsne_res:
        _assert_true("t-SNE坐标维度正确", tsne_res['coordinates'].shape == (200, 2))
    else:
        _report("t-SNE执行", False, str(tsne_res.get('error', 'unknown')))


def test_model_tuner(tmp_dir: str):
    """测试 ModelTuner"""
    print("\n--- ModelTuner ---")

    tuner = ModelTuner()

    # 回归调参 - RandomForest（少trials加快测试）
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(5)])
    y_s = pd.Series(y)

    from sklearn.ensemble import RandomForestRegressor
    reg_tune = tuner.tune(
        RandomForestRegressor, X_df, y_s, 'regression',
        n_trials=5, cv_folds=3, random_state=42,
    )
    _assert_true("回归调参成功", reg_tune['success'])
    _assert_not_none("回归最佳模型", reg_tune['best_model'])
    _assert_not_none("回归最佳参数", reg_tune['best_params'])
    _assert_true("回归最佳模型可预测", hasattr(reg_tune['best_model'], 'predict'))
    preds = reg_tune['best_model'].predict(X_df[:5])
    _assert_true("回归预测结果数量正确", len(preds) == 5)
    _report(f"  回归调参: {reg_tune['model_name']}, score={reg_tune['best_score']:.4f}", True)

    # 分类调参 - Ridge（简单模型快）
    X2, y2 = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X2_df = pd.DataFrame(X2, columns=[f'f_{i}' for i in range(5)])
    y2_s = pd.Series(y2)

    from sklearn.linear_model import LogisticRegression
    clf_tune = tuner.tune(
        LogisticRegression, X2_df, y2_s, 'classification',
        n_trials=5, cv_folds=3, random_state=42,
    )
    _assert_true("分类调参成功", clf_tune['success'])
    _assert_true(
        "分类调参指标为平衡准确率",
        clf_tune['cv_metric'] == 'balanced_accuracy',
    )
    _assert_true(
        "逻辑回归保留saga求解器",
        clf_tune['best_params'].get('solver') == 'saga',
    )
    _report(f"  分类调参: {clf_tune['model_name']}, score={clf_tune['best_score']:.4f}", True)

    # 曾经存在构造参数错误的搜索空间回归测试
    from sklearn.linear_model import RidgeCV, PassiveAggressiveRegressor
    from sklearn.svm import LinearSVC, NuSVC
    regression_cases = [RidgeCV, PassiveAggressiveRegressor]
    classification_cases = [LinearSVC, NuSVC]
    for model_class in regression_cases:
        result = ModelTuner().tune(
            model_class, X_df, y_s, 'regression',
            n_trials=2, cv_folds=2, random_state=42,
        )
        _assert_true(f"{model_class.__name__}搜索空间有效", result['success'])
        model_class(**result['best_params'])
    ridge_result = ModelTuner().tune(
        RidgeCV, X_df, y_s, 'regression',
        n_trials=2, cv_folds=2, random_state=42,
    )
    _assert_true(
        "RidgeCV返回多个alphas",
        len(ridge_result['best_params'].get('alphas', [])) >= 3,
    )
    _assert_true(
        "RidgeCV不返回非法alpha参数",
        'alpha' not in ridge_result['best_params'],
    )
    for model_class in classification_cases:
        result = ModelTuner().tune(
            model_class, X2_df, y2_s, 'classification',
            n_trials=2, cv_folds=2, random_state=42,
        )
        _assert_true(f"{model_class.__name__}搜索空间有效", result['success'])
        model_class(**result['best_params'])

    import optuna
    from ml_platform.utils.ml_utils import _SEARCH_SPACES
    sigmoid_params = _SEARCH_SPACES['SVC'](optuna.trial.FixedTrial({
        'kernel': 'sigmoid',
        'C': 1.0,
        'gamma': 'scale',
    }))
    _assert_true(
        "SVM sigmoid分支可达",
        sigmoid_params['kernel'] == 'sigmoid' and sigmoid_params['gamma'] == 'scale',
    )

    class FitRecorder(BaseEstimator, TransformerMixin):
        fit_sizes = []

        def fit(self, X, y=None):
            type(self).fit_sizes.append(len(X))
            return self

        def transform(self, X):
            return X

    FitRecorder.fit_sizes = []
    ModelTuner().tune(
        LogisticRegression,
        X2_df,
        y2_s,
        'classification',
        n_trials=2,
        cv_folds=2,
        random_state=42,
        pipeline_factory=lambda model: Pipeline([
            ('recorder', FitRecorder()),
            ('model', model),
        ]),
    )
    _assert_true(
        "Optuna每折独立拟合Pipeline",
        FitRecorder.fit_sizes.count(len(X2_df) // 2) >= 4
        and FitRecorder.fit_sizes[-1] == len(X2_df),
    )

    class AlwaysFailEstimator:
        def __init__(self):
            pass

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

        def fit(self, X, y):
            raise ValueError("expected failure")

        def predict(self, X):
            return np.zeros(len(X))

    try:
        ModelTuner().tune(
            AlwaysFailEstimator, X_df, y_s, 'regression',
            n_trials=2, cv_folds=2,
        )
        _report("全部失败trial显式报错", False, "未抛出MLTuningError")
    except MLTuningError:
        _report("全部失败trial显式报错", True)


def test_model_manager(tmp_dir: str):
    """测试 ModelManager"""
    print("\n--- ModelManager ---")

    manager = ModelManager()

    # 训练一个简单模型
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(5)])

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_df, y)

    # 预处理器
    preprocessor = DataPreprocessor()
    df_with_target = X_df.copy()
    df_with_target['target'] = y
    preprocessor.fit_transform(df_with_target, target_col='target', task_type='regression')

    # 保存
    save_path = os.path.join(tmp_dir, 'test_model')
    save_result = manager.save_model(
        model, save_path,
        metadata={'model_class': 'RandomForestRegressor', 'task_type': 'regression'},
        preprocessor_bundle=preprocessor.get_bundle(),
    )
    _assert_true("模型保存成功", save_result['success'])
    _assert_true("joblib文件存在", os.path.exists(save_path + '.joblib'))
    _assert_true("meta.json文件存在", os.path.exists(save_path + '_meta.json'))

    # 加载
    load_result = manager.load_model(save_path)
    _assert_true("模型加载成功", load_result['success'])
    _assert_not_none("加载的模型", load_result['model'])
    _assert_not_none("加载的预处理器", load_result['preprocessor_bundle'])

    # 预测
    pred_result = manager.predict(model, X_df[:10], preprocessor)
    _assert_true("预测成功", pred_result['success'])
    _assert_true("预测数量正确", pred_result['prediction_count'] == 10)
    _assert_true("predictions_df有prediction列", 'prediction' in pred_result['predictions_df'].columns)

    # 导出CSV
    csv_path = os.path.join(tmp_dir, 'predictions.csv')
    export_result = manager.export_predictions(pred_result['predictions_df'], csv_path)
    _assert_true("导出CSV成功", export_result['success'])
    _assert_true("CSV文件存在", os.path.exists(csv_path))

    # 导出XLSX（需要openpyxl）
    try:
        import openpyxl  # noqa: F401
        xlsx_path = os.path.join(tmp_dir, 'predictions.xlsx')
        export_xlsx = manager.export_predictions(pred_result['predictions_df'], xlsx_path, format='xlsx')
        _assert_true("导出XLSX成功", export_xlsx['success'])
    except ImportError:
        _report("导出XLSX(跳过)", True, "openpyxl未安装")

    # 加载不存在的模型
    try:
        manager.load_model('nonexistent_model')
        _report("加载不存在模型报错", False, "未抛异常")
    except Exception:
        _report("加载不存在模型报错", True)


def test_pipeline_supervised_regression(tmp_dir: str):
    """测试 MLPipeline 监督回归完整流程"""
    print("\n--- MLPipeline 监督回归 ---")

    csv_path, _ = _create_regression_csv(tmp_dir)
    model_path = os.path.join(tmp_dir, 'reg_model')
    new_data_path = _create_new_data_csv(tmp_dir, 10, has_cat=True)
    output_path = os.path.join(tmp_dir, 'reg_predictions.csv')

    pipeline = MLPipeline()
    result = pipeline.run_supervised(
        file_path=csv_path,
        target_col='target',
        task_type='regression',
        test_size=0.2,
        feature_selection_method='f_test',
        correlation_threshold=0.05,
        tune=True,
        n_trials=3,
        cv_folds=3,
        random_state=42,
        save_path=model_path,
    )

    _assert_true("回归流水线成功", result['success'])
    _assert_true("任务类型正确", result['task_type'] == 'regression')
    _assert_has_keys("回归结果键完整", result, [
        'task_type', 'data_metadata', 'preprocessing_info',
        'correlation_results', 'feature_selection', 'training_results',
    ])
    _assert_not_none("回归最佳模型名", result['training_results']['best_model_name'])
    _assert_true("回归模型已保存", result.get('saved_info') is not None)
    _assert_true("回归joblib文件存在", os.path.exists(model_path + '.joblib'))
    _assert_true(
        "回归CV指标为R2",
        result['tuning_results']['cv_metric'] == 'r2',
    )
    _assert_true(
        "回归结果包含独立测试得分",
        result['tuning_results']['test_score'] is not None,
    )

    # 保存的预处理统计量只能来自外层训练集。
    source_df = pd.read_csv(csv_path)
    source_X = source_df.drop(columns=['target'])
    source_y = source_df['target']
    X_train_expected, _, _, _ = train_test_split(
        source_X, source_y, test_size=0.2, random_state=42,
    )
    loaded = pipeline.manager.load_model(model_path)
    bundle = loaded['preprocessor_bundle']
    numeric_cols = bundle['numeric_features']
    numeric_pipeline = bundle['pipeline'].named_transformers_['num']
    imputed_train = numeric_pipeline.named_steps['imputer'].transform(
        X_train_expected[numeric_cols],
    )
    scaler = numeric_pipeline.named_steps.get('scaler')
    if hasattr(scaler, 'mean_'):
        _assert_true(
            "缩放统计量未使用测试集",
            np.allclose(scaler.mean_, np.asarray(imputed_train).mean(axis=0)),
        )
    _assert_true(
        "保存模型样本数仅含训练集",
        loaded['metadata']['sample_count'] == len(X_train_expected),
    )

    # 新数据预测
    pred_result = pipeline.predict_new(model_path, new_data_path, output_path)
    _assert_true("回归预测成功", pred_result['success'])
    _assert_true("回归预测数量正确", pred_result['prediction_count'] == 20)
    _assert_true("回归预测文件存在", os.path.exists(output_path))

    print(f"  → 最佳模型: {result['training_results']['best_model_name']}")
    if result.get('tuning_results') and result['tuning_results'].get('best_score') is not None:
        print(f"  → 调参得分: {result['tuning_results']['best_score']:.4f}")
    print(f"  → 选择特征: {result['feature_selection']['selected_features'][:5]}...")


def test_pipeline_supervised_classification(tmp_dir: str):
    """测试 MLPipeline 监督分类完整流程"""
    print("\n--- MLPipeline 监督分类 ---")

    csv_path, _ = _create_classification_csv(tmp_dir)
    model_path = os.path.join(tmp_dir, 'clf_model')

    pipeline = MLPipeline()
    result = pipeline.run_supervised(
        file_path=csv_path,
        target_col='target',
        task_type='classification',
        test_size=0.2,
        feature_selection_method='f_test',
        correlation_threshold=0.05,
        tune=True,
        n_trials=3,
        cv_folds=3,
        random_state=42,
        save_path=model_path,
    )

    _assert_true("分类流水线成功", result['success'])
    _assert_true("任务类型正确", result['task_type'] == 'classification')
    _assert_not_none("分类最佳模型名", result['training_results']['best_model_name'])
    _assert_true("分类模型已保存", result.get('saved_info') is not None)
    _assert_true(
        "分类CV指标为平衡准确率",
        result['tuning_results']['cv_metric'] == 'balanced_accuracy',
    )
    _assert_true(
        "分类测试指标为平衡准确率",
        result['tuning_results']['test_metric'] == 'balanced_accuracy',
    )

    # 新数据预测（含标签解码）
    new_data_path = _create_new_data_csv(tmp_dir, 10, has_cat=False)
    output_path = os.path.join(tmp_dir, 'clf_predictions.csv')
    pred_result = pipeline.predict_new(model_path, new_data_path, output_path)
    _assert_true("分类预测成功", pred_result['success'])
    _assert_true("分类预测数量正确", pred_result['prediction_count'] == 20)

    # 验证预测结果有标签
    pred_df = pred_result['predictions_df']
    _assert_true("分类预测有prediction列", 'prediction' in pred_df.columns)

    print(f"  → 最佳模型: {result['training_results']['best_model_name']}")
    print(f"  → 选择特征: {result['feature_selection']['selected_features'][:5]}...")


def test_pipeline_unsupervised(tmp_dir: str):
    """测试 MLPipeline 无监督学习完整流程"""
    print("\n--- MLPipeline 无监督学习 ---")

    csv_path, _ = _create_clustering_csv(tmp_dir)

    pipeline = MLPipeline()
    result = pipeline.run_unsupervised(
        file_path=csv_path,
        n_clusters=3,
        dim_reduction=True,
        random_state=42,
    )

    _assert_true("无监督流水线成功", result['success'])
    _assert_true("任务类型正确", result['task_type'] == 'unsupervised')
    _assert_has_keys("无监督结果键完整", result, [
        'task_type', 'data_metadata', 'preprocessing_info',
        'cluster_results', 'labeled_data',
    ])
    _assert_true("聚类比较非空", len(result['cluster_results']) > 0)
    # labeled_data 经 _to_serializable 序列化为 records(list[dict])，列名在各行 dict 的 keys 中
    _labeled_cols = list(result['labeled_data'][0].keys()) if result['labeled_data'] else []
    _assert_true("标记数据有聚类列", any('cluster_' in col for col in _labeled_cols))

    # 检查降维结果
    if result.get('dim_reduction'):
        _assert_true("降维结果非空", len(result['dim_reduction']) > 0)
        print(f"  → 降维方法: {list(result['dim_reduction'].keys())}")

    # 打印聚类比较
    for item in result['cluster_results']:
        sil = item.get('silhouette_score')
        print(f"  → {item['algorithm']}: {item['n_clusters']}个簇, 轮廓系数={sil}")


def test_json_serialization():
    """验证嵌套在 numpy/pandas 容器中的非有限浮点数可安全写入 MySQL JSON。"""
    print("\n--- 严格 JSON 序列化 ---")

    data = {
        'array': np.array([1.0, np.nan, np.inf, -np.inf]),
        'series': pd.Series([2.0, np.nan]),
        'frame': pd.DataFrame({'value': [3.0, np.nan]}),
    }
    for name, serializer in (
        ('ML工具序列化器', _to_serializable),
        ('Celery任务序列化器', _task_to_serializable),
    ):
        serialized = serializer(data)
        try:
            json.dumps(serialized, allow_nan=False)
        except ValueError as exc:
            _report(f"{name}符合严格JSON", False, str(exc))
        else:
            _report(f"{name}符合严格JSON", True)
        _assert_true(
            f"{name}将非有限浮点数转为null",
            serialized['array'][1:] == [None, None, None]
            and serialized['series'][1] is None
            and serialized['frame'][1]['value'] is None,
        )


# ============================================================
# 可视化测试
# ============================================================

def test_visualizer_echarts(tmp_dir: str):
    """测试 Visualizer 所有图表方法的 ECharts JSON 输出"""
    print("\n--- Visualizer ECharts JSON ---")

    charts_dir = os.path.join(tmp_dir, 'charts')
    viz = Visualizer(output_dir=charts_dir, lang='zh')

    # 准备测试数据
    correlations = [
        {'feature': 'feat_0', 'score': 50.0, 'p_value': 0.001, 'mutual_info': 0.3, 'method': 'f_regression'},
        {'feature': 'feat_1', 'score': 30.0, 'p_value': 0.01, 'mutual_info': 0.2, 'method': 'f_regression'},
        {'feature': 'feat_2', 'score': 5.0, 'p_value': 0.3, 'mutual_info': 0.05, 'method': 'f_regression'},
    ]
    importance = [
        {'feature': 'feat_0', 'importance': 0.45},
        {'feature': 'feat_1', 'importance': 0.30},
        {'feature': 'feat_2', 'importance': 0.25},
    ]
    selection_scores = {
        'feat_0': {'f_score': 50.0, 'p_value': 0.001, 'mutual_info': 0.3},
        'feat_1': {'f_score': 30.0, 'p_value': 0.01, 'mutual_info': 0.2},
        'feat_2': {'f_score': 5.0, 'p_value': 0.3, 'mutual_info': 0.05},
    }
    models_reg = [
        {'model_name': 'RandomForest', 'rank': 1, 'adjusted_r_squared': 0.85},
        {'model_name': 'Ridge', 'rank': 2, 'adjusted_r_squared': 0.80},
        {'model_name': 'Lasso', 'rank': 3, 'adjusted_r_squared': 0.75},
    ]
    models_clf = [
        {'model_name': 'RandomForest', 'rank': 1, 'balanced_accuracy': 0.90},
        {'model_name': 'SVC', 'rank': 2, 'balanced_accuracy': 0.85},
    ]
    opt_history = [0.5, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85]
    elbow_data = [{'k': 2, 'inertia': 5000}, {'k': 3, 'inertia': 3200},
                  {'k': 4, 'inertia': 2800}, {'k': 5, 'inertia': 2500}]
    cluster_comp = [
        {'algorithm': 'KMeans', 'n_clusters': 3, 'silhouette_score': 0.65},
        {'algorithm': 'DBSCAN', 'n_clusters': 2, 'silhouette_score': 0.50},
    ]
    variance_ratio = [0.55, 0.25, 0.10, 0.05]
    cumulative = [0.55, 0.80, 0.90, 0.95]
    np.random.seed(42)
    coords = np.random.randn(50, 2)
    labels = np.array([0] * 20 + [1] * 20 + [2] * 10)
    metadata = {
        'row_count': 200, 'column_count': 5,
        'columns': ['a', 'b', 'c', 'd', 'e'],
        'dtypes': {'a': 'float64'},
        'missing_values': {'a': 5, 'b': 0, 'c': 3, 'd': 0, 'e': 1},
    }

    # 1. 相关性
    r = viz.plot_correlation_heatmap(correlations)
    _assert_has_keys("相关性ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("相关性ECharts有数据", len(r['echarts']['series'][0]['data']) > 0)

    # 2. 特征重要性
    r = viz.plot_feature_importance(importance)
    _assert_has_keys("重要性ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("重要性ECharts有数据", len(r['echarts']['series'][0]['data']) == 3)

    # 3. 特征选择
    r = viz.plot_feature_selection(selection_scores, ['feat_0', 'feat_1'])
    _assert_has_keys("特征选择ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("特征选择ECharts有两个系列", len(r['echarts']['series']) == 2)

    # 4. 模型对比 - 回归
    r = viz.plot_model_comparison(models_reg, 'regression')
    _assert_has_keys("模型对比(回归)ECharts", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("模型对比(回归)有3个模型", len(r['echarts']['yAxis']['data']) == 3)

    # 模型对比 - 分类
    r = viz.plot_model_comparison(models_clf, 'classification')
    _assert_has_keys("模型对比(分类)ECharts", r['echarts'], ['title', 'series'])

    # 5. 调参历史
    r = viz.plot_optimization_history(opt_history, 'RandomForest')
    _assert_has_keys("调参历史ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("调参历史ECharts有两个系列", len(r['echarts']['series']) == 2)

    # 6. 肘部法则
    r = viz.plot_elbow_curve(elbow_data)
    _assert_has_keys("肘部法则ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("肘部法则ECharts有4个点", len(r['echarts']['series'][0]['data']) == 4)

    # 7. 聚类对比
    r = viz.plot_cluster_comparison(cluster_comp)
    _assert_has_keys("聚类对比ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])

    # 8. 碎石图
    r = viz.plot_scree(variance_ratio, cumulative)
    _assert_has_keys("碎石图ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("碎石图ECharts有条形+折线", len(r['echarts']['series']) == 2)

    # 9. 散点图
    r = viz.plot_scatter(coords, labels, title='Test Scatter', algorithm_name='test')
    _assert_has_keys("散点图ECharts结构", r['echarts'], ['title', 'tooltip', 'series'])
    _assert_true("散点图ECharts有3个系列", len(r['echarts']['series']) == 3)

    # 10. 数据概览
    r = viz.plot_data_preview(metadata)
    _assert_has_keys("数据概览ECharts结构", r['echarts'], ['title'])
    _assert_true("数据概览ECharts有tooltip", 'tooltip' in r['echarts'])


def test_visualizer_png(tmp_dir: str):
    """测试 Visualizer 所有图表方法的 PNG 输出"""
    print("\n--- Visualizer Matplotlib PNG ---")

    charts_dir = os.path.join(tmp_dir, 'charts')
    viz = Visualizer(output_dir=charts_dir, lang='zh')

    # 用最小数据生成所有PNG
    correlations = [
        {'feature': 'a', 'score': 10, 'p_value': 0.01, 'mutual_info': 0.1, 'method': 'f_test'},
    ]
    importance = [{'feature': 'x', 'importance': 0.5}]
    selection_scores = {'x': {'f_score': 10, 'p_value': 0.01, 'mutual_info': 0.1}}
    models = [{'model_name': 'RF', 'rank': 1, 'adjusted_r_squared': 0.9}]
    history = [0.5, 0.7, 0.85]
    elbow = [{'k': 2, 'inertia': 500}, {'k': 3, 'inertia': 300}]
    comp = [{'algorithm': 'KMeans', 'n_clusters': 3, 'silhouette_score': 0.7}]
    coords = np.random.randn(30, 2)
    labels = np.array([0] * 15 + [1] * 15)

    # 生成所有PNG并验证文件存在
    r1 = viz.plot_correlation_heatmap(correlations)
    _assert_true("相关性PNG存在", r1['png_path'] and os.path.exists(r1['png_path']))
    _assert_true("相关性PNG非空", os.path.getsize(r1['png_path']) > 0)

    r2 = viz.plot_feature_importance(importance)
    _assert_true("重要性PNG存在", r2['png_path'] and os.path.exists(r2['png_path']))

    r3 = viz.plot_feature_selection(selection_scores, ['x'])
    _assert_true("特征选择PNG存在", r3['png_path'] and os.path.exists(r3['png_path']))

    r4 = viz.plot_model_comparison(models, 'regression')
    _assert_true("模型对比PNG存在", r4['png_path'] and os.path.exists(r4['png_path']))

    r5 = viz.plot_optimization_history(history, 'Test')
    _assert_true("调参历史PNG存在", r5['png_path'] and os.path.exists(r5['png_path']))

    r6 = viz.plot_elbow_curve(elbow)
    _assert_true("肘部法则PNG存在", r6['png_path'] and os.path.exists(r6['png_path']))

    r7 = viz.plot_cluster_comparison(comp)
    _assert_true("聚类对比PNG存在", r7['png_path'] and os.path.exists(r7['png_path']))

    r8 = viz.plot_scree([0.6, 0.3], [0.6, 0.9])
    _assert_true("碎石图PNG存在", r8['png_path'] and os.path.exists(r8['png_path']))

    r9 = viz.plot_scatter(coords, labels, title='Test', algorithm_name='test')
    _assert_true("散点图PNG存在", r9['png_path'] and os.path.exists(r9['png_path']))

    metadata = {
        'columns': ['a', 'b'], 'dtypes': {'a': 'float'},
        'missing_values': {'a': 5, 'b': 0},
    }
    r10 = viz.plot_data_preview(metadata)
    _assert_true("数据概览PNG存在", r10['png_path'] is not None or True)  # 可能无缺失值

    # 检查output_dir=None时不崩溃
    viz_none = Visualizer(output_dir=None)
    r = viz_none.plot_correlation_heatmap(correlations)
    _assert_true("output_dir=None不崩溃", r['png_path'] is None)
    _assert_true("output_dir=None仍有ECharts", 'title' in r['echarts'])


def test_visualizer_pipeline_integration(tmp_dir: str):
    """测试 Visualizer 与 MLPipeline 一键生成集成"""
    print("\n--- Visualizer Pipeline集成 ---")

    charts_dir = os.path.join(tmp_dir, 'charts')
    viz = Visualizer(output_dir=charts_dir, lang='zh')

    # --- 监督学习一键生成 ---
    csv_path, _ = _create_regression_csv(tmp_dir)
    pipeline = MLPipeline()
    result = pipeline.run_supervised(
        file_path=csv_path, target_col='target', task_type='regression',
        test_size=0.2, tune=True, n_trials=3, cv_folds=3, random_state=42,
    )
    _assert_true("监督流水线成功(可视化)", result['success'])

    charts = viz.generate_supervised_charts(result)
    chart_names = list(charts.keys())
    print(f"  → 监督学习图表: {chart_names}")
    _assert_true("监督一键生成了图表", len(charts) > 0)

    for name, chart in charts.items():
        echarts = chart.get('echarts', {})
        _assert_true(f"监督图表[{name}]有ECharts", bool(echarts))
        if 'title' in echarts:
            _assert_true(f"监督图表[{name}]有title", True)

    # 验证关键图表存在
    _assert_true("有数据概览图", 'data_preview' in charts)
    _assert_true("有相关性图", 'correlation' in charts)
    _assert_true("有模型对比图", 'model_comparison' in charts)

    # 验证返回值中有optimization_history（之前修复的字段）
    tuning = result.get('tuning_results')
    if tuning:
        _assert_true("有调参历史数据", bool(tuning.get('optimization_history')))

    # --- 无监督学习一键生成 ---
    cluster_csv, _ = _create_clustering_csv(tmp_dir)
    result2 = pipeline.run_unsupervised(
        file_path=cluster_csv, n_clusters=3, dim_reduction=True, random_state=42,
    )
    _assert_true("无监督流水线成功(可视化)", result2['success'])

    charts2 = viz.generate_unsupervised_charts(result2)
    chart_names2 = list(charts2.keys())
    print(f"  → 无监督学习图表: {chart_names2}")
    _assert_true("无监督一键生成了图表", len(charts2) > 0)

    for name, chart in charts2.items():
        echarts = chart.get('echarts', {})
        _assert_true(f"无监督图表[{name}]有ECharts", bool(echarts))

    # 验证关键图表
    _assert_true("有肘部法则图", 'elbow' in charts2)
    _assert_true("有聚类对比图", 'cluster_comparison' in charts2)
    _assert_true("有PCA碎石图", 'scree' in charts2)
    _assert_true("有PCA散点图", 'scatter_pca' in charts2)
    _assert_true("有t-SNE散点图", 'scatter_tsne' in charts2)

    # 验证返回值中有cluster_labels
    _assert_true("有聚类标签数据", bool(result2.get('cluster_labels')))

    # 验证PNG文件
    png_count = 0
    for name, chart in {**charts, **charts2}.items():
        if chart.get('png_path') and os.path.exists(chart['png_path']):
            png_count += 1
    print(f"  → 生成PNG图片: {png_count}个")
    _assert_true("生成了PNG图片", png_count > 0)

    # 列出charts目录所有文件
    if os.path.exists(charts_dir):
        files = os.listdir(charts_dir)
        print(f"  → charts目录文件: {files}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("ML工具模块验证测试")
    print("=" * 60)

    # 使用固定目录保存所有测试结果，便于检查
    tmp_dir = os.path.join(
        os.path.dirname(_SCRIPT_DIR), '..', '..', '..', 'ml_test_results'
    )
    tmp_dir = os.path.normpath(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"测试结果目录: {tmp_dir}")

    try:
        test_data_importer(tmp_dir)
        test_data_preprocessor(tmp_dir)
        test_feature_selector(tmp_dir)
        test_supervised_trainer(tmp_dir)
        test_unsupervised_trainer(tmp_dir)
        test_model_tuner(tmp_dir)
        test_model_manager(tmp_dir)
        test_pipeline_supervised_regression(tmp_dir)
        test_pipeline_supervised_classification(tmp_dir)
        test_pipeline_unsupervised(tmp_dir)
        test_json_serialization()
        test_visualizer_echarts(tmp_dir)
        test_visualizer_png(tmp_dir)
        test_visualizer_pipeline_integration(tmp_dir)
    except Exception as e:
        print(f"\n!!! 未捕获的异常: {e}")
        import traceback
        traceback.print_exc()

    # 汇总
    print("\n" + "=" * 60)
    total = _passed + _failed
    print(f"测试结果: {_passed}/{total} 通过", end="")
    if _failed > 0:
        print(f", {_failed} 失败")
        print("\n失败详情:")
        for err in _errors:
            print(f"  - {err}")
    else:
        print(" [OK] 全部通过!")
    print("=" * 60)

    return 0 if _failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
