"""
商品期货机器学习量化模型 - 主流程
ML Pipeline for commodity futures quantitative trading.

完整流程:
1. 数据加载与预处理
2. 特征工程（计算衍生指标）
3. 模型训练与预测
4. 回测与特征选择
5. 结果评估

使用方法:
    python ml_pipeline.py --period 5min --target future_direction
"""

import argparse
import logging

import numpy as np
import pandas as pd

from features.feature_engineering import compute_all_features, compute_prediction_targets, get_feature_hierarchy
from models.ml_models import create_model, LightGBMModel, XGBoostModel
from models.ensemble_model import EnsembleModel
from models.position_sizing import RiskBudgetManager
from backtest.feature_selector import FeatureSelector, get_recommended_feature_groups
from config import ML_FRAMEWORKS, PREDICTION_TARGETS, BACKTEST_CONFIG, ENSEMBLE_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_data(n_rows=5000, period="5min"):
    """
    生成模拟K线数据用于演示。

    Parameters
    ----------
    n_rows : int
        数据行数
    period : str
        K线周期

    Returns
    -------
    pd.DataFrame
        模拟K线数据
    """
    np.random.seed(42)

    freq_map = {"1min": "1min", "5min": "5min", "15min": "15min"}
    freq = freq_map.get(period, "5min")

    dates = pd.date_range(start="2024-01-01", periods=n_rows, freq=freq)

    # 模拟价格序列（随机游走）
    returns = np.random.normal(0, 0.002, n_rows)
    close = 5000 * np.exp(np.cumsum(returns))

    # 生成OHLCV数据
    noise = np.random.uniform(0.001, 0.005, n_rows)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_price = close * (1 + np.random.normal(0, 0.001, n_rows))

    volume = np.random.randint(100, 10000, n_rows).astype(float)
    open_interest = 50000 + np.cumsum(np.random.randint(-100, 100, n_rows)).astype(float)
    open_interest = np.maximum(open_interest, 1000)  # 确保持仓量为正

    df = pd.DataFrame({
        "datetime": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "open_interest": open_interest,
    })
    df.set_index("datetime", inplace=True)
    return df


def prepare_data(df, period="5min", target_name="future_direction", horizon=None):
    """
    准备训练数据：计算特征和目标变量。

    Parameters
    ----------
    df : pd.DataFrame
        原始K线数据
    period : str
        K线周期
    target_name : str
        预测目标名称
    horizon : int, optional
        预测周期

    Returns
    -------
    tuple
        (X, y) 特征矩阵和目标变量
    """
    logger.info(f"计算 {period} 周期衍生指标...")
    features = compute_all_features(df, period=period)

    if horizon is None:
        horizon = PREDICTION_TARGETS.get(target_name, {}).get("horizon", {}).get(period, 5)

    logger.info(f"计算预测目标: {target_name}, 预测周期: {horizon}")
    targets = compute_prediction_targets(df, horizon=horizon)

    # 合并特征和目标
    data = pd.concat([features, targets], axis=1)
    data.dropna(inplace=True)

    X = data[features.columns]
    y = data[target_name]

    # 替换无穷值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().fillna(0)

    logger.info(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")

    # 显示分层特征结构
    hierarchy = get_feature_hierarchy(X)
    for level, info in hierarchy.items():
        n = len(info["features"])
        logger.info(f"  {level} ({info['description']}): {n}个特征")

    return X, y


def train_and_evaluate(X, y, period="5min", task="classification"):
    """
    训练模型并评估。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    period : str
        K线周期
    task : str
        任务类型

    Returns
    -------
    tuple
        (model, metrics) 训练好的模型和评估指标
    """
    train_ratio = BACKTEST_CONFIG["train_ratio"]
    val_ratio = BACKTEST_CONFIG["validation_ratio"]

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    logger.info(f"创建 {period} 周期模型 (推荐框架: {ML_FRAMEWORKS[period]})...")
    # 15分钟周期使用LSTM不参与集成，集成仅用于LightGBM+XGBoost周期
    use_ensemble = ENSEMBLE_CONFIG.get("enabled", False) and period != "15min"
    if use_ensemble:
        logger.info("使用 LightGBM+XGBoost 集成模型...")
        model = EnsembleModel(task=task)
    else:
        model = create_model(period, task=task)

    logger.info("训练模型...")
    model.train(X_train, y_train, X_val, y_val)

    logger.info("评估模型...")
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(f"训练集指标: {train_metrics}")
    logger.info(f"测试集指标: {test_metrics}")

    # 特征重要性
    importance = model.get_feature_importance()
    if importance is not None:
        logger.info(f"Top 10 重要特征:\n{importance.head(10)}")

    # 集成模型权重
    if use_ensemble:
        logger.info(f"集成模型权重: {model.get_model_weights()}")

    # 仓位管理建议
    if task == "classification" and hasattr(model, 'predict_proba'):
        try:
            probas = model.predict_proba(X_test)[:, 1]
            test_volatility = X_test["atr"].values if "atr" in X_test.columns else np.ones(len(X_test)) * 0.01
            risk_mgr = RiskBudgetManager()
            signals = risk_mgr.compute_signals(probas, test_volatility)
            n_trades = np.sum(signals["signal"] != 0)
            logger.info(f"仓位管理: 测试集产生 {n_trades} 个交易信号")
        except Exception:
            pass

    return model, {"train": train_metrics, "test": test_metrics}


def run_feature_selection(X, y, period="5min", task="classification"):
    """
    运行特征选择流程。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    period : str
        K线周期
    task : str
        任务类型

    Returns
    -------
    dict
        特征选择结果
    """
    logger.info("=" * 60)
    logger.info("开始回测特征选择...")
    logger.info("=" * 60)

    # 对LSTM使用XGBoost做特征选择（LSTM没有直接的特征重要性）
    model_map = {
        "1min": LightGBMModel,
        "5min": XGBoostModel,
        "15min": XGBoostModel,
    }
    model_class = model_map[period]

    selector = FeatureSelector(
        n_splits=BACKTEST_CONFIG["n_splits"],
        min_success_rate=BACKTEST_CONFIG["min_success_rate"],
    )

    # 1. 评估各特征组的成功率
    logger.info("评估各特征组的成功率...")
    feature_groups = get_recommended_feature_groups()
    group_results = selector.evaluate_feature_subsets(
        model_class, X, y, feature_groups, task=task
    )
    logger.info(f"特征组评估结果:\n{group_results[['group', 'n_features', 'accuracy']].to_string()}")

    # 2. 综合特征选择
    logger.info("综合特征选择...")
    best_features = selector.select_best_features(
        model_class, X, y, task=task, top_n=20
    )
    logger.info(f"最佳特征数量: {best_features['n_features']}")
    logger.info(f"最佳特征: {best_features['selected_features']}")
    if task == "classification":
        logger.info(f"最终成功率: {best_features['final_success_rate']:.4f}")

    return {
        "group_results": group_results,
        "best_features": best_features,
        "selector": selector,
    }


def main():
    """主流程入口"""
    parser = argparse.ArgumentParser(description="商品期货机器学习量化模型")
    parser.add_argument("--period", type=str, default="5min",
                        choices=["1min", "5min", "15min"],
                        help="K线周期 (default: 5min)")
    parser.add_argument("--target", type=str, default="future_direction",
                        choices=["future_return", "future_direction", "future_regime"],
                        help="预测目标 (default: future_direction)")
    parser.add_argument("--n-rows", type=int, default=5000,
                        help="模拟数据行数 (default: 5000)")
    parser.add_argument("--skip-feature-selection", action="store_true",
                        help="跳过特征选择步骤")
    args = parser.parse_args()

    task = "classification" if args.target in ("future_direction", "future_regime") else "regression"

    logger.info("=" * 60)
    logger.info("商品期货机器学习量化模型")
    logger.info(f"周期: {args.period}, 目标: {args.target}, 任务: {task}")
    logger.info(f"推荐ML框架: {ML_FRAMEWORKS[args.period]}")
    logger.info("=" * 60)

    # 1. 生成/加载数据
    logger.info("生成模拟数据...")
    df = generate_sample_data(n_rows=args.n_rows, period=args.period)
    logger.info(f"数据形状: {df.shape}")

    # 2. 准备数据
    X, y = prepare_data(df, period=args.period, target_name=args.target)

    # 3. 训练和评估
    model, metrics = train_and_evaluate(X, y, period=args.period, task=task)

    # 4. 特征选择（可选）
    if not args.skip_feature_selection:
        selection_results = run_feature_selection(X, y, period=args.period, task=task)

    logger.info("=" * 60)
    logger.info("流程完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
