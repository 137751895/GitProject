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
from models.position_sizing import RiskBudgetManager, compute_performance_metrics
from models.adaptive_learning import AdaptiveModelManager
from backtest.feature_selector import FeatureSelector, get_recommended_feature_groups
from config import (ML_FRAMEWORKS, PREDICTION_TARGETS, BACKTEST_CONFIG,
                    ENSEMBLE_CONFIG, POSITION_CONFIG, ADAPTIVE_CONFIG,
                    MULTI_TIMEFRAME_CONFIG)

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

    # 仓位管理建议（增强版）
    if task == "classification" and hasattr(model, 'predict_proba'):
        try:
            probas = model.predict_proba(X_test)[:, 1]
            test_volatility = X_test["atr"].values if "atr" in X_test.columns else np.ones(len(X_test)) * 0.01
            risk_mgr = RiskBudgetManager(
                account_risk=POSITION_CONFIG.get("account_risk", 0.02),
                max_position=POSITION_CONFIG.get("max_position", 1.0),
                base_threshold=POSITION_CONFIG.get("base_threshold", 0.55),
                max_drawdown=POSITION_CONFIG.get("max_drawdown", 0.15),
                drawdown_warning=POSITION_CONFIG.get("drawdown_warning", 0.08),
                max_daily_loss=POSITION_CONFIG.get("max_daily_loss", 0.03),
                max_daily_trades=POSITION_CONFIG.get("max_daily_trades", 20),
            )
            signals = risk_mgr.compute_signals(probas, test_volatility)
            n_trades = np.sum(signals["signal"] != 0)
            logger.info(f"仓位管理: 测试集产生 {n_trades} 个交易信号")
            logger.info(f"  回撤保护乘数: {signals['drawdown_multiplier']:.2f}")

            # 模拟交易绩效
            trade_mask = signals["signal"] != 0
            if np.sum(trade_mask) > 0:
                # 模拟收益: 将二分类标签(0/1)映射为方向(-1/+1)，乘以模拟单笔收益0.1%
                actual_returns = (y_test.values[trade_mask] * 2 - 1) * 0.001
                perf = compute_performance_metrics(actual_returns)
                logger.info(f"  模拟绩效: 胜率={perf['win_rate']:.2%}, "
                            f"盈亏比={perf['profit_factor']:.2f}, "
                            f"Sharpe={perf['sharpe_ratio']:.2f}")
        except Exception:
            pass

    # 自适应模型管理演示
    if task == "classification" and period != "15min":
        try:
            adaptive_mgr = AdaptiveModelManager(
                base_model=model if not use_ensemble else model.lgbm,
                drift_window=ADAPTIVE_CONFIG.get("drift_window", 100),
                drift_warning=ADAPTIVE_CONFIG.get("drift_warning_threshold", 0.05),
                drift_threshold=ADAPTIVE_CONFIG.get("drift_threshold", 0.10),
                update_interval=ADAPTIVE_CONFIG.get("update_interval", 500),
                n_incremental_trees=ADAPTIVE_CONFIG.get("n_incremental_trees", 50),
                base_threshold=POSITION_CONFIG.get("base_threshold", 0.55),
            )
            y_pred_test = model.predict(X_test)
            test_vol = X_test["atr"].values if "atr" in X_test.columns else np.ones(len(X_test)) * 0.01
            vol_mean = float(np.nanmean(test_vol))

            # 模拟逐步喂入数据
            batch_size = min(100, len(X_test))
            status = adaptive_mgr.step(
                y_true=y_test.values[:batch_size],
                y_pred=y_pred_test[:batch_size],
                current_volatility=float(np.nanmean(test_vol[:batch_size])),
                mean_volatility=vol_mean,
            )
            summary = adaptive_mgr.get_status_summary()
            logger.info(f"自适应模块: 漂移={status['drift_level']}, "
                        f"阈值={status['threshold']:.3f}, "
                        f"需要重训={status['needs_retrain']}")
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


def run_multi_timeframe(n_rows=5000, target="future_direction"):
    """
    运行多时间框架协同交易演示。

    训练15分钟、5分钟、1分钟三个周期的模型，
    然后通过 MultiTimeframeCoordinator 协同生成交易信号。

    Parameters
    ----------
    n_rows : int
        模拟数据行数
    target : str
        预测目标名称
    """
    from models.multi_timeframe import MultiTimeframeCoordinator

    logger.info("=" * 60)
    logger.info("多时间框架协同交易系统")
    logger.info("=" * 60)

    task = "classification" if target in ("future_direction", "future_regime") else "regression"

    # Step 1: 生成三个周期的数据
    logger.info("生成三个周期的模拟数据...")
    df_15m = generate_sample_data(n_rows=n_rows, period="15min")
    df_5m = generate_sample_data(n_rows=n_rows * 3, period="5min")
    df_1m = generate_sample_data(n_rows=n_rows * 15, period="1min")

    # Step 2: 各周期独立训练
    logger.info("训练15分钟模型（趋势方向）...")
    X_15m, y_15m = prepare_data(df_15m, period="15min", target_name=target)
    # 多时间框架模式使用XGBoost作为15分钟模型，因为协同系统需要predict_proba
    # 且XGBoost无需TensorFlow依赖，适合轻量级部署
    model_15m = XGBoostModel(task=task)
    train_ratio = BACKTEST_CONFIG["train_ratio"]
    val_ratio = BACKTEST_CONFIG["validation_ratio"]
    n_15 = len(X_15m)
    tr_end_15 = int(n_15 * train_ratio)
    val_end_15 = int(n_15 * (train_ratio + val_ratio))
    model_15m.train(X_15m.iloc[:tr_end_15], y_15m.iloc[:tr_end_15],
                    X_15m.iloc[tr_end_15:val_end_15], y_15m.iloc[tr_end_15:val_end_15])
    metrics_15m = model_15m.evaluate(X_15m.iloc[val_end_15:], y_15m.iloc[val_end_15:])
    logger.info(f"  15分钟模型测试集: {metrics_15m}")

    logger.info("训练5分钟模型（入场时机）...")
    X_5m, y_5m = prepare_data(df_5m, period="5min", target_name=target)
    model_5m, _ = train_and_evaluate(X_5m, y_5m, period="5min", task=task)

    logger.info("训练1分钟模型（入场价格）...")
    X_1m, y_1m = prepare_data(df_1m, period="1min", target_name=target)
    model_1m, _ = train_and_evaluate(X_1m, y_1m, period="1min", task=task)

    # Step 3: 创建协同系统
    logger.info("=" * 60)
    logger.info("创建多时间框架协同系统...")
    mtf_config = MULTI_TIMEFRAME_CONFIG
    coordinator = MultiTimeframeCoordinator(
        neutral_zone=mtf_config.get("neutral_zone", 0.10),
        entry_threshold=mtf_config.get("entry_threshold", 0.60),
        optimization_threshold=mtf_config.get("optimization_threshold", 0.55),
        min_grade=mtf_config.get("min_grade", "C"),
    )
    coordinator.set_models(
        model_15min=model_15m,
        model_5min=model_5m,
        model_1min=model_1m,
    )

    # Step 4: 使用测试集数据生成协同信号
    # 取各周期测试集的尾部数据（对齐最小长度）
    test_start = int(len(X_15m) * 0.85)
    X_15m_test = X_15m.iloc[test_start:]
    test_start_5m = int(len(X_5m) * 0.85)
    X_5m_test = X_5m.iloc[test_start_5m:]
    test_start_1m = int(len(X_1m) * 0.85)
    X_1m_test = X_1m.iloc[test_start_1m:]

    # 对齐到最小长度
    min_len = min(len(X_15m_test), len(X_5m_test), len(X_1m_test))
    X_15m_test = X_15m_test.iloc[:min_len]
    X_5m_test = X_5m_test.iloc[:min_len]
    X_1m_test = X_1m_test.iloc[:min_len]

    logger.info(f"协同信号生成: {min_len} 个样本")
    result = coordinator.generate_signals(X_15m_test, X_5m_test, X_1m_test)

    # Step 5: 输出协同信号统计
    summary = coordinator.get_signal_summary()
    logger.info("=" * 60)
    logger.info("多时间框架协同信号统计:")
    logger.info(f"  总样本数: {summary['n_total']}")
    logger.info(f"  做多信号: {summary['n_long']}")
    logger.info(f"  做空信号: {summary['n_short']}")
    logger.info(f"  中性(无信号): {summary['n_neutral']}")
    logger.info(f"  A级信号(三周期共振): {summary.get('grade_A', 0)}")
    logger.info(f"  B级信号(双周期确认): {summary.get('grade_B', 0)}")
    logger.info(f"  平均信号强度: {summary['avg_signal_strength']:.3f}")
    logger.info(f"  平均一致性评分: {summary['avg_agreement']:.3f}")
    logger.info("=" * 60)
    logger.info("多时间框架协同交易系统完成!")


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
    parser.add_argument("--multi-timeframe", action="store_true",
                        help="运行多时间框架协同交易系统")
    args = parser.parse_args()

    # 多时间框架协同模式
    if args.multi_timeframe:
        run_multi_timeframe(n_rows=args.n_rows, target=args.target)
        return

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
