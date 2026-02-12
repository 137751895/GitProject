#!/usr/bin/env python
"""
商品期货机器学习量化模型 - 真实数据全自动流水线
Full automated pipeline: data load → features → selection → train → backtest → predict.

使用方法:
    python run_real_data_pipeline.py --period 5min
    python run_real_data_pipeline.py --period 15min --xgb-n-features 30
    python run_real_data_pipeline.py --period 1min --skip-feature-selection --n-trials 10
"""

import argparse
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(period: str):
    """配置日志输出到控制台和文件。"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"real_data_pipeline_{period}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # clear existing handlers
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

from scripts.load_real_data import load_or_generate
from ml_pipeline import prepare_data, train_and_evaluate, generate_sample_data
from features.feature_engineering import compute_all_features, compute_prediction_targets
from models.ml_models import create_model
from models.position_sizing import RiskBudgetManager, compute_performance_metrics
from backtest.feature_selector import FeatureSelector, get_recommended_feature_groups
from config import (
    ML_FRAMEWORKS, PREDICTION_TARGETS, BACKTEST_CONFIG,
    POSITION_CONFIG, COST_CONFIG, LSTM_FEATURE_SELECTION_CONFIG,
)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join("data", "mx")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# ===========================================================================
# Step 3.1 — Data loading
# ===========================================================================

def step_load_data(period: str, n_rows: int, logger):
    """加载真实数据或回退到模拟数据。"""
    logger.info("=" * 60)
    logger.info("Step 1: 数据加载与预处理")
    logger.info("=" * 60)
    df, is_real = load_or_generate(period, n_rows=n_rows)
    data_type = "真实数据" if is_real else "模拟数据"
    logger.info(f"  数据类型: {data_type}")
    logger.info(f"  数据形状: {df.shape}")
    logger.info(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")
    logger.info(f"  列: {list(df.columns)}")
    return df, is_real


# ===========================================================================
# Step 3.2 — Feature engineering
# ===========================================================================

def step_feature_engineering(df, period, target_name, logger):
    """计算全量特征和预测目标。"""
    logger.info("=" * 60)
    logger.info("Step 2: 特征工程全量计算")
    logger.info("=" * 60)
    X, y = prepare_data(df, period=period, target_name=target_name)
    logger.info(f"  特征数量: {X.shape[1]}")
    logger.info(f"  样本数量: {X.shape[0]}")
    logger.info(f"  目标分布:\n{y.value_counts().to_string()}")
    return X, y


# ===========================================================================
# Step 3.3 — Feature selection
# ===========================================================================

def step_feature_selection(X, y, period, task, xgb_n_features, logger):
    """特征筛选与稳定性评估。15min额外执行XGBoost预筛选。"""
    logger.info("=" * 60)
    logger.info("Step 3: 特征筛选与稳定性评估")
    logger.info("=" * 60)

    report_lines = []
    report_lines.append(f"# 特征筛选报告 — {period}\n")
    report_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"原始特征数量: {X.shape[1]}\n")

    # --- ExtraTrees importance + stability ---
    logger.info("  运行 ExtraTrees 重要性 + 稳定性筛选...")
    selector = FeatureSelector(
        n_splits=BACKTEST_CONFIG["n_splits"],
        min_success_rate=BACKTEST_CONFIG["min_success_rate"],
    )
    et_df = selector.extra_trees_importance_stability(X, y, top_n=50, task=task)
    logger.info(f"  ExtraTrees Top 10:\n{et_df.head(10).to_string()}")

    report_lines.append("\n## ExtraTrees 特征重要性 + 稳定性 (Top 30)\n")
    report_lines.append("| 排名 | 特征 | 重要性 | 标准差 | 稳定性得分 |")
    report_lines.append("|------|------|--------|--------|------------|")
    for i, (_, row) in enumerate(et_df.head(30).iterrows(), 1):
        report_lines.append(
            f"| {i} | {row['feature']} | {row['importance']:.6f} | "
            f"{row['std']:.6f} | {row['stability_score']:.2f} |"
        )

    # --- RandomForest importance ---
    logger.info("  运行 RandomForest 特征重要性...")
    rf_imp = selector.random_forest_importance(X, y, top_n=50, task=task)
    logger.info(f"  RF Top 10:\n{rf_imp.head(10).to_string()}")

    report_lines.append("\n## RandomForest 特征重要性 (Top 30)\n")
    report_lines.append("| 排名 | 特征 | 重要性 |")
    report_lines.append("|------|------|--------|")
    for i, (feat, imp) in enumerate(rf_imp.head(30).items(), 1):
        report_lines.append(f"| {i} | {feat} | {imp:.6f} |")

    # Combined stable features from ET
    stable_features = et_df.head(40)["feature"].tolist()

    # --- 15min XGBoost pre-screening ---
    xgb_selected = None
    if period == "15min":
        logger.info("  [15min专用] 运行 XGBoost 特征预筛选...")
        from models.xgb_feature_selector import XGBFeatureSelector

        sel_cfg = LSTM_FEATURE_SELECTION_CONFIG["selection"]
        n_feat = xgb_n_features if xgb_n_features > 0 else sel_cfg["n_features"]
        xgb_params = LSTM_FEATURE_SELECTION_CONFIG.get("xgb_params", {})

        xgb_selector = XGBFeatureSelector(
            n_features=n_feat,
            threshold=sel_cfg.get("threshold", "median"),
            cv_splits=sel_cfg.get("cv_splits", 5),
            xgb_params=xgb_params,
        )
        xgb_selected, xgb_imp_df = xgb_selector.fit_select(X, y)
        logger.info(f"  XGBoost 筛选出 {len(xgb_selected)} 个特征")

        # Dynamic search
        dynamic_result = xgb_selector.dynamic_feature_count(
            X, y, min_features=10, max_features=min(40, X.shape[1]), step=5,
        )
        optimal_n = dynamic_result["optimal_n_features"]
        logger.info(f"  动态搜索建议最佳特征数: {optimal_n}")

        # If user set 0 (auto), re-select with optimal count
        if xgb_n_features == 0 and optimal_n != n_feat:
            logger.info(f"  使用动态建议数量 {optimal_n} 重新筛选...")
            xgb_selector2 = XGBFeatureSelector(
                n_features=optimal_n, cv_splits=sel_cfg.get("cv_splits", 5),
                xgb_params=xgb_params,
            )
            xgb_selected, xgb_imp_df = xgb_selector2.fit_select(X, y)

        # Feature group analysis
        feature_groups = get_recommended_feature_groups()
        group_analysis = xgb_selector.get_feature_groups_importance(feature_groups)

        report_lines.append("\n## 15分钟 XGBoost 特征预筛选结果\n")
        report_lines.append(f"筛选特征数量: {len(xgb_selected)}")
        report_lines.append(f"动态建议最佳数量: {optimal_n}\n")
        report_lines.append("### 入选特征列表\n")
        report_lines.append("| 排名 | 特征 | 重要性 | 标准差 |")
        report_lines.append("|------|------|--------|--------|")
        selected_set = set(xgb_selected)
        for i, (_, row) in enumerate(xgb_imp_df.iterrows(), 1):
            if row["feature"] in selected_set:
                report_lines.append(
                    f"| {i} | {row['feature']} | {row['importance']:.6f} | "
                    f"{row['std']:.6f} |"
                )
        report_lines.append("\n### 特征组重要性\n")
        report_lines.append("| 特征组 | 评分 | 总特征数 | 入选数 |")
        report_lines.append("|--------|------|----------|--------|")
        for grp, row in group_analysis.iterrows():
            report_lines.append(
                f"| {grp} | {row['score']:.4f} | "
                f"{int(row['n_features'])} | {int(row['n_selected'])} |"
            )
        report_lines.append(f"\n### 动态搜索结果\n")
        report_lines.append("| 特征数 | 准确率 | 标准差 |")
        report_lines.append("|--------|--------|--------|")
        for _, row in dynamic_result["search_results"].iterrows():
            marker = " ←最优" if int(row["n_features"]) == optimal_n else ""
            report_lines.append(
                f"| {int(row['n_features'])} | {row['accuracy']:.4f} | "
                f"{row['std']:.4f} |{marker}"
            )

        # Store selector for saving later
        stable_features = xgb_selected

    # Save report
    report_path = os.path.join(OUTPUT_DIR, f"feature_selection_report_{period}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"  特征筛选报告已保存: {report_path}")

    return stable_features, xgb_selected


# ===========================================================================
# Step 3.4 — Model training
# ===========================================================================

def step_train_model(X, y, period, task, selected_features, logger):
    """训练模型（15min用筛选后的特征子集训练LSTM）。"""
    logger.info("=" * 60)
    logger.info("Step 4: 模型训练")
    logger.info(f"  周期: {period}, 框架: {ML_FRAMEWORKS[period]}")
    logger.info("=" * 60)

    if selected_features is not None:
        X_use = X[selected_features]
        logger.info(f"  使用筛选后的 {len(selected_features)} 个特征")
    else:
        X_use = X
        logger.info(f"  使用全部 {X.shape[1]} 个特征")

    model, metrics = train_and_evaluate(X_use, y, period=period, task=task)
    logger.info(f"  训练集指标: {metrics['train']}")
    logger.info(f"  测试集指标: {metrics['test']}")

    return model, metrics, X_use


# ===========================================================================
# Step 3.5 — Backtest
# ===========================================================================

def step_backtest(X_use, y, model, period, task, metrics, is_real, logger):
    """回测与绩效评估。"""
    logger.info("=" * 60)
    logger.info("Step 5: 回测与绩效评估")
    logger.info("=" * 60)

    train_ratio = BACKTEST_CONFIG["train_ratio"]
    val_ratio = BACKTEST_CONFIG["validation_ratio"]
    n = len(X_use)
    val_end = int(n * (train_ratio + val_ratio))
    X_test = X_use.iloc[val_end:]
    y_test = y.iloc[val_end:]

    # Predictions
    y_pred = model.predict(X_test)
    test_metrics = metrics.get("test", {})

    # Performance metrics from simulated trades
    perf = {}
    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)[:, 1]
            trade_mask = probas > POSITION_CONFIG.get("base_threshold", 0.55)
            if np.sum(trade_mask) > 0:
                actual_returns = (y_test.values[trade_mask] * 2 - 1) * 0.001
                cost = COST_CONFIG.get("buy_rate", 0.0003)
                actual_returns = actual_returns - cost
                perf = compute_performance_metrics(actual_returns)
        except Exception:
            pass

    # Generate backtest report
    report_lines = []
    report_lines.append(f"# 回测报告 — {period}\n")
    report_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"数据类型: {'真实数据 (白银ag)' if is_real else '模拟数据'}\n")

    report_lines.append("## 模型评估指标\n")
    report_lines.append("| 指标 | 值 |")
    report_lines.append("|------|------|")
    for k, v in test_metrics.items():
        report_lines.append(f"| {k} | {v:.4f} |")

    report_lines.append("\n## 交易绩效\n")
    report_lines.append("| 指标 | 值 |")
    report_lines.append("|------|------|")
    for k, v in perf.items():
        if isinstance(v, float):
            report_lines.append(f"| {k} | {v:.4f} |")
        else:
            report_lines.append(f"| {k} | {v} |")

    report_lines.append(f"\n## 资金曲线说明\n")
    report_lines.append(f"- 测试集样本数: {len(X_test)}")
    report_lines.append(f"- 模型框架: {ML_FRAMEWORKS[period]}")
    report_lines.append(f"- 特征数量: {X_use.shape[1]}")
    report_lines.append(f"- 交易成本: 单边 {COST_CONFIG.get('buy_rate', 0.0003)*100:.2f}%")
    report_lines.append(f"- 最大仓位: {POSITION_CONFIG.get('max_position_usage', 0.9)*100:.0f}%")

    report_path = os.path.join(OUTPUT_DIR, f"backtest_report_{period}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"  回测报告已保存: {report_path}")

    return test_metrics, perf


# ===========================================================================
# Step 3.6 — Predictions & model save
# ===========================================================================

def step_predict_and_save(X_use, y, model, period, task, selected_features,
                          logger):
    """保存预测结果和模型。"""
    logger.info("=" * 60)
    logger.info("Step 6: 模型保存与预测输出")
    logger.info("=" * 60)

    train_ratio = BACKTEST_CONFIG["train_ratio"]
    val_ratio = BACKTEST_CONFIG["validation_ratio"]
    n = len(X_use)
    val_end = int(n * (train_ratio + val_ratio))
    X_test = X_use.iloc[val_end:]
    y_test = y.iloc[val_end:]

    y_pred = model.predict(X_test)

    # Save predictions
    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
    }, index=X_test.index)

    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            if probas.shape[1] == 2:
                pred_df["probability"] = probas[:, 1]
            else:
                for c in range(probas.shape[1]):
                    pred_df[f"prob_class_{c}"] = probas[:, c]
        except Exception:
            pass

    pred_path = os.path.join(OUTPUT_DIR, f"predictions_{period}.csv")
    pred_df.to_csv(pred_path)
    logger.info(f"  预测结果已保存: {pred_path}")

    # Save model
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    model_type = ML_FRAMEWORKS[period]
    model_filename = f"{period}_{model_type}_{today_str}"

    try:
        import joblib
        model_path = os.path.join(MODELS_DIR, f"{model_filename}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"  模型已保存: {model_path}")
    except ImportError:
        import pickle
        model_path = os.path.join(MODELS_DIR, f"{model_filename}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"  模型已保存 (pickle): {model_path}")

    # Save feature list
    feat_path = os.path.join(MODELS_DIR, f"{model_filename}_features.txt")
    features_to_save = selected_features if selected_features is not None else list(X_use.columns)
    with open(feat_path, "w") as f:
        for feat in features_to_save:
            f.write(feat + "\n")
    logger.info(f"  特征列表已保存: {feat_path}")

    # Save scaler if available (LSTM models have scalers)
    if hasattr(model, "scaler") and model.scaler is not None:
        try:
            import joblib
            scaler_path = os.path.join(MODELS_DIR, f"{model_filename}_scaler.pkl")
            joblib.dump(model.scaler, scaler_path)
            logger.info(f"  标准化器已保存: {scaler_path}")
        except Exception:
            pass

    return pred_df


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(period: str, target_name: str = "future_direction",
                 n_rows: int = 5000, skip_feature_selection: bool = False,
                 n_trials: int = 30, xgb_n_features: int = 25):
    """运行完整的真实数据流水线。

    Parameters
    ----------
    period : str
        K线周期: 1min, 5min, 15min
    target_name : str
        预测目标
    n_rows : int
        模拟数据行数（真实数据存在时忽略）
    skip_feature_selection : bool
        跳过特征筛选
    n_trials : int
        Optuna超参搜索次数
    xgb_n_features : int
        15min XGBoost预筛选特征数（0=自动搜索）
    """
    logger = setup_logging(period)

    logger.info("=" * 70)
    logger.info("  商品期货ML量化模型 — 真实数据全自动流水线")
    logger.info(f"  周期: {period}  目标: {target_name}  框架: {ML_FRAMEWORKS[period]}")
    logger.info("=" * 70)

    ensure_output_dirs()

    task = "classification" if target_name in ("future_direction", "future_regime") else "regression"

    # Step 1: 数据加载
    try:
        df, is_real = step_load_data(period, n_rows, logger)
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return

    # Step 2: 特征工程
    try:
        X, y = step_feature_engineering(df, period, target_name, logger)
    except Exception as e:
        logger.error(f"特征工程失败: {e}")
        return

    # Step 3: 特征筛选
    selected_features = None
    xgb_selected = None
    if not skip_feature_selection:
        try:
            selected_features, xgb_selected = step_feature_selection(
                X, y, period, task, xgb_n_features, logger,
            )
        except Exception as e:
            logger.warning(f"特征筛选出错: {e}，使用全部特征继续")
            selected_features = None
    else:
        logger.info("跳过特征筛选步骤")

    # For 15min, use XGBoost-selected features for LSTM
    if period == "15min" and xgb_selected is not None:
        selected_features = xgb_selected

    # Step 4: 模型训练
    try:
        model, metrics, X_use = step_train_model(
            X, y, period, task, selected_features, logger,
        )
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        return

    # Step 5: 回测
    try:
        test_metrics, perf = step_backtest(
            X_use, y, model, period, task, metrics, is_real, logger,
        )
    except Exception as e:
        logger.warning(f"回测步骤出错: {e}")
        test_metrics, perf = {}, {}

    # Step 6: 预测 & 保存
    try:
        pred_df = step_predict_and_save(
            X_use, y, model, period, task, selected_features, logger,
        )
    except Exception as e:
        logger.warning(f"预测/保存步骤出错: {e}")

    # Final summary
    logger.info("=" * 70)
    logger.info("  关键绩效摘要")
    logger.info("=" * 70)
    logger.info(f"  周期: {period}")
    logger.info(f"  框架: {ML_FRAMEWORKS[period]}")
    logger.info(f"  数据: {'真实数据 (白银ag)' if is_real else '模拟数据'}")
    logger.info(f"  特征数量: {X_use.shape[1]} / 原始 {X.shape[1]}")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    for k, v in perf.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info("=" * 70)
    logger.info("流水线完成!")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="商品期货ML量化模型 — 真实数据全自动流水线",
    )
    parser.add_argument("--period", type=str, default="5min",
                        choices=["1min", "5min", "15min"],
                        help="K线周期 (default: 5min)")
    parser.add_argument("--target", type=str, default="future_direction",
                        choices=["future_return", "future_direction", "future_regime"],
                        help="预测目标 (default: future_direction)")
    parser.add_argument("--n-rows", type=int, default=5000,
                        help="模拟数据行数，真实数据存在时忽略 (default: 5000)")
    parser.add_argument("--skip-feature-selection", action="store_true",
                        help="跳过特征筛选步骤")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna超参搜索次数 (default: 30)")
    parser.add_argument("--xgb-n-features", type=int, default=25,
                        help="15min XGBoost预筛选特征数 (0=自动搜索, default: 25)")
    args = parser.parse_args()

    run_pipeline(
        period=args.period,
        target_name=args.target,
        n_rows=args.n_rows,
        skip_feature_selection=args.skip_feature_selection,
        n_trials=args.n_trials,
        xgb_n_features=args.xgb_n_features,
    )


if __name__ == "__main__":
    main()
