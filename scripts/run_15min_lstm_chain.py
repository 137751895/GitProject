import argparse
import datetime
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from run_real_data_pipeline import (
    setup_logging,
    ensure_output_dirs,
    step_load_data,
    step_feature_engineering,
    step_train_model,
    step_backtest,
    step_predict_and_save,
    OUTPUT_DIR,
)
from models.xgb_feature_selector import XGBFeatureSelector
from config import LSTM_FEATURE_SELECTION_CONFIG


def run_chain(
    target: str = "future_direction",
    xgb_n_features: int = 25,
    max_rows: int | None = None,
):
    logger = setup_logging("15min")
    ensure_output_dirs()

    logger.info("=" * 70)
    logger.info("15min LSTM 特殊链路开始：XGBoost预筛选 -> LSTM训练")
    logger.info("=" * 70)

    # Step 1: load real data
    df, is_real = step_load_data("15min", n_rows=5000, logger=logger)
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).copy()
        logger.info(f"  使用最近 {max_rows} 行运行链路")

    # Step 2: feature engineering
    X, y = step_feature_engineering(df, "15min", target, logger)

    # Step 3: XGBoost pre-selection for LSTM
    logger.info("=" * 60)
    logger.info("Step 3: 15min XGBoost特征预筛选")
    logger.info("=" * 60)

    sel_cfg = LSTM_FEATURE_SELECTION_CONFIG.get("selection", {})
    xgb_params = LSTM_FEATURE_SELECTION_CONFIG.get("xgb_params", {})

    n_features = xgb_n_features if xgb_n_features > 0 else int(sel_cfg.get("n_features", 25))
    selector = XGBFeatureSelector(
        n_features=n_features,
        threshold=sel_cfg.get("threshold", "median"),
        cv_splits=int(sel_cfg.get("cv_splits", 5)),
        xgb_params=xgb_params,
    )

    selected_features, xgb_imp_df = selector.fit_select(X, y)

    dynamic_result = selector.dynamic_feature_count(
        X,
        y,
        min_features=int(sel_cfg.get("dynamic_min_features", 10)),
        max_features=min(int(sel_cfg.get("dynamic_max_features", 40)), X.shape[1]),
        step=int(sel_cfg.get("dynamic_step", 5)),
    )
    optimal_n = int(dynamic_result["optimal_n_features"])

    if xgb_n_features == 0 and optimal_n != len(selected_features):
        logger.info(f"  使用动态建议特征数重筛选: {optimal_n}")
        selector2 = XGBFeatureSelector(
            n_features=optimal_n,
            cv_splits=int(sel_cfg.get("cv_splits", 5)),
            xgb_params=xgb_params,
        )
        selected_features, xgb_imp_df = selector2.fit_select(X, y)

    logger.info(f"  XGBoost筛选特征数: {len(selected_features)}")
    logger.info(f"  动态搜索建议特征数: {optimal_n}")

    # Save dedicated feature selection report for this chain
    fs_report = os.path.join(OUTPUT_DIR, "feature_selection_report_15min_lstm_chain.md")
    lines = [
        "# 15min LSTM特征预筛选报告（XGBoost代理）",
        "",
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"原始特征数量: {X.shape[1]}",
        f"筛选特征数量: {len(selected_features)}",
        f"动态建议特征数量: {optimal_n}",
        "",
        "## 入选特征（按重要性）",
    ]
    selected_set = set(selected_features)
    rank = 1
    for _, row in xgb_imp_df.iterrows():
        feat = row["feature"]
        if feat in selected_set:
            lines.append(f"{rank}. {feat} (importance={row['importance']:.6f}, std={row['std']:.6f})")
            rank += 1
    with open(fs_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  链路特征筛选报告已保存: {fs_report}")

    # Step 4/5/6: train, backtest, predict
    model, metrics, X_use = step_train_model(
        X, y, "15min", "classification", selected_features, logger
    )
    test_metrics, perf = step_backtest(
        X_use, y, model, "15min", "classification", metrics, is_real, logger
    )
    pred_df = step_predict_and_save(
        X_use, y, model, "15min", "classification", selected_features, logger
    )

    # Save chain summary
    summary_path = os.path.join(OUTPUT_DIR, "lstm_chain_summary_15min.md")
    summary_lines = [
        "# 15min LSTM特殊处理链路总结",
        "",
        "流程: XGBoost特征预筛选 -> LSTM训练 -> 回测 -> 预测保存",
        "",
        f"数据类型: {'真实数据' if is_real else '模拟数据'}",
        f"样本数: {X_use.shape[0]}",
        f"原始特征数: {X.shape[1]}",
        f"筛选特征数: {len(selected_features)}",
        f"预测行数: {len(pred_df)}",
        "",
        "## 测试集指标",
    ]
    for k, v in test_metrics.items():
        if isinstance(v, float):
            summary_lines.append(f"- {k}: {v:.6f}")
        else:
            summary_lines.append(f"- {k}: {v}")

    if perf:
        summary_lines.append("")
        summary_lines.append("## 交易绩效")
        for k, v in perf.items():
            if isinstance(v, float):
                summary_lines.append(f"- {k}: {v:.6f}")
            else:
                summary_lines.append(f"- {k}: {v}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    logger.info(f"  链路总结已保存: {summary_path}")
    logger.info("15min LSTM 特殊链路完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="15min: XGBoost特征预筛选 -> LSTM训练链路")
    parser.add_argument("--target", default="future_direction", choices=["future_direction", "future_return", "future_regime"])
    parser.add_argument("--xgb-n-features", type=int, default=25, help="XGBoost筛选特征数，0=自动使用动态建议")
    parser.add_argument("--max-rows", type=int, default=10000, help="使用最近N行，0表示使用全量")
    args = parser.parse_args()

    mr = None if args.max_rows == 0 else args.max_rows
    run_chain(target=args.target, xgb_n_features=args.xgb_n_features, max_rows=mr)
