import datetime
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from run_real_data_pipeline import (
    setup_logging,
    step_load_data,
    step_feature_engineering,
    step_train_model,
    step_backtest,
    step_predict_and_save,
    ensure_output_dirs,
)
from backtest.feature_selector import FeatureSelector
from models.xgb_feature_selector import XGBFeatureSelector
from config import LSTM_FEATURE_SELECTION_CONFIG


def run_light_feature_selection(X, y, period, logger, xgb_n_features=25):
    if period == "15min":
        sel_cfg = LSTM_FEATURE_SELECTION_CONFIG.get("selection", {})
        xgb_params = LSTM_FEATURE_SELECTION_CONFIG.get("xgb_params", {})
        n_features = xgb_n_features if xgb_n_features > 0 else int(sel_cfg.get("n_features", 25))

        xgb_selector = XGBFeatureSelector(
            n_features=n_features,
            threshold=sel_cfg.get("threshold", "median"),
            cv_splits=int(sel_cfg.get("cv_splits", 5)),
            xgb_params=xgb_params,
        )
        selected_features, xgb_imp_df = xgb_selector.fit_select(X, y)
        if selected_features is None or len(selected_features) == 0:
            raise ValueError("15min 必须使用 XGBoost 预筛选获得有效特征")

        report_lines = []
        report_lines.append(f"# 特征筛选报告(15min专用) — {period}\n")
        report_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("流程: XGBoost 特征预筛选 -> LSTM 训练")
        report_lines.append(f"原始特征数量: {X.shape[1]}")
        report_lines.append(f"筛选后特征数量: {len(selected_features)}\n")
        report_lines.append("## XGBoost 重要性 (Top 30)")
        report_lines.append("| 排名 | 特征 | 重要性 | 标准差 |")
        report_lines.append("|------|------|--------|--------|")
        selected_set = set(selected_features)
        rank = 1
        for _, row in xgb_imp_df.iterrows():
            if row["feature"] in selected_set and rank <= 30:
                report_lines.append(
                    f"| {rank} | {row['feature']} | {row['importance']:.6f} | {row['std']:.6f} |"
                )
                rank += 1

        output_dir = os.path.join("data", "mx")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"feature_selection_report_{period}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"  15min XGBoost特征筛选报告已保存: {report_path}")
        return selected_features

    selector = FeatureSelector(n_splits=5, min_success_rate=0.55)
    et_df = selector.extra_trees_importance_stability(X, y, top_n=50, task="classification")
    selected_features = et_df.head(40)["feature"].tolist()

    report_lines = []
    report_lines.append(f"# 特征筛选报告(轻量-全流程) — {period}\n")
    report_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"原始特征数量: {X.shape[1]}")
    report_lines.append(f"筛选后特征数量: {len(selected_features)}\n")
    report_lines.append("## ExtraTrees 重要性 + 稳定性 (Top 30)")
    report_lines.append("| 排名 | 特征 | 重要性 | 标准差 | 稳定性得分 |")
    report_lines.append("|------|------|--------|--------|------------|")

    for i, (_, row) in enumerate(et_df.head(30).iterrows(), 1):
        report_lines.append(
            f"| {i} | {row['feature']} | {row['importance']:.6f} | {row['std']:.6f} | {row['stability_score']:.2f} |"
        )

    output_dir = os.path.join("data", "mx")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"feature_selection_report_{period}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"  轻量特征筛选报告已保存: {report_path}")
    return selected_features


def run_full_pipeline_all_periods(periods=None, target_name="future_direction"):
    if periods is None:
        periods = ["1min", "5min", "15min"]

    ensure_output_dirs()

    max_rows_map = {
        "1min": 10000,
        "5min": 10000,
        "15min": 10000,
    }

    summary = []

    for period in periods:
        logger = setup_logging(period)
        logger.info("=" * 70)
        logger.info(f"Full pipeline start: {period}")
        logger.info("=" * 70)

        row = {
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "period": period,
            "status": "success",
            "n_samples": "",
            "n_features": "",
            "n_selected_features": "",
            "test_accuracy": "",
            "test_f1": "",
            "win_rate": "",
            "profit_factor": "",
            "message": "",
        }

        try:
            df, is_real = step_load_data(period, n_rows=5000, logger=logger)
            max_rows = max_rows_map.get(period)
            if max_rows is not None and len(df) > max_rows:
                df = df.tail(max_rows).copy()
                logger.info(f"  使用最近 {max_rows} 行运行全流程")

            X, y = step_feature_engineering(df, period, target_name, logger)
            selected_features = run_light_feature_selection(X, y, period, logger)

            model, metrics, X_use = step_train_model(
                X, y, period, "classification", selected_features, logger
            )
            test_metrics, perf = step_backtest(
                X_use, y, model, period, "classification", metrics, is_real, logger
            )
            step_predict_and_save(
                X_use, y, model, period, "classification", selected_features, logger
            )

            row["n_samples"] = str(X_use.shape[0])
            row["n_features"] = str(X.shape[1])
            row["n_selected_features"] = str(len(selected_features))
            row["test_accuracy"] = str(round(float(test_metrics.get("accuracy", 0.0)), 6))
            row["test_f1"] = str(round(float(test_metrics.get("f1_score", 0.0)), 6))
            row["win_rate"] = str(round(float(perf.get("win_rate", 0.0)), 6))
            row["profit_factor"] = str(round(float(perf.get("profit_factor", 0.0)), 6))
            row["message"] = "ok"

        except Exception as e:
            row["status"] = "failed"
            row["message"] = str(e)
            logger.exception(f"Full pipeline failed for {period}: {e}")

        summary.append(row)

    out_path = os.path.join("data", "mx", "full_pipeline_summary.csv")
    headers = [
        "datetime",
        "period",
        "status",
        "n_samples",
        "n_features",
        "n_selected_features",
        "test_accuracy",
        "test_f1",
        "win_rate",
        "profit_factor",
        "message",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in summary:
            values = [str(r[h]).replace(",", " ").replace("\n", " ") for h in headers]
            f.write(",".join(values) + "\n")

    print(f"saved: {out_path}")
    for r in summary:
        print(r)


if __name__ == "__main__":
    run_full_pipeline_all_periods()
