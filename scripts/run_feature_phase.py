import datetime
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from run_real_data_pipeline import setup_logging, step_load_data, step_feature_engineering
from backtest.feature_selector import FeatureSelector


def run_light_feature_selection(X, y, period, xgb_n_features, logger):
    selector = FeatureSelector(n_splits=5, min_success_rate=0.55)
    et_df = selector.extra_trees_importance_stability(X, y, top_n=50, task="classification")
    selected_features = et_df.head(40)["feature"].tolist()

    report_lines = []
    report_lines.append(f"# 特征筛选报告(轻量) — {period}\n")
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

    if period == "15min":
        from models.xgb_feature_selector import XGBFeatureSelector
        xgb_selector = XGBFeatureSelector(n_features=xgb_n_features, cv_splits=3)
        xgb_selected, _ = xgb_selector.fit_select(X, y)
        if not xgb_selected:
            raise ValueError("15min 周期 XGBoost 预筛选未返回有效特征，禁止回退")
        selected_features = xgb_selected
        report_lines.append("\n## 15min XGBoost 预筛选")
        report_lines.append("流程要求：XGBoost预筛选 -> LSTM训练")
        report_lines.append(f"XGBoost筛选特征数量: {len(xgb_selected)}")

    output_dir = os.path.join("data", "mx")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"feature_selection_report_{period}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"  轻量特征筛选报告已保存: {report_path}")

    return selected_features


def run_feature_phase(periods=None, target_name="future_direction", xgb_n_features=25):
    if periods is None:
        periods = ["1min", "5min", "15min"]

    max_rows_map = {
        "1min": 10000,
        "5min": 10000,
        "15min": 10000,
    }

    summary = []

    for period in periods:
        logger = setup_logging(period)
        logger.info("=" * 70)
        logger.info(f"Feature phase start: {period}")
        logger.info("=" * 70)

        row = {
            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "period": period,
            "status": "success",
            "n_samples": "",
            "n_features": "",
            "n_selected_features": "",
            "message": "",
        }

        try:
            df, _ = step_load_data(period, n_rows=5000, logger=logger)
            max_rows = max_rows_map.get(period)
            if max_rows is not None and len(df) > max_rows:
                df = df.tail(max_rows).copy()
                logger.info(f"  使用最近 {max_rows} 行进行特征阶段计算")
            X, y = step_feature_engineering(df, period, target_name, logger)
            selected_features = run_light_feature_selection(X, y, period, xgb_n_features, logger)

            row["n_samples"] = str(X.shape[0])
            row["n_features"] = str(X.shape[1])
            row["n_selected_features"] = str(len(selected_features) if selected_features is not None else 0)
            row["message"] = "ok"
        except Exception as e:
            row["status"] = "failed"
            row["message"] = str(e)
            logger.exception(f"Feature phase failed for {period}: {e}")

        summary.append(row)

    output_dir = os.path.join("data", "mx")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "feature_phase_summary.csv")

    headers = [
        "datetime",
        "period",
        "status",
        "n_samples",
        "n_features",
        "n_selected_features",
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
    run_feature_phase()
