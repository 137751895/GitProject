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
)
from backtest.feature_selector import FeatureSelector


def run():
    logger = setup_logging("15min")

    df, is_real = step_load_data("15min", n_rows=5000, logger=logger)
    if len(df) > 10000:
        df = df.tail(10000).copy()
        logger.info("  使用最近 10000 行运行 15min 轻量全流程")

    X, y = step_feature_engineering(df, "15min", "future_direction", logger)

    selector = FeatureSelector(n_splits=5, min_success_rate=0.55)
    et_df = selector.extra_trees_importance_stability(X, y, top_n=50, task="classification")
    selected_features = et_df.head(40)["feature"].tolist()

    model, metrics, X_use = step_train_model(X, y, "15min", "classification", selected_features, logger)
    test_metrics, perf = step_backtest(X_use, y, model, "15min", "classification", metrics, is_real, logger)
    pred_df = step_predict_and_save(X_use, y, model, "15min", "classification", selected_features, logger)

    print({
        "status": "success",
        "n_samples": int(X_use.shape[0]),
        "n_features": int(X.shape[1]),
        "n_selected": int(len(selected_features)),
        "accuracy": float(test_metrics.get("accuracy", 0.0)),
        "f1": float(test_metrics.get("f1_score", 0.0)),
        "win_rate": float(perf.get("win_rate", 0.0)),
        "pred_rows": int(len(pred_df)),
    })


if __name__ == "__main__":
    run()
