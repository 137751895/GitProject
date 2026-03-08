import os
import sys
import traceback
import tempfile
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

results = []


def record(name, ok, msg=""):
    results.append({"module": name, "status": "PASS" if ok else "FAIL", "message": msg})


def make_df(n=300):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    base = np.cumsum(np.random.randn(n) * 0.5) + 5000
    df = pd.DataFrame({
        "open": base + np.random.randn(n) * 0.2,
        "high": base + np.abs(np.random.randn(n) * 0.5),
        "low": base - np.abs(np.random.randn(n) * 0.5),
        "close": base,
        "volume": np.random.randint(100, 10000, n).astype(float),
        "open_interest": np.random.randint(10000, 12000, n).astype(float),
    }, index=idx)
    return df


try:
    from scripts.load_real_data import load_kline_csv
    path = os.path.join(ROOT_DIR, "data", "klines", "KQi@SHFEag", "KQi@SHFEag_5min.csv")
    _ = load_kline_csv(path)
    record("load_real_data", True)
except Exception as e:
    record("load_real_data", False, str(e))

try:
    from features.feature_engineering import compute_all_features, compute_prediction_targets
    df = make_df()
    feat = compute_all_features(df, period="5min")
    tgt = compute_prediction_targets(df, horizon=3)
    assert len(feat) == len(df)
    assert len(tgt) == len(df)
    record("feature_engineering", True)
except Exception as e:
    record("feature_engineering", False, str(e))

try:
    from backtest.feature_selector import FeatureSelector
    selector = FeatureSelector(n_splits=3)
    data = pd.concat([feat, tgt[["future_direction"]]], axis=1).dropna()
    X = data[feat.columns].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y = data["future_direction"]
    et = selector.extra_trees_importance_stability(X, y, top_n=10, task="classification")
    assert not et.empty
    record("feature_selector", True)
except Exception as e:
    record("feature_selector", False, str(e))

try:
    from models.smart_labels import SmartLabelGenerator
    gen = SmartLabelGenerator(horizon=3)
    lab = gen.create_labels(make_df())
    assert "trading_signal" in lab.columns
    record("smart_labels", True)
except Exception as e:
    record("smart_labels", False, str(e))

try:
    from models.hybrid_trading import HybridTradingSystem
    sysm = HybridTradingSystem()
    out = sysm.predict(make_df(400))
    assert "final_signal" in out
    record("hybrid_trading", True)
except Exception as e:
    record("hybrid_trading", False, str(e))

try:
    from models.multi_timeframe import MultiTimeframeCoordinator
    m = MultiTimeframeCoordinator()
    n = 120
    x = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    r = m.generate_signals(x, x, x)
    assert "final_signal" in r
    record("multi_timeframe", True)
except Exception as e:
    record("multi_timeframe", False, str(e))

try:
    from models.cost_model import CostConfig, StopConfig, calc_buy_cost, calc_sell_cash_in, should_stop
    c = CostConfig()
    s = StopConfig()
    _ = calc_buy_cost(100000, c)
    _ = calc_sell_cash_in(100000, c)
    _ = should_stop(100, 97, 1, s)
    record("cost_model", True)
except Exception as e:
    record("cost_model", False, str(e))

try:
    from models.var_stress import RiskModels
    rm = RiskModels()
    rets = (np.random.randn(400) * 0.01).tolist()
    vr = rm.calculate_comprehensive_var(rets)
    assert isinstance(vr, list)
    record("var_stress", True)
except Exception as e:
    record("var_stress", False, str(e))

try:
    from signal_postprocess import apply_topn_by_date_rank
    preds = pd.DataFrame({
        "date": ["2025-01-01"] * 5 + ["2025-01-02"] * 5,
        "pred": np.random.rand(10),
    })
    out = apply_topn_by_date_rank(preds, topn=2)
    assert "signal" in out.columns
    record("signal_postprocess", True)
except Exception as e:
    record("signal_postprocess", False, str(e))

try:
    from factor_loader import load_factors
    with tempfile.TemporaryDirectory() as td:
        idx = pd.DataFrame({
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["20250101", "20250102", "20250103"],
        })
        idx.index = pd.Index([0, 1, 2])
        idx_path = os.path.join(td, "index.pkl")
        idx.to_pickle(idx_path)

        f1 = pd.DataFrame({"alpha1": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        f1.to_pickle(os.path.join(td, "alpha1.pkl"))

        out = load_factors(idx_path, td, ["alpha1"])
        assert not out.empty
        record("factor_loader", True)
except Exception as e:
    record("factor_loader", False, str(e))

try:
    from hyperopt_runner import build_l2_pipeline_lgbm, run_optuna_for_hfml
    _ = build_l2_pipeline_lgbm(n_estimators=20)
    _ = run_optuna_for_hfml("5min", "future_direction", n_trials=1)
    record("hyperopt_runner", True)
except Exception as e:
    record("hyperopt_runner", False, str(e))

try:
    from models.adaptive_learning import ConceptDriftDetector
    d = ConceptDriftDetector(window_size=20)
    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.randint(0, 2, 50)
    _ = d.update(y_true, y_pred)
    record("adaptive_learning", True)
except Exception as e:
    record("adaptive_learning", False, str(e))

out_dir = os.path.join(ROOT_DIR, "data", "mx")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "module_smoke_results.csv")
pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"saved: {out_path}")
for r in results:
    print(r)
