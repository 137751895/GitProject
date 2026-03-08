# ===== 修改说明 =====
# 依据《当前项目性能提升指南-最终版v2.md》改进点 3.1 / 3.3 / 5.1
# 新增内容已用 # [新增] 标记
# ===================
"""
商品期货机器学习量化模型 - 超参优化模块
Hyperparameter optimization with Optuna and Bayesian Optimization.
"""

import optuna  # [新增]

from sklearn.model_selection import TimeSeriesSplit, cross_val_score  # [新增]
from sklearn.pipeline import Pipeline  # [新增]
from sklearn.preprocessing import MinMaxScaler, Normalizer  # [新增]

import lightgbm as lgb  # [新增]

from models.ml_models import create_model  # [新增]


# --- 改进点 3.1: Optuna 超参搜索 ---

def run_optuna_for_hfml(period: str, target_name: str, n_trials: int = 50):  # [新增]
    from ml_pipeline import generate_sample_data, prepare_data  # [新增]  # [BUGFIX] P1-2: lazy import to avoid circular dependency
    df = generate_sample_data(n_rows=8000, period=period)  # [新增]
    X, y = prepare_data(df, period=period, target_name=target_name)  # [新增]

    task = "classification" if "direction" in target_name or "regime" in target_name else "regression"  # [新增]

    tscv = TimeSeriesSplit(n_splits=5)  # [新增]
    train_idx, val_idx = list(tscv.split(X))[-1]  # [新增]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]  # [新增]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]  # [新增]

    def objective(trial: optuna.Trial) -> float:  # [新增]
        params = {  # [新增]
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),  # [新增]
            "max_depth": trial.suggest_int("max_depth", 3, 10),  # [新增]
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),  # [新增]
        }  # [新增]
        model = create_model(period=period, task=task, params=params)  # [新增]
        model.train(X_train, y_train, X_val=X_val, y_val=y_val)  # [新增]
        metrics = model.evaluate(X_val, y_val)  # [新增]
        score = float(metrics["accuracy"]) if task == "classification" else -float(metrics["rmse"])  # [新增]
        return float(score)  # [新增]

    study = optuna.create_study(direction="maximize")  # [新增]
    study.optimize(objective, n_trials=int(n_trials))  # [新增]

    return study.best_params, study.best_value  # [新增]


# --- 改进点 3.3: 贝叶斯优化 + Pipeline ---

def run_bayesopt_pipeline_lightgbm(X, y, n_iter: int = 30):  # [新增]
    from bayes_opt import BayesianOptimization  # [新增]

    tscv = TimeSeriesSplit(n_splits=5)  # [新增]

    def bayesian_model_cv(n_estimators, learning_rate, subsample, max_depth, min_child_samples, num_leaves, max_bin):  # [新增]
        model = Pipeline([  # [新增]
            ("Minmax", MinMaxScaler()),  # [新增]
            ("lightgbm", lgb.LGBMRegressor(  # [新增]
                boosting_type="gbdt", objective="regression", metric="rmse",  # [新增]
                n_estimators=int(n_estimators),  # [新增]
                learning_rate=float(learning_rate),  # [新增]
                subsample=float(subsample),  # [新增]
                max_depth=int(max_depth),  # [新增]
                min_child_samples=int(min_child_samples),  # [新增]
                num_leaves=int(num_leaves),  # [新增]
                max_bin=int(max_bin),  # [新增]
            )),  # [新增]
        ])  # [新增]

        val = cross_val_score(  # [新增]
            model, X, y, scoring="neg_mean_squared_error", cv=tscv, n_jobs=1  # [新增]
        ).mean()  # [新增]
        return float(val)  # [新增]

    pbounds = {  # [新增]
        "n_estimators": (200, 1200),  # [新增]
        "learning_rate": (0.005, 0.2),  # [新增]
        "subsample": (0.6, 1.0),  # [新增]
        "max_depth": (3, 10),  # [新增]
        "min_child_samples": (5, 50),  # [新增]
        "num_leaves": (16, 256),  # [新增]
        "max_bin": (64, 512),  # [新增]
    }  # [新增]

    bo = BayesianOptimization(f=bayesian_model_cv, pbounds=pbounds, random_state=42)  # [新增]
    bo.maximize(n_iter=int(n_iter))  # [新增]
    return bo.max  # [新增]


# --- 改进点 5.1: L2 Normalize Pipeline ---

def build_l2_pipeline_lgbm(**lgb_params):  # [新增]
    return Pipeline([  # [新增]
        ("l2", Normalizer(norm="l2")),  # [新增]
        ("lgbm", lgb.LGBMClassifier(**lgb_params)),  # [新增]
    ])  # [新增]
