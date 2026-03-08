"""
商品期货机器学习量化模型 - XGBoost特征预筛选模块
XGBoost-based feature pre-screening for LSTM model (15min period).

由于LSTM模型无法直接评估特征重要性，使用XGBoost作为特征筛选器，
选择对15分钟周期预测最有效的特征子集，再将这些特征输入LSTM进行训练。

流程:
    原始数据 → 特征工程(62-89个特征) → XGBoost筛选(Top N) → LSTM训练

使用方法:
    selector = XGBFeatureSelector(n_features=25)
    selected, importance_df = selector.fit_select(X, y)
    X_selected = X[selected]
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class XGBFeatureSelector:
    """基于XGBoost的特征筛选器，主要用于15分钟LSTM模型的特征预筛选。

    Parameters
    ----------
    n_features : int
        最终选择的特征数量 (default: 25)
    threshold : str or float
        重要性阈值 ('median', 'mean', 或具体数值)
    cv_splits : int
        时间序列交叉验证折数 (default: 5)
    xgb_params : dict, optional
        XGBoost模型参数覆盖
    """

    def __init__(self, n_features=25, threshold="median", cv_splits=5,
                 xgb_params=None):
        self.n_features = n_features
        self.threshold = threshold
        self.cv_splits = cv_splits
        self.selected_features = []
        self.feature_importance_df = None

        self.xgb_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if xgb_params:
            self.xgb_params.update(xgb_params)

    def fit_select(self, X, y):
        """训练XGBoost并筛选特征。

        使用时间序列交叉验证，在每一折中训练XGBoost并记录特征重要性，
        最终取各折的平均重要性来排序筛选。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量

        Returns
        -------
        tuple
            (selected_features, importance_df)
            - selected_features: 筛选后的特征名列表
            - importance_df: 特征重要性数据框(含importance, std)
        """
        import xgboost as xgb

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        importance_scores = {}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            n_classes = len(np.unique(y_train))
            if n_classes > 2:
                model = xgb.XGBClassifier(
                    eval_metric="mlogloss", **self.xgb_params
                )
            else:
                model = xgb.XGBClassifier(
                    eval_metric="logloss", **self.xgb_params
                )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            fold_importance = dict(zip(
                X.columns,
                model.feature_importances_,
            ))

            for feature, importance in fold_importance.items():
                if feature not in importance_scores:
                    importance_scores[feature] = []
                importance_scores[feature].append(importance)

        # 计算平均重要性和标准差
        avg_importance = {
            feature: np.mean(scores)
            for feature, scores in importance_scores.items()
        }

        self.feature_importance_df = pd.DataFrame({
            "feature": list(avg_importance.keys()),
            "importance": list(avg_importance.values()),
            "std": [np.std(importance_scores[f]) for f in avg_importance],
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # 选择Top N特征
        n_select = min(self.n_features, len(self.feature_importance_df))
        self.selected_features = (
            self.feature_importance_df
            .nlargest(n_select, "importance")["feature"]
            .tolist()
        )

        return self.selected_features, self.feature_importance_df

    def get_feature_groups_importance(self, feature_groups):
        """按特征组评估重要性。

        Parameters
        ----------
        feature_groups : dict
            特征分组，格式: {"group_name": [feature_list]}

        Returns
        -------
        pd.DataFrame
            各特征组的重要性汇总（score, n_features, n_selected）
        """
        if self.feature_importance_df is None:
            raise RuntimeError("请先调用 fit_select() 计算特征重要性")

        group_importance = {}
        all_features = set(self.feature_importance_df["feature"].values)

        for group_name, features in feature_groups.items():
            group_features = [f for f in features if f in all_features]
            if not group_features:
                continue
            group_score = self.feature_importance_df[
                self.feature_importance_df["feature"].isin(group_features)
            ]["importance"].mean()
            group_importance[group_name] = {
                "score": group_score,
                "n_features": len(group_features),
                "n_selected": sum(
                    1 for f in group_features if f in self.selected_features
                ),
            }

        result = (
            pd.DataFrame(group_importance)
            .T.sort_values("score", ascending=False)
        )
        return result

    def dynamic_feature_count(self, X, y, min_features=10, max_features=40,
                              step=5):
        """动态确定最佳特征数量。

        在不同特征数量下使用TimeSeriesSplit评估，选择达到最佳准确率
        95%水平时所需的最少特征数量（即在精度几乎不损失的前提下
        使用尽可能少的特征）。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        min_features : int
            最少特征数
        max_features : int
            最多特征数
        step : int
            步长

        Returns
        -------
        dict
            包含 optimal_n_features, search_results
        """
        import xgboost as xgb

        if self.feature_importance_df is None:
            self.fit_select(X, y)

        ranked_features = self.feature_importance_df["feature"].tolist()
        max_features = min(max_features, len(ranked_features))

        search_results = []
        for n in range(min_features, max_features + 1, step):
            top_n_features = ranked_features[:n]
            X_subset = X[top_n_features]

            model = xgb.XGBClassifier(
                n_estimators=50, random_state=42, n_jobs=-1, verbosity=0,
                eval_metric="logloss",
            )
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, test_idx in tscv.split(X_subset):
                model.fit(X_subset.iloc[train_idx], y.iloc[train_idx])
                score = model.score(X_subset.iloc[test_idx], y.iloc[test_idx])
                scores.append(score)

            search_results.append({
                "n_features": n,
                "accuracy": np.mean(scores),
                "std": np.std(scores),
            })

        results_df = pd.DataFrame(search_results)

        # 选择达到最佳准确率95%水平时所需的最少特征数量
        max_acc = results_df["accuracy"].max()
        threshold_acc = max_acc * 0.95
        optimal_rows = results_df[results_df["accuracy"] >= threshold_acc]
        if len(optimal_rows) > 0:
            optimal_n = int(optimal_rows.iloc[0]["n_features"])
        else:
            optimal_n = int(results_df.loc[
                results_df["accuracy"].idxmax(), "n_features"
            ])

        return {
            "optimal_n_features": optimal_n,
            "search_results": results_df,
            "threshold_accuracy": threshold_acc,
        }
