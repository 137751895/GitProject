"""
商品期货机器学习量化模型 - 回测与特征选择模块
Backtesting and feature selection for commodity futures ML model.

通过回测挑选成功率高的特征的方法:
1. 时间序列交叉验证 (TimeSeriesSplit) - 避免前视偏差
2. 特征重要性排序 - 基于模型的特征重要性
3. 递归特征消除 (RFE) - 逐步移除不重要特征
4. 前向特征选择 - 逐步添加最优特征
5. 基于成功率的特征筛选 - 以预测准确率为标准
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class FeatureSelector:
    """
    特征选择器 - 通过回测挑选成功率高的特征。

    使用时间序列交叉验证避免前视偏差，
    综合多种方法评估特征重要性和预测成功率。
    """

    def __init__(self, n_splits=5, min_success_rate=0.55):
        """
        Parameters
        ----------
        n_splits : int
            时间序列交叉验证折数
        min_success_rate : float
            最低成功率阈值（用于筛选特征子集）
        """
        self.n_splits = n_splits
        self.min_success_rate = min_success_rate
        self.results = {}

    def time_series_cv_evaluate(self, model_class, X, y, task="classification", model_params=None):
        """
        时间序列交叉验证评估。

        使用 TimeSeriesSplit 确保训练数据始终在测试数据之前，
        避免前视偏差（look-ahead bias）。

        Parameters
        ----------
        model_class : class
            模型类
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        task : str
            任务类型
        model_params : dict, optional
            模型参数

        Returns
        -------
        dict
            交叉验证结果，包含每折的指标和平均成功率
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = model_class(task=task, params=model_params)
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            metrics["fold"] = fold
            fold_metrics.append(metrics)

        results = pd.DataFrame(fold_metrics)
        summary = {
            "fold_metrics": results,
            "mean_metrics": results.drop(columns=["fold"]).mean().to_dict(),
        }

        if task == "classification":
            summary["mean_success_rate"] = results["accuracy"].mean()
            summary["std_success_rate"] = results["accuracy"].std()
        else:
            summary["mean_r2"] = results["r2"].mean()

        return summary

    def feature_importance_ranking(self, model_class, X, y, task="classification", model_params=None):
        """
        基于模型的特征重要性排序。

        使用全部训练数据训练模型，提取特征重要性排名。

        Parameters
        ----------
        model_class : class
            模型类（需支持 get_feature_importance 方法）
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量

        Returns
        -------
        pd.Series
            特征重要性排序（降序）
        """
        model = model_class(task=task, params=model_params)
        model.train(X, y)
        importance = model.get_feature_importance()
        if importance is not None:
            self.results["feature_importance"] = importance
            return importance
        return None

    def mutual_information_ranking(self, X, y, task="classification"):
        """
        基于互信息的特征排序。

        互信息衡量特征与目标变量之间的统计依赖性，
        不假设线性关系，适合发现非线性特征。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        task : str
            任务类型

        Returns
        -------
        pd.Series
            互信息排序（降序）
        """
        if task == "classification":
            mi = mutual_info_classif(X, y, random_state=42)
        else:
            mi = mutual_info_regression(X, y, random_state=42)

        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        self.results["mutual_information"] = mi_series
        return mi_series

    def forward_feature_selection(self, model_class, X, y, max_features=20, task="classification", model_params=None):
        """
        前向特征选择 - 逐步添加最优特征。

        每一步添加能使成功率提升最大的特征，
        直到达到最大特征数或成功率不再提升。

        Parameters
        ----------
        model_class : class
            模型类
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        max_features : int
            最大特征数量
        task : str
            任务类型
        model_params : dict, optional
            模型参数

        Returns
        -------
        dict
            包含选中特征列表和各步的成功率
        """
        selected_features = []
        remaining_features = list(X.columns)
        selection_history = []
        best_score = 0

        for step in range(min(max_features, len(remaining_features))):
            best_feature = None
            best_step_score = 0

            for feature in remaining_features:
                trial_features = selected_features + [feature]
                X_subset = X[trial_features]

                cv_result = self.time_series_cv_evaluate(
                    model_class, X_subset, y, task=task, model_params=model_params
                )

                if task == "classification":
                    score = cv_result["mean_success_rate"]
                else:
                    score = cv_result["mean_r2"]

                if score > best_step_score:
                    best_step_score = score
                    best_feature = feature

            if best_feature is None or best_step_score <= best_score:
                break

            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_score = best_step_score

            selection_history.append({
                "step": step + 1,
                "feature": best_feature,
                "score": best_step_score,
                "n_features": len(selected_features),
            })

        result = {
            "selected_features": selected_features,
            "selection_history": pd.DataFrame(selection_history),
            "final_score": best_score,
        }
        self.results["forward_selection"] = result
        return result

    def evaluate_feature_subsets(self, model_class, X, y, feature_groups, task="classification", model_params=None):
        """
        评估不同特征子集的成功率。

        Parameters
        ----------
        model_class : class
            模型类
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        feature_groups : dict
            特征分组，格式: {"group_name": [feature_list]}
        task : str
            任务类型
        model_params : dict, optional
            模型参数

        Returns
        -------
        pd.DataFrame
            各特征组的评估结果
        """
        group_results = []

        for group_name, features in feature_groups.items():
            valid_features = [f for f in features if f in X.columns]
            if not valid_features:
                continue

            X_subset = X[valid_features]
            cv_result = self.time_series_cv_evaluate(
                model_class, X_subset, y, task=task, model_params=model_params
            )

            result = {
                "group": group_name,
                "n_features": len(valid_features),
                "features": valid_features,
            }
            result.update(cv_result["mean_metrics"])

            if task == "classification":
                result["success_rate"] = cv_result["mean_success_rate"]

            group_results.append(result)

        results_df = pd.DataFrame(group_results)
        if task == "classification" and not results_df.empty:
            results_df = results_df.sort_values("success_rate", ascending=False)

        self.results["feature_subsets"] = results_df
        return results_df

    def stability_selection(self, X, y, n_bootstrap=20, sample_fraction=0.7,
                            threshold=0.6, lasso_cv=3, lasso_max_iter=2000):
        """
        稳定性特征选择 (Stability Selection)。

        多次随机采样训练LassoCV，统计每个特征被选中的频率，
        频率高于阈值的特征被认为是稳定且重要的特征。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        n_bootstrap : int
            采样次数
        sample_fraction : float
            每次采样比例
        threshold : float
            特征选择频率阈值

        Returns
        -------
        pd.DataFrame
            特征稳定性得分
        """
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=X.index
        )

        selection_counts = pd.Series(0, index=X.columns, dtype=float)
        n_samples = len(X_scaled)
        sample_size = int(n_samples * sample_fraction)

        for i in range(n_bootstrap):
            # 随机采样（保持时间顺序的采样）
            rng = np.random.RandomState(i)
            indices = rng.choice(n_samples, size=sample_size, replace=False)
            indices.sort()

            X_sample = X_scaled.iloc[indices]
            y_sample = y.iloc[indices]

            lasso = LassoCV(cv=lasso_cv, random_state=i, max_iter=lasso_max_iter)
            lasso.fit(X_sample, y_sample)

            selected = np.abs(lasso.coef_) > 0
            selection_counts[selected] += 1

        stability_scores = selection_counts / n_bootstrap
        result = pd.DataFrame({
            "feature": X.columns,
            "stability_score": stability_scores.values,
            "selected": stability_scores.values >= threshold,
        }).sort_values("stability_score", ascending=False)

        self.results["stability_selection"] = result
        return result

    def recursive_feature_elimination(self, model_class, X, y,
                                       min_features=10, step=5,
                                       task="classification",
                                       model_params=None):
        """
        递归特征消除 (RFE) 与交叉验证结合。

        逐步移除最不重要的特征，在每一步使用时间序列CV评估，
        找到最佳特征数量。

        Parameters
        ----------
        model_class : class
            模型类
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        min_features : int
            最少保留的特征数
        step : int
            每步移除的特征数
        task : str
            任务类型
        model_params : dict, optional
            模型参数

        Returns
        -------
        dict
            RFE结果，包含最佳特征集和各步评估
        """
        current_features = list(X.columns)
        rfe_history = []
        best_score = -np.inf
        best_features = current_features.copy()

        while len(current_features) >= min_features:
            X_subset = X[current_features]
            cv_result = self.time_series_cv_evaluate(
                model_class, X_subset, y, task=task, model_params=model_params
            )

            if task == "classification":
                score = cv_result["mean_success_rate"]
            else:
                score = cv_result.get("mean_r2", 0)

            rfe_history.append({
                "n_features": len(current_features),
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_features = current_features.copy()

            # 获取特征重要性并移除最不重要的
            model = model_class(task=task, params=model_params)
            model.train(X_subset, y)
            importance = model.get_feature_importance()
            if importance is None:
                break

            # 移除importance最低的特征，确保不低于min_features
            n_to_remove = min(step, len(current_features) - min_features)
            if n_to_remove <= 0:
                break
            features_to_remove = list(importance.tail(n_to_remove).index)
            current_features = [f for f in current_features
                                if f not in features_to_remove]

        result = {
            "best_features": best_features,
            "best_score": best_score,
            "n_best_features": len(best_features),
            "rfe_history": pd.DataFrame(rfe_history),
        }
        self.results["rfe"] = result
        return result

    def select_best_features(self, model_class, X, y, task="classification", model_params=None, top_n=20):
        """
        综合多种方法挑选最佳特征。

        流程:
        1. 计算特征重要性排名
        2. 计算互信息排名
        3. 综合排名取交集
        4. 使用时间序列CV验证最终特征集的成功率

        Parameters
        ----------
        model_class : class
            模型类
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        task : str
            任务类型
        model_params : dict, optional
            模型参数
        top_n : int
            选取的特征数量

        Returns
        -------
        dict
            最佳特征列表和评估结果
        """
        # 1. 模型特征重要性
        importance = self.feature_importance_ranking(
            model_class, X, y, task=task, model_params=model_params
        )
        if importance is not None:
            top_by_importance = set(importance.head(top_n).index)
        else:
            top_by_importance = set(X.columns[:top_n])

        # 2. 互信息排名
        mi = self.mutual_information_ranking(X, y, task=task)
        top_by_mi = set(mi.head(top_n).index)

        # 3. 综合排名（取并集再排序）
        combined_features = list(top_by_importance | top_by_mi)
        if len(combined_features) > top_n:
            # 基于两种排名的综合分数选取
            if importance is not None:
                imp_rank = importance.rank(ascending=False)
            else:
                imp_rank = pd.Series(range(len(X.columns)), index=X.columns)
            mi_rank = mi.rank(ascending=False)
            combined_rank = (imp_rank + mi_rank) / 2
            combined_features = list(combined_rank.sort_values().head(top_n).index)

        # 4. 最终验证
        X_final = X[combined_features]
        final_cv = self.time_series_cv_evaluate(
            model_class, X_final, y, task=task, model_params=model_params
        )

        result = {
            "selected_features": combined_features,
            "n_features": len(combined_features),
            "cv_results": final_cv,
        }

        if task == "classification":
            result["final_success_rate"] = final_cv["mean_success_rate"]

        self.results["best_features"] = result
        return result


def get_recommended_feature_groups():
    """
    获取推荐的特征分组。

    Returns
    -------
    dict
        推荐的特征分组字典
    """
    return {
        "价格趋势": [
            "ma_5", "ma_10", "ma_20", "ma_60",
            "ema_5", "ema_10", "ema_20", "ema_60",
            "price_position", "dist_to_high", "dist_to_low",
        ],
        "布林带": [
            "boll_upper", "boll_mid", "boll_lower",
            "boll_width", "boll_pct_b",
        ],
        "动量指标": [
            "rsi_14", "rsi_6",
            "macd_dif", "macd_dea", "macd_hist",
            "kdj_k", "kdj_d", "kdj_j",
            "cci", "williams_r", "roc_12", "roc_6",
        ],
        "成交量": [
            "vol_ma_5", "vol_ma_10", "vol_ma_20",
            "vol_ratio_5", "vol_ratio_10", "vol_ratio_20",
            "vol_change", "obv", "vwap",
        ],
        "波动率": [
            "tr", "atr",
            "volatility_5", "volatility_10", "volatility_20",
            "return_ma_5", "return_ma_10", "return_ma_20",
            "log_return",
        ],
        "持仓量(仓差)": [
            "oi_change", "oi_change_pct",
            "oi_ma_5", "oi_ma_10", "oi_ma_20",
            "oi_change_ma_5", "oi_change_ma_10", "oi_change_ma_20",
            "vol_oi_ratio",
        ],
        "K线形态": [
            "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
            "candle_direction", "amplitude", "gap", "gap_ratio",
        ],
        "市场状态": [
            "regime_volatility", "trend_strength", "volatility_rank",
            "market_regime", "trend_direction", "volatility_change",
            "trend_acceleration",
        ],
    }
