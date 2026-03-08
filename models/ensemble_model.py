"""
商品期货机器学习量化模型 - 模型集成模块
Ensemble model combining LightGBM and XGBoost for improved stability.

集成策略:
- 加权投票 (分类任务): 基于验证集表现动态分配权重
- 加权平均 (回归任务): 基于验证集R²分配权重
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score


class EnsembleModel:
    """
    LightGBM + XGBoost 加权集成模型。

    通过组合两种不同的梯度提升模型提升预测稳定性，
    权重基于验证集表现自动分配。
    """

    def __init__(self, task="classification", lgbm_params=None, xgb_params=None):
        """
        Parameters
        ----------
        task : str
            任务类型 ("classification" 或 "regression")
        lgbm_params : dict, optional
            LightGBM超参数
        xgb_params : dict, optional
            XGBoost超参数
        """
        from models.ml_models import LightGBMModel, XGBoostModel

        self.task = task
        self.lgbm = LightGBMModel(task=task, params=lgbm_params)
        self.xgb = XGBoostModel(task=task, params=xgb_params)
        self.weights = [0.5, 0.5]  # 默认等权
        self.feature_names = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练集成模型并基于验证集调整权重。

        Parameters
        ----------
        X_train : pd.DataFrame
            训练特征
        y_train : pd.Series
            训练目标
        X_val : pd.DataFrame, optional
            验证特征（用于权重分配）
        y_val : pd.Series, optional
            验证目标
        """
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        self.lgbm.train(X_train, y_train, X_val, y_val)
        self.xgb.train(X_train, y_train, X_val, y_val)

        # 基于验证集表现调整权重
        if X_val is not None and y_val is not None:
            self._calibrate_weights(X_val, y_val)

        return self

    def _calibrate_weights(self, X_val, y_val):
        """基于验证集表现计算模型权重"""
        lgbm_pred = self.lgbm.predict(X_val)
        xgb_pred = self.xgb.predict(X_val)

        if self.task == "classification":
            lgbm_score = accuracy_score(y_val, lgbm_pred)
            xgb_score = accuracy_score(y_val, xgb_pred)
        else:
            lgbm_score = max(r2_score(y_val, lgbm_pred), 0.01)
            xgb_score = max(r2_score(y_val, xgb_pred), 0.01)

        total = lgbm_score + xgb_score
        if total > 0:
            self.weights = [lgbm_score / total, xgb_score / total]

    def predict(self, X):
        """加权预测"""
        if self.task == "classification":
            lgbm_proba = self.lgbm.predict_proba(X)
            xgb_proba = self.xgb.predict_proba(X)
            ensemble_proba = self.weights[0] * lgbm_proba + self.weights[1] * xgb_proba
            return np.argmax(ensemble_proba, axis=1)
        else:
            lgbm_pred = self.lgbm.predict(X)
            xgb_pred = self.xgb.predict(X)
            return self.weights[0] * lgbm_pred + self.weights[1] * xgb_pred

    def predict_proba(self, X):
        """加权预测概率"""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification tasks")
        lgbm_proba = self.lgbm.predict_proba(X)
        xgb_proba = self.xgb.predict_proba(X)
        return self.weights[0] * lgbm_proba + self.weights[1] * xgb_proba

    def get_feature_importance(self):
        """获取加权平均特征重要性"""
        lgbm_imp = self.lgbm.get_feature_importance()
        xgb_imp = self.xgb.get_feature_importance()
        if lgbm_imp is not None and xgb_imp is not None:
            # 归一化后加权平均
            lgbm_norm = lgbm_imp / lgbm_imp.sum() if lgbm_imp.sum() > 0 else lgbm_imp
            xgb_norm = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp
            combined = self.weights[0] * lgbm_norm + self.weights[1] * xgb_norm
            return combined.sort_values(ascending=False)
        return lgbm_imp if lgbm_imp is not None else xgb_imp

    def evaluate(self, X, y):
        """评估集成模型"""
        from models.ml_models import get_classification_metrics, get_regression_metrics

        y_pred = self.predict(X)
        if self.task == "classification":
            return get_classification_metrics(y, y_pred)
        return get_regression_metrics(y, y_pred)

    def get_model_weights(self):
        """获取各模型权重"""
        return {"lightgbm": self.weights[0], "xgboost": self.weights[1]}
