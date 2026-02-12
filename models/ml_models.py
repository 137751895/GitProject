"""
商品期货机器学习量化模型 - 机器学习模型模块
ML models for commodity futures quantitative trading.

各周期推荐的机器学习框架:
- 1分钟K线: LightGBM
  - 数据量最大，LightGBM训练速度快、内存效率高
  - 基于直方图的算法适合大规模数据
  - 支持类别特征，无需额外编码

- 5分钟K线: XGBoost
  - 数据量适中，XGBoost精度高且稳定
  - 正则化能力强，不易过拟合
  - 特征重要性评估完善

- 15分钟K线: LSTM (深度学习)
  - 数据量较少但序列特性更明显
  - LSTM能捕捉长期时序依赖关系
  - 适合学习复杂的非线性模式
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_classification_metrics(y_true, y_pred):
    """计算分类指标"""
    n_classes = len(np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
    average = "binary" if n_classes <= 2 else "weighted"
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average=average),
        "recall": recall_score(y_true, y_pred, zero_division=0, average=average),
        "f1_score": f1_score(y_true, y_pred, zero_division=0, average=average),
    }


def get_regression_metrics(y_true, y_pred):
    """计算回归指标"""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


class LightGBMModel:
    """
    LightGBM模型 - 推荐用于1分钟K线数据。

    优势: 训练速度快、内存占用低、支持大规模数据。
    """

    def __init__(self, task="classification", params=None):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        import lightgbm as lgb

        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率（仅分类任务）"""
        if self.task == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only available for classification tasks")

    def get_feature_importance(self):
        """获取特征重要性"""
        importance = self.model.feature_importances_
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return importance

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        if self.task == "classification":
            return get_classification_metrics(y, y_pred)
        return get_regression_metrics(y, y_pred)


class XGBoostModel:
    """
    XGBoost模型 - 推荐用于5分钟K线数据。

    优势: 精度高、正则化强、不易过拟合。
    """

    def __init__(self, task="classification", params=None):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        default_params = {
            "n_estimators": 300,
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
        if params:
            default_params.update(params)
        self.params = default_params

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        import xgboost as xgb

        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        if self.task == "classification":
            n_classes = len(np.unique(y_train))
            if n_classes > 2:
                self.model = xgb.XGBClassifier(
                    eval_metric="mlogloss", **self.params
                )
            else:
                self.model = xgb.XGBClassifier(
                    eval_metric="logloss", use_label_encoder=False, **self.params
                )
        else:
            self.model = xgb.XGBRegressor(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率（仅分类任务）"""
        if self.task == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba only available for classification tasks")

    def get_feature_importance(self):
        """获取特征重要性"""
        importance = self.model.feature_importances_
        if self.feature_names:
            return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        return importance

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        if self.task == "classification":
            return get_classification_metrics(y, y_pred)
        return get_regression_metrics(y, y_pred)


class LSTMModel:
    """
    LSTM模型 - 推荐用于15分钟K线数据。

    优势: 擅长捕捉时序依赖关系和复杂非线性模式。
    """

    def __init__(self, task="classification", params=None):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        default_params = {
            "sequence_length": 20,
            "hidden_units": 64,
            "dropout_rate": 0.2,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "patience": 10,  # [新增]
        }
        if params:
            default_params.update(params)
        self.params = default_params

    def _build_model(self, input_shape):
        """构建LSTM模型"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(
                self.params["hidden_units"],
                input_shape=input_shape,
                return_sequences=True,
            ),
            Dropout(self.params["dropout_rate"]),
            LSTM(self.params["hidden_units"] // 2, return_sequences=False),
            Dropout(self.params["dropout_rate"]),
            Dense(32, activation="relu"),
        ])

        if self.task == "classification":
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                optimizer=Adam(learning_rate=self.params["learning_rate"]),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        else:
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=self.params["learning_rate"]),
                loss="mse",
                metrics=["mae"],
            )
        return model

    def _create_sequences(self, X, y=None):
        """创建时序序列数据"""
        seq_len = self.params["sequence_length"]
        X_seq = []
        y_seq = []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len: i])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 创建序列
        X_seq, y_seq = self._create_sequences(X_train_scaled, y_train.values if hasattr(y_train, "values") else y_train)

        # 构建模型
        input_shape = (self.params["sequence_length"], X_seq.shape[2])
        self.model = self._build_model(input_shape)

        fit_params = {
            "epochs": self.params["epochs"],
            "batch_size": self.params["batch_size"],
            "verbose": 0,
        }

        callbacks = []  # [新增]

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values if hasattr(y_val, "values") else y_val)
            if len(X_val_seq) > 0:
                fit_params["validation_data"] = (X_val_seq, y_val_seq)

                from tensorflow.keras.callbacks import EarlyStopping  # [新增]
                callbacks.append(  # [新增]
                    EarlyStopping(  # [新增]
                        monitor="val_loss",  # [新增]
                        patience=int(self.params.get("patience", 10)),  # [新增]
                        restore_best_weights=True,  # [新增]
                    )  # [新增]
                )  # [新增]
                fit_params["callbacks"] = callbacks  # [新增]

        self.model.fit(X_seq, y_seq, **fit_params)
        return self

    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        raw_pred = self.model.predict(X_seq, verbose=0)
        if self.task == "classification":
            return (raw_pred > 0.5).astype(int).flatten()
        return raw_pred.flatten()

    def predict_proba(self, X):
        """预测概率（仅分类任务）"""
        if self.task == "classification":
            X_scaled = self.scaler.transform(X)
            X_seq = self._create_sequences(X_scaled)
            proba = self.model.predict(X_seq, verbose=0).flatten()
            return np.column_stack([1 - proba, proba])
        raise ValueError("predict_proba only available for classification tasks")

    def evaluate(self, X, y):
        """评估模型"""
        X_scaled = self.scaler.transform(X)
        X_seq, y_seq = self._create_sequences(X_scaled, y.values if hasattr(y, "values") else y)
        y_pred_raw = self.model.predict(X_seq, verbose=0)
        if self.task == "classification":
            y_pred = (y_pred_raw > 0.5).astype(int).flatten()
            return get_classification_metrics(y_seq, y_pred)
        return get_regression_metrics(y_seq, y_pred_raw.flatten())

    def get_feature_importance(self):
        """LSTM无直接特征重要性，返回None"""
        return None


def create_model(period, task="classification", params=None):
    """
    根据K线周期创建推荐的模型。

    Parameters
    ----------
    period : str
        K线周期 ("1min", "5min", "15min")
    task : str
        任务类型 ("classification" 或 "regression")
    params : dict, optional
        模型参数

    Returns
    -------
    model
        对应周期推荐的ML模型实例
    """
    model_map = {
        "1min": LightGBMModel,
        "5min": XGBoostModel,
        "15min": LSTMModel,
    }
    model_class = model_map.get(period)
    if model_class is None:
        raise ValueError(f"不支持的周期: {period}，支持: {list(model_map.keys())}")
    return model_class(task=task, params=params)
