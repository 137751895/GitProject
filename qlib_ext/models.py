"""
HFML-Qlib 模型封装模块
======================
HFMLLSTMQlibModel  : 将 HFML LSTMModel 包装为 Qlib Model 协议（方案 B 过渡实现）。
HFMLLGBMQlibModel  : 将 HFML LightGBMModel 包装为 Qlib Model 协议。
HFMLXGBQlibModel   : 将 HFML XGBoostModel 包装为 Qlib Model 协议。

所有模型均继承 qlib.model.base.Model，实现 fit / predict 接口。
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基础包装器 Mixin
# ---------------------------------------------------------------------------

class _HFMLModelMixin:
    """公共辅助方法：从 dataset 提取 X/y，生成带 MultiIndex 的预测序列。"""

    def _prepare_Xy(self, dataset, segments, col_set=("feature", "label"), data_key=None):
        """从 DatasetH/DataHandlerLP 提取特征与标签。

        Returns
        -------
        tuple of (X_list, y_list)
            每个元素对应一个 segment 的 X/y DataFrame/Series。
        """
        try:
            from qlib.data.dataset.handler import DataHandlerLP
            dk = data_key or DataHandlerLP.DK_L
        except ImportError:
            dk = "learn"

        results_X, results_y = [], []
        for seg in segments:
            try:
                seg_data = dataset.prepare(
                    seg, col_set=list(col_set), data_key=dk
                )
                if isinstance(seg_data, tuple):
                    X, y = seg_data
                else:
                    X = seg_data.get("feature", seg_data)
                    y = seg_data.get("label", None)
                results_X.append(X)
                results_y.append(y)
            except Exception as exc:
                logger.warning(f"prepare segment '{seg}' 失败: {exc}")
                results_X.append(pd.DataFrame())
                results_y.append(None)
        return results_X, results_y

    def _make_pred_series(self, scores: np.ndarray, index: pd.Index,
                          name: str = "score") -> pd.Series:
        """将预测分数封装为带 MultiIndex 的 pd.Series。"""
        if not isinstance(index, pd.MultiIndex):
            return pd.Series(scores, index=index, name=name)
        return pd.Series(scores, index=index, name=name)


# ---------------------------------------------------------------------------
# HFMLLSTMQlibModel
# ---------------------------------------------------------------------------

try:
    from qlib.model.base import Model

    class HFMLLSTMQlibModel(Model, _HFMLModelMixin):
        """将 HFML LSTMModel（TensorFlow）包装为 Qlib Model（方案 B 过渡版）。

        训练时从 dataset 的 train/valid 段提取特征与标签，调用
        LSTMModel.train。预测时调用 LSTMModel.predict_proba，并对齐索引。

        Parameters
        ----------
        params : dict, optional
            传递给 LSTMModel 的参数（sequence_length, hidden_units 等）。
        task : str
            "classification" 或 "regression"。
        label_col : str
            标签列名，默认 "trading_signal"。
        """

        def __init__(self, params=None, task="classification", label_col="trading_signal"):
            from models.ml_models import LSTMModel
            self.params = params or {}
            self.task = task
            self.label_col = label_col
            self.inner = LSTMModel(task=task, params=self.params)

        def fit(self, dataset, reweighter=None):
            """训练 LSTM 模型。"""
            from qlib.data.dataset.handler import DataHandlerLP

            [X_train, X_valid], [y_train_df, y_valid_df] = self._prepare_Xy(
                dataset,
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            y_train = self._extract_label(y_train_df)
            y_valid = self._extract_label(y_valid_df)

            if X_train.empty or y_train is None:
                raise ValueError("训练数据为空，请检查 DataHandler 配置")

            logger.info(f"LSTM 训练数据: X_train={X_train.shape}, y_train={y_train.shape}")
            self.inner.train(
                X_train, y_train,
                X_val=X_valid if not X_valid.empty else None,
                y_val=y_valid,
            )

        def predict(self, dataset, segment="test"):
            """预测并返回带正确 MultiIndex 的 pd.Series。"""
            from qlib.data.dataset.handler import DataHandlerLP

            [X_test], _ = self._prepare_Xy(
                dataset,
                [segment],
                col_set=["feature"],
                data_key=DataHandlerLP.DK_I,
            )
            if X_test.empty:
                return pd.Series(dtype=float, name="score")

            seq_len = self.inner.params.get("sequence_length", 20)
            try:
                proba = self.inner.predict_proba(X_test)
                scores = proba[:, 1]
            except Exception as exc:
                logger.warning(f"predict_proba 失败，改用 predict: {exc}")
                scores = self.inner.predict(X_test).astype(float)

            # LSTM 内部裁掉了前 seq_len 行，对齐索引
            valid_index = X_test.index[seq_len:]
            n = min(len(scores), len(valid_index))
            if len(scores) != len(valid_index):
                logger.debug(
                    f"HFMLLSTMQlibModel.predict: 预测长度 {len(scores)} 与"
                    f" 有效索引长度 {len(valid_index)} 不一致，取前 {n} 个对齐"
                )
            return self._make_pred_series(scores[:n], valid_index[:n])

        def _extract_label(self, label_df):
            """从标签 DataFrame 提取目标列。"""
            if label_df is None:
                return None
            if isinstance(label_df, pd.Series):
                return label_df
            if isinstance(label_df, pd.DataFrame):
                if self.label_col in label_df.columns:
                    return label_df[self.label_col]
                # MultiIndex 列
                if isinstance(label_df.columns, pd.MultiIndex):
                    for top, sub in label_df.columns:
                        if sub == self.label_col:
                            return label_df[(top, sub)]
                return label_df.iloc[:, 0]
            return label_df

    class HFMLLGBMQlibModel(Model, _HFMLModelMixin):
        """将 HFML LightGBMModel 包装为 Qlib Model（1 分钟周期推荐）。

        Parameters
        ----------
        params : dict, optional
            传递给 LightGBMModel 的参数。
        task : str
            "classification" 或 "regression"。
        label_col : str
            标签列名，默认 "trading_signal"。
        """

        def __init__(self, params=None, task="classification", label_col="trading_signal"):
            from models.ml_models import LightGBMModel
            self.params = params or {}
            self.task = task
            self.label_col = label_col
            self.inner = LightGBMModel(task=task, params=self.params)

        def fit(self, dataset, reweighter=None):
            from qlib.data.dataset.handler import DataHandlerLP

            [X_train, X_valid], [y_train_df, y_valid_df] = self._prepare_Xy(
                dataset,
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            y_train = self._extract_label(y_train_df)
            y_valid = self._extract_label(y_valid_df)
            self.inner.train(
                X_train, y_train,
                X_val=X_valid if not X_valid.empty else None,
                y_val=y_valid,
            )

        def predict(self, dataset, segment="test"):
            from qlib.data.dataset.handler import DataHandlerLP

            [X_test], _ = self._prepare_Xy(
                dataset, [segment], col_set=["feature"], data_key=DataHandlerLP.DK_I
            )
            if X_test.empty:
                return pd.Series(dtype=float, name="score")
            try:
                scores = self.inner.predict_proba(X_test)[:, 1]
            except Exception:
                scores = self.inner.predict(X_test).astype(float)
            return self._make_pred_series(scores, X_test.index)

        def _extract_label(self, label_df):
            if label_df is None:
                return None
            if isinstance(label_df, pd.Series):
                return label_df
            if isinstance(label_df, pd.DataFrame):
                if self.label_col in label_df.columns:
                    return label_df[self.label_col]
                if isinstance(label_df.columns, pd.MultiIndex):
                    for top, sub in label_df.columns:
                        if sub == self.label_col:
                            return label_df[(top, sub)]
                return label_df.iloc[:, 0]
            return label_df

    class HFMLXGBQlibModel(Model, _HFMLModelMixin):
        """将 HFML XGBoostModel 包装为 Qlib Model（5 分钟周期推荐）。

        Parameters
        ----------
        params : dict, optional
            传递给 XGBoostModel 的参数。
        task : str
            "classification" 或 "regression"。
        label_col : str
            标签列名，默认 "trading_signal"。
        """

        def __init__(self, params=None, task="classification", label_col="trading_signal"):
            from models.ml_models import XGBoostModel
            self.params = params or {}
            self.task = task
            self.label_col = label_col
            self.inner = XGBoostModel(task=task, params=self.params)

        def fit(self, dataset, reweighter=None):
            from qlib.data.dataset.handler import DataHandlerLP

            [X_train, X_valid], [y_train_df, y_valid_df] = self._prepare_Xy(
                dataset,
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            y_train = self._extract_label(y_train_df)
            y_valid = self._extract_label(y_valid_df)
            self.inner.train(
                X_train, y_train,
                X_val=X_valid if not X_valid.empty else None,
                y_val=y_valid,
            )

        def predict(self, dataset, segment="test"):
            from qlib.data.dataset.handler import DataHandlerLP

            [X_test], _ = self._prepare_Xy(
                dataset, [segment], col_set=["feature"], data_key=DataHandlerLP.DK_I
            )
            if X_test.empty:
                return pd.Series(dtype=float, name="score")
            try:
                scores = self.inner.predict_proba(X_test)[:, 1]
            except Exception:
                scores = self.inner.predict(X_test).astype(float)
            return self._make_pred_series(scores, X_test.index)

        def _extract_label(self, label_df):
            if label_df is None:
                return None
            if isinstance(label_df, pd.Series):
                return label_df
            if isinstance(label_df, pd.DataFrame):
                if self.label_col in label_df.columns:
                    return label_df[self.label_col]
                if isinstance(label_df.columns, pd.MultiIndex):
                    for top, sub in label_df.columns:
                        if sub == self.label_col:
                            return label_df[(top, sub)]
                return label_df.iloc[:, 0]
            return label_df

except ImportError:
    # Qlib 未安装时的占位实现
    class HFMLLSTMQlibModel(_HFMLModelMixin):  # type: ignore[no-redef]
        def __init__(self, params=None, task="classification", label_col="trading_signal"):
            from models.ml_models import LSTMModel
            self.inner = LSTMModel(task=task, params=params or {})
            self.label_col = label_col

        def fit(self, dataset, reweighter=None):
            pass

        def predict(self, dataset, segment="test"):
            return pd.Series(dtype=float, name="score")

    class HFMLLGBMQlibModel(_HFMLModelMixin):  # type: ignore[no-redef]
        def __init__(self, params=None, task="classification", label_col="trading_signal"):
            from models.ml_models import LightGBMModel
            self.inner = LightGBMModel(task=task, params=params or {})
            self.label_col = label_col

        def fit(self, dataset, reweighter=None):
            pass

        def predict(self, dataset, segment="test"):
            return pd.Series(dtype=float, name="score")

    class HFMLXGBQlibModel(_HFMLModelMixin):  # type: ignore[no-redef]
        def __init__(self, params=None, task="classification", label_col="trading_signal"):
            from models.ml_models import XGBoostModel
            self.inner = XGBoostModel(task=task, params=params or {})
            self.label_col = label_col

        def fit(self, dataset, reweighter=None):
            pass

        def predict(self, dataset, segment="test"):
            return pd.Series(dtype=float, name="score")
