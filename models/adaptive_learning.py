"""
商品期货机器学习量化模型 - 实时自适应模块
Adaptive learning module for handling market structure changes.

核心功能:
1. 在线学习 (Online Learning): 增量更新模型参数，不需要完全重训
2. 概念漂移检测 (Concept Drift Detection): 监控模型性能衰减
3. 自适应阈值 (Adaptive Threshold): 根据波动率和模型表现动态调整交易信号阈值

设计思路:
- 市场结构会随时间变化（如波动率变化、相关性变化等），固定模型可能逐渐失效
- 通过滑动窗口监控模型准确率，当性能下降超过阈值时触发警告或重训
- 通过增量学习保持模型对最新数据的适应性
"""

import numpy as np
import pandas as pd
from collections import deque


class ConceptDriftDetector:
    """
    概念漂移检测器。

    通过滑动窗口监控模型预测性能，当性能持续下降或
    偏离历史基准时触发漂移警告，提示需要重训模型。

    检测方法:
    - 滑动窗口准确率追踪
    - 与历史基准的对比 (基准衰减检测)
    - 连续下降趋势检测

    漂移等级:
    - "none": 无漂移，模型正常
    - "warning": 轻微漂移，建议关注
    - "drift": 严重漂移，建议重训模型
    """

    def __init__(self, window_size=100, warning_threshold=0.05,
                 drift_threshold=0.10, baseline_window=500):
        """
        Parameters
        ----------
        window_size : int
            滑动窗口大小（最近N个预测样本）
        warning_threshold : float
            准确率下降超过此值时触发warning（相对于基准）
        drift_threshold : float
            准确率下降超过此值时触发drift（相对于基准）
        baseline_window : int
            基准性能统计窗口大小
        """
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.baseline_window = baseline_window

        # 最近的预测正确性记录 (1=正确, 0=错误)
        self._recent_correct = deque(maxlen=window_size)
        # 所有历史准确率（用于建立基准）
        self._all_correct = deque(maxlen=baseline_window)
        # 漂移历史
        self._drift_history = []

    @property
    def n_samples(self):
        """已收到的样本数"""
        return len(self._all_correct)

    @property
    def current_accuracy(self):
        """当前滑动窗口准确率"""
        if len(self._recent_correct) == 0:
            return np.nan
        return np.mean(self._recent_correct)

    @property
    def baseline_accuracy(self):
        """历史基准准确率"""
        if len(self._all_correct) < self.window_size:
            return np.nan
        return np.mean(self._all_correct)

    def update(self, y_true, y_pred):
        """
        更新检测器，输入一批新的真实值和预测值。

        Parameters
        ----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签

        Returns
        -------
        dict
            漂移检测结果，包含:
            - drift_level: "none" / "warning" / "drift"
            - current_accuracy: 当前窗口准确率
            - baseline_accuracy: 历史基准准确率
            - accuracy_drop: 准确率下降幅度
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        correct = (y_true == y_pred).astype(float)

        for c in correct:
            self._recent_correct.append(c)
            self._all_correct.append(c)

        cur_acc = self.current_accuracy
        base_acc = self.baseline_accuracy

        # 计算漂移等级
        if np.isnan(base_acc) or np.isnan(cur_acc):
            drift_level = "none"
            accuracy_drop = 0.0
        else:
            accuracy_drop = base_acc - cur_acc
            if accuracy_drop >= self.drift_threshold:
                drift_level = "drift"
            elif accuracy_drop >= self.warning_threshold:
                drift_level = "warning"
            else:
                drift_level = "none"

        result = {
            "drift_level": drift_level,
            "current_accuracy": cur_acc,
            "baseline_accuracy": base_acc,
            "accuracy_drop": accuracy_drop,
            "n_samples": self.n_samples,
        }

        self._drift_history.append(result)
        return result

    def get_drift_history(self):
        """获取漂移检测历史记录"""
        if not self._drift_history:
            return pd.DataFrame()
        return pd.DataFrame(self._drift_history)

    def reset(self):
        """重置检测器状态（通常在模型重训后调用）"""
        self._recent_correct.clear()
        self._all_correct.clear()
        self._drift_history.clear()


class OnlineLearner:
    """
    在线学习器。

    对LightGBM/XGBoost模型执行增量更新（warm-start），
    在新数据到来时不需要完全重训，而是在已有模型基础上继续训练。

    工作模式:
    - 收集新的数据样本（达到 update_interval 后触发更新）
    - 在已有模型基础上增量训练少量树
    - 记录每次更新后的模型性能

    注意事项:
    - LSTM模型不支持在线学习，需完全重训
    - 增量更新的树数量 (n_incremental_trees) 应远小于初始模型
    - 建议定期（如每月）完全重训模型以避免增量偏差累积
    """

    def __init__(self, base_model, update_interval=500,
                 n_incremental_trees=50, max_buffer_size=5000):
        """
        Parameters
        ----------
        base_model : object
            已训练的基础模型 (LightGBMModel 或 XGBoostModel)
        update_interval : int
            每收集多少新样本后触发一次增量更新
        n_incremental_trees : int
            每次增量更新新增的树数量
        max_buffer_size : int
            数据缓冲区最大大小
        """
        self.base_model = base_model
        self.update_interval = update_interval
        self.n_incremental_trees = n_incremental_trees
        self.max_buffer_size = max_buffer_size

        self._buffer_X = []
        self._buffer_y = []
        self._update_count = 0
        self._update_history = []

    @property
    def total_samples_received(self):
        """累计接收的样本总数"""
        return self._update_count * self.update_interval + len(self._buffer_X)

    def add_samples(self, X_new, y_new):
        """
        添加新数据样本到缓冲区。

        当缓冲区大小达到 update_interval 时自动触发增量更新。

        Parameters
        ----------
        X_new : pd.DataFrame or np.ndarray
            新特征数据
        y_new : pd.Series or np.ndarray
            新目标数据

        Returns
        -------
        dict or None
            如果触发了增量更新返回更新结果，否则返回None
        """
        if isinstance(X_new, pd.DataFrame):
            self._buffer_X.append(X_new)
        else:
            self._buffer_X.append(pd.DataFrame(X_new))

        if isinstance(y_new, pd.Series):
            self._buffer_y.append(y_new)
        else:
            self._buffer_y.append(pd.Series(y_new))

        total_buffered = sum(len(x) for x in self._buffer_X)
        if total_buffered >= self.update_interval:
            return self._incremental_update()
        return None

    def _incremental_update(self):
        """
        执行增量更新。

        在已有模型基础上继续训练少量新树。

        Returns
        -------
        dict
            更新结果
        """
        X_update = pd.concat(self._buffer_X, ignore_index=True)
        y_update = pd.concat(self._buffer_y, ignore_index=True)

        # 限制缓冲区大小（保留最近的数据，因为增量学习关注最新的市场状态）
        if len(X_update) > self.max_buffer_size:
            X_update = X_update.tail(self.max_buffer_size)
            y_update = y_update.tail(self.max_buffer_size)

        model = self.base_model
        model_type = type(model).__name__

        updated = False
        if model_type == "LightGBMModel" and model.model is not None:
            # LightGBM支持init_model增量训练
            import lightgbm as lgb
            old_model = model.model
            if hasattr(old_model, 'booster_'):
                new_params = model.params.copy()
                new_params["n_estimators"] = self.n_incremental_trees
                if model.task == "classification":
                    new_model = lgb.LGBMClassifier(**new_params)
                else:
                    new_model = lgb.LGBMRegressor(**new_params)
                new_model.fit(X_update, y_update, init_model=old_model)
                model.model = new_model
                updated = True

        elif model_type == "XGBoostModel" and model.model is not None:
            # XGBoost支持xgb_model增量训练
            import xgboost as xgb
            old_model = model.model
            new_params = model.params.copy()
            new_params["n_estimators"] = self.n_incremental_trees
            if model.task == "classification":
                n_classes = getattr(old_model, 'n_classes_', 2)
                if n_classes > 2:
                    new_model = xgb.XGBClassifier(
                        eval_metric="mlogloss", **new_params
                    )
                else:
                    new_model = xgb.XGBClassifier(
                        eval_metric="logloss", **new_params
                    )
            else:
                new_model = xgb.XGBRegressor(**new_params)
            new_model.fit(X_update, y_update, xgb_model=old_model)
            model.model = new_model
            updated = True

        self._update_count += 1
        result = {
            "update_id": self._update_count,
            "n_samples": len(X_update),
            "model_type": model_type,
            "updated": updated,
        }
        self._update_history.append(result)

        # 清空缓冲区
        self._buffer_X.clear()
        self._buffer_y.clear()

        return result

    def get_update_history(self):
        """获取增量更新历史"""
        if not self._update_history:
            return pd.DataFrame()
        return pd.DataFrame(self._update_history)


class AdaptiveThresholdManager:
    """
    自适应阈值管理器。

    根据波动率水平和模型近期表现动态调整交易信号阈值:
    - 高波动率时提高阈值（更保守，减少假信号）
    - 低波动率时降低阈值（捕捉更多机会）
    - 模型性能下降时提高阈值（减少暴露）

    阈值范围始终限制在 [min_threshold, max_threshold] 之间。
    """

    def __init__(self, base_threshold=0.55, vol_sensitivity=1.0,
                 performance_sensitivity=0.5,
                 min_threshold=0.50, max_threshold=0.85):
        """
        Parameters
        ----------
        base_threshold : float
            基础阈值
        vol_sensitivity : float
            波动率灵敏度因子 (>1更敏感, <1更稳定)
        performance_sensitivity : float
            性能灵敏度因子 (>1更敏感, <1更稳定)
        min_threshold : float
            最小阈值（下限）
        max_threshold : float
            最大阈值（上限）
        """
        self.base_threshold = base_threshold
        self.vol_sensitivity = vol_sensitivity
        self.performance_sensitivity = performance_sensitivity
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._threshold_history = []

    def compute_threshold(self, current_volatility, mean_volatility,
                          current_accuracy=None, baseline_accuracy=None):
        """
        计算自适应阈值。

        公式:
        threshold = base + vol_adjustment + performance_adjustment
        - vol_adjustment = vol_sensitivity × (vol_ratio - 1.0) × 0.1
        - perf_adjustment = perf_sensitivity × max(0, accuracy_drop) × 0.5

        Parameters
        ----------
        current_volatility : float
            当前波动率
        mean_volatility : float
            波动率均值（历史参考水平）
        current_accuracy : float, optional
            模型当前准确率
        baseline_accuracy : float, optional
            模型基准准确率

        Returns
        -------
        dict
            包含 threshold, vol_adjustment, perf_adjustment 的字典
        """
        threshold = self.base_threshold

        # 1. 波动率调整
        vol_adjustment = 0.0
        if mean_volatility > 0:
            vol_ratio = current_volatility / mean_volatility
            vol_adjustment = self.vol_sensitivity * (vol_ratio - 1.0) * 0.1
        threshold += vol_adjustment

        # 2. 性能衰减调整
        perf_adjustment = 0.0
        if (current_accuracy is not None and baseline_accuracy is not None
                and baseline_accuracy > 0):
            accuracy_drop = max(0.0, baseline_accuracy - current_accuracy)
            perf_adjustment = self.performance_sensitivity * accuracy_drop * 0.5
        threshold += perf_adjustment

        # 限制范围
        threshold = float(np.clip(threshold, self.min_threshold,
                                  self.max_threshold))

        result = {
            "threshold": threshold,
            "base_threshold": self.base_threshold,
            "vol_adjustment": vol_adjustment,
            "perf_adjustment": perf_adjustment,
        }
        self._threshold_history.append(result)
        return result

    def get_threshold_history(self):
        """获取阈值调整历史"""
        if not self._threshold_history:
            return pd.DataFrame()
        return pd.DataFrame(self._threshold_history)


class AdaptiveModelManager:
    """
    自适应模型管理器。

    整合概念漂移检测、在线学习和自适应阈值三个组件，
    提供统一的模型管理接口。

    工作流:
    1. 接收新的预测结果和真实值 → 更新漂移检测器
    2. 收集新数据 → 在线学习器增量更新
    3. 基于漂移状态和波动率 → 动态调整阈值
    4. 漂移严重时 → 触发完全重训信号

    使用示例:
    ```python
    manager = AdaptiveModelManager(trained_model)
    # 在每个预测步骤后:
    status = manager.step(
        y_true=actual_labels,
        y_pred=predicted_labels,
        X_new=new_features,
        y_new=new_targets,
        current_volatility=0.02,
        mean_volatility=0.015,
    )
    print(status["drift_level"])    # "none" / "warning" / "drift"
    print(status["threshold"])       # 当前交易阈值
    print(status["needs_retrain"])   # 是否需要完全重训
    ```
    """

    def __init__(self, base_model,
                 drift_window=100, drift_warning=0.05, drift_threshold=0.10,
                 update_interval=500, n_incremental_trees=50,
                 base_threshold=0.55):
        """
        Parameters
        ----------
        base_model : object
            已训练的基础模型
        drift_window : int
            漂移检测窗口
        drift_warning : float
            漂移warning阈值
        drift_threshold : float
            漂移drift阈值
        update_interval : int
            在线学习更新间隔
        n_incremental_trees : int
            增量更新树数量
        base_threshold : float
            基础交易阈值
        """
        self.drift_detector = ConceptDriftDetector(
            window_size=drift_window,
            warning_threshold=drift_warning,
            drift_threshold=drift_threshold,
        )
        self.online_learner = OnlineLearner(
            base_model=base_model,
            update_interval=update_interval,
            n_incremental_trees=n_incremental_trees,
        )
        self.threshold_manager = AdaptiveThresholdManager(
            base_threshold=base_threshold,
        )
        self._step_history = []

    def step(self, y_true, y_pred, X_new=None, y_new=None,
             current_volatility=0.0, mean_volatility=0.0):
        """
        执行一步自适应更新。

        Parameters
        ----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        X_new : pd.DataFrame, optional
            新特征数据（用于增量学习）
        y_new : pd.Series, optional
            新目标数据
        current_volatility : float
            当前波动率
        mean_volatility : float
            历史波动率均值

        Returns
        -------
        dict
            自适应状态信息
        """
        # 1. 漂移检测
        drift_result = self.drift_detector.update(y_true, y_pred)

        # 2. 在线学习
        update_result = None
        if X_new is not None and y_new is not None:
            update_result = self.online_learner.add_samples(X_new, y_new)

        # 3. 自适应阈值
        threshold_result = self.threshold_manager.compute_threshold(
            current_volatility=current_volatility,
            mean_volatility=mean_volatility,
            current_accuracy=drift_result["current_accuracy"],
            baseline_accuracy=drift_result["baseline_accuracy"],
        )

        status = {
            "drift_level": drift_result["drift_level"],
            "current_accuracy": drift_result["current_accuracy"],
            "accuracy_drop": drift_result["accuracy_drop"],
            "threshold": threshold_result["threshold"],
            "vol_adjustment": threshold_result["vol_adjustment"],
            "perf_adjustment": threshold_result["perf_adjustment"],
            "incremental_update": update_result is not None,
            "needs_retrain": drift_result["drift_level"] == "drift",
        }
        self._step_history.append(status)
        return status

    def get_status_summary(self):
        """
        获取当前系统状态摘要。

        Returns
        -------
        dict
            系统状态摘要
        """
        return {
            "n_samples": self.drift_detector.n_samples,
            "current_accuracy": self.drift_detector.current_accuracy,
            "baseline_accuracy": self.drift_detector.baseline_accuracy,
            "n_updates": self.online_learner._update_count,
            "latest_threshold": (
                self._step_history[-1]["threshold"]
                if self._step_history else None
            ),
            "latest_drift": (
                self._step_history[-1]["drift_level"]
                if self._step_history else "none"
            ),
        }

    def get_step_history(self):
        """获取所有步骤的历史记录"""
        if not self._step_history:
            return pd.DataFrame()
        return pd.DataFrame(self._step_history)
