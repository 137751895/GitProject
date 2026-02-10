"""
商品期货机器学习量化模型 - 多时间框架协同模块
Multi-timeframe coordination module for commodity futures ML model.

解决各周期模型独立运行的问题，通过层级化信号融合提升交易质量。

协同策略:
- 15分钟模型确定 **趋势方向** (大方向过滤)
- 5分钟模型寻找 **入场时机** (信号触发)
- 1分钟模型优化 **入场价格** (精确入场)

信号融合规则:
1. 15分钟趋势方向为中性(neutral)时不开仓
2. 5分钟信号必须与15分钟方向一致才触发
3. 1分钟信号用于在5分钟触发窗口内寻找最优入场点
4. 三个周期信号一致时产生最强信号(resonance)
"""

import numpy as np
import pandas as pd


class TrendDirectionModel:
    """
    15分钟趋势方向模型。

    职责: 确定大方向（看多/看空/中性），作为交易方向的过滤器。
    只有当大方向明确时，才允许下级周期发出交易信号。

    输出:
    - direction: +1(看多), -1(看空), 0(中性)
    - confidence: 方向确信度 (0~1)
    - regime: 当前市场状态 (trend/range)
    """

    def __init__(self, neutral_zone=0.10):
        """
        Parameters
        ----------
        neutral_zone : float
            中性区间宽度。当模型输出概率在 (0.5-zone, 0.5+zone) 之间时
            判定为中性方向，不提供方向指引。
        """
        self.neutral_zone = neutral_zone
        self.model = None
        self.last_direction = 0
        self.last_confidence = 0.0

    def set_model(self, model):
        """
        设置已训练的15分钟模型。

        Parameters
        ----------
        model : object
            已训练的模型（LSTMModel 或其他），需具备 predict_proba 方法
        """
        self.model = model

    def predict_direction(self, X_15min):
        """
        预测趋势方向。

        Parameters
        ----------
        X_15min : pd.DataFrame or np.ndarray
            15分钟特征数据

        Returns
        -------
        dict
            direction: +1(看多) / -1(看空) / 0(中性)
            confidence: 方向确信度(0~1)
            probabilities: 原始概率值
        """
        if self.model is None:
            return self._default_result(len(X_15min))

        try:
            probas = self.model.predict_proba(X_15min)
            if probas.ndim == 2:
                up_prob = probas[:, 1]
            else:
                up_prob = probas
        except (ValueError, AttributeError):
            return self._default_result(len(X_15min))

        n = len(up_prob)
        direction = np.zeros(n, dtype=int)
        confidence = np.zeros(n, dtype=float)

        upper = 0.5 + self.neutral_zone
        lower = 0.5 - self.neutral_zone

        bullish = up_prob > upper
        bearish = up_prob < lower

        direction[bullish] = 1
        direction[bearish] = -1

        # 确信度 = 概率偏离0.5的程度，归一化到0~1
        confidence = np.abs(up_prob - 0.5) * 2.0
        confidence = np.clip(confidence, 0.0, 1.0)

        self.last_direction = direction[-1] if n > 0 else 0
        self.last_confidence = confidence[-1] if n > 0 else 0.0

        return {
            "direction": direction,
            "confidence": confidence,
            "probabilities": up_prob,
        }

    def _default_result(self, n):
        """无模型时返回中性"""
        return {
            "direction": np.zeros(n, dtype=int),
            "confidence": np.zeros(n, dtype=float),
            "probabilities": np.full(n, 0.5),
        }


class EntryTimingModel:
    """
    5分钟入场时机模型。

    职责: 在15分钟确定方向后，寻找合适的入场时机。
    只有当5分钟信号与15分钟方向一致时才触发入场。

    输出:
    - signal: +1(做多入场) / -1(做空入场) / 0(等待)
    - strength: 信号强度 (0~1)
    - timing_quality: 入场时机质量评估
    """

    def __init__(self, entry_threshold=0.60):
        """
        Parameters
        ----------
        entry_threshold : float
            入场信号阈值。5分钟模型概率超过此值才产生入场信号。
        """
        self.entry_threshold = entry_threshold
        self.model = None

    def set_model(self, model):
        """设置已训练的5分钟模型"""
        self.model = model

    def predict_entry(self, X_5min, trend_direction):
        """
        预测入场时机。

        Parameters
        ----------
        X_5min : pd.DataFrame or np.ndarray
            5分钟特征数据
        trend_direction : np.ndarray
            15分钟模型给出的趋势方向 (+1/-1/0)

        Returns
        -------
        dict
            signal: 入场信号 (+1/-1/0)
            strength: 信号强度 (0~1)
            raw_probabilities: 原始概率
        """
        n = len(X_5min) if hasattr(X_5min, '__len__') else 1
        trend_direction = np.asarray(trend_direction).ravel()

        if self.model is None:
            return self._default_result(n)

        try:
            probas = self.model.predict_proba(X_5min)
            if probas.ndim == 2:
                up_prob = probas[:, 1]
            else:
                up_prob = probas
        except (ValueError, AttributeError):
            return self._default_result(n)

        # 对齐长度
        min_len = min(len(up_prob), len(trend_direction))
        up_prob = up_prob[:min_len]
        trend_dir = trend_direction[:min_len]

        signal = np.zeros(min_len, dtype=int)
        strength = np.zeros(min_len, dtype=float)

        # 做多条件: 15分钟看多 + 5分钟上涨概率超过阈值
        long_mask = (trend_dir == 1) & (up_prob > self.entry_threshold)
        signal[long_mask] = 1
        strength[long_mask] = (up_prob[long_mask] - 0.5) * 2.0

        # 做空条件: 15分钟看空 + 5分钟下跌概率超过阈值
        short_mask = (trend_dir == -1) & (up_prob < (1 - self.entry_threshold))
        signal[short_mask] = -1
        strength[short_mask] = (0.5 - up_prob[short_mask]) * 2.0

        strength = np.clip(strength, 0.0, 1.0)

        return {
            "signal": signal,
            "strength": strength,
            "raw_probabilities": up_prob,
        }

    def _default_result(self, n):
        """无模型时返回无信号"""
        return {
            "signal": np.zeros(n, dtype=int),
            "strength": np.zeros(n, dtype=float),
            "raw_probabilities": np.full(n, 0.5),
        }


class EntryPriceOptimizer:
    """
    1分钟入场价格优化模型。

    职责: 在5分钟触发入场信号后，利用1分钟数据寻找最优入场点。
    通过评估短期价格走势，在入场窗口内选择最有利的价格。

    输出:
    - entry_score: 入场质量评分 (0~1, 越高越好)
    - optimal_entry: 是否为最优入场点
    """

    def __init__(self, optimization_threshold=0.55):
        """
        Parameters
        ----------
        optimization_threshold : float
            入场优化阈值。1分钟模型概率超过此值时认为是好的入场点。
        """
        self.optimization_threshold = optimization_threshold
        self.model = None

    def set_model(self, model):
        """设置已训练的1分钟模型"""
        self.model = model

    def optimize_entry(self, X_1min, entry_signal):
        """
        在入场窗口内寻找最优入场点。

        Parameters
        ----------
        X_1min : pd.DataFrame or np.ndarray
            1分钟特征数据
        entry_signal : np.ndarray
            5分钟模型给出的入场信号 (+1/-1/0)

        Returns
        -------
        dict
            entry_score: 入场质量评分(0~1)
            optimal_entry: 是否为优质入场点(bool)
            final_signal: 最终入场信号（经优化后的）
        """
        n = len(X_1min) if hasattr(X_1min, '__len__') else 1
        entry_signal = np.asarray(entry_signal).ravel()

        if self.model is None:
            return self._default_result(n, entry_signal)

        try:
            probas = self.model.predict_proba(X_1min)
            if probas.ndim == 2:
                up_prob = probas[:, 1]
            else:
                up_prob = probas
        except (ValueError, AttributeError):
            return self._default_result(n, entry_signal)

        min_len = min(len(up_prob), len(entry_signal))
        up_prob = up_prob[:min_len]
        entry_sig = entry_signal[:min_len]

        entry_score = np.zeros(min_len, dtype=float)
        optimal_entry = np.zeros(min_len, dtype=bool)
        final_signal = np.zeros(min_len, dtype=int)

        # 做多入场优化: 1分钟也确认上涨
        long_mask = entry_sig == 1
        if np.any(long_mask):
            long_scores = up_prob[long_mask]
            entry_score[long_mask] = long_scores
            optimal_entry[long_mask] = long_scores > self.optimization_threshold
            final_signal[long_mask & optimal_entry] = 1

        # 做空入场优化: 1分钟也确认下跌
        short_mask = entry_sig == -1
        if np.any(short_mask):
            short_scores = 1.0 - up_prob[short_mask]
            entry_score[short_mask] = short_scores
            optimal_entry[short_mask] = short_scores > self.optimization_threshold
            final_signal[short_mask & optimal_entry] = -1

        return {
            "entry_score": entry_score,
            "optimal_entry": optimal_entry,
            "final_signal": final_signal,
        }

    def _default_result(self, n, entry_signal):
        """无模型时直接传递5分钟信号"""
        entry_signal = np.asarray(entry_signal).ravel()[:n]
        return {
            "entry_score": np.full(n, 0.5),
            "optimal_entry": np.ones(n, dtype=bool),
            "final_signal": entry_signal.copy(),
        }


class MultiTimeframeCoordinator:
    """
    多时间框架协同交易系统。

    将三个周期的模型组织为层级决策链:

    ┌─────────────────┐
    │  15分钟模型       │ ← 确定趋势方向（看多/看空/中性）
    │  TrendDirection  │   中性时不开仓
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  5分钟模型        │ ← 寻找入场时机（信号必须与大方向一致）
    │  EntryTiming     │   方向不一致时过滤掉
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  1分钟模型        │ ← 优化入场价格（在入场窗口内择时）
    │  EntryPrice      │   选择最优入场点
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  最终交易信号     │   方向 + 强度 + 入场质量
    └─────────────────┘

    信号质量评级:
    - A级 (resonance): 三个周期信号完全一致，最强信号
    - B级 (confirmed): 15分钟+5分钟一致，1分钟部分确认
    - C级 (partial): 仅15分钟+5分钟一致，1分钟未确认
    - 无信号: 方向不一致或中性

    使用方式:
    ```python
    coordinator = MultiTimeframeCoordinator()
    coordinator.set_models(model_15m, model_5m, model_1m)
    result = coordinator.generate_signals(X_15m, X_5m, X_1m)
    print(result["final_signal"])
    print(result["signal_grade"])
    ```
    """

    def __init__(self, neutral_zone=0.10, entry_threshold=0.60,
                 optimization_threshold=0.55, min_grade="C"):
        """
        Parameters
        ----------
        neutral_zone : float
            15分钟中性区间宽度
        entry_threshold : float
            5分钟入场阈值
        optimization_threshold : float
            1分钟入场优化阈值
        min_grade : str
            最低信号质量等级 ("A"/"B"/"C")，低于此等级的信号被过滤
        """
        self.trend_model = TrendDirectionModel(neutral_zone=neutral_zone)
        self.entry_model = EntryTimingModel(entry_threshold=entry_threshold)
        self.price_model = EntryPriceOptimizer(
            optimization_threshold=optimization_threshold
        )
        self.min_grade = min_grade
        self._signal_history = []

    def set_models(self, model_15min=None, model_5min=None, model_1min=None):
        """
        设置各周期已训练的模型。

        Parameters
        ----------
        model_15min : object, optional
            15分钟周期模型
        model_5min : object, optional
            5分钟周期模型
        model_1min : object, optional
            1分钟周期模型
        """
        if model_15min is not None:
            self.trend_model.set_model(model_15min)
        if model_5min is not None:
            self.entry_model.set_model(model_5min)
        if model_1min is not None:
            self.price_model.set_model(model_1min)

    def generate_signals(self, X_15min, X_5min, X_1min):
        """
        生成多时间框架协同交易信号。

        流程:
        1. 15分钟模型判断趋势方向
        2. 5分钟模型在趋势方向上寻找入场时机
        3. 1分钟模型在入场窗口内优化入场价格
        4. 综合三个层级生成最终信号

        Parameters
        ----------
        X_15min : pd.DataFrame
            15分钟特征数据
        X_5min : pd.DataFrame
            5分钟特征数据
        X_1min : pd.DataFrame
            1分钟特征数据

        Returns
        -------
        dict
            final_signal: 最终交易信号 (+1/-1/0)
            signal_grade: 信号质量等级 ("A"/"B"/"C"/"-")
            signal_strength: 综合信号强度 (0~1)
            trend_direction: 15分钟趋势方向
            trend_confidence: 15分钟方向确信度
            entry_signal: 5分钟入场信号
            entry_strength: 5分钟信号强度
            entry_score: 1分钟入场质量评分
            optimal_entry: 1分钟是否最优入场
            agreement_score: 三周期一致性评分 (0~1)
        """
        # Step 1: 15分钟 → 趋势方向
        trend_result = self.trend_model.predict_direction(X_15min)

        # Step 2: 5分钟 → 入场时机 (受15分钟方向约束)
        entry_result = self.entry_model.predict_entry(
            X_5min, trend_result["direction"]
        )

        # Step 3: 1分钟 → 入场价格优化 (在5分钟信号基础上)
        price_result = self.price_model.optimize_entry(
            X_1min, entry_result["signal"]
        )

        # Step 4: 综合信号生成
        n = min(
            len(trend_result["direction"]),
            len(entry_result["signal"]),
            len(price_result["final_signal"]),
        )

        trend_dir = trend_result["direction"][:n]
        trend_conf = trend_result["confidence"][:n]
        entry_sig = entry_result["signal"][:n]
        entry_str = entry_result["strength"][:n]
        entry_score = price_result["entry_score"][:n]
        optimal = price_result["optimal_entry"][:n]
        final_sig = price_result["final_signal"][:n]

        # 信号质量评级
        signal_grade = np.full(n, "-", dtype="<U1")
        signal_strength = np.zeros(n, dtype=float)

        for i in range(n):
            if final_sig[i] != 0 and optimal[i]:
                # A级: 三个周期完全一致 (resonance)
                signal_grade[i] = "A"
                signal_strength[i] = (
                    0.4 * trend_conf[i] + 0.35 * entry_str[i]
                    + 0.25 * entry_score[i]
                )
            elif entry_sig[i] != 0 and trend_dir[i] != 0:
                if entry_sig[i] == trend_dir[i]:
                    # B级: 15min+5min一致, 1min部分确认
                    signal_grade[i] = "B"
                    signal_strength[i] = (
                        0.5 * trend_conf[i] + 0.5 * entry_str[i]
                    )
                    # B级时使用5分钟信号方向
                    final_sig[i] = entry_sig[i]

        # 应用最低等级过滤
        grade_order = {"A": 3, "B": 2, "C": 1, "-": 0}
        min_level = grade_order.get(self.min_grade, 1)
        for i in range(n):
            if grade_order.get(signal_grade[i], 0) < min_level:
                final_sig[i] = 0
                signal_strength[i] = 0.0

        # 三周期一致性评分
        agreement_score = self._compute_agreement(
            trend_dir, entry_sig, price_result["final_signal"][:n]
        )

        result = {
            "final_signal": final_sig,
            "signal_grade": signal_grade,
            "signal_strength": signal_strength,
            "trend_direction": trend_dir,
            "trend_confidence": trend_conf,
            "entry_signal": entry_sig,
            "entry_strength": entry_str,
            "entry_score": entry_score,
            "optimal_entry": optimal,
            "agreement_score": agreement_score,
        }
        self._signal_history.append(result)
        return result

    @staticmethod
    def _compute_agreement(trend_dir, entry_signal, price_signal):
        """
        计算三个周期的信号一致性评分。

        Parameters
        ----------
        trend_dir : np.ndarray
            15分钟趋势方向
        entry_signal : np.ndarray
            5分钟入场信号
        price_signal : np.ndarray
            1分钟入场信号

        Returns
        -------
        np.ndarray
            一致性评分 (0~1)
        """
        n = min(len(trend_dir), len(entry_signal), len(price_signal))
        score = np.zeros(n, dtype=float)

        for i in range(n):
            td = trend_dir[i]
            es = entry_signal[i]
            ps = price_signal[i]

            if td == 0 and es == 0 and ps == 0:
                score[i] = 0.0
                continue

            agreements = 0
            total_pairs = 0

            if td != 0 or es != 0:
                total_pairs += 1
                if td == es:
                    agreements += 1

            if td != 0 or ps != 0:
                total_pairs += 1
                if td == ps:
                    agreements += 1

            if es != 0 or ps != 0:
                total_pairs += 1
                if es == ps:
                    agreements += 1

            score[i] = agreements / total_pairs if total_pairs > 0 else 0.0

        return score

    def get_signal_summary(self):
        """
        获取最近一次信号生成的摘要统计。

        Returns
        -------
        dict
            信号统计摘要
        """
        if not self._signal_history:
            return {"n_signals": 0}

        last = self._signal_history[-1]
        final = last["final_signal"]
        grades = last["signal_grade"]

        n_long = int(np.sum(final == 1))
        n_short = int(np.sum(final == -1))
        n_neutral = int(np.sum(final == 0))

        grade_counts = {}
        for g in ["A", "B", "C", "-"]:
            grade_counts[f"grade_{g}"] = int(np.sum(grades == g))

        avg_agreement = float(np.nanmean(last["agreement_score"]))
        avg_strength = float(np.nanmean(
            last["signal_strength"][last["final_signal"] != 0]
        )) if np.any(final != 0) else 0.0

        return {
            "n_total": len(final),
            "n_long": n_long,
            "n_short": n_short,
            "n_neutral": n_neutral,
            "avg_agreement": avg_agreement,
            "avg_signal_strength": avg_strength,
            **grade_counts,
        }

    def get_signal_history_df(self):
        """
        获取信号历史的DataFrame。

        Returns
        -------
        pd.DataFrame
            信号历史记录
        """
        if not self._signal_history:
            return pd.DataFrame()

        records = []
        for i, h in enumerate(self._signal_history):
            records.append({
                "batch": i,
                "n_signals": int(np.sum(h["final_signal"] != 0)),
                "n_long": int(np.sum(h["final_signal"] == 1)),
                "n_short": int(np.sum(h["final_signal"] == -1)),
                "avg_agreement": float(np.nanmean(h["agreement_score"])),
            })
        return pd.DataFrame(records)
