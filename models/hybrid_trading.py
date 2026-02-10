"""
商品期货机器学习量化模型 - 混合智能交易系统
Hybrid intelligent trading system with multiple expert strategies.

三层决策架构:
1. 信号生成层 — 五个独立交易专家各自生成信号
2. 信号融合层 — 加权投票融合多专家信号
3. 风险管理层 — 自适应风控调整最终仓位

五个交易专家:
- TrendFollowingExpert: 趋势跟踪（MA交叉 + 趋势持续性）
- MeanReversionExpert: 均值回归（RSI超买超卖 + 布林带偏离）
- BreakoutExpert: 突破交易（N日高低点突破）
- VolumePriceExpert: 量价关系（价格变化 + 成交量放大）
- OIConfirmationExpert: 持仓确认（资金流向与价格的协同性）

设计理念:
    不依赖单一模型，而是让多个"交易员视角"的专家各司其职，
    通过加权投票决定最终信号，提高信号的稳健性和多样性。
"""

import numpy as np
import pandas as pd


class TrendFollowingExpert:
    """
    趋势跟踪专家。

    基于移动平均线金叉/死叉系统判断趋势方向，
    并通过趋势持续性（方向一致性）衡量置信度。

    信号逻辑:
    - MA_fast > MA_slow → 看多 (+1)
    - MA_fast < MA_slow → 看空 (-1)
    - 置信度 = 趋势强度 × 方向一致性
    """

    def __init__(self, fast_window=5, slow_window=20, consistency_window=10):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.consistency_window = consistency_window

    def generate_signal(self, df):
        """
        生成趋势信号。

        Parameters
        ----------
        df : pd.DataFrame
            K线数据，需包含 close 列

        Returns
        -------
        tuple of (float, float)
            (signal, confidence) — 信号方向和置信度
        """
        close = df["close"]
        ma_fast = close.rolling(self.fast_window, min_periods=1).mean()
        ma_slow = close.rolling(self.slow_window, min_periods=1).mean()

        trend_signal = np.sign(ma_fast - ma_slow)
        trend_strength = (ma_fast - ma_slow).abs() / (close + 1e-8)

        # 方向一致性
        price_changes = close.diff()
        current_dir = trend_signal.iloc[-1] if len(trend_signal) > 0 else 0
        recent_changes = price_changes.tail(self.consistency_window)
        if len(recent_changes) > 0 and current_dir != 0:
            consistency = float(
                (np.sign(recent_changes) == current_dir).sum()
                / len(recent_changes)
            )
        else:
            consistency = 0.0

        signal = float(trend_signal.iloc[-1]) if len(trend_signal) > 0 else 0.0
        strength = float(trend_strength.iloc[-1]) if len(trend_strength) > 0 else 0.0
        confidence = strength * consistency

        return signal, min(confidence, 1.0)


class MeanReversionExpert:
    """
    均值回归专家。

    基于RSI超买超卖信号判断价格过度偏离，
    预期价格将回归均值。

    信号逻辑:
    - RSI > overbought → 超买，卖出 (-1)
    - RSI < oversold → 超卖，买入 (+1)
    - 置信度 = RSI偏离50的程度
    """

    def __init__(self, rsi_window=14, overbought=70, oversold=30):
        self.rsi_window = rsi_window
        self.overbought = overbought
        self.oversold = oversold

    def generate_signal(self, df):
        """生成均值回归信号"""
        close = df["close"]
        rsi = self._calc_rsi(close, self.rsi_window)

        if len(rsi) == 0 or np.isnan(rsi.iloc[-1]):
            return 0.0, 0.0

        current_rsi = float(rsi.iloc[-1])

        if current_rsi > self.overbought:
            signal = -1.0  # 超买 → 卖出
        elif current_rsi < self.oversold:
            signal = 1.0   # 超卖 → 买入
        else:
            signal = 0.0

        confidence = abs(current_rsi - 50) / 50.0
        return signal, min(confidence, 1.0)

    @staticmethod
    def _calc_rsi(close, window=14):
        """计算RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


class BreakoutExpert:
    """
    突破交易专家。

    基于价格突破N日高低点判断趋势启动，
    突破幅度越大置信度越高。

    信号逻辑:
    - close > 前N日最高价 → 向上突破 (+1)
    - close < 前N日最低价 → 向下突破 (-1)
    - 置信度 = 突破幅度 / 前价格
    """

    def __init__(self, lookback=20):
        self.lookback = lookback

    def generate_signal(self, df):
        """生成突破信号"""
        if len(df) < self.lookback + 1:
            return 0.0, 0.0

        close = df["close"]
        current_close = float(close.iloc[-1])

        # 排除当前K线的历史高低点
        recent = df.iloc[-(self.lookback + 1):-1]
        high_n = float(recent["high"].max()) if "high" in df.columns else float(recent["close"].max())
        low_n = float(recent["low"].min()) if "low" in df.columns else float(recent["close"].min())

        if current_close > high_n:
            signal = 1.0
            confidence = (current_close - high_n) / (high_n + 1e-8)
        elif current_close < low_n:
            signal = -1.0
            confidence = (low_n - current_close) / (current_close + 1e-8)
        else:
            signal = 0.0
            confidence = 0.0

        return signal, min(confidence, 1.0)


class VolumePriceExpert:
    """
    量价关系专家。

    基于价格变化与成交量放大的协同性判断趋势有效性。

    信号逻辑:
    - 价格涨 + 放量 → 看多 (+1)
    - 价格跌 + 放量 → 看空 (-1)
    - 无放量或价格无明显变化 → 中性 (0)
    - 置信度 = 成交量超出均值的标准差数
    """

    def __init__(self, vol_ma_window=20):
        self.vol_ma_window = vol_ma_window

    def generate_signal(self, df):
        """生成量价信号"""
        if "volume" not in df.columns or len(df) < self.vol_ma_window:
            return 0.0, 0.0

        close = df["close"]
        volume = df["volume"]

        price_up = float(close.diff().iloc[-1]) > 0
        vol_ma = float(volume.rolling(self.vol_ma_window, min_periods=5).mean().iloc[-1])
        vol_std = float(volume.rolling(self.vol_ma_window, min_periods=5).std().iloc[-1])
        current_vol = float(volume.iloc[-1])

        volume_up = current_vol > vol_ma

        if price_up and volume_up:
            signal = 1.0
        elif not price_up and volume_up:
            signal = -1.0
        else:
            signal = 0.0

        # 置信度 = 成交量偏离均值的程度
        if vol_std > 0:
            confidence = abs(current_vol - vol_ma) / vol_std
        else:
            confidence = 0.0

        return signal, min(confidence / 2.0, 1.0)  # 归一化到0~1


class OIConfirmationExpert:
    """
    持仓确认专家。

    基于持仓量变化与价格变化的协同性判断趋势延续或反转。

    信号逻辑:
    - 持仓增 + 价格涨 → 趋势延续, 看多 (+1)
    - 持仓增 + 价格跌 → 趋势延续, 看空 (-1)
    - 持仓减 + 价格涨/跌 → 可能反转, 反向信号
    - 置信度 = 协同一致性 × 持仓变化幅度
    """

    def __init__(self, alignment_window=3, reference_window=20):
        self.alignment_window = alignment_window
        self.reference_window = reference_window

    def generate_signal(self, df):
        """生成持仓确认信号"""
        if "open_interest" not in df.columns or len(df) < self.reference_window:
            return 0.0, 0.0

        close = df["close"]
        oi = df["open_interest"]

        price_change = close.diff()
        oi_change = oi.diff()

        # 协同性 = sign(持仓变化) × sign(价格变化)
        alignment = np.sign(price_change) * np.sign(oi_change)
        recent_alignment = float(
            alignment.tail(self.alignment_window).mean()
        )

        current_price_dir = float(np.sign(price_change.iloc[-1]))

        if recent_alignment > 0.5:
            signal = current_price_dir  # 趋势延续
        elif recent_alignment < -0.5:
            signal = -current_price_dir  # 可能反转
        else:
            signal = 0.0

        # 置信度
        oi_mean = float(oi_change.abs().rolling(
            self.reference_window, min_periods=5
        ).mean().iloc[-1])
        oi_current = float(oi_change.tail(self.alignment_window).abs().mean())
        magnitude = oi_current / (oi_mean + 1e-8)

        # 一致性
        recent = alignment.tail(5)
        if len(recent) > 0:
            consistency = float((recent > 0).sum() / len(recent))
        else:
            consistency = 0.5

        confidence = consistency * min(magnitude, 2.0) / 2.0
        return signal, min(confidence, 1.0)


class HybridTradingSystem:
    """
    混合智能交易系统。

    整合五个独立交易专家，通过加权投票融合信号，
    并经过自适应风控调整生成最终交易决策。

    三层架构:
    ┌────────────────────────┐
    │ 信号生成层 (5个专家)     │
    │  趋势 | 回归 | 突破     │
    │  量价 | 持仓             │
    └────────┬───────────────┘
             ▼
    ┌────────────────────────┐
    │ 信号融合层 (加权投票)     │
    │  权重 × 信号 → 融合信号  │
    └────────┬───────────────┘
             ▼
    ┌────────────────────────┐
    │ 风险管理层 (自适应)       │
    │  波动率调整 + 仓位控制   │
    └────────────────────────┘

    使用示例:
    ```python
    system = HybridTradingSystem()
    result = system.predict(df)
    print(result["final_signal"])      # +1/-1/0
    print(result["position_size"])     # 0~0.2
    print(result["confidence"])        # 0~1
    print(result["experts_breakdown"]) # 各专家信号明细
    ```
    """

    def __init__(self, signal_threshold=0.3, max_position=0.2,
                 target_win_rate=0.7, target_win_loss_ratio=2.0):
        """
        Parameters
        ----------
        signal_threshold : float
            最低信号强度阈值（融合信号绝对值低于此值则不交易）
        max_position : float
            最大仓位比例（0.2 = 20%）
        target_win_rate : float
            目标胜率（用于Kelly公式）
        target_win_loss_ratio : float
            目标盈亏比（用于Kelly公式）
        """
        self.signal_threshold = signal_threshold
        self.max_position = max_position
        self.target_win_rate = target_win_rate
        self.target_win_loss_ratio = target_win_loss_ratio

        # 五个专家
        self.experts = {
            "trend": TrendFollowingExpert(),
            "mean_reversion": MeanReversionExpert(),
            "breakout": BreakoutExpert(),
            "volume_price": VolumePriceExpert(),
            "oi_confirmation": OIConfirmationExpert(),
        }

    def predict(self, df):
        """
        生成混合交易信号。

        流程:
        1. 各专家独立生成信号和置信度
        2. 加权投票融合信号
        3. 风险调整决定最终仓位

        Parameters
        ----------
        df : pd.DataFrame
            K线数据，需包含 close 列，可选 high, low, volume, open_interest

        Returns
        -------
        dict
            final_signal: 最终交易方向 (+1/-1/0)
            position_size: 建议仓位大小 (0~max_position)
            confidence: 综合置信度 (0~1)
            fused_signal: 融合信号原始值 (-1~+1)
            agreement: 专家一致性比例 (0~1)
            experts_breakdown: dict, 各专家的信号和置信度
        """
        # Step 1: 各专家生成信号
        expert_signals = {}
        expert_confidences = {}

        for name, expert in self.experts.items():
            signal, confidence = expert.generate_signal(df)
            expert_signals[name] = signal
            expert_confidences[name] = confidence

        # Step 2: 加权投票融合
        fused = self._ensemble_fusion(expert_signals, expert_confidences)

        # Step 3: 风险调整
        final = self._risk_adjustment(fused, df)

        return {
            "final_signal": final["signal"],
            "position_size": final["size"],
            "confidence": final["confidence"],
            "fused_signal": fused["signal"],
            "agreement": fused["agreement"],
            "experts_breakdown": {
                name: {"signal": expert_signals[name],
                       "confidence": expert_confidences[name]}
                for name in self.experts
            },
        }

    def predict_batch(self, df, window=100):
        """
        批量生成信号（滚动窗口）。

        Parameters
        ----------
        df : pd.DataFrame
            完整K线数据
        window : int
            每次计算使用的窗口大小

        Returns
        -------
        pd.DataFrame
            每个时间点的信号和置信度
        """
        results = []
        n = len(df)

        for i in range(window, n):
            df_window = df.iloc[i - window:i + 1]
            result = self.predict(df_window)
            results.append({
                "signal": result["final_signal"],
                "position_size": result["position_size"],
                "confidence": result["confidence"],
                "agreement": result["agreement"],
                "fused_signal": result["fused_signal"],
            })

        result_df = pd.DataFrame(results, index=df.index[window:])
        return result_df

    def _ensemble_fusion(self, signals, confidences):
        """
        加权投票融合。

        使用置信度作为投票权重，将多个专家信号融合为单一信号。

        Parameters
        ----------
        signals : dict
            各专家信号
        confidences : dict
            各专家置信度

        Returns
        -------
        dict
            signal: 融合信号 (-1~+1)
            agreement: 专家一致性比例
            total_confidence: 总置信度
        """
        sig_list = list(signals.values())
        conf_list = list(confidences.values())

        weighted_sum = sum(s * c for s, c in zip(sig_list, conf_list))
        total_confidence = sum(conf_list)

        if total_confidence > 0:
            fused_signal = weighted_sum / total_confidence
        else:
            fused_signal = 0.0

        fused_signal = float(np.clip(fused_signal, -1.0, 1.0))

        # 一致性（有多大比例的专家同意融合方向）
        if fused_signal != 0:
            fused_sign = np.sign(fused_signal)
            active_experts = [s for s in sig_list if s != 0]
            if active_experts:
                agreement = sum(
                    1 for s in active_experts if np.sign(s) == fused_sign
                ) / len(active_experts)
            else:
                agreement = 0.0
        else:
            agreement = 0.0

        return {
            "signal": fused_signal,
            "agreement": agreement,
            "total_confidence": total_confidence,
        }

    def _risk_adjustment(self, fused, df):
        """
        自适应风险调整。

        基于Kelly公式、波动率和信号质量计算最终仓位。

        Parameters
        ----------
        fused : dict
            融合信号结果
        df : pd.DataFrame
            K线数据

        Returns
        -------
        dict
            signal: 最终方向 (+1/-1/0)
            size: 仓位大小
            confidence: 综合置信度
        """
        close = df["close"]

        # 当前波动率
        returns = close.pct_change()
        volatility = float(returns.tail(20).std()) if len(returns) > 20 else 0.01

        # 信号强度
        signal_strength = abs(fused["signal"])

        # Kelly公式
        kelly_fraction = (
            self.target_win_rate
            - (1 - self.target_win_rate) / self.target_win_loss_ratio
        )

        # 波动率调整（高波动率时缩减仓位）
        vol_adjustment = 1.0 / (1.0 + volatility * 10.0)

        # 信号质量
        n_experts = len(self.experts)
        signal_quality = (
            fused["agreement"]
            * fused["total_confidence"]
            / n_experts
        )

        # 最终仓位
        position_size = kelly_fraction * 0.3 * vol_adjustment * signal_quality
        position_size = float(np.clip(
            position_size, 0.0, self.max_position
        ))

        # 信号方向（弱信号过滤）
        if signal_strength > self.signal_threshold:
            final_signal = int(np.sign(fused["signal"]))
        else:
            final_signal = 0
            position_size = 0.0

        return {
            "signal": final_signal,
            "size": position_size,
            "confidence": signal_quality,
        }
