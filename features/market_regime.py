"""
商品期货机器学习量化模型 - 市场状态识别模块
Market regime detection for commodity futures ML model.

识别趋势/震荡/反转等不同市场状态，为模型提供额外特征。

市场状态分类（五状态模型）:
- 0: low_vol_range  — 低波动震荡市（窄幅整理，等待突破）
- 1: low_vol_trend  — 低波动趋势市（平稳趋势，顺势操作）
- 2: high_vol_range — 高波动震荡市（宽幅震荡，区间操作）
- 3: high_vol_trend — 高波动趋势市（强势行情，趋势跟踪）
- 4: reversal       — 反转信号（趋势即将反转，警惕风险）

反转检测基于以下信号的综合判断：
- RSI极端值（超买>70 或 超卖<30）
- 价格偏离均线过远
- 趋势强度突然减弱（趋势加速度反向）
"""

import numpy as np
import pandas as pd


# 市场状态标签映射
REGIME_LABELS = {
    0: "low_vol_range",
    1: "low_vol_trend",
    2: "high_vol_range",
    3: "high_vol_trend",
    4: "reversal",
}


class MarketRegimeDetector:
    """
    市场状态识别器。

    基于波动率水平、趋势强度和反转信号将市场分为五种状态，
    为模型提供市场环境的上下文信息。

    五种状态及其交易建议:
    - low_vol_range (0):  低波动震荡 → 观望或做区间交易
    - low_vol_trend (1):  低波动趋势 → 顺势轻仓操作
    - high_vol_range (2): 高波动震荡 → 宽幅区间交易、注意止损
    - high_vol_trend (3): 高波动趋势 → 趋势跟踪、加大仓位
    - reversal (4):       反转信号   → 平仓或反向操作
    """

    def __init__(self, vol_window=20, trend_window=20,
                 vol_quantile_high=0.7, vol_quantile_low=0.3,
                 trend_threshold=0.3,
                 rsi_window=14, rsi_overbought=70, rsi_oversold=30,
                 ma_deviation_threshold=0.02,
                 reversal_lookback=5,
                 trend_accel_threshold=0.1,
                 reversal_weights=(0.4, 0.3, 0.3)):
        """
        Parameters
        ----------
        vol_window : int
            波动率计算窗口
        trend_window : int
            趋势强度计算窗口
        vol_quantile_high : float
            高波动率分位数阈值
        vol_quantile_low : float
            低波动率分位数阈值
        trend_threshold : float
            趋势强度阈值
        rsi_window : int
            RSI计算窗口（反转检测用）
        rsi_overbought : float
            RSI超买阈值
        rsi_oversold : float
            RSI超卖阈值
        ma_deviation_threshold : float
            价格偏离均线阈值（百分比，如0.02表示2%）
        reversal_lookback : int
            反转信号回顾窗口
        trend_accel_threshold : float
            趋势加速度反向阈值
        reversal_weights : tuple of float
            反转信号三个分量的权重 (RSI权重, 均线偏离权重, 趋势减弱权重)
        """
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_quantile_high = vol_quantile_high
        self.vol_quantile_low = vol_quantile_low
        self.trend_threshold = trend_threshold
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.ma_deviation_threshold = ma_deviation_threshold
        self.reversal_lookback = reversal_lookback
        self.trend_accel_threshold = trend_accel_threshold
        self.reversal_weights = reversal_weights

    def detect_regime(self, df):
        """
        识别市场状态（含反转检测）。

        Parameters
        ----------
        df : pd.DataFrame
            K线数据，必须包含 close 列

        Returns
        -------
        pd.DataFrame
            包含市场状态特征的DataFrame，输出列:
            - regime_volatility: 滚动波动率
            - trend_strength: 趋势强度（-1~+1，Pearson相关系数）
            - volatility_rank: 波动率百分位排名
            - market_regime: 五状态编码（0~4）
            - trend_direction: 趋势方向（-1/0/+1）
            - volatility_change: 波动率5周期变化率
            - trend_acceleration: 趋势加速度
            - reversal_score: 反转信号强度（0~1）
            - regime_duration: 当前状态持续K线数
            - regime_change_prob: 状态转换概率（滚动）
        """
        close = df["close"]
        returns = close.pct_change()

        result = pd.DataFrame(index=df.index)

        # ── 1. 波动率水平 ──
        volatility = returns.rolling(window=self.vol_window).std()
        result["regime_volatility"] = volatility

        # ── 2. 趋势强度（收益率与时间的相关性）──
        trend_strength = close.rolling(window=self.trend_window).apply(
            self._calc_trend_strength, raw=True
        )
        result["trend_strength"] = trend_strength

        # ── 3. 波动率分位数（滚动）──
        vol_expanding = volatility.expanding(min_periods=self.vol_window)
        vol_rank = vol_expanding.rank(pct=True)
        result["volatility_rank"] = vol_rank

        # ── 4. 反转信号检测 ──
        reversal_score = self._detect_reversal(df, trend_strength)
        result["reversal_score"] = reversal_score

        # ── 5. 五状态分类 ──
        high_vol = vol_rank > self.vol_quantile_high
        low_vol = vol_rank < self.vol_quantile_low
        strong_trend = trend_strength.abs() > self.trend_threshold
        is_reversal = reversal_score >= 0.5

        # 五状态分类:
        # 反转信号优先级最高（当反转分数>=0.5时覆盖其他状态）
        conditions = [
            is_reversal,                  # reversal (4)
            high_vol & strong_trend,      # high_vol_trend (3)
            high_vol & ~strong_trend,     # high_vol_range (2)
            low_vol & strong_trend,       # low_vol_trend (1)
            low_vol & ~strong_trend,      # low_vol_range (0)
        ]
        choices = [4, 3, 2, 1, 0]
        result["market_regime"] = np.select(conditions, choices, default=1)

        # ── 6. 趋势方向 ──
        result["trend_direction"] = np.sign(trend_strength)

        # ── 7. 波动率变化率 ──
        result["volatility_change"] = volatility.pct_change(periods=5)

        # ── 8. 趋势加速度 ──
        result["trend_acceleration"] = trend_strength.diff(periods=5)

        # ── 9. 状态持续时间 ──
        result["regime_duration"] = self._calc_regime_duration(
            result["market_regime"]
        )

        # ── 10. 状态转换概率 ──
        result["regime_change_prob"] = self._calc_regime_change_prob(
            result["market_regime"], window=self.trend_window
        )

        return result

    def _detect_reversal(self, df, trend_strength):
        """
        检测反转信号。

        综合以下三种信号判断反转可能性:
        1. RSI极端值（超买/超卖）
        2. 价格偏离均线过远
        3. 趋势加速度反向（趋势在减弱）

        Parameters
        ----------
        df : pd.DataFrame
            K线数据
        trend_strength : pd.Series
            趋势强度序列

        Returns
        -------
        pd.Series
            反转分数（0~1），值越大反转可能性越高
        """
        close = df["close"]
        n = len(close)
        score = pd.Series(0.0, index=df.index)
        w_rsi, w_dev, w_trend = self.reversal_weights

        # 信号1: RSI超买超卖
        rsi = self._calc_rsi(close, self.rsi_window)
        rsi_extreme = pd.Series(0.0, index=df.index)
        rsi_extreme[rsi > self.rsi_overbought] = (
            (rsi[rsi > self.rsi_overbought] - self.rsi_overbought)
            / (100 - self.rsi_overbought)
        )
        rsi_extreme[rsi < self.rsi_oversold] = (
            (self.rsi_oversold - rsi[rsi < self.rsi_oversold])
            / self.rsi_oversold
        )
        score += rsi_extreme * w_rsi

        # 信号2: 价格偏离均线
        ma = close.rolling(window=self.trend_window).mean()
        deviation = (close - ma) / ma.replace(0, np.nan)
        dev_signal = (deviation.abs() > self.ma_deviation_threshold).astype(float)
        max_dev_multiplier = 2.0
        dev_magnitude = (
            deviation.abs() / self.ma_deviation_threshold
        ).clip(upper=max_dev_multiplier) / max_dev_multiplier
        score += dev_signal * dev_magnitude * w_dev

        # 信号3: 趋势加速度反向（趋势正在减弱）
        trend_accel = trend_strength.diff(periods=self.reversal_lookback)
        trend_weakening = (
            (trend_strength > self.trend_threshold) & (trend_accel < -self.trend_accel_threshold)
        ) | (
            (trend_strength < -self.trend_threshold) & (trend_accel > self.trend_accel_threshold)
        )
        score += trend_weakening.astype(float) * w_trend

        return score.clip(lower=0.0, upper=1.0)

    @staticmethod
    def _calc_rsi(close, window=14):
        """计算RSI指标（独立实现，不依赖外部模块）"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calc_regime_duration(regime_series):
        """
        计算当前市场状态持续的K线数量。

        Parameters
        ----------
        regime_series : pd.Series
            市场状态编码序列

        Returns
        -------
        pd.Series
            每根K线上当前状态已持续的周期数
        """
        values = regime_series.values
        n = len(values)
        duration = np.ones(n, dtype=float)
        for i in range(1, n):
            if values[i] == values[i - 1]:
                duration[i] = duration[i - 1] + 1
            else:
                duration[i] = 1
        return pd.Series(duration, index=regime_series.index)

    @staticmethod
    def _calc_regime_change_prob(regime_series, window=20):
        """
        计算市场状态转换概率（滚动窗口内状态改变的频率）。

        Parameters
        ----------
        regime_series : pd.Series
            市场状态编码序列
        window : int
            滚动窗口

        Returns
        -------
        pd.Series
            滚动窗口内的状态转换概率（0~1）
        """
        changes = (regime_series != regime_series.shift(1)).astype(float)
        prob = changes.rolling(window=window, min_periods=1).mean()
        return prob

    @staticmethod
    def _calc_trend_strength(values):
        """计算趋势强度（价格与时间序列的相关系数）"""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = values.mean()
        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sqrt(
            np.sum((x - x_mean) ** 2) * np.sum((values - y_mean) ** 2)
        )
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def get_regime_label(code):
        """
        获取市场状态编码对应的文本标签。

        Parameters
        ----------
        code : int
            状态编码（0~4）

        Returns
        -------
        str
            状态标签
        """
        return REGIME_LABELS.get(code, "unknown")
