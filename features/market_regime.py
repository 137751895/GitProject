"""
商品期货机器学习量化模型 - 市场状态识别模块
Market regime detection for commodity futures ML model.

识别趋势/震荡/反转等不同市场状态，为模型提供额外特征。
市场状态分类:
- high_vol_trend: 高波动趋势市
- low_vol_range: 低波动震荡市
- high_vol_range: 高波动震荡市（反转可能）
- low_vol_trend: 低波动趋势市（平稳趋势）
"""

import numpy as np
import pandas as pd


class MarketRegimeDetector:
    """
    市场状态识别器。

    基于波动率水平和趋势强度将市场分为四种状态，
    为模型提供市场环境的上下文信息。
    """

    def __init__(self, vol_window=20, trend_window=20,
                 vol_quantile_high=0.7, vol_quantile_low=0.3,
                 trend_threshold=0.3):
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
        """
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_quantile_high = vol_quantile_high
        self.vol_quantile_low = vol_quantile_low
        self.trend_threshold = trend_threshold

    def detect_regime(self, df):
        """
        识别市场状态。

        Parameters
        ----------
        df : pd.DataFrame
            K线数据，必须包含 close 列

        Returns
        -------
        pd.DataFrame
            包含市场状态特征的DataFrame
        """
        close = df["close"]
        returns = close.pct_change()

        result = pd.DataFrame(index=df.index)

        # 波动率水平
        volatility = returns.rolling(window=self.vol_window).std()
        result["regime_volatility"] = volatility

        # 趋势强度（收益率与时间的相关性）
        trend_strength = close.rolling(window=self.trend_window).apply(
            self._calc_trend_strength, raw=True
        )
        result["trend_strength"] = trend_strength

        # 波动率分位数（滚动）
        vol_expanding = volatility.expanding(min_periods=self.vol_window)
        vol_rank = vol_expanding.rank(pct=True)
        result["volatility_rank"] = vol_rank

        # 市场状态编码
        high_vol = vol_rank > self.vol_quantile_high
        low_vol = vol_rank < self.vol_quantile_low
        strong_trend = trend_strength.abs() > self.trend_threshold

        # 四状态分类: 0=low_vol_range, 1=low_vol_trend, 2=high_vol_range, 3=high_vol_trend
        conditions = [
            high_vol & strong_trend,      # high_vol_trend
            low_vol & ~strong_trend,      # low_vol_range
            high_vol & ~strong_trend,     # high_vol_range
            low_vol & strong_trend,       # low_vol_trend
        ]
        choices = [3, 0, 2, 1]
        result["market_regime"] = np.select(conditions, choices, default=1)

        # 趋势方向
        result["trend_direction"] = np.sign(trend_strength)

        # 波动率变化率
        result["volatility_change"] = volatility.pct_change(periods=5)

        # 趋势加速度（趋势强度的变化）
        result["trend_acceleration"] = trend_strength.diff(periods=5)

        return result

    @staticmethod
    def _calc_trend_strength(values):
        """计算趋势强度（价格与时间序列的相关系数）"""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        # Pearson相关系数
        x_mean = x.mean()
        y_mean = values.mean()
        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((values - y_mean) ** 2))
        if denominator == 0:
            return 0.0
        return numerator / denominator
