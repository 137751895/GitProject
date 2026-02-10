"""
商品期货机器学习量化模型 - 智能标签生成模块
Smart label generator for multi-dimensional trading signals.

从简单二元分类升级为智能信号标签系统:
- 五级信号: strong_buy(2), weak_buy(1), neutral(0), weak_sell(-1), strong_sell(-2)
- 信号质量评分: 综合收益率幅度、持仓确认、成交量确认、波动率环境
- 动态阈值: 基于波动率自适应调整信号阈值
- 持仓确认: 持仓量变化方向与价格方向的协同性

设计理念:
    70%胜率不是靠预测涨跌，而是靠:
    1. 选择正确的交易时机（过滤低质量信号）
    2. 识别高盈亏比机会
    3. 考虑交易成本
"""

import numpy as np
import pandas as pd


class SmartLabelGenerator:
    """
    智能标签生成器。

    生成多维度交易信号标签，综合价格变化、持仓量确认和成交量确认，
    输出五级信号和信号质量评分。

    五级信号:
    - +2 (strong_buy): 价格大涨 + 持仓增 + 放量 → 强势看多
    - +1 (weak_buy): 价格上涨但缺乏持仓/成交量确认 → 弱看多
    -  0 (neutral): 变化幅度不足 → 不交易
    - -1 (weak_sell): 价格下跌但缺乏确认 → 弱看空
    - -2 (strong_sell): 价格大跌 + 持仓减 + 放量 → 强势看空

    信号质量评分 (0~1):
    - 0.4 × 收益率幅度（波动率调整后）
    - 0.3 × 持仓变化强度
    - 0.2 × 成交量确认度
    - 0.1 × 波动率环境适宜度

    使用示例:
    ```python
    generator = SmartLabelGenerator(horizon=3)
    labels = generator.create_labels(df)
    # labels包含: trading_signal, signal_strength, signal_quality,
    #             expected_return, oi_confirmation
    ```
    """

    def __init__(self, horizon=3, base_threshold=0.001,
                 strong_multiplier=2.0, vol_window=20,
                 oi_window=None, volume_window=60,
                 quality_weights=None):
        """
        Parameters
        ----------
        horizon : int
            预测未来K线数量
        base_threshold : float
            基础信号阈值（收益率绝对值超过此值才产生信号）
        strong_multiplier : float
            强信号阈值倍数（强信号 = base_threshold × multiplier）
        vol_window : int
            波动率计算窗口
        oi_window : int, optional
            持仓量滚动窗口（默认与horizon相同）
        volume_window : int
            成交量参考窗口
        quality_weights : tuple of float, optional
            信号质量评分权重 (return, oi, volume, vol_env)
            默认 (0.4, 0.3, 0.2, 0.1)
        """
        self.horizon = horizon
        self.base_threshold = base_threshold
        self.strong_multiplier = strong_multiplier
        self.vol_window = vol_window
        self.oi_window = oi_window if oi_window is not None else horizon
        self.volume_window = volume_window
        self.quality_weights = quality_weights or (0.4, 0.3, 0.2, 0.1)

    def create_labels(self, df):
        """
        生成多维度交易信号标签。

        Parameters
        ----------
        df : pd.DataFrame
            K线数据，必须包含 close 列，
            可选 open_interest 和 volume 列

        Returns
        -------
        pd.DataFrame
            trading_signal: 五级信号 (-2/-1/0/+1/+2)
            signal_strength: 信号强度 (0~2)
            signal_quality: 信号质量评分 (0~1)
            expected_return: 预期收益率
            oi_confirmation: 持仓确认方向 (-1/0/+1)
        """
        close = df["close"]

        # 1. 未来收益率
        future_price = close.shift(-self.horizon)
        raw_return = future_price / close - 1

        # 2. 波动率自适应阈值
        volatility = close.pct_change().rolling(
            self.vol_window, min_periods=5
        ).std()
        vol_median = volatility.expanding(min_periods=self.vol_window).median()
        # 波动率比值裁剪至[0.5, 2.0]，防止极端波动率产生不合理的阈值
        dynamic_threshold = self.base_threshold * (
            volatility / (vol_median + 1e-8)
        ).clip(lower=0.5, upper=2.0)

        strong_threshold = dynamic_threshold * self.strong_multiplier

        # 3. 持仓量确认
        has_oi = "open_interest" in df.columns
        if has_oi:
            oi_change = df["open_interest"].diff()
            oi_rolling = oi_change.rolling(
                self.oi_window, min_periods=1
            ).sum()
            oi_signal = np.sign(oi_rolling)
        else:
            oi_signal = pd.Series(0.0, index=df.index)

        # 4. 成交量确认
        has_vol = "volume" in df.columns
        if has_vol:
            vol_ma = df["volume"].rolling(
                self.volume_window, min_periods=10
            ).mean()
            volume_confirmed = df["volume"] > vol_ma
        else:
            volume_confirmed = pd.Series(False, index=df.index)

        # 5. 生成五级信号
        labels = pd.Series(0, index=df.index, dtype=int)

        # 强买入: 大涨 + 持仓增 + 放量
        strong_buy = (
            (raw_return > strong_threshold)
            & (oi_signal > 0)
            & volume_confirmed
        )
        labels[strong_buy] = 2

        # 弱买入: 上涨但不满足强买入条件
        weak_buy = (
            (raw_return > dynamic_threshold)
            & ~strong_buy
        )
        labels[weak_buy] = 1

        # 强卖出: 大跌 + 持仓减 + 放量
        strong_sell = (
            (raw_return < -strong_threshold)
            & (oi_signal < 0)
            & volume_confirmed
        )
        labels[strong_sell] = -2

        # 弱卖出: 下跌但不满足强卖出条件
        weak_sell = (
            (raw_return < -dynamic_threshold)
            & ~strong_sell
        )
        labels[weak_sell] = -1

        # 6. 信号质量评分
        signal_quality = self._compute_signal_quality(
            raw_return, oi_signal if has_oi else None,
            volume_confirmed if has_vol else None,
            volatility, vol_median,
        )

        return pd.DataFrame({
            "trading_signal": labels,
            "signal_strength": labels.abs(),
            "signal_quality": signal_quality,
            "expected_return": raw_return,
            "oi_confirmation": oi_signal,
        }, index=df.index)

    def _compute_signal_quality(self, returns, oi_signal, vol_confirmed,
                                volatility, vol_median):
        """
        计算信号质量评分 (0~1)。

        综合四个维度:
        1. 收益率幅度（波动率调整后，越大越好）
        2. 持仓确认度（持仓变化越大越确定）
        3. 成交量确认（有放量确认为1，否则为0）
        4. 波动率环境（适中的波动率环境最好）

        Parameters
        ----------
        returns : pd.Series
            收益率
        oi_signal : pd.Series or None
            持仓确认方向
        vol_confirmed : pd.Series or None
            是否有成交量确认
        volatility : pd.Series
            波动率
        vol_median : pd.Series
            波动率中位数

        Returns
        -------
        pd.Series
            信号质量评分 (0~1)
        """
        w_ret, w_oi, w_vol, w_env = self.quality_weights

        # 1. 收益率幅度（波动率调整后）
        return_magnitude = returns.abs() / (volatility + 1e-8)
        score_return = return_magnitude.clip(upper=3.0) / 3.0

        # 2. 持仓确认度
        if oi_signal is not None:
            score_oi = oi_signal.abs().clip(upper=1.0)
        else:
            score_oi = pd.Series(0.0, index=returns.index)

        # 3. 成交量确认
        if vol_confirmed is not None:
            score_vol = vol_confirmed.astype(float)
        else:
            score_vol = pd.Series(0.0, index=returns.index)

        # 4. 波动率环境适宜度（偏离中位数越远越差）
        vol_ratio = volatility / (vol_median + 1e-8)
        score_env = (1 - (vol_ratio - 1).abs()).clip(lower=0.0, upper=1.0)

        quality = (
            w_ret * score_return
            + w_oi * score_oi
            + w_vol * score_vol
            + w_env * score_env
        )

        return quality.clip(lower=0.0, upper=1.0)

    @staticmethod
    def get_signal_description(signal_value):
        """
        获取信号值对应的文本描述。

        Parameters
        ----------
        signal_value : int
            信号值 (-2, -1, 0, 1, 2)

        Returns
        -------
        str
            信号描述
        """
        descriptions = {
            2: "strong_buy (强势看多)",
            1: "weak_buy (弱看多)",
            0: "neutral (中性/不交易)",
            -1: "weak_sell (弱看空)",
            -2: "strong_sell (强势看空)",
        }
        return descriptions.get(signal_value, "unknown")
