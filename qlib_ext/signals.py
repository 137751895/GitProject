"""
HFML-Qlib 信号模块
==================
HFMLHybridSignal : 将 HybridTradingSystem 封装为可在 Strategy 内调用的信号生成器。

注意：此模块提供信号计算能力，实际交易决策由 strategies.py 中的 Strategy 完成。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class HFMLHybridSignal:
    """HFML 混合专家系统信号封装。

    将 HybridTradingSystem 的多专家融合信号暴露为可在
    HFMLHybridStrategy.generate_trade_decision 中调用的接口。

    Parameters
    ----------
    window : int
        滚动窗口大小（取最近 N 行行情计算信号）。
    **kwargs :
        传递给 HybridTradingSystem 的参数（signal_threshold,
        max_position, target_win_rate, target_win_loss_ratio 等）。
    """

    def __init__(self, window: int = 100, **kwargs):
        from models.hybrid_trading import HybridTradingSystem
        self.window = window
        self.system = HybridTradingSystem(**kwargs)
        self._market_data: pd.DataFrame = pd.DataFrame()

    def set_market_data(self, market_data: pd.DataFrame) -> None:
        """设置行情数据（在 Strategy 初始化或每 bar 更新时调用）。

        Parameters
        ----------
        market_data : pd.DataFrame
            K 线数据，索引为 datetime，列包含 close/open/high/low/volume 等。
        """
        self._market_data = market_data.copy()

    def get_signal(self, end_time=None) -> dict:
        """计算当前时刻的混合信号。

        Parameters
        ----------
        end_time : datetime-like or None
            截止时间点，默认使用全部行情数据。

        Returns
        -------
        dict
            包含 final_signal, position_size, confidence,
            fused_signal, agreement, experts_breakdown 等键。
        """
        if self._market_data.empty:
            logger.warning("HFMLHybridSignal: 行情数据为空，返回中性信号")
            return {"final_signal": 0, "position_size": 0.0, "confidence": 0.0}

        df = self._market_data
        if end_time is not None:
            df = df[df.index <= pd.Timestamp(end_time)]
        df = df.tail(self.window)

        try:
            result = self.system.predict(df)
            return result if isinstance(result, dict) else {"final_signal": 0, "position_size": 0.0}
        except Exception as exc:
            logger.warning(f"HFMLHybridSignal.get_signal 异常: {exc}")
            return {"final_signal": 0, "position_size": 0.0, "confidence": 0.0}
