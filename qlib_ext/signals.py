"""
HFML-Qlib 信号模块（修正版）
============================
HFMLHybridSignal : 继承 Qlib Signal 基类，将 HybridTradingSystem 封装为标准 Qlib 信号。

修正点（对应 7.6 问题）：
- HFMLHybridSignal 现在继承 qlib.backtest.signal.Signal，符合 Qlib 标准扩展点
- 保留向后兼容的 get_signal() 便捷接口供 HFMLHybridStrategy 直接调用
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


try:
    from qlib.backtest.signal import Signal as QlibSignal

    class HFMLHybridSignal(QlibSignal):
        """HFML 混合专家系统信号封装（继承 Qlib Signal 基类）。

        将 HybridTradingSystem 的多专家融合信号暴露为 Qlib 标准 Signal 扩展点，
        同时保留 get_signal() 接口供 HFMLHybridStrategy 内部调用。

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
            """设置行情数据。"""
            self._market_data = market_data.copy()

        def get_signal(self, end_time=None) -> dict:
            """计算当前时刻的混合信号（便捷接口，供策略层调用）。

            Parameters
            ----------
            end_time : datetime-like or None
                截止时间点，默认使用全部行情数据。

            Returns
            -------
            dict
                包含 final_signal, position_size, confidence 等键。
            """
            return self._get_hybrid_signal(end_time=end_time)

        # ------------------------------------------------------------------
        # Qlib Signal 协议实现
        # ------------------------------------------------------------------

        def get_signal(  # type: ignore[no-redef]
            self,
            trade_start_time=None,
            trade_end_time=None,
            trade_exchange=None,
        ):
            """Qlib Signal.get_signal 协议方法（覆盖上方便捷方法）。

            Qlib 在回测时会调用此方法以获取每个 bar 的信号。
            本实现委托给内部 HybridTradingSystem。

            Returns
            -------
            pd.Series or None
                若无信号则返回 None；有信号则返回带标的代码索引的 Series。
            """
            sig = self._get_hybrid_signal(end_time=trade_end_time)
            if sig is None or sig.get("final_signal", 0) == 0:
                return None
            # 返回格式：pd.Series，索引为 instrument，值为信号分数
            return pd.Series(
                [float(sig.get("fused_signal", sig.get("final_signal", 0.0)))],
                index=["HFML_FUTURES"],
                name="signal",
            )

        def _get_hybrid_signal(self, end_time=None) -> dict:
            """内部辅助，等同于旧版 get_signal()。"""
            if self._market_data.empty:
                return {"final_signal": 0, "position_size": 0.0, "confidence": 0.0}

            df = self._market_data
            if end_time is not None:
                try:
                    df = df[df.index <= pd.Timestamp(end_time)]
                except Exception:
                    pass
            df = df.tail(self.window)

            try:
                result = self.system.predict(df)
                return result if isinstance(result, dict) else {"final_signal": 0, "position_size": 0.0}
            except Exception as exc:
                logger.warning(f"HFMLHybridSignal._get_hybrid_signal 异常: {exc}")
                return {"final_signal": 0, "position_size": 0.0, "confidence": 0.0}

except ImportError:
    # Qlib 未安装时的纯 Python 实现
    class HFMLHybridSignal:  # type: ignore[no-redef]
        """占位实现（Qlib 未安装）。"""

        def __init__(self, window: int = 100, **kwargs):
            from models.hybrid_trading import HybridTradingSystem
            self.window = window
            self.system = HybridTradingSystem(**kwargs)
            self._market_data: pd.DataFrame = pd.DataFrame()

        def set_market_data(self, market_data: pd.DataFrame) -> None:
            self._market_data = market_data.copy()

        def get_signal(self, end_time=None) -> dict:
            if self._market_data.empty:
                return {"final_signal": 0, "position_size": 0.0, "confidence": 0.0}
            df = self._market_data
            if end_time is not None:
                try:
                    df = df[df.index <= pd.Timestamp(end_time)]
                except Exception:
                    pass
            df = df.tail(self.window)
            try:
                result = self.system.predict(df)
                return result if isinstance(result, dict) else {"final_signal": 0, "position_size": 0.0}
            except Exception as exc:
                logger.warning(f"HFMLHybridSignal.get_signal 异常: {exc}")
                return {"final_signal": 0, "position_size": 0.0, "confidence": 0.0}

        def _get_hybrid_signal(self, end_time=None) -> dict:
            return self.get_signal(end_time=end_time)
