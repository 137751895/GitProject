"""
HFML-Qlib 策略模块
==================
HFMLHybridStrategy         : 基于 HFMLHybridSignal 的单时间框架混合专家策略。
HFMLMultiTimeframeStrategy : 基于 MultiTimeframeCoordinator 的三周期协同策略。

两个策略均继承 qlib.strategy.base.BaseStrategy，实现
generate_trade_decision 接口。
"""

import copy
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _load_pred_from_recorder(recorder_id: str, artifact_name: str = "pred.pkl") -> pd.DataFrame:
    """从 Qlib Recorder 加载预测结果。

    Parameters
    ----------
    recorder_id : str
        实验记录 ID。
    artifact_name : str
        artifact 文件名，默认 "pred.pkl"。

    Returns
    -------
    pd.DataFrame
        带 MultiIndex(datetime, instrument) 的预测 DataFrame。
    """
    from qlib.workflow import R

    rec = R.get_recorder(recorder_id=recorder_id)
    pred = rec.load_object(artifact_name)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if not isinstance(pred.index, pd.MultiIndex):
        raise ValueError(
            f"recorder '{recorder_id}' 的预测结果索引必须为 MultiIndex(datetime, instrument)"
        )
    return pred.sort_index()


def _normalize_pred(pred) -> pd.DataFrame:
    """规范化预测对象为带 MultiIndex 的 DataFrame。"""
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if isinstance(pred, pd.DataFrame):
        return pred.sort_index()
    raise TypeError(f"不支持的预测类型: {type(pred)}")


def _slice_pred(pred_df: pd.DataFrame, dt, instrument: str):
    """从 MultiIndex 预测 DataFrame 中取特定时刻与标的的值。"""
    try:
        row = pred_df.loc[(dt, instrument)]
        return float(row.iloc[0]) if isinstance(row, pd.Series) else float(row)
    except KeyError:
        # 尝试 asof 风格查找（最近可用预测）
        try:
            sub = pred_df.xs(instrument, level=1)
            loc = sub.index.get_indexer([dt], method="ffill")[0]
            if loc >= 0:
                return float(sub.iloc[loc, 0])
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# HFMLHybridStrategy
# ---------------------------------------------------------------------------

try:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.decision import Order, TradeDecisionWO

    class HFMLHybridStrategy(BaseStrategy):
        """基于 HFMLHybridSignal 的单时间框架混合专家策略。

        Parameters
        ----------
        signal : HFMLHybridSignal
            已配置好行情数据的信号对象。
        instrument : str
            交易标的代码（如 "KQi@SHFEag"）。
        market_data : pd.DataFrame, optional
            若提供，调用 signal.set_market_data 初始化行情。
        **kwargs :
            传递给 BaseStrategy 的参数（trade_exchange 等）。
        """

        def __init__(self, signal, instrument: str, market_data=None, **kwargs):
            super().__init__(**kwargs)
            self.signal = signal
            self.instrument = instrument
            if market_data is not None:
                self.signal.set_market_data(market_data)

        def generate_trade_decision(self, execute_result=None):
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

            sig = self.signal.get_signal(end_time=trade_end_time)
            return self._build_trade_decision(sig, trade_start_time, trade_end_time)

        def _build_trade_decision(self, sig: dict, trade_start_time, trade_end_time):
            if sig is None:
                return TradeDecisionWO([], self)

            instrument = self.instrument
            final_signal = int(sig.get("final_signal", 0))
            target_ratio = float(
                max(0.0, min(1.0, sig.get("position_size", 0.0)))
            )

            # 当前持仓
            try:
                current_pos = copy.deepcopy(self.trade_position)
                current_amount = current_pos.get_stock_amount(instrument)
                cash = current_pos.get_cash()
            except Exception as exc:
                logger.warning(f"获取持仓失败: {exc}")
                return TradeDecisionWO([], self)

            if final_signal == 0:
                # 中性信号：平掉现有仓位
                if current_amount <= 0:
                    return TradeDecisionWO([], self)
                try:
                    deal_price = self.trade_exchange.get_deal_price(
                        stock_id=instrument,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.SELL,
                    )
                except Exception:
                    deal_price = None
                if deal_price is None or deal_price <= 0:
                    return TradeDecisionWO([], self)
                try:
                    contract_factor = self.trade_exchange.get_factor(
                        stock_id=instrument,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    )
                    sell_amount = self.trade_exchange.round_amount_by_trade_unit(
                        current_amount, contract_factor
                    )
                except Exception:
                    sell_amount = current_amount
                if sell_amount > 0:
                    close_order = Order(
                        stock_id=instrument,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.SELL,
                    )
                    try:
                        if self.trade_exchange.check_order(close_order):
                            return TradeDecisionWO([close_order], self)
                    except Exception:
                        return TradeDecisionWO([close_order], self)
                return TradeDecisionWO([], self)

            # 成交价
            direction = Order.BUY if final_signal > 0 else Order.SELL
            try:
                deal_price = self.trade_exchange.get_deal_price(
                    stock_id=instrument,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=direction,
                )
            except Exception:
                deal_price = None

            if deal_price is None or deal_price <= 0:
                return TradeDecisionWO([], self)

            # 合约乘数 / 最小交易单位
            try:
                contract_factor = self.trade_exchange.get_factor(
                    stock_id=instrument,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:
                contract_factor = 1

            target_value = cash * target_ratio
            target_amount = target_value / deal_price
            try:
                target_amount = self.trade_exchange.round_amount_by_trade_unit(
                    target_amount, contract_factor
                )
            except Exception:
                pass

            order_list = []
            if final_signal > 0:
                delta = max(0.0, target_amount - current_amount)
                if delta > 0:
                    order_list.append(Order(
                        stock_id=instrument,
                        amount=delta,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.BUY,
                    ))
            elif final_signal < 0:
                sell_amount = current_amount if current_amount > 0 else target_amount
                try:
                    sell_amount = self.trade_exchange.round_amount_by_trade_unit(
                        sell_amount, contract_factor
                    )
                except Exception:
                    pass
                if sell_amount > 0:
                    order_list.append(Order(
                        stock_id=instrument,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.SELL,
                    ))

            # 过滤不可执行订单
            executable = []
            for order in order_list:
                try:
                    if self.trade_exchange.check_order(order):
                        executable.append(order)
                except Exception:
                    executable.append(order)

            return TradeDecisionWO(executable, self)

    class HFMLMultiTimeframeStrategy(BaseStrategy):
        """三周期协同策略，融合 15 分钟/5 分钟/1 分钟预测。

        Parameters
        ----------
        pred_15m : pd.DataFrame or pd.Series, optional
            直接传入的 15 分钟预测结果。
        pred_5m : pd.DataFrame or pd.Series, optional
            直接传入的 5 分钟预测结果。
        pred_1m : pd.DataFrame or pd.Series, optional
            直接传入的 1 分钟预测结果。
        recorder_id_15m : str, optional
            15 分钟实验的 recorder_id（与 pred_15m 二选一）。
        recorder_id_5m : str, optional
            5 分钟实验的 recorder_id。
        recorder_id_1m : str, optional
            1 分钟实验的 recorder_id。
        coordinator_kwargs : dict, optional
            传递给 MultiTimeframeCoordinator 的参数。
        instrument : str
            交易标的代码。
        **kwargs :
            传递给 BaseStrategy 的参数。
        """

        def __init__(
            self,
            instrument: str,
            pred_15m=None,
            pred_5m=None,
            pred_1m=None,
            recorder_id_15m=None,
            recorder_id_5m=None,
            recorder_id_1m=None,
            coordinator_kwargs=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.instrument = instrument
            self.pred_15m = self._init_pred(pred_15m, recorder_id_15m)
            self.pred_5m = self._init_pred(pred_5m, recorder_id_5m)
            self.pred_1m = self._init_pred(pred_1m, recorder_id_1m)

            from models.multi_timeframe import MultiTimeframeCoordinator
            self.coordinator = MultiTimeframeCoordinator(**(coordinator_kwargs or {}))

        def _init_pred(self, pred, recorder_id):
            if pred is not None:
                return _normalize_pred(pred)
            if recorder_id is None:
                return pd.DataFrame()
            return _load_pred_from_recorder(recorder_id)

        def generate_trade_decision(self, execute_result=None):
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
            instrument = self.instrument

            pred_15m = _slice_pred(self.pred_15m, trade_start_time, instrument)
            pred_5m = _slice_pred(self.pred_5m, trade_start_time, instrument)
            pred_1m = _slice_pred(self.pred_1m, trade_start_time, instrument)

            if any(p is None for p in [pred_15m, pred_5m, pred_1m]):
                return TradeDecisionWO([], self)

            try:
                fused = self.coordinator.generate_signals(
                    trend_pred=pred_15m,
                    entry_pred=pred_5m,
                    execution_pred=pred_1m,
                )
            except Exception as exc:
                logger.warning(f"MultiTimeframeCoordinator.generate_signals 异常: {exc}")
                return TradeDecisionWO([], self)

            final_signal = fused.get("final_signal", 0)
            if final_signal == 0:
                return TradeDecisionWO([], self)

            amount = max(0.0, float(fused.get("position_size", 0.0)))
            direction = Order.BUY if final_signal > 0 else Order.SELL
            order = Order(
                stock_id=instrument,
                amount=amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=direction,
            )
            return TradeDecisionWO([order], self)

except ImportError:
    # Qlib 未安装时的占位实现
    class HFMLHybridStrategy:  # type: ignore[no-redef]
        def __init__(self, signal, instrument, market_data=None, **kwargs):
            self.signal = signal
            self.instrument = instrument

        def generate_trade_decision(self, execute_result=None):
            return []

    class HFMLMultiTimeframeStrategy:  # type: ignore[no-redef]
        def __init__(self, instrument, **kwargs):
            self.instrument = instrument

        def generate_trade_decision(self, execute_result=None):
            return []
