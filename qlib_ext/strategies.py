"""
HFML-Qlib 策略模块（修正版）
============================
HFMLHybridStrategy         : 基于 HFMLHybridSignal 的单时间框架混合专家策略（含风控）。
HFMLMultiTimeframeStrategy : 基于 MultiTimeframeCoordinator 的三周期协同策略（修正版）。

修正点：
7.5 - HFMLMultiTimeframeStrategy：不再将标量分数传给 generate_signals()，
      改为构建轻量级"预测包装模型"并通过 coordinator.set_models() 注入，
      再传入适当大小的特征 stub 调用 generate_signals()。
7.7 - HFMLHybridStrategy：接入 RiskBudgetManager / DrawdownTracker / DailyRiskLimit，
      实现期货多空执行语义（开多/平多/开空/平空）。
"""

import copy
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _load_pred_from_recorder(recorder_id: str, artifact_name: str = "pred.pkl") -> pd.DataFrame:
    """从 Qlib Recorder 加载预测结果。"""
    from qlib.workflow import R
    rec = R.get_recorder(recorder_id=recorder_id)
    pred = rec.load_object(artifact_name)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if not isinstance(pred.index, pd.MultiIndex):
        raise ValueError(f"recorder '{recorder_id}' 的预测结果索引必须为 MultiIndex(datetime, instrument)")
    return pred.sort_index()


def _normalize_pred(pred) -> pd.DataFrame:
    """规范化预测对象为带 MultiIndex 的 DataFrame。"""
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if isinstance(pred, pd.DataFrame):
        return pred.sort_index()
    raise TypeError(f"不支持的预测类型: {type(pred)}")


def _slice_pred(pred_df: pd.DataFrame, dt, instrument: str) -> Optional[float]:
    """从 MultiIndex 预测 DataFrame 中取特定时刻与标的的值（含向前填充回退）。"""
    try:
        row = pred_df.loc[(dt, instrument)]
        return float(row.iloc[0]) if isinstance(row, pd.Series) else float(row)
    except KeyError:
        pass
    try:
        sub = pred_df.xs(instrument, level=1)
        loc = sub.index.get_indexer([dt], method="ffill")[0]
        if loc >= 0:
            return float(sub.iloc[loc, 0])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 轻量级"预测分数注入"包装模型
# 用于将标量 score 注入 MultiTimeframeCoordinator 的内部子模型
# ---------------------------------------------------------------------------

class _ScorePredictorWrapper:
    """将单个预测分数包装为 predict_proba 兼容的伪模型。

    MultiTimeframeCoordinator 的各子模型（TrendDirectionModel,
    EntryTimingModel, EntryPriceOptimizer）内部通过
    model.predict_proba(X) 获取概率。本包装器存储预先计算的标量 score，
    并在调用 predict_proba 时返回等长的概率数组，无论 X 的内容如何。
    """

    def __init__(self, score: float):
        """
        Parameters
        ----------
        score : float
            [0, 1] 区间内的上涨概率。
        """
        self.score = float(score)

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        col_pos = np.full(n, self.score)
        col_neg = 1.0 - col_pos
        return np.column_stack([col_neg, col_pos])

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return (np.full(n, self.score) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# HFMLHybridStrategy（含风控）
# ---------------------------------------------------------------------------

try:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.decision import Order, TradeDecisionWO

    class HFMLHybridStrategy(BaseStrategy):
        """基于 HFMLHybridSignal 的单时间框架混合专家策略（含期货风控）。

        修正内容：
        - 接入 RiskBudgetManager（仓位计算、自适应阈值、回撤保护）
        - 接入 DailyRiskLimit（日内亏损与交易次数限制）
        - 实现期货执行语义：区分开多/平多/开空/平空

        Parameters
        ----------
        signal : HFMLHybridSignal
            已配置好行情数据的信号对象。
        instrument : str
            交易标的代码。
        market_data : pd.DataFrame, optional
            若提供，调用 signal.set_market_data 初始化行情。
        risk_kwargs : dict, optional
            传递给 RiskBudgetManager 的参数（account_risk, max_position,
            base_threshold, max_drawdown, max_daily_loss 等）。
        **kwargs :
            传递给 BaseStrategy 的参数。
        """

        def __init__(self, signal, instrument: str, market_data=None,
                     risk_kwargs: dict = None, **kwargs):
            super().__init__(**kwargs)
            self.signal = signal
            self.instrument = instrument
            if market_data is not None:
                self.signal.set_market_data(market_data)

            # 风控初始化
            from models.position_sizing import RiskBudgetManager
            self._risk_mgr = RiskBudgetManager(**(risk_kwargs or {}))
            self._last_trade_date: Optional[date] = None

        def generate_trade_decision(self, execute_result=None):
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

            # 处理执行结果（更新风控状态）
            if execute_result:
                for order, trade_val, cost, trade_price in execute_result:
                    if order is not None:
                        pnl = (trade_val - cost) if order.direction == Order.BUY else (trade_val + cost)
                        try:
                            _td = trade_start_time.date() if hasattr(trade_start_time, "date") else None
                            self._risk_mgr.record_trade_result(pnl, trade_date=_td)
                        except Exception:
                            pass

            # 日内风控检查
            try:
                current_equity = self._get_current_equity()
                _td = trade_start_time.date() if hasattr(trade_start_time, "date") else None
                self._risk_mgr.daily_limit.check_and_reset(_td, current_equity)
                if not self._risk_mgr.daily_limit.can_trade:
                    logger.info("日内风控触发（亏损/次数限额），跳过本 bar 交易")
                    return TradeDecisionWO([], self)
            except Exception:
                pass

            # 获取信号
            sig = self.signal._get_hybrid_signal(end_time=trade_end_time)
            return self._build_trade_decision(sig, trade_start_time, trade_end_time)

        def _get_current_equity(self) -> float:
            try:
                return float(self.trade_position.calculate_value())
            except Exception:
                return float(self._risk_mgr.drawdown_tracker.equity)

        def _build_trade_decision(self, sig: dict, trade_start_time, trade_end_time):
            if sig is None:
                return TradeDecisionWO([], self)

            instrument    = self.instrument
            final_signal  = int(sig.get("final_signal", 0))
            raw_pos_size  = float(sig.get("position_size", 0.0))

            # 通过 RiskBudgetManager 调整仓位大小（含回撤保护）
            dd_multiplier = self._risk_mgr.drawdown_tracker.position_multiplier
            target_ratio  = float(np.clip(raw_pos_size * dd_multiplier, 0.0, self._risk_mgr.max_position))

            # 获取当前持仓（期货多头/空头方向）
            try:
                current_long  = self.trade_position.get_stock_amount(instrument)
            except Exception:
                current_long = 0.0

            # 中性信号：平掉现有仓位
            if final_signal == 0:
                return self._close_all(instrument, current_long, trade_start_time, trade_end_time)

            # 获取成交价
            direction = Order.BUY if final_signal > 0 else Order.SELL
            deal_price = self._get_deal_price(instrument, trade_start_time, trade_end_time, direction)
            if deal_price is None or deal_price <= 0:
                return TradeDecisionWO([], self)

            # 合约乘数
            try:
                contract_factor = self.trade_exchange.get_factor(
                    stock_id=instrument,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:
                contract_factor = 1

            # 计算目标仓位金额
            try:
                cash = self.trade_position.get_cash()
            except Exception:
                cash = self._risk_mgr.drawdown_tracker.equity
            target_value  = cash * target_ratio
            target_amount = target_value / deal_price if deal_price > 0 else 0.0
            target_amount = self._round_amount(target_amount, contract_factor)

            orders = []

            if final_signal > 0:
                # 期货做多：区分"开多"和"平空回补"
                if current_long < 0:
                    # 当前持空头，先平空（买入回补）
                    close_short = abs(current_long)
                    close_short = self._round_amount(close_short, contract_factor)
                    if close_short > 0:
                        orders.append(Order(
                            stock_id=instrument, amount=close_short,
                            start_time=trade_start_time, end_time=trade_end_time,
                            direction=Order.BUY,
                        ))
                # 开多（或增多）
                delta = max(0.0, target_amount - max(0.0, current_long))
                if delta > 0:
                    orders.append(Order(
                        stock_id=instrument, amount=delta,
                        start_time=trade_start_time, end_time=trade_end_time,
                        direction=Order.BUY,
                    ))

            elif final_signal < 0:
                # 期货做空：区分"开空"和"平多"
                if current_long > 0:
                    # 当前持多头，先平多（卖出）
                    close_long = current_long
                    close_long = self._round_amount(close_long, contract_factor)
                    if close_long > 0:
                        orders.append(Order(
                            stock_id=instrument, amount=close_long,
                            start_time=trade_start_time, end_time=trade_end_time,
                            direction=Order.SELL,
                        ))
                # 开空（卖出）
                short_amount = self._round_amount(target_amount, contract_factor)
                if short_amount > 0:
                    orders.append(Order(
                        stock_id=instrument, amount=short_amount,
                        start_time=trade_start_time, end_time=trade_end_time,
                        direction=Order.SELL,
                    ))

            # 过滤不可执行订单
            executable = []
            for order in orders:
                try:
                    if self.trade_exchange.check_order(order):
                        executable.append(order)
                except Exception:
                    executable.append(order)

            return TradeDecisionWO(executable, self)

        def _close_all(self, instrument, current_long, trade_start_time, trade_end_time):
            """平掉所有仓位（多头卖出，空头回补）。"""
            if abs(current_long) < 1e-9:
                return TradeDecisionWO([], self)

            direction = Order.SELL if current_long > 0 else Order.BUY
            deal_price = self._get_deal_price(instrument, trade_start_time, trade_end_time, direction)
            if deal_price is None or deal_price <= 0:
                return TradeDecisionWO([], self)

            try:
                contract_factor = self.trade_exchange.get_factor(
                    stock_id=instrument,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            except Exception:
                contract_factor = 1

            close_amount = self._round_amount(abs(current_long), contract_factor)
            if close_amount <= 0:
                return TradeDecisionWO([], self)

            order = Order(
                stock_id=instrument, amount=close_amount,
                start_time=trade_start_time, end_time=trade_end_time,
                direction=direction,
            )
            try:
                if not self.trade_exchange.check_order(order):
                    return TradeDecisionWO([], self)
            except Exception:
                pass
            return TradeDecisionWO([order], self)

        def _get_deal_price(self, instrument, start_time, end_time, direction):
            try:
                return self.trade_exchange.get_deal_price(
                    stock_id=instrument,
                    start_time=start_time,
                    end_time=end_time,
                    direction=direction,
                )
            except Exception:
                return None

        def _round_amount(self, amount, contract_factor=1):
            try:
                return self.trade_exchange.round_amount_by_trade_unit(amount, contract_factor)
            except Exception:
                return amount

    # ---------------------------------------------------------------------------
    # HFMLMultiTimeframeStrategy（修正版）
    # ---------------------------------------------------------------------------

    class HFMLMultiTimeframeStrategy(BaseStrategy):
        """三周期协同策略（修正版）。

        修正点（对应 7.5）：
        - 不再把标量分数直接传给 generate_signals()
        - 使用 _ScorePredictorWrapper 将每个周期的预测分数注入为伪模型
        - 通过 coordinator.set_models() 设置三路伪模型
        - 构造只有1行的 feature stub DataFrame，触发 generate_signals() 执行
        - 取结果的最后一个元素（与当前 bar 对应）

        Parameters
        ----------
        instrument : str
            交易标的代码。
        pred_15m / pred_5m / pred_1m : pd.DataFrame or pd.Series, optional
            直接传入的各周期预测结果。
        recorder_id_15m / recorder_id_5m / recorder_id_1m : str, optional
            各周期 recorder ID（与 pred_Xm 二选一）。
        coordinator_kwargs : dict, optional
            传递给 MultiTimeframeCoordinator 的参数（neutral_zone,
            entry_threshold, optimization_threshold, min_grade）。
        risk_kwargs : dict, optional
            传递给 RiskBudgetManager 的参数。
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
            risk_kwargs=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.instrument = instrument
            self.pred_15m = self._init_pred(pred_15m, recorder_id_15m)
            self.pred_5m  = self._init_pred(pred_5m,  recorder_id_5m)
            self.pred_1m  = self._init_pred(pred_1m,  recorder_id_1m)

            from models.multi_timeframe import MultiTimeframeCoordinator
            self.coordinator = MultiTimeframeCoordinator(**(coordinator_kwargs or {}))

            from models.position_sizing import RiskBudgetManager
            self._risk_mgr = RiskBudgetManager(**(risk_kwargs or {}))

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

            # 获取各周期预测分数（标量）
            score_15m = _slice_pred(self.pred_15m, trade_start_time, instrument)
            score_5m  = _slice_pred(self.pred_5m,  trade_start_time, instrument)
            score_1m  = _slice_pred(self.pred_1m,  trade_start_time, instrument)

            if any(s is None for s in [score_15m, score_5m, score_1m]):
                return TradeDecisionWO([], self)

            # 日内风控检查
            try:
                if not self._risk_mgr.daily_limit.can_trade:
                    return TradeDecisionWO([], self)
            except Exception:
                pass

            # 将标量分数注入协调器作为伪模型
            self.coordinator.set_models(
                model_15min=_ScorePredictorWrapper(score_15m),
                model_5min=_ScorePredictorWrapper(score_5m),
                model_1min =_ScorePredictorWrapper(score_1m),
            )

            # 构造1行 feature stub（内容无实际意义，伪模型忽略特征）
            stub = pd.DataFrame({"_x": [0.0]})
            try:
                result = self.coordinator.generate_signals(
                    X_15min=stub,
                    X_5min=stub,
                    X_1min=stub,
                )
            except Exception as exc:
                logger.warning(f"MultiTimeframeCoordinator.generate_signals 异常: {exc}")
                return TradeDecisionWO([], self)

            # 取生成结果的最后一个信号
            final_sig_arr = result.get("final_signal", [0])
            final_signal  = int(final_sig_arr[-1]) if hasattr(final_sig_arr, "__len__") else int(final_sig_arr)

            if final_signal == 0:
                return TradeDecisionWO([], self)

            # 回撤保护
            dd_mult   = self._risk_mgr.drawdown_tracker.position_multiplier
            sig_grade = result.get("signal_grade", ["-"])
            grade     = sig_grade[-1] if hasattr(sig_grade, "__len__") else sig_grade
            strength_arr = result.get("signal_strength", [0.5])
            strength  = float(strength_arr[-1]) if hasattr(strength_arr, "__len__") else float(strength_arr)

            # 仓位大小 = 信号强度 × 回撤系数
            position_ratio = float(np.clip(strength * dd_mult, 0.0, self._risk_mgr.max_position))

            direction = Order.BUY if final_signal > 0 else Order.SELL
            order = Order(
                stock_id=instrument,
                amount=position_ratio,       # 实际使用时由 Exchange 结合账户资金换算
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=direction,
            )
            return TradeDecisionWO([order], self)

except ImportError:
    # Qlib 未安装时的占位实现
    class HFMLHybridStrategy:  # type: ignore[no-redef]
        def __init__(self, signal, instrument, market_data=None, risk_kwargs=None, **kwargs):
            self.signal     = signal
            self.instrument = instrument

        def generate_trade_decision(self, execute_result=None):
            return []

    class HFMLMultiTimeframeStrategy:  # type: ignore[no-redef]
        def __init__(self, instrument, **kwargs):
            self.instrument = instrument

        def generate_trade_decision(self, execute_result=None):
            return []
