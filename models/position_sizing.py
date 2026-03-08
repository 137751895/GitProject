"""
商品期货机器学习量化模型 - 风险预算与仓位管理模块
Position sizing and risk budget management.

将机器学习信号转化为实际仓位建议，
基于改进的Kelly公式和波动率自适应。

功能模块:
1. Kelly公式仓位计算 — 基于信号强度和波动率
2. 波动率自适应阈值 — 根据市场波动率动态调整
3. 回撤管理 — 跟踪账户回撤，在高回撤时自动缩减仓位
4. 日内风险限额 — 限制单日最大亏损和交易次数
5. 绩效指标 — 盈亏比(Profit Factor)、Calmar比率等
"""

import numpy as np
import pandas as pd


def calculate_position_size(signal_strength, volatility, account_risk=0.02,
                            max_position=1.0):
    """
    基于信号强度和波动率计算仓位大小。

    使用改进的Kelly公式: position = (signal * risk_budget) / volatility

    Parameters
    ----------
    signal_strength : float or np.ndarray
        模型信号强度 (0~1之间的概率值)，0.5为中性
    volatility : float or np.ndarray
        当前波动率（ATR或收益率标准差）
    account_risk : float
        每笔交易的最大风险比例（默认2%）
    max_position : float
        最大仓位限制（1.0 = 100%）

    Returns
    -------
    float or np.ndarray
        建议仓位大小 (-max_position ~ +max_position)
    """
    # 将信号强度映射为方向和强度 (-1 ~ +1)
    directional_signal = 2 * (signal_strength - 0.5)

    # Kelly公式变体
    safe_vol = np.where(
        np.asarray(volatility) > 0, volatility, 1e-6
    )
    position = (directional_signal * account_risk) / safe_vol

    return np.clip(position, -max_position, max_position)


def adaptive_threshold(base_threshold, volatility, vol_mean,
                       sensitivity=1.0):
    """
    根据波动率自适应调整交易信号阈值。

    高波动率时提高阈值（减少交易），低波动率时降低阈值。

    Parameters
    ----------
    base_threshold : float
        基础阈值（如0.55）
    volatility : float or np.ndarray
        当前波动率
    vol_mean : float
        波动率均值（参考水平）
    sensitivity : float
        灵敏度因子

    Returns
    -------
    float or np.ndarray
        调整后的阈值
    """
    if vol_mean <= 0:
        return base_threshold
    vol_ratio = volatility / vol_mean
    adjustment = sensitivity * (vol_ratio - 1.0) * 0.1
    adjusted = base_threshold + adjustment
    return np.clip(adjusted, 0.5, 0.9)


class DrawdownTracker:
    """
    回撤跟踪器。

    实时跟踪账户权益的最大回撤，当回撤超过阈值时
    自动缩减仓位比例，实现风险保护。

    缩减公式:
    - 回撤 < warning_level: 正常仓位 (乘数=1.0)
    - warning_level <= 回撤 < max_drawdown: 线性缩减 (1.0→0.0)
    - 回撤 >= max_drawdown: 仓位为零 (乘数=0.0)
    """

    def __init__(self, max_drawdown=0.15, warning_level=0.08,
                 initial_equity=1000000.0):
        """
        Parameters
        ----------
        max_drawdown : float
            最大允许回撤比例（默认15%，达到后仓位降为零）
        warning_level : float
            回撤预警水平（默认8%，开始缩减仓位）
        initial_equity : float
            初始权益
        """
        self.max_drawdown = max_drawdown
        self.warning_level = warning_level
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self._history = []

    @property
    def current_drawdown(self):
        """当前回撤比例"""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def position_multiplier(self):
        """
        基于回撤的仓位缩减乘数 (0.0~1.0)。

        - 回撤在warning_level以下: 1.0（全仓可用）
        - 回撤在warning_level到max_drawdown之间: 线性缩减
        - 回撤达到max_drawdown: 0.0（停止开仓）
        """
        dd = self.current_drawdown
        if dd <= self.warning_level:
            return 1.0
        if dd >= self.max_drawdown:
            return 0.0
        # 线性插值
        return (self.max_drawdown - dd) / (self.max_drawdown - self.warning_level)

    def update_equity(self, pnl):
        """
        更新权益。

        Parameters
        ----------
        pnl : float
            本期盈亏金额

        Returns
        -------
        dict
            回撤状态
        """
        self.equity += pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        status = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.current_drawdown,
            "position_multiplier": self.position_multiplier,
        }
        self._history.append(status)
        return status

    def get_history(self):
        """获取回撤历史"""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history)

    def reset(self, equity=None):
        """重置跟踪器"""
        if equity is not None:
            self.equity = equity
        self.peak_equity = self.equity
        self._history.clear()


class DailyRiskLimit:
    """
    日内风险限额管理。

    限制单个交易日内的最大亏损和最大交易次数，
    防止在极端行情中过度暴露。

    达到限额后的行为:
    - 达到亏损限额: 当日停止开新仓，已有仓位保持
    - 达到交易次数限额: 当日停止新交易
    - 新的交易日自动重置
    """

    def __init__(self, max_daily_loss=0.03, max_daily_trades=20):
        """
        Parameters
        ----------
        max_daily_loss : float
            单日最大亏损比例（相对于初始权益，默认3%）
        max_daily_trades : int
            单日最大交易次数
        """
        self.max_daily_loss = max_daily_loss
        self.max_daily_trades = max_daily_trades
        self._current_date = None
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._initial_equity = 0.0

    @property
    def is_loss_limit_reached(self):
        """是否达到日内亏损限额"""
        if self._initial_equity <= 0:
            return False
        return (-self._daily_pnl / self._initial_equity) >= self.max_daily_loss

    @property
    def is_trade_limit_reached(self):
        """是否达到日内交易次数限额"""
        return self._daily_trades >= self.max_daily_trades

    @property
    def can_trade(self):
        """是否可以开新仓"""
        return not self.is_loss_limit_reached and not self.is_trade_limit_reached

    def check_and_reset(self, current_date, current_equity):
        """
        检查日期变化并在新交易日重置计数。

        Parameters
        ----------
        current_date : date-like
            当前日期
        current_equity : float
            当前权益

        Returns
        -------
        bool
            是否发生了日期重置
        """
        if self._current_date is None or current_date != self._current_date:
            self._current_date = current_date
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._initial_equity = current_equity
            return True
        return False

    def record_trade(self, pnl):
        """
        记录一笔交易。

        Parameters
        ----------
        pnl : float
            交易盈亏

        Returns
        -------
        dict
            日内风险状态
        """
        self._daily_trades += 1
        self._daily_pnl += pnl
        return {
            "daily_trades": self._daily_trades,
            "daily_pnl": self._daily_pnl,
            "loss_limit_reached": self.is_loss_limit_reached,
            "trade_limit_reached": self.is_trade_limit_reached,
            "can_trade": self.can_trade,
        }


def compute_performance_metrics(returns):
    """
    计算交易绩效指标。

    Parameters
    ----------
    returns : np.ndarray or pd.Series
        交易收益率序列

    Returns
    -------
    dict
        绩效指标:
        - profit_factor: 盈亏比（总盈利/总亏损，>1.8为优秀）
        - calmar_ratio: Calmar比率（年化收益/最大回撤）
        - win_rate: 胜率
        - avg_win: 平均盈利
        - avg_loss: 平均亏损
        - max_drawdown: 最大回撤
        - sharpe_ratio: 夏普比率（年化）
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return {
            "profit_factor": 0.0, "calmar_ratio": 0.0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_drawdown": 0.0, "sharpe_ratio": 0.0,
        }

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    # 盈亏比
    total_profit = np.sum(wins) if len(wins) > 0 else 0.0
    total_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else 999.0

    # 胜率
    n_trades = np.sum(returns != 0)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

    # 平均盈利/亏损
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    # 最大回撤
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # 年化收益率和夏普比率（假设252个交易日）
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    annualized_return = mean_return * 252
    sharpe_ratio = (mean_return / std_return * np.sqrt(252)
                    if std_return > 0 else 0.0)

    # Calmar比率
    calmar_ratio = (annualized_return / max_drawdown
                    if max_drawdown > 0 else 0.0)

    return {
        "profit_factor": float(profit_factor),
        "calmar_ratio": float(calmar_ratio),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
    }


class RiskBudgetManager:
    """
    风险预算管理器。

    综合信号强度、波动率和账户风险约束，
    计算最优仓位大小和交易阈值。

    增强功能:
    - 回撤保护: 当回撤超过阈值时自动缩减仓位
    - 日内限额: 限制单日最大亏损和交易次数
    - 绩效监控: 实时计算盈亏比、Calmar比率等
    """

    def __init__(self, account_risk=0.02, max_position=1.0,
                 base_threshold=0.55,
                 max_drawdown=0.15, drawdown_warning=0.08,
                 max_daily_loss=0.03, max_daily_trades=20, max_position_usage=0.9):  # [新增]
        """
        Parameters
        ----------
        account_risk : float
            每笔交易最大风险比例
        max_position : float
            最大仓位限制
        base_threshold : float
            基础信号阈值
        max_drawdown : float
            最大允许回撤（默认15%）
        drawdown_warning : float
            回撤预警水平（默认8%）
        max_daily_loss : float
            单日最大亏损比例（默认3%）
        max_daily_trades : int
            单日最大交易次数
        """
        self.account_risk = account_risk
        self.max_position = max_position
        self.base_threshold = base_threshold
        self.max_position_usage = float(max_position_usage)  # [新增]
        if not 0 < self.max_position_usage <= 1:  # [新增]
            self.max_position_usage = 0.9  # [新增]
        self.drawdown_tracker = DrawdownTracker(
            max_drawdown=max_drawdown,
            warning_level=drawdown_warning,
        )
        self.daily_limit = DailyRiskLimit(
            max_daily_loss=max_daily_loss,
            max_daily_trades=max_daily_trades,
        )
        self._trade_returns = []

    def compute_signals(self, probabilities, volatility):
        """
        将模型预测概率转化为交易信号和仓位。

        Parameters
        ----------
        probabilities : np.ndarray
            模型预测概率（上涨概率）
        volatility : np.ndarray
            波动率序列

        Returns
        -------
        dict
            包含 position_size, threshold, signal 的字典
        """
        vol_mean = np.nanmean(volatility)

        # 自适应阈值
        threshold = adaptive_threshold(
            self.base_threshold, volatility, vol_mean
        )

        # 交易信号
        signal = np.where(probabilities > threshold, 1,
                          np.where(probabilities < (1 - threshold), -1, 0))

        # 仓位大小
        position = calculate_position_size(
            probabilities, volatility,
            self.account_risk, self.max_position
        )

        # 无信号时仓位为零
        position = np.where(signal != 0, position, 0.0)

        # 回撤保护缩减
        dd_multiplier = self.drawdown_tracker.position_multiplier
        position = position * dd_multiplier

        position = position * self.max_position_usage  # [新增]

        return {
            "position_size": position,
            "threshold": threshold,
            "signal": signal,
            "probabilities": probabilities,
            "drawdown_multiplier": dd_multiplier,
            "current_drawdown": self.drawdown_tracker.current_drawdown,
        }

    def record_trade_result(self, pnl, trade_date=None):
        """
        记录交易结果，更新回撤和日内限额。

        Parameters
        ----------
        pnl : float
            交易盈亏
        trade_date : date-like, optional
            交易日期（用于日内限额重置）

        Returns
        -------
        dict
            风险管理状态
        """
        # 更新回撤
        dd_status = self.drawdown_tracker.update_equity(pnl)

        # 记录收益率用于绩效计算（基于交易前权益）
        pre_trade_equity = self.drawdown_tracker.equity - pnl
        if pre_trade_equity > 0:
            self._trade_returns.append(pnl / pre_trade_equity)

        # 更新日内限额
        daily_status = None
        if trade_date is not None:
            current_equity = self.drawdown_tracker.equity
            self.daily_limit.check_and_reset(trade_date, current_equity)
            daily_status = self.daily_limit.record_trade(pnl)

        return {
            "drawdown": dd_status,
            "daily_limit": daily_status,
            "can_trade": self.daily_limit.can_trade,
        }

    def get_performance_metrics(self):
        """
        获取当前绩效指标。

        Returns
        -------
        dict
            绩效指标（盈亏比、Calmar比率、胜率等）
        """
        if not self._trade_returns:
            return compute_performance_metrics(np.array([]))
        return compute_performance_metrics(np.array(self._trade_returns))

    def get_risk_summary(self):
        """
        获取风险管理综合摘要。

        Returns
        -------
        dict
            综合摘要
        """
        perf = self.get_performance_metrics()
        return {
            "equity": self.drawdown_tracker.equity,
            "peak_equity": self.drawdown_tracker.peak_equity,
            "current_drawdown": self.drawdown_tracker.current_drawdown,
            "position_multiplier": self.drawdown_tracker.position_multiplier,
            "n_trades": len(self._trade_returns),
            "can_trade": self.daily_limit.can_trade,
            **perf,
        }
