"""
商品期货机器学习量化模型 - 风险预算与仓位管理模块
Position sizing and risk budget management.

将机器学习信号转化为实际仓位建议，
基于改进的Kelly公式和波动率自适应。
"""

import numpy as np


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


class RiskBudgetManager:
    """
    风险预算管理器。

    综合信号强度、波动率和账户风险约束，
    计算最优仓位大小和交易阈值。
    """

    def __init__(self, account_risk=0.02, max_position=1.0,
                 base_threshold=0.55):
        """
        Parameters
        ----------
        account_risk : float
            每笔交易最大风险比例
        max_position : float
            最大仓位限制
        base_threshold : float
            基础信号阈值
        """
        self.account_risk = account_risk
        self.max_position = max_position
        self.base_threshold = base_threshold

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

        return {
            "position_size": position,
            "threshold": threshold,
            "signal": signal,
            "probabilities": probabilities,
        }
