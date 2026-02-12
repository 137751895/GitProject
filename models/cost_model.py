# ===== 修改说明 =====
# 依据《当前项目性能提升指南-最终版v2.md》改进点 1.1
# 新增内容已用 # [新增] 标记
# ===================
"""
商品期货机器学习量化模型 - 交易成本与止盈止损模块
Cost model and stop-loss/take-profit rules.
"""

from __future__ import annotations  # [新增]

from dataclasses import dataclass  # [新增]
from typing import Tuple  # [BUGFIX] P2-1: import Tuple for <3.9 compat


@dataclass(frozen=True)  # [新增]
class CostConfig:  # [新增]
    buy_rate: float = 0.0003  # [新增]
    buy_min: float = 5.0  # [新增]
    sell_rate: float = 0.0003  # [新增]
    sell_min: float = 5.0  # [新增]
    stamp_duty: float = 0.001  # [新增]


@dataclass(frozen=True)  # [新增]
class StopConfig:  # [新增]
    stop_loss_rate: float = -0.03  # [新增]
    stop_profit_rate: float = 0.05  # [新增]


def calc_buy_cost(trade_value: float, cfg: CostConfig) -> float:  # [新增]
    service_change = trade_value * cfg.buy_rate  # [新增]
    if service_change < cfg.buy_min:  # [新增]
        service_change = cfg.buy_min  # [新增]
    return float(service_change)  # [新增]


def calc_sell_cash_in(trade_value: float, cfg: CostConfig) -> Tuple[float, float, float]:  # [新增]  # [BUGFIX] P2-1: Tuple from typing
    service_change = trade_value * cfg.sell_rate  # [新增]
    if service_change < cfg.sell_min:  # [新增]
        service_change = cfg.sell_min  # [新增]
    stamp_duty = cfg.stamp_duty * trade_value  # [新增]
    cash_in = trade_value - service_change - stamp_duty  # [新增]
    total_cost = service_change + stamp_duty  # [新增]
    return float(cash_in), float(total_cost), float(stamp_duty)  # [新增]


def should_stop(entry_price: float, current_price: float, side: int, cfg: StopConfig) -> Tuple[bool, float]:  # [新增]  # [BUGFIX] P2-1
    if entry_price <= 0:  # [新增]
        return False, 0.0  # [新增]
    raw_ret = (current_price - entry_price) / entry_price  # [新增]
    signed_ret = raw_ret if side > 0 else -raw_ret  # [新增]
    hit_profit = signed_ret >= cfg.stop_profit_rate  # [新增]
    hit_loss = signed_ret <= cfg.stop_loss_rate  # [新增]
    return bool(hit_profit or hit_loss), float(signed_ret)  # [新增]
