"""
商品期货机器学习量化模型 - Numba加速滚动计算模块
Numba-accelerated rolling computations for high-frequency feature engineering.

提供比pandas原生函数快10-50倍的滚动计算实现。
当numba未安装时自动降级为纯numpy实现。
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def deco(func):
            return func
        return deco


@njit(cache=True)
def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Numba加速的滚动均值"""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        v = x[i]
        s += v
        if i >= window:
            s -= x[i - window]
        if i + 1 < window:
            out[i] = np.nan
        else:
            out[i] = s / window
    return out


@njit(cache=True)
def _rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    """Numba加速的滚动方差"""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    s = 0.0
    s2 = 0.0
    for i in range(n):
        v = x[i]
        s += v
        s2 += v * v
        if i >= window:
            v0 = x[i - window]
            s -= v0
            s2 -= v0 * v0
        if i + 1 < window:
            out[i] = np.nan
        else:
            mu = s / window
            out[i] = max(0.0, (s2 / window) - mu * mu)
    return out


@njit(cache=True)
def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Numba加速的滚动标准差"""
    var = _rolling_var(x, window)
    n = var.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = var[i]
        out[i] = math.sqrt(v) if not math.isnan(v) else np.nan
    return out


@njit(cache=True)
def _ewm_mean(x: np.ndarray, alpha: float) -> np.ndarray:
    """Numba加速的指数加权均值"""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    m = np.nan
    for i in range(n):
        v = x[i]
        if math.isnan(v):
            out[i] = np.nan
            continue
        if math.isnan(m):
            m = v
        else:
            m = alpha * v + (1.0 - alpha) * m
        out[i] = m
    return out


@njit(cache=True)
def _diff(x: np.ndarray, lag: int) -> np.ndarray:
    """Numba加速的差分"""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < lag:
            out[i] = np.nan
        else:
            out[i] = x[i] - x[i - lag]
    return out


@njit(cache=True)
def _pct_change(x: np.ndarray, lag: int) -> np.ndarray:
    """Numba加速的百分比变化"""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < lag:
            out[i] = np.nan
        else:
            denom = x[i - lag]
            out[i] = (x[i] / denom - 1.0) if denom != 0.0 else np.nan
    return out


@njit(cache=True)
def _log_return(x: np.ndarray, lag: int) -> np.ndarray:
    """Numba加速的对数收益率"""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < lag:
            out[i] = np.nan
        else:
            x0 = x[i - lag]
            out[i] = math.log(x[i] / x0) if x0 > 0.0 and x[i] > 0.0 else np.nan
    return out


@njit(cache=True)
def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    """Numba加速的RSI计算"""
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    gain = 0.0
    loss = 0.0
    for i in range(n):
        if i == 0:
            out[i] = np.nan
            continue
        chg = close[i] - close[i - 1]
        g = chg if chg > 0.0 else 0.0
        l_val = -chg if chg < 0.0 else 0.0
        gain += g
        loss += l_val
        if i >= window:
            chg0 = close[i - window + 1] - close[i - window]
            g0 = chg0 if chg0 > 0.0 else 0.0
            l0 = -chg0 if chg0 < 0.0 else 0.0
            gain -= g0
            loss -= l0
        if i + 1 < window:
            out[i] = np.nan
        else:
            avg_gain = gain / window
            avg_loss = loss / window
            if avg_loss == 0.0:
                out[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


@njit(cache=True)
def _macd_hist(close: np.ndarray, fast: int, slow: int, signal: int):
    """Numba加速的MACD计算，返回 (macd_line, signal_line, histogram)"""
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    alpha_sig = 2.0 / (signal + 1.0)

    ema_fast = _ewm_mean(close, alpha_fast)
    ema_slow = _ewm_mean(close, alpha_slow)

    n = close.shape[0]
    macd = np.empty(n, dtype=np.float64)
    for i in range(n):
        macd[i] = ema_fast[i] - ema_slow[i]

    sig = _ewm_mean(macd, alpha_sig)
    hist = np.empty(n, dtype=np.float64)
    for i in range(n):
        hist[i] = macd[i] - sig[i]
    return macd, sig, hist


@njit(cache=True)
def _bollinger_bandwidth(close: np.ndarray, window: int, n_std: float) -> np.ndarray:
    """Numba加速的布林带宽度"""
    ma = _rolling_mean(close, window)
    sd = _rolling_std(close, window)
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if math.isnan(ma[i]) or math.isnan(sd[i]) or ma[i] == 0.0:
            out[i] = np.nan
        else:
            upper = ma[i] + n_std * sd[i]
            lower = ma[i] - n_std * sd[i]
            out[i] = (upper - lower) / ma[i]
    return out


@njit(cache=True)
def _signed_volume_imbalance(open_: np.ndarray, close: np.ndarray,
                              volume: np.ndarray, window: int) -> np.ndarray:
    """Numba加速的签名成交量不平衡"""
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    s_signed = 0.0
    s_vol = 0.0
    for i in range(n):
        sign = 1.0 if close[i] > open_[i] else (-1.0 if close[i] < open_[i] else 0.0)
        sv = sign * volume[i]
        s_signed += sv
        s_vol += volume[i]
        if i >= window:
            sign0 = 1.0 if close[i - window] > open_[i - window] else (
                -1.0 if close[i - window] < open_[i - window] else 0.0)
            sv0 = sign0 * volume[i - window]
            s_signed -= sv0
            s_vol -= volume[i - window]
        if i + 1 < window or s_vol == 0.0:
            out[i] = np.nan
        else:
            out[i] = s_signed / s_vol
    return out


# ─── Public API (type-safe wrappers) ───


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """滚动均值（Numba加速）"""
    return _rolling_mean(x.astype(np.float64), int(window))


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差（Numba加速）"""
    return _rolling_std(x.astype(np.float64), int(window))


def pct_change(x: np.ndarray, lag: int = 1) -> np.ndarray:
    """百分比变化（Numba加速）"""
    return _pct_change(x.astype(np.float64), int(lag))


def log_return(x: np.ndarray, lag: int = 1) -> np.ndarray:
    """对数收益率（Numba加速）"""
    return _log_return(x.astype(np.float64), int(lag))


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    """RSI指标（Numba加速）"""
    return _rsi(close.astype(np.float64), int(window))


def macd_hist(close: np.ndarray, fast: int = 12, slow: int = 26,
              signal: int = 9) -> np.ndarray:
    """MACD柱状图（Numba加速）"""
    return _macd_hist(close.astype(np.float64), int(fast), int(slow), int(signal))[2]


def bollinger_bandwidth(close: np.ndarray, window: int = 20,
                        n_std: float = 2.0) -> np.ndarray:
    """布林带宽度（Numba加速）"""
    return _bollinger_bandwidth(close.astype(np.float64), int(window), float(n_std))


def signed_volume_imbalance(open_: np.ndarray, close: np.ndarray,
                             volume: np.ndarray, window: int = 20) -> np.ndarray:
    """签名成交量不平衡（Numba加速）"""
    return _signed_volume_imbalance(
        open_.astype(np.float64), close.astype(np.float64),
        volume.astype(np.float64), int(window)
    )
