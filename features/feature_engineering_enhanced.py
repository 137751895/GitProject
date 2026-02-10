"""
商品期货机器学习量化模型 - 增强特征工程模块
Enhanced feature engineering with microstructure, multi-timeframe, and numba-accelerated features.

新增特征类别:
- 微观结构特征: divergence, vwap_dev, vol_state, mom_slope, close_pos, rsi_slope, vol_zscore
- 高级波动率特征: volatility_regime, vol_ratio, atr_pct, range_pct
- 缺口衰减特征: gap_decay (针对期货夜盘)
"""

import numpy as np
import pandas as pd

from features.rolling_numba import (
    rolling_mean,
    rolling_std,
    rsi as nb_rsi,
)


def compute_microstructure_features(df):
    """
    计算微观结构特征。

    Parameters
    ----------
    df : pd.DataFrame
        K线数据，必须包含 open, high, low, close, volume, open_interest

    Returns
    -------
    pd.DataFrame
        微观结构特征
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    oi = df["open_interest"].values.astype(np.float64)

    result = {}

    # 1. divergence: 资金流向与价格背离
    pos_chg = np.diff(oi, prepend=np.nan)
    price_chg = np.diff(close, prepend=np.nan)
    result["divergence"] = np.sign(pos_chg) * np.sign(price_chg)

    # 2. vwap_dev: 价格与成交量加权均价的偏离
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    vwap = cum_tp_vol / (cum_vol + 1e-12)
    result["vwap_dev"] = (close - vwap) / (vwap + 1e-12)

    # 3. vol_state: 波动率状态（ATR14 / 60日均价，标准化）
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.nan_to_num(tr, nan=0.0)
    atr_14 = rolling_mean(tr, 14)
    result["vol_state"] = atr_14 / (rolling_mean(close, 60) + 1e-12)

    # 4. mom_slope: 动量斜率（5周期价格斜率 / 当前价格）
    slope = _rolling_slope(close, 5)
    result["mom_slope"] = slope / (close + 1e-12)

    # 5. close_pos: K线内相对位置
    hl_range = high - low
    result["close_pos"] = (close - low) / (hl_range + 1e-12)

    # 6. rsi_slope: RSI的5周期斜率
    rsi_14 = nb_rsi(close, 14)
    result["rsi_slope"] = _rolling_slope(rsi_14, 5)

    # 7. vol_zscore: 成交量Z分数
    vol_ma = rolling_mean(volume, 20)
    vol_sd = rolling_std(volume, 20)
    result["vol_zscore"] = (volume - vol_ma) / (vol_sd + 1e-12)

    return pd.DataFrame(result, index=df.index)


def compute_advanced_volatility_features(df):
    """
    计算高级波动率特征。

    Parameters
    ----------
    df : pd.DataFrame
        K线数据

    Returns
    -------
    pd.DataFrame
        高级波动率特征
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    result = {}

    # 收益率
    ret = np.empty_like(close)
    ret[0] = np.nan
    ret[1:] = np.log(close[1:] / close[:-1])

    # 1. volatility_regime: 收益率的20周期滚动标准差
    result["volatility_regime"] = rolling_std(ret, 20)

    # 2. vol_ratio: ATR14 / ATR14的20周期均值
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - prev_close), np.abs(low - prev_close)
    ))
    tr = np.nan_to_num(tr, nan=0.0)
    atr_14 = rolling_mean(tr, 14)
    atr_14_filled = np.nan_to_num(atr_14, nan=0.0)
    result["vol_ratio"] = atr_14 / (rolling_mean(atr_14_filled, 20) + 1e-12)

    # 3. atr_pct: ATR / 价格 (百分比化)
    result["atr_pct"] = atr_14 / (close + 1e-12)

    # 4. range_pct: 振幅百分比
    result["range_pct"] = (high - low) / (close + 1e-12)

    return pd.DataFrame(result, index=df.index)


def compute_gap_decay_features(df):
    """
    计算缺口衰减特征（针对期货夜盘）。

    夜盘开盘后的缺口会随时间指数衰减，
    衰减因子 = exp(-经过秒数 / decay_constant)

    Parameters
    ----------
    df : pd.DataFrame
        K线数据，index需为DatetimeIndex

    Returns
    -------
    pd.DataFrame
        缺口衰减特征
    """
    from config import ENHANCED_FEATURE_CONFIG
    gap_cfg = ENHANCED_FEATURE_CONFIG.get("gap_decay", {})
    night_hour = gap_cfg.get("night_session_start_hour", 21)
    night_max_min = gap_cfg.get("night_session_start_max_minute", 30)
    decay_const = gap_cfg.get("decay_constant", 300)

    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)

    gap_decay = np.zeros(len(df), dtype=np.float64)

    # 尝试获取datetime信息
    if isinstance(df.index, pd.DatetimeIndex):
        dt = df.index
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
    else:
        # 无法确定时间，返回全零
        return pd.DataFrame({"gap_decay": gap_decay}, index=df.index)

    for i in range(len(df)):
        t = dt[i]
        if t.hour == night_hour and t.minute < night_max_min:
            prev_close = close[i - 1] if i > 0 else np.nan
            if np.isfinite(prev_close) and prev_close > 0:
                gap = (open_[i] - prev_close) / prev_close
                seconds = t.minute * 60 + t.second
                gap_decay[i] = gap * float(np.exp(-seconds / decay_const))

    return pd.DataFrame({"gap_decay": gap_decay}, index=df.index)


def _rolling_slope(y: np.ndarray, window: int) -> np.ndarray:
    """
    计算滚动线性回归斜率。

    Parameters
    ----------
    y : np.ndarray
        输入数组
    window : int
        窗口大小

    Returns
    -------
    np.ndarray
        滚动斜率
    """
    n = y.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2:
        return out
    x = np.arange(window, dtype=np.float64)
    x_mean = (window - 1) / 2.0
    var_x = np.sum((x - x_mean) ** 2)
    if abs(var_x) < 1e-12:
        return out
    for i in range(window - 1, n):
        y_win = y[i - window + 1: i + 1]
        if np.any(~np.isfinite(y_win)):
            continue
        y_mean = float(np.mean(y_win))
        cov = float(np.dot(x - x_mean, y_win - y_mean))
        out[i] = cov / var_x
    return out
