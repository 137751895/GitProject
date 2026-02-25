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

    if "amount" in df.columns and "vol" in df.columns:  # [新增]
        amount = df["amount"].values.astype(np.float64)  # [新增]
        vol = df["vol"].values.astype(np.float64)  # [新增]
        # [BUGFIX] P1-4: document unit conversion constants
        # amount is in 元, vol is in 手; 1手=100股, amount单位千元→元需×1000
        vwap_amount = (amount * 1000.0) / (vol * 100.0 + 1e-12)  # [新增]  # [BUGFIX] P1-4: use epsilon instead of +1.0
        result["vwap_amount"] = vwap_amount  # [新增]
        result["vwap_amount_dev"] = (close - vwap_amount) / (vwap_amount + 1e-12)  # [新增]

    required = {"best_bid", "best_ask", "buy_depth", "sell_depth"}  # [新增]
    if required.issubset(set(df.columns)):  # [新增]
        best_bid = df["best_bid"].values.astype(np.float64)  # [新增]
        best_ask = df["best_ask"].values.astype(np.float64)  # [新增]
        buy_depth = df["buy_depth"].values.astype(np.float64)  # [新增]
        sell_depth = df["sell_depth"].values.astype(np.float64)  # [新增]
        mid = (best_bid + best_ask) / 2.0  # [新增]
        micro_num = best_ask * buy_depth + best_bid * sell_depth  # [新增]
        micro_den = buy_depth + sell_depth  # [新增]
        microprice = micro_num / (micro_den + 1e-12)  # [新增]
        result["micro_bias"] = (microprice - mid) / (mid + 1e-12)  # [新增]
        ofi_raw = np.diff(buy_depth, prepend=np.nan) - np.diff(sell_depth, prepend=np.nan)  # [新增]
        ofi_raw = np.nan_to_num(ofi_raw, nan=0.0)  # [新增]
        result["ofi_raw"] = ofi_raw  # [新增]
        result["ofi_ema"] = pd.Series(ofi_raw, index=df.index).ewm(span=8, adjust=False).mean().values  # [新增]

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
    from features.feature_context import FeatureContext  # [新增]

    ctx = FeatureContext(df)  # [新增]
    close = ctx.close  # [新增]
    high = ctx.high  # [新增]
    low = ctx.low  # [新增]
    prev_close = ctx.prev_close  # [新增]
    tr = ctx.tr  # [新增]
    atr_14 = ctx.atr_14  # [新增]

    result = {}

    # 收益率
    ret = np.empty_like(close)
    ret[0] = np.nan
    ret[1:] = np.log(close[1:] / close[:-1])

    # 1. volatility_regime: 收益率的20周期滚动标准差
    result["volatility_regime"] = rolling_std(ret, 20)

    # 2. vol_ratio: ATR14 / ATR14的20周期均值
    atr_14_filled = np.nan_to_num(atr_14, nan=0.0)
    result["vol_ratio"] = atr_14 / (rolling_mean(atr_14_filled, 20) + 1e-12)

    # 3. atr_pct: ATR / 价格 (百分比化)
    result["atr_pct"] = atr_14 / (close + 1e-12)

    # 4. range_pct: 振幅百分比
    result["range_pct"] = (high - low) / (close + 1e-12)

    result["trange"] = tr  # [新增]
    result["natr_14"] = atr_14 / (close + 1e-12)  # [新增]
    result["trange_pct"] = tr / (close + 1e-12)  # [新增]

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


try:                                                              # [新增]
    from numba import njit as _njit                               # [新增]
except ImportError:                                               # [新增]
    def _njit(*args, **kwargs):                                   # [新增]
        def deco(func):                                           # [新增]
            return func                                           # [新增]
        return deco                                               # [新增]


@_njit(cache=True)                                                # [新增]
def _rolling_slope_numba(arr, window):                            # [新增]
    """Numba加速版本的滚动斜率计算。

    Parameters
    ----------
    arr : np.ndarray
        输入数组 (float64)
    window : int
        窗口大小

    Returns
    -------
    np.ndarray
        滚动斜率
    """                                                           # [新增]
    n = len(arr)                                                  # [新增]
    out = np.full(n, np.nan, dtype=np.float64)                    # [新增]
    if window < 2:                                                # [新增]
        return out                                                # [新增]
                                                                  # [新增]
    x = np.arange(window, dtype=np.float64)                       # [新增]
    x_mean = (window - 1) / 2.0                                   # [新增]
    var_x = 0.0                                                   # [新增]
    for j in range(window):                                       # [新增]
        var_x += (x[j] - x_mean) ** 2                             # [新增]
    if abs(var_x) < 1e-12:                                        # [新增]
        return out                                                # [新增]
                                                                  # [新增]
    for i in range(window - 1, n):                                # [新增]
        y = arr[i - window + 1: i + 1]                            # [新增]
        has_nan = False                                           # [新增]
        for j in range(window):                                   # [新增]
            if np.isnan(y[j]):                                    # [新增]
                has_nan = True                                    # [新增]
                break                                             # [新增]
        if has_nan:                                               # [新增]
            continue                                              # [新增]
        y_mean = 0.0                                              # [新增]
        for j in range(window):                                   # [新增]
            y_mean += y[j]                                        # [新增]
        y_mean /= window                                          # [新增]
        cov = 0.0                                                 # [新增]
        for j in range(window):                                   # [新增]
            cov += (x[j] - x_mean) * (y[j] - y_mean)             # [新增]
        out[i] = cov / var_x                                      # [新增]
    return out                                                    # [新增]


def _rolling_slope(y, window):                                    # [新增]
    """计算滚动线性回归斜率（自动使用Numba加速）。

    Parameters
    ----------
    y : array-like
        输入数组
    window : int
        窗口大小

    Returns
    -------
    np.ndarray
        滚动斜率
    """                                                           # [新增]
    arr = np.asarray(y, dtype=np.float64)                         # [新增]
    return _rolling_slope_numba(arr, window)                      # [新增]


@_njit(cache=True)                                                # [新增]
def _rolling_autocorr_numba(arr, window, lag=1):                  # [新增]
    """Numba加速的滚动自相关系数计算。

    Parameters
    ----------
    arr : np.ndarray
        输入数组 (float64)
    window : int
        窗口大小
    lag : int
        滞后阶数

    Returns
    -------
    np.ndarray
        滚动自相关系数
    """                                                           # [新增]
    n = len(arr)                                                  # [新增]
    out = np.full(n, np.nan, dtype=np.float64)                    # [新增]
                                                                  # [新增]
    for i in range(window, n):                                    # [新增]
        y = arr[i - window: i]                                    # [新增]
                                                                  # [新增]
        has_nan = False                                           # [新增]
        for j in range(window):                                   # [新增]
            if np.isnan(y[j]):                                    # [新增]
                has_nan = True                                    # [新增]
                break                                             # [新增]
        if has_nan:                                               # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        mean = 0.0                                                # [新增]
        for j in range(window):                                   # [新增]
            mean += y[j]                                          # [新增]
        mean /= window                                            # [新增]
                                                                  # [新增]
        var = 0.0                                                 # [新增]
        for j in range(window):                                   # [新增]
            var += (y[j] - mean) ** 2                             # [新增]
        if var < 1e-12:                                           # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        cov = 0.0                                                 # [新增]
        for j in range(window - lag):                             # [新增]
            cov += (y[j] - mean) * (y[j + lag] - mean)           # [新增]
                                                                  # [新增]
        out[i] = cov / var                                        # [新增]
                                                                  # [新增]
    return out                                                    # [新增]


@_njit(cache=True)                                                # [新增]
def _rolling_corr_numba(x, y, window):                            # [新增]
    """Numba加速的滚动相关系数计算。

    Parameters
    ----------
    x, y : np.ndarray
        输入数组 (float64)
    window : int
        窗口大小

    Returns
    -------
    np.ndarray
        滚动相关系数
    """                                                           # [新增]
    n = len(x)                                                    # [新增]
    result = np.full(n, np.nan)                                   # [新增]
                                                                  # [新增]
    for i in range(window - 1, n):                                # [新增]
        x_window = x[i - window + 1 : i + 1]                     # [新增]
        y_window = y[i - window + 1 : i + 1]                     # [新增]
                                                                  # [新增]
        has_nan = False                                           # [新增]
        for j in range(window):                                   # [新增]
            if np.isnan(x_window[j]) or np.isnan(y_window[j]):   # [新增]
                has_nan = True                                    # [新增]
                break                                             # [新增]
        if has_nan:                                               # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        x_mean = 0.0                                              # [新增]
        y_mean = 0.0                                              # [新增]
        for j in range(window):                                   # [新增]
            x_mean += x_window[j]                                 # [新增]
            y_mean += y_window[j]                                 # [新增]
        x_mean /= window                                          # [新增]
        y_mean /= window                                          # [新增]
                                                                  # [新增]
        cov = 0.0                                                 # [新增]
        x_var = 0.0                                               # [新增]
        y_var = 0.0                                               # [新增]
        for j in range(window):                                   # [新增]
            x_dev = x_window[j] - x_mean                          # [新增]
            y_dev = y_window[j] - y_mean                          # [新增]
            cov += x_dev * y_dev                                  # [新增]
            x_var += x_dev * x_dev                                # [新增]
            y_var += y_dev * y_dev                                # [新增]
                                                                  # [新增]
        if x_var > 0 and y_var > 0:                               # [新增]
            result[i] = cov / np.sqrt(x_var * y_var)              # [新增]
                                                                  # [新增]
    return result                                                 # [新增]


@_njit(cache=True)                                                # [新增]
def _variance_ratio_numba(returns, q, window):                    # [新增]
    """Numba加速的方差比率计算。

    VR > 1 表示趋势, VR < 1 表示均值回归, VR ≈ 1 表示随机游走。

    Parameters
    ----------
    returns : np.ndarray
        收益率序列
    q : int
        聚合期数
    window : int
        滚动窗口大小

    Returns
    -------
    np.ndarray
        方差比率
    """                                                           # [新增]
    n = len(returns)                                              # [新增]
    result = np.full(n, np.nan)                                   # [新增]
                                                                  # [新增]
    if n < window + q:                                            # [新增]
        return result                                             # [新增]
                                                                  # [新增]
    for i in range(window + q, n):                                # [新增]
        y = returns[i - window : i]                               # [新增]
                                                                  # [新增]
        var_1 = 0.0                                               # [新增]
        count_1 = 0                                               # [新增]
        for j in range(1, window):                                # [新增]
            if not np.isnan(y[j]) and not np.isnan(y[j - 1]):    # [新增]
                var_1 += y[j] * y[j]                              # [新增]
                count_1 += 1                                      # [新增]
                                                                  # [新增]
        if count_1 < 10:                                          # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        var_1 = var_1 / count_1                                   # [新增]
                                                                  # [新增]
        q_returns = np.zeros(window - q + 1)                      # [新增]
        for j in range(window - q):                               # [新增]
            q_sum = 0.0                                           # [新增]
            for k in range(q):                                    # [新增]
                if not np.isnan(y[j + k + 1]):                    # [新增]
                    q_sum += y[j + k + 1]                         # [新增]
            q_returns[j] = q_sum                                  # [新增]
                                                                  # [新增]
        q_mean = 0.0                                              # [新增]
        q_count = 0                                               # [新增]
        for j in range(window - q):                               # [新增]
            if not np.isnan(q_returns[j]):                        # [新增]
                q_mean += q_returns[j]                             # [新增]
                q_count += 1                                      # [新增]
                                                                  # [新增]
        if q_count < 5:                                           # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        q_mean /= q_count                                         # [新增]
                                                                  # [新增]
        var_q = 0.0                                               # [新增]
        for j in range(window - q):                               # [新增]
            if not np.isnan(q_returns[j]):                        # [新增]
                var_q += (q_returns[j] - q_mean) ** 2             # [新增]
                                                                  # [新增]
        if q_count > 0:                                           # [新增]
            var_q = var_q / q_count                               # [新增]
                                                                  # [新增]
        if var_1 > 0 and var_q > 0:                               # [新增]
            result[i] = var_q / (q * var_1)                       # [新增]
                                                                  # [新增]
    return result                                                 # [新增]


# ============================================================     # [新增]
# 第三辑辅助函数：分形分析、信息熵、去趋势波动                     # [新增]
# ============================================================     # [新增]


def _fractal_dimension_numba(arr, window):                          # [新增]
    """分形维数计算（Higuchi算法简化版）"""                          # [新增]
    n = len(arr)                                                    # [新增]
    result = np.full(n, np.nan)                                     # [新增]
                                                                    # [新增]
    for i in range(window * 2, n):                                  # [新增]
        y = arr[i - window + 1 : i + 1]                             # [新增]
                                                                    # [新增]
        has_nan = False                                             # [新增]
        for j in range(window):                                     # [新增]
            if np.isnan(y[j]):                                      # [新增]
                has_nan = True                                      # [新增]
                break                                               # [新增]
        if has_nan:                                                 # [新增]
            continue                                                # [新增]
                                                                    # [新增]
        kmax = min(10, window // 4)                                 # [新增]
        L = np.zeros(kmax)                                          # [新增]
                                                                    # [新增]
        for k in range(1, kmax + 1):                                # [新增]
            Lk = 0.0                                                # [新增]
            for m in range(k):                                      # [新增]
                Nk = (window - m - 1) // k                          # [新增]
                if Nk > 1:                                          # [新增]
                    sum_abs = 0.0                                   # [新增]
                    for j2 in range(1, Nk):                         # [新增]
                        idx1 = m + j2 * k                           # [新增]
                        idx2 = m + (j2 - 1) * k                    # [新增]
                        if idx1 < window and idx2 < window:         # [新增]
                            sum_abs += abs(y[idx1] - y[idx2])       # [新增]
                    if Nk > 0:                                      # [新增]
                        Lk += sum_abs * (window - 1) / (Nk * k * k)  # [新增]
            if Lk > 0:                                              # [新增]
                L[k - 1] = Lk                                       # [新增]
                                                                    # [新增]
        valid_k = []                                                # [新增]
        valid_L = []                                                # [新增]
        for k_idx in range(kmax):                                   # [新增]
            if L[k_idx] > 0:                                        # [新增]
                valid_k.append(np.log(1.0 / (k_idx + 1)))          # [新增]
                valid_L.append(np.log(L[k_idx]))                   # [新增]
                                                                    # [新增]
        if len(valid_k) > 2:                                        # [新增]
            xm = sum(valid_k) / len(valid_k)                        # [新增]
            ym = sum(valid_L) / len(valid_L)                        # [新增]
            cov = sum((valid_k[j3] - xm) * (valid_L[j3] - ym) for j3 in range(len(valid_k)))  # [新增]
            var = sum((valid_k[j3] - xm) ** 2 for j3 in range(len(valid_k)))  # [新增]
            if var > 0:                                             # [新增]
                result[i] = cov / var                               # [新增]
                                                                    # [新增]
    return result                                                   # [新增]


def _approximate_entropy_calc(arr, window, m=2, r_factor=0.2):     # [新增]
    """近似熵计算"""                                                # [新增]
    n = len(arr)                                                    # [新增]
    result = np.full(n, np.nan)                                     # [新增]
                                                                    # [新增]
    for i in range(window + m, n):                                  # [新增]
        y = arr[i - window + 1 : i + 1]                             # [新增]
                                                                    # [新增]
        has_nan = False                                             # [新增]
        for j in range(window):                                     # [新增]
            if np.isnan(y[j]):                                      # [新增]
                has_nan = True                                      # [新增]
                break                                               # [新增]
        if has_nan:                                                 # [新增]
            continue                                                # [新增]
                                                                    # [新增]
        std = np.std(y)                                             # [新增]
        r = r_factor * std                                          # [新增]
        if r < 1e-12:                                               # [新增]
            continue                                                # [新增]
                                                                    # [新增]
        def _phi(m_val):                                            # [新增]
            N = window - m_val + 1                                  # [新增]
            C = np.zeros(N)                                         # [新增]
            for j2 in range(N):                                     # [新增]
                count = 0                                           # [新增]
                for k2 in range(N):                                 # [新增]
                    max_diff = 0.0                                  # [新增]
                    for l_idx in range(m_val):                      # [新增]
                        diff = abs(y[j2 + l_idx] - y[k2 + l_idx])  # [新增]
                        if diff > max_diff:                         # [新增]
                            max_diff = diff                         # [新增]
                    if max_diff <= r:                                # [新增]
                        count += 1                                  # [新增]
                if count > 0:                                       # [新增]
                    C[j2] = count / N                               # [新增]
            sum_log = 0.0                                           # [新增]
            for j2 in range(N):                                     # [新增]
                if C[j2] > 0:                                       # [新增]
                    sum_log += np.log(C[j2])                        # [新增]
            return sum_log / N                                      # [新增]
                                                                    # [新增]
        phi_m = _phi(m)                                             # [新增]
        phi_m1 = _phi(m + 1)                                        # [新增]
        if not np.isnan(phi_m) and not np.isnan(phi_m1):            # [新增]
            result[i] = phi_m - phi_m1                              # [新增]
                                                                    # [新增]
    return result                                                   # [新增]


def _sample_entropy_calc(arr, window, m=2, r_factor=0.2):          # [新增]
    """样本熵计算"""                                                # [新增]
    n = len(arr)                                                    # [新增]
    result = np.full(n, np.nan)                                     # [新增]
                                                                    # [新增]
    for i in range(window + m, n):                                  # [新增]
        y = arr[i - window + 1 : i + 1]                             # [新增]
                                                                    # [新增]
        has_nan = False                                             # [新增]
        for j in range(window):                                     # [新增]
            if np.isnan(y[j]):                                      # [新增]
                has_nan = True                                      # [新增]
                break                                               # [新增]
        if has_nan:                                                 # [新增]
            continue                                                # [新增]
                                                                    # [新增]
        std = np.std(y)                                             # [新增]
        r = r_factor * std                                          # [新增]
        if r < 1e-12:                                               # [新增]
            continue                                                # [新增]
                                                                    # [新增]
        N = window - m                                              # [新增]
        B = 0                                                       # [新增]
        A = 0                                                       # [新增]
        for j2 in range(N):                                         # [新增]
            for k2 in range(j2 + 1, N):                             # [新增]
                d_m = 0.0                                           # [新增]
                for l_idx in range(m):                              # [新增]
                    diff = abs(y[j2 + l_idx] - y[k2 + l_idx])      # [新增]
                    if diff > d_m:                                  # [新增]
                        d_m = diff                                  # [新增]
                if d_m <= r:                                        # [新增]
                    B += 1                                          # [新增]
                    if j2 + m < window and k2 + m < window:         # [新增]
                        d_m1 = max(d_m, abs(y[j2 + m] - y[k2 + m]))  # [新增]
                        if d_m1 <= r:                               # [新增]
                            A += 1                                  # [新增]
        if B > 0 and A > 0:                                         # [新增]
            result[i] = -np.log(A / B)                              # [新增]
                                                                    # [新增]
    return result                                                   # [新增]


def _permutation_entropy_calc(arr, window, order=3, delay=1):      # [新增]
    """排列熵计算"""                                                # [新增]
    from collections import Counter                                 # [新增]
    n = len(arr)                                                    # [新增]
    result = np.full(n, np.nan)                                     # [新增]
                                                                    # [新增]
    for i in range(window + order * delay, n):                      # [新增]
        y = arr[i - window + 1 : i + 1]                             # [新增]
                                                                    # [新增]
        has_nan = False                                             # [新增]
        for j in range(window):                                     # [新增]
            if np.isnan(y[j]):                                      # [新增]
                has_nan = True                                      # [新增]
                break                                               # [新增]
        if has_nan:                                                 # [新增]
            continue                                                # [新增]
                                                                    # [新增]
        N = window - (order - 1) * delay                            # [新增]
        patterns = []                                               # [新增]
        for j2 in range(N):                                         # [新增]
            pattern = [y[j2 + k2 * delay] for k2 in range(order)]   # [新增]
            indices = tuple(np.argsort(pattern))                    # [新增]
            patterns.append(indices)                                # [新增]
                                                                    # [新增]
        counts = Counter(patterns)                                  # [新增]
        entropy = 0.0                                               # [新增]
        for count in counts.values():                               # [新增]
            prob = count / N                                        # [新增]
            if prob > 0:                                            # [新增]
                entropy -= prob * np.log(prob)                      # [新增]
        max_entropy = np.log(len(counts)) if len(counts) > 0 else 1.0  # [新增]
        if max_entropy > 0:                                         # [新增]
            result[i] = entropy / max_entropy                       # [新增]
                                                                    # [新增]
    return result                                                   # [新增]


def _detrended_fluctuation_calc(arr, window, min_box=4, max_box=None):  # [新增]
    """去趋势波动分析"""                                            # [新增]
    n = len(arr)                                                    # [新增]
    result = np.full(n, np.nan)                                     # [新增]
    if max_box is None:                                             # [新增]
        max_box = min(window // 4, 50)                              # [新增]
                                                                    # [新增]
    for i in range(window, n):                                      # [新增]
        y = arr[i - window + 1 : i + 1]                             # [新增]
                                                                    # [新增]
        has_nan = False                                             # [新增]
        for j in range(window):                                     # [新增]
            if np.isnan(y[j]):                                      # [新增]
                has_nan = True                                      # [新增]
                break                                               # [新增]
        if has_nan:                                                 # [新增]
            continue                                                # [新增]
                                                                    # [新增]
        y_cum = np.cumsum(y)                                        # [新增]
        box_sizes = []                                              # [新增]
        fluctuations = []                                           # [新增]
                                                                    # [新增]
        for box_size in range(min_box, max_box):                    # [新增]
            n_box = window // box_size                              # [新增]
            if n_box < 2:                                           # [新增]
                continue                                            # [新增]
            F2 = 0.0                                                # [新增]
            for b in range(n_box):                                  # [新增]
                start = b * box_size                                # [新增]
                end = min((b + 1) * box_size, window)               # [新增]
                seg_len = end - start                               # [新增]
                if seg_len < 2:                                     # [新增]
                    continue                                        # [新增]
                y_seg = y_cum[start:end]                            # [新增]
                x_mean = (seg_len - 1) / 2.0                       # [新增]
                y_mean = np.mean(y_seg)                             # [新增]
                var_x = sum((j2 - x_mean) ** 2 for j2 in range(seg_len))  # [新增]
                cov = sum((j2 - x_mean) * (y_seg[j2] - y_mean) for j2 in range(seg_len))  # [新增]
                if var_x > 0:                                       # [新增]
                    slope = cov / var_x                             # [新增]
                    intercept = y_mean - slope * x_mean             # [新增]
                    residual = sum((y_seg[j2] - (slope * j2 + intercept)) ** 2 for j2 in range(seg_len))  # [新增]
                    F2 += residual                                  # [新增]
            if n_box > 0 and F2 > 0:                                # [新增]
                box_sizes.append(np.log(box_size))                  # [新增]
                fluctuations.append(np.log(np.sqrt(F2 / n_box)))   # [新增]
                                                                    # [新增]
        if len(box_sizes) > 3:                                      # [新增]
            xm = sum(box_sizes) / len(box_sizes)                    # [新增]
            ym = sum(fluctuations) / len(fluctuations)              # [新增]
            cov = sum((box_sizes[j3] - xm) * (fluctuations[j3] - ym) for j3 in range(len(box_sizes)))  # [新增]
            var = sum((box_sizes[j3] - xm) ** 2 for j3 in range(len(box_sizes)))  # [新增]
            if var > 0:                                             # [新增]
                result[i] = cov / var                               # [新增]
                                                                    # [新增]
    return result                                                   # [新增]
