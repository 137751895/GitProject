"""
商品期货机器学习量化模型 - 特征工程模块
Feature engineering for commodity futures ML quantitative model.

支持 1分钟、5分钟、15分钟 K线数据的衍生指标计算。
输入数据格式: OHLCV + 持仓量(open_interest) + 仓差(oi_change)

推荐技术指标和特征:
- 价格类: 移动平均线(MA/EMA), 布林带(BOLL), SAR
- 动量类: RSI, MACD, KDJ, ROC, CCI, Williams %R
- 成交量类: 量比, OBV, VWAP, 成交量MA
- 波动率类: ATR, 波动率标准差, 真实波幅
- 持仓量类: 仓差变化率, 持仓量MA, 量仓比
- 形态类: K线实体比, 上下影线比, 缺口
"""

import numpy as np
import pandas as pd


def compute_ma(series, windows=None):
    """计算移动平均线 (MA)"""
    if windows is None:
        windows = [5, 10, 20, 60]
    result = {}
    for w in windows:
        result[f"ma_{w}"] = series.rolling(window=w).mean()
    return pd.DataFrame(result, index=series.index)


def compute_ema(series, windows=None):
    """计算指数移动平均线 (EMA)"""
    if windows is None:
        windows = [5, 10, 20, 60]
    result = {}
    for w in windows:
        result[f"ema_{w}"] = series.ewm(span=w, adjust=False).mean()
    return pd.DataFrame(result, index=series.index)


def compute_bollinger_bands(series, window=20, num_std=2):
    """计算布林带 (Bollinger Bands)"""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma
    pct_b = (series - lower) / (upper - lower)
    return pd.DataFrame({
        "boll_upper": upper,
        "boll_mid": ma,
        "boll_lower": lower,
        "boll_width": width,
        "boll_pct_b": pct_b,
    }, index=series.index)


def compute_rsi(series, window=14):
    """计算相对强弱指标 (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return pd.DataFrame({
        "macd_dif": dif,
        "macd_dea": dea,
        "macd_hist": macd_hist,
    }, index=series.index)


def compute_kdj(high, low, close, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100

    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({
        "kdj_k": k,
        "kdj_d": d,
        "kdj_j": j,
    }, index=close.index)


def compute_atr(high, low, close, window=14):
    """计算平均真实波幅 (ATR)"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return pd.DataFrame({
        "tr": tr,
        "atr": atr,
    }, index=close.index)


def compute_cci(high, low, close, window=14):
    """计算CCI指标"""
    tp = (high + low + close) / 3
    ma = tp.rolling(window=window).mean()
    md = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - ma) / (0.015 * md)
    return cci


def compute_williams_r(high, low, close, window=14):
    """计算威廉指标 (Williams %R)"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    wr = (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan) * -100
    return wr


def compute_roc(series, window=12):
    """计算变动率 (ROC)"""
    return (series - series.shift(window)) / series.shift(window).replace(0, np.nan) * 100


def compute_obv(close, volume):
    """计算能量潮 (OBV)"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    return obv


def compute_vwap(high, low, close, volume):
    """计算成交量加权平均价 (VWAP)"""
    tp = (high + low + close) / 3
    cumulative_tp_vol = (tp * volume).cumsum()
    cumulative_vol = volume.cumsum()
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    return vwap


def compute_volume_features(volume, windows=None):
    """计算成交量相关特征"""
    if windows is None:
        windows = [5, 10, 20]
    result = {}
    for w in windows:
        vol_ma = volume.rolling(window=w).mean()
        result[f"vol_ma_{w}"] = vol_ma
        result[f"vol_ratio_{w}"] = volume / vol_ma.replace(0, np.nan)
    result["vol_change"] = volume.pct_change()
    return pd.DataFrame(result, index=volume.index)


def compute_oi_features(open_interest, volume, windows=None):
    """计算持仓量相关特征（仓差分析）"""
    if windows is None:
        windows = [5, 10, 20]
    result = {}
    # 仓差（持仓量变化）
    result["oi_change"] = open_interest.diff()
    result["oi_change_pct"] = open_interest.pct_change()

    for w in windows:
        result[f"oi_ma_{w}"] = open_interest.rolling(window=w).mean()
        result[f"oi_change_ma_{w}"] = result["oi_change"].rolling(window=w).mean()

    # 量仓比
    result["vol_oi_ratio"] = volume / open_interest.replace(0, np.nan)
    return pd.DataFrame(result, index=open_interest.index)


def compute_candle_features(open_price, high, low, close):
    """计算K线形态特征"""
    body = close - open_price
    body_abs = body.abs()
    total_range = (high - low).replace(0, np.nan)

    result = {}
    # K线实体比（实体占总振幅的比例）
    result["body_ratio"] = body_abs / total_range
    # 上影线比
    result["upper_shadow_ratio"] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / total_range
    # 下影线比
    result["lower_shadow_ratio"] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / total_range
    # 涨跌方向
    result["candle_direction"] = np.sign(body)
    # 振幅
    result["amplitude"] = total_range / close.shift(1).replace(0, np.nan)
    # 缺口
    result["gap"] = open_price - close.shift(1)
    result["gap_ratio"] = result["gap"] / close.shift(1).replace(0, np.nan)
    return pd.DataFrame(result, index=close.index)


def compute_volatility_features(close, windows=None):
    """计算波动率特征"""
    if windows is None:
        windows = [5, 10, 20]
    returns = close.pct_change()
    result = {}
    for w in windows:
        result[f"volatility_{w}"] = returns.rolling(window=w).std()
        result[f"return_ma_{w}"] = returns.rolling(window=w).mean()
    result["log_return"] = np.log(close / close.shift(1))
    return pd.DataFrame(result, index=close.index)


def compute_price_position(close, high, low, window=20):
    """计算价格位置特征"""
    highest = high.rolling(window=window).max()
    lowest = low.rolling(window=window).min()
    price_range = (highest - lowest).replace(0, np.nan)
    position = (close - lowest) / price_range
    return pd.DataFrame({
        "price_position": position,
        "dist_to_high": (highest - close) / close.replace(0, np.nan),
        "dist_to_low": (close - lowest) / close.replace(0, np.nan),
    }, index=close.index)


def compute_all_features(df, period="5min"):
    """
    计算所有衍生指标和特征。

    Parameters
    ----------
    df : pd.DataFrame
        K线数据，必须包含列: open, high, low, close, volume, open_interest
    period : str
        K线周期 ("1min", "5min", "15min")

    Returns
    -------
    pd.DataFrame
        包含所有特征的DataFrame
    """
    required_cols = {"open", "high", "low", "close", "volume", "open_interest"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需的列: {missing}")

    o = df["open"]
    h = df["high"]
    low = df["low"]
    c = df["close"]
    v = df["volume"]
    oi = df["open_interest"]

    # 根据周期选择不同的窗口参数
    if period == "1min":
        ma_windows = [5, 10, 20, 60, 120]
        vol_windows = [5, 10, 20, 60]
        oi_windows = [5, 10, 20, 60]
    elif period == "5min":
        ma_windows = [5, 10, 20, 60]
        vol_windows = [5, 10, 20]
        oi_windows = [5, 10, 20]
    else:  # 15min
        ma_windows = [5, 10, 20, 40]
        vol_windows = [5, 10, 20]
        oi_windows = [5, 10, 20]

    features = pd.DataFrame(index=df.index)

    # 价格类指标
    features = pd.concat([features, compute_ma(c, ma_windows)], axis=1)
    features = pd.concat([features, compute_ema(c, ma_windows)], axis=1)
    features = pd.concat([features, compute_bollinger_bands(c)], axis=1)
    features = pd.concat([features, compute_price_position(c, h, low)], axis=1)

    # 动量类指标
    features["rsi_14"] = compute_rsi(c, 14)
    features["rsi_6"] = compute_rsi(c, 6)
    features = pd.concat([features, compute_macd(c)], axis=1)
    features = pd.concat([features, compute_kdj(h, low, c)], axis=1)
    features["cci"] = compute_cci(h, low, c)
    features["williams_r"] = compute_williams_r(h, low, c)
    features["roc_12"] = compute_roc(c, 12)
    features["roc_6"] = compute_roc(c, 6)

    # 成交量类指标
    features = pd.concat([features, compute_volume_features(v, vol_windows)], axis=1)
    features["obv"] = compute_obv(c, v)
    features["vwap"] = compute_vwap(h, low, c, v)

    # 波动率类指标
    features = pd.concat([features, compute_atr(h, low, c)], axis=1)
    features = pd.concat([features, compute_volatility_features(c, vol_windows)], axis=1)

    # 持仓量类指标（仓差）
    features = pd.concat([features, compute_oi_features(oi, v, oi_windows)], axis=1)

    # K线形态特征
    features = pd.concat([features, compute_candle_features(o, h, low, c)], axis=1)

    # 市场状态特征
    from features.market_regime import MarketRegimeDetector
    detector = MarketRegimeDetector()
    regime_features = detector.detect_regime(df)
    features = pd.concat([features, regime_features], axis=1)

    return features


def compute_prediction_targets(df, horizon=5):
    """
    计算预测目标。

    推荐预测目标:
    - future_return: 未来N根K线的收益率（回归任务）
    - future_direction: 未来价格方向（分类任务，最推荐）
    - future_volatility: 未来波动率

    Parameters
    ----------
    df : pd.DataFrame
        K线数据，必须包含 close 列
    horizon : int
        预测的未来K线数量

    Returns
    -------
    pd.DataFrame
        预测目标
    """
    close = df["close"]
    targets = pd.DataFrame(index=df.index)

    # 未来收益率
    targets["future_return"] = close.shift(-horizon) / close - 1

    # 未来方向 (1=涨, 0=跌)
    targets["future_direction"] = (targets["future_return"] > 0).astype(int)

    # 未来波动率
    returns = close.pct_change()
    targets["future_volatility"] = returns.shift(-1).rolling(window=horizon).std()

    # 多分类预测目标（五分位）
    bins = [-np.inf, -0.005, -0.001, 0.001, 0.005, np.inf]
    labels = [0, 1, 2, 3, 4]  # strong_down, weak_down, neutral, weak_up, strong_up
    targets["future_regime"] = pd.cut(
        targets["future_return"], bins=bins, labels=labels
    ).astype(float)

    return targets
