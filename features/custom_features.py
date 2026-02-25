"""
商品期货机器学习量化模型 - 自定义特征
Custom features registered via @register_feature decorator.

依据《hfml特征工程增强报告-56项目挖掘-最终可执行版.md》集成的9个特征因子，
依据《hfml特征工程增强报告-精选10特征-可执行版.md》集成的10个精选特征因子，
加上2个原有示例特征，共21个注册特征。

每个函数通过 @register_feature 装饰器自动注册到全局 FeatureRegistry，
添加后无需修改任何下游代码（特征筛选、模型训练、回测等）。

新增特征步骤:
    1. 在本文件中定义计算函数，并用 @register_feature 装饰
    2. 完成！系统自动发现、计算、参与筛选和训练
"""

import numpy as np                                            # [新增]
import pandas as pd                                           # [新增]

from features.feature_registry import register_feature        # [新增]


# ============================================================
# 原有示例特征
# ============================================================

@register_feature(                                            # [新增]
    group="动量指标",                                          # [新增]
    level="level3_momentum",                                  # [新增]
    description="RSI 14周期指标的5周期斜率，捕捉RSI变化速率",  # [新增]
    depends_on=["rsi_14"],                                    # [新增]
    output_names=["rsi_14_slope_custom"],                     # [新增]
)                                                             # [新增]
def compute_rsi_14_slope(df, features_df=None, **kwargs):     # [新增]
    """计算RSI(14)的5周期斜率。

    RSI的变化速率比RSI绝对值更能提前预警趋势反转:
    - 正斜率: RSI正在上升，动量增强
    - 负斜率: RSI正在下降，动量减弱
    - 斜率由正转负: 可能出现顶部反转

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    features_df : pd.DataFrame
        已计算的特征矩阵（包含rsi_14）。

    Returns
    -------
    pd.DataFrame
        包含 rsi_14_slope_custom 列
    """                                                       # [新增]
    slope = features_df["rsi_14"].diff(5) / 5.0               # [新增]
    return pd.DataFrame(                                      # [新增]
        {"rsi_14_slope_custom": slope}, index=df.index        # [新增]
    )                                                         # [新增]


@register_feature(                                            # [新增]
    group="成交量",                                            # [新增]
    level="level4_micro",                                     # [新增]
    description="成交量加速度，放量突破信号",                  # [新增]
    depends_on=["vol_change"],                                # [新增]
    output_names=["volume_acceleration"],                     # [新增]
)                                                             # [新增]
def compute_volume_acceleration(df, features_df=None, **kwargs):  # [新增]
    """计算成交量变化的加速度（二阶导数）。

    成交量加速度可以更早地捕捉放量突破信号:
    - 加速度为正: 成交量增速加快（可能突破）
    - 加速度为负: 成交量增速放缓（可能回归常态）

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    features_df : pd.DataFrame
        已计算的特征矩阵（包含vol_change）。

    Returns
    -------
    pd.DataFrame
        包含 volume_acceleration 列
    """                                                       # [新增]
    vol_chg = features_df["vol_change"]                       # [新增]
    accel = vol_chg.diff(3) / 3.0                             # [新增]
    return pd.DataFrame(                                      # [新增]
        {"volume_acceleration": accel}, index=df.index        # [新增]
    )                                                         # [新增]


# ============================================================
# 9个特征因子（依据《hfml特征工程增强报告》集成）
# ============================================================
# 注：divergence, vwap_dev, vol_state, mom_slope, rsi_slope,
#     vol_zscore, gap_decay 已在 feature_engineering_enhanced.py
#     中通过 compute_microstructure_features / compute_gap_decay_features
#     计算。这里注册它们以提供元数据（分组、层级、依赖声明），
#     compute_all_features 的去重逻辑会跳过已存在的同名列。
#     oi_price_alignment 和 oi_price_magnitude 为真正新增的特征。
# ============================================================


# ---- 特征1: divergence (资金-价格方向一致性) ----           # [新增]
@register_feature(                                            # [新增]
    group="微观结构",                                          # [新增]
    level="level4_micro",                                     # [新增]
    description="资金-价格方向一致性 (综合评分22/25)",         # [新增]
    depends_on=[],                                            # [新增]
    output_names=["divergence"],                              # [新增]
)                                                             # [新增]
def compute_divergence(df, features_df=None, **kwargs):       # [新增]
    """计算持仓变化与价格变化的方向一致性。

    - +1: 价格和持仓同向变化（趋势确认）
    - -1: 价格和持仓反向变化（可能反转）
    -  0: 至少一方无变化

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据，需含 close 和 open_interest 列。

    Returns
    -------
    pd.DataFrame
        包含 divergence 列
    """                                                       # [新增]
    result = np.sign(df["close"].diff()) * np.sign(           # [新增]
        df["open_interest"].diff()                            # [新增]
    )                                                         # [新增]
    return pd.DataFrame({"divergence": result}, index=df.index)  # [新增]


# ---- 特征2: vwap_dev (VWAP偏离度) ----                     # [新增]
@register_feature(                                            # [新增]
    group="量价关系",                                          # [新增]
    level="level4_micro",                                     # [新增]
    description="价格相对VWAP偏离 (综合评分21/25)",           # [新增]
    depends_on=[],                                            # [新增]
    output_names=["vwap_dev"],                                # [新增]
)                                                             # [新增]
def compute_vwap_dev(df, features_df=None, **kwargs):         # [新增]
    """计算价格与成交量加权均价(VWAP)的偏离度。

    正值表示价格高于VWAP（买盘较强），负值表示低于VWAP。

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据，需含 high, low, close, volume 列。

    Returns
    -------
    pd.DataFrame
        包含 vwap_dev 列
    """                                                       # [新增]
    tp = (df["high"] + df["low"] + df["close"]) / 3           # [新增]
    vwap = (tp * df["volume"]).cumsum() / (                   # [新增]
        df["volume"].cumsum() + 1e-12                         # [新增]
    )                                                         # [新增]
    return pd.DataFrame(                                      # [新增]
        {"vwap_dev": (df["close"] - vwap) / (vwap + 1e-12)}, # [新增]
        index=df.index,                                       # [新增]
    )                                                         # [新增]


# ---- 特征3: vol_state (ATR归一化波动状态) ----              # [新增]
@register_feature(                                            # [新增]
    group="波动率",                                            # [新增]
    level="level5_cross",                                     # [新增]
    description="ATR归一化波动状态 (综合评分20/25)",          # [新增]
    depends_on=[],                                            # [新增]
    output_names=["vol_state"],                               # [新增]
)                                                             # [新增]
def compute_vol_state(                                        # [新增]
    df, features_df=None, atr_window=14, norm_window=60, **kwargs  # [新增]
):                                                            # [新增]
    """计算波动率状态: ATR(14) / 60日均价。

    数值越高表示波动率相对价格水平越大。

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    atr_window : int
        ATR窗口 (默认14)。
    norm_window : int
        归一化基准窗口 (默认60)。

    Returns
    -------
    pd.DataFrame
        包含 vol_state 列
    """                                                       # [新增]
    prev_close = df["close"].shift(1)                         # [新增]
    tr = pd.concat([                                          # [新增]
        df["high"] - df["low"],                               # [新增]
        (df["high"] - prev_close).abs(),                      # [新增]
        (df["low"] - prev_close).abs(),                       # [新增]
    ], axis=1).max(axis=1).fillna(0)                          # [新增]
    atr = tr.rolling(atr_window).mean()                       # [新增]
    base = df["close"].rolling(norm_window).mean()            # [新增]
    return pd.DataFrame(                                      # [新增]
        {"vol_state": atr / (base + 1e-12)}, index=df.index   # [新增]
    )                                                         # [新增]


# ---- 特征4: mom_slope (动量斜率) ----                       # [新增]
@register_feature(                                            # [新增]
    group="动量指标",                                          # [新增]
    level="level3_momentum",                                  # [新增]
    description="价格滚动线性斜率归一化 (综合评分19/25)",     # [新增]
    depends_on=[],                                            # [新增]
    output_names=["mom_slope"],                               # [新增]
)                                                             # [新增]
def compute_mom_slope(df, features_df=None, window=5, **kwargs):  # [新增]
    """计算价格的滚动线性回归斜率，归一化到当前价格。

    正值表示上升趋势加速，负值表示下降趋势加速。

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    window : int
        滚动窗口 (默认5)。

    Returns
    -------
    pd.DataFrame
        包含 mom_slope 列
    """                                                       # [新增]
    from features.feature_engineering_enhanced import (        # [新增]
        _rolling_slope,                                       # [新增]
    )                                                         # [新增]
    slope = _rolling_slope(df["close"].values, window)        # [新增]
    return pd.DataFrame(                                      # [新增]
        {"mom_slope": slope / (df["close"].values + 1e-12)},  # [新增]
        index=df.index,                                       # [新增]
    )                                                         # [新增]


# ---- 特征5: rsi_slope (RSI滚动斜率) ----                   # [新增]
@register_feature(                                            # [新增]
    group="动量指标",                                          # [新增]
    level="level3_momentum",                                  # [新增]
    description="RSI滚动斜率 (综合评分19/25)",               # [新增]
    depends_on=[],                                            # [新增]
    output_names=["rsi_slope"],                               # [新增]
)                                                             # [新增]
def compute_rsi_slope(                                        # [新增]
    df, features_df=None, rsi_window=14, slope_window=5, **kwargs  # [新增]
):                                                            # [新增]
    """计算RSI指标的滚动斜率。

    RSI斜率比RSI绝对值更能提前预警超买/超卖反转。

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    rsi_window : int
        RSI窗口 (默认14)。
    slope_window : int
        斜率窗口 (默认5)。

    Returns
    -------
    pd.DataFrame
        包含 rsi_slope 列
    """                                                       # [新增]
    from features.rolling_numba import rsi as nb_rsi          # [新增]
    from features.feature_engineering_enhanced import (        # [新增]
        _rolling_slope,                                       # [新增]
    )                                                         # [新增]
    rsi_14 = nb_rsi(df["close"].values.astype(np.float64),    # [新增]
                    rsi_window)                                # [新增]
    result = _rolling_slope(rsi_14, slope_window)             # [新增]
    return pd.DataFrame({"rsi_slope": result}, index=df.index)  # [新增]


# ---- 特征6: vol_zscore (成交量Z分数) ----                  # [新增]
@register_feature(                                            # [新增]
    group="成交量",                                            # [新增]
    level="level4_micro",                                     # [新增]
    description="成交量ZScore (综合评分20/25)",               # [新增]
    depends_on=[],                                            # [新增]
    output_names=["vol_zscore"],                              # [新增]
)                                                             # [新增]
def compute_vol_zscore(df, features_df=None, window=20, **kwargs):  # [新增]
    """计算成交量的Z分数标准化。

    正值表示放量，负值表示缩量，绝对值>2为异常。

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    window : int
        滚动窗口 (默认20)。

    Returns
    -------
    pd.DataFrame
        包含 vol_zscore 列
    """                                                       # [新增]
    vol = df["volume"].astype(float)                          # [新增]
    vol_ma = vol.rolling(window).mean()                       # [新增]
    vol_sd = vol.rolling(window).std()                        # [新增]
    return pd.DataFrame(                                      # [新增]
        {"vol_zscore": (vol - vol_ma) / (vol_sd + 1e-12)},   # [新增]
        index=df.index,                                       # [新增]
    )                                                         # [新增]


# ---- 特征7: gap_decay (夜盘缺口衰减) ----                  # [新增]
@register_feature(                                            # [新增]
    group="价格形态",                                          # [新增]
    level="level4_micro",                                     # [新增]
    description="夜盘缺口衰减 (综合评分18/25)",              # [新增]
    depends_on=[],                                            # [新增]
    output_names=["gap_decay"],                               # [新增]
)                                                             # [新增]
def compute_gap_decay(                                        # [新增]
    df, features_df=None,                                     # [新增]
    night_session_start_hour=21,                              # [新增]
    night_session_start_max_minute=30,                        # [新增]
    decay_constant=300,                                       # [新增]
    **kwargs,                                                 # [新增]
):                                                            # [新增]
    """计算夜盘缺口的指数衰减因子。

    在夜盘开盘后，缺口信号随时间按 exp(-t/τ) 衰减。

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据，index需为DatetimeIndex。
    night_session_start_hour : int
        夜盘开始小时 (默认21)。
    night_session_start_max_minute : int
        夜盘前N分钟才计算 (默认30)。
    decay_constant : float
        衰减时间常数/秒 (默认300)。

    Returns
    -------
    pd.DataFrame
        包含 gap_decay 列
    """                                                       # [新增]
    gap_decay = np.zeros(len(df), dtype=np.float64)           # [新增]
    if isinstance(df.index, pd.DatetimeIndex):                # [新增]
        dt = df.index                                         # [新增]
    elif "datetime" in df.columns:                            # [新增]
        dt = pd.to_datetime(df["datetime"])                   # [新增]
    else:                                                     # [新增]
        return pd.DataFrame(                                  # [新增]
            {"gap_decay": gap_decay}, index=df.index          # [新增]
        )                                                     # [新增]
    close = df["close"].values.astype(np.float64)             # [新增]
    open_ = df["open"].values.astype(np.float64)              # [新增]
    for i in range(len(df)):                                  # [新增]
        t = dt[i]                                             # [新增]
        if (t.hour == night_session_start_hour                # [新增]
                and t.minute < night_session_start_max_minute):  # [新增]
            prev_close = close[i - 1] if i > 0 else np.nan   # [新增]
            if np.isfinite(prev_close) and prev_close > 0:    # [新增]
                gap = (open_[i] - prev_close) / prev_close    # [新增]
                seconds = t.minute * 60 + t.second            # [新增]
                gap_decay[i] = gap * np.exp(                  # [新增]
                    -seconds / decay_constant                 # [新增]
                )                                             # [新增]
    return pd.DataFrame(                                      # [新增]
        {"gap_decay": gap_decay}, index=df.index              # [新增]
    )                                                         # [新增]


# ---- 特征8: oi_price_alignment (持仓-价格一致性) ----      # [新增]
@register_feature(                                            # [新增]
    group="跨周期结构",                                        # [新增]
    level="level5_cross",                                     # [新增]
    description="持仓与价格方向一致性 (综合评分20/25)",       # [新增]
    depends_on=["oi_change"],                                 # [新增]
    output_names=["oi_price_alignment"],                      # [新增]
)                                                             # [新增]
def compute_oi_price_alignment(df, features_df=None, **kwargs):  # [新增]
    """计算持仓量变化与价格变化的方向一致性。

    - +1: 持仓增加+价格上涨 或 持仓减少+价格下跌 (趋势确认)
    - -1: 持仓增加+价格下跌 或 持仓减少+价格上涨 (可能反转)

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    features_df : pd.DataFrame
        已计算的特征矩阵（包含oi_change）。

    Returns
    -------
    pd.DataFrame
        包含 oi_price_alignment 列
    """                                                       # [新增]
    if features_df is not None and "oi_change" in features_df.columns:  # [新增]
        oi_change = features_df["oi_change"]                  # [新增]
    else:                                                     # [新增]
        oi_change = df["open_interest"].diff()                # [新增]
    returns = df["close"].pct_change()                        # [新增]
    return pd.DataFrame(                                      # [新增]
        {"oi_price_alignment": np.sign(oi_change) * np.sign(returns)},  # [新增]
        index=df.index,                                       # [新增]
    )                                                         # [新增]


# ---- 特征9: oi_price_magnitude (OI/价格幅度比) ----        # [新增]
@register_feature(                                            # [新增]
    group="跨周期结构",                                        # [新增]
    level="level5_cross",                                     # [新增]
    description="OI变化相对价格变化强度 (综合评分20/25)",     # [新增]
    depends_on=["oi_change"],                                 # [新增]
    output_names=["oi_price_magnitude"],                      # [新增]
)                                                             # [新增]
def compute_oi_price_magnitude(                               # [新增]
    df, features_df=None, eps=1e-8, clip_upper=100.0, **kwargs  # [新增]
):                                                            # [新增]
    """计算持仓量变化幅度与价格变化幅度的比率。

    衡量"资金推动效率"——每单位价格变化伴随多少持仓量变化。
    结果使用clip截尾以防止极端值。

    Parameters
    ----------
    df : pd.DataFrame
        原始OHLCV数据。
    features_df : pd.DataFrame
        已计算的特征矩阵（包含oi_change）。
    eps : float
        分母保护 (默认1e-8)。
    clip_upper : float
        上限截尾 (默认100.0)。

    Returns
    -------
    pd.DataFrame
        包含 oi_price_magnitude 列
    """                                                       # [新增]
    if features_df is not None and "oi_change" in features_df.columns:  # [新增]
        oi_change = features_df["oi_change"].abs()            # [新增]
    else:                                                     # [新增]
        oi_change = df["open_interest"].diff().abs()          # [新增]
    ret_abs = df["close"].pct_change().abs()                  # [新增]
    val = (oi_change / (ret_abs + eps)).clip(upper=clip_upper)  # [新增]
    return pd.DataFrame(                                      # [新增]
        {"oi_price_magnitude": val}, index=df.index           # [新增]
    )                                                         # [新增]


# ============================================================
# 10个精选特征因子（依据《hfml特征工程增强报告-精选10特征》集成）
# ============================================================

try:                                                              # [新增]
    from numba import njit as _njit_custom                        # [新增]
except ImportError:                                               # [新增]
    def _njit_custom(func=None, **kwargs):                        # [新增]
        """Numba njit fallback when numba is not installed."""     # [新增]
        if func is not None:                                      # [新增]
            return func                                           # [新增]
        def deco(f):                                              # [新增]
            return f                                              # [新增]
        return deco                                               # [新增]


# ---- 精选特征1: buy_sell_pressure (买卖压力指标) ----         # [新增]
@_njit_custom                                                     # [新增]
def _buy_sell_pressure_numba(high, low, close, volume):           # [新增]
    """Numba加速的买卖压力计算"""                                 # [新增]
    n = len(close)                                                # [新增]
    result = np.full(n, np.nan)                                   # [新增]
                                                                  # [新增]
    for i in range(n):                                            # [新增]
        hl_range = high[i] - low[i]                               # [新增]
        if hl_range > 1e-8:                                       # [新增]
            buy_pressure = volume[i] * (close[i] - low[i]) / hl_range   # [新增]
            sell_pressure = volume[i] * (high[i] - close[i]) / hl_range  # [新增]
            total = buy_pressure + sell_pressure                  # [新增]
            if total > 1e-8:                                      # [新增]
                result[i] = (buy_pressure - sell_pressure) / total  # [新增]
                                                                  # [新增]
    return result                                                 # [新增]


@register_feature(                                                # [新增]
    group="微观结构",                                              # [新增]
    level="level4_micro",                                         # [新增]
    description="基于K线位置的买卖净压力，正值表示买方主导，负值表示卖方主导",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["buy_sell_pressure"],                           # [新增]
)                                                                 # [新增]
def compute_buy_sell_pressure(df, features_df=None, **kwargs):    # [新增]
    """计算买卖压力指标。

    K线收盘位置反映买卖双方博弈结果：收盘靠近高价表示买方主导，
    收盘靠近低价表示卖方主导。净压力在[-1,1]区间。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: high, low, close, volume

    Returns
    -------
    pd.DataFrame
        包含 buy_sell_pressure 列
    """                                                           # [新增]
    high = df["high"].values.astype(np.float64)                   # [新增]
    low = df["low"].values.astype(np.float64)                     # [新增]
    close = df["close"].values.astype(np.float64)                 # [新增]
    volume = df["volume"].values.astype(np.float64)               # [新增]
                                                                  # [新增]
    result = _buy_sell_pressure_numba(high, low, close, volume)   # [新增]
    return pd.DataFrame(                                          # [新增]
        {"buy_sell_pressure": result}, index=df.index             # [新增]
    )                                                             # [新增]


# ---- 精选特征2: volatility_skew (波动率偏度) ----             # [新增]
@register_feature(                                                # [新增]
    group="波动率",                                                # [新增]
    level="level5_cross",                                         # [新增]
    description="收益率的滚动偏度，正偏表示上涨动能强，负偏表示下跌动能强",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["volatility_skew"],                             # [新增]
)                                                                 # [新增]
def compute_volatility_skew(df, features_df=None, window=20, **kwargs):  # [新增]
    """计算收益率的滚动偏度。

    偏度衡量收益分布的不对称性：正偏表示右尾更长（上涨趋势），
    负偏表示左尾更长（下跌趋势）。偏度变化往往先于价格反转。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小

    Returns
    -------
    pd.DataFrame
        包含 volatility_skew 列
    """                                                           # [新增]
    returns = df["close"].pct_change()                            # [新增]
    skew = returns.rolling(                                       # [新增]
        window, min_periods=max(5, window // 3)                   # [新增]
    ).skew()                                                      # [新增]
    return pd.DataFrame(                                          # [新增]
        {"volatility_skew": skew}, index=df.index                 # [新增]
    )                                                             # [新增]


# ---- 精选特征3: momentum_cross (多周期动量差) ----             # [新增]
@register_feature(                                                # [新增]
    group="动量指标",                                              # [新增]
    level="level5_cross",                                         # [新增]
    description="快慢周期动量差值，正值表示加速，负值表示减速",   # [新增]
    depends_on=[],                                                # [新增]
    output_names=["momentum_cross"],                              # [新增]
)                                                                 # [新增]
def compute_momentum_cross(                                       # [新增]
    df, features_df=None, fast_window=5, slow_window=20, **kwargs  # [新增]
):                                                                # [新增]
    """计算快慢周期动量差值。

    短期动量 > 长期动量 → 趋势加速；反之 → 趋势减速。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    fast_window : int
        短期动量窗口
    slow_window : int
        长期动量窗口

    Returns
    -------
    pd.DataFrame
        包含 momentum_cross 列
    """                                                           # [新增]
    fast_mom = df["close"].pct_change(fast_window)                # [新增]
    slow_mom = df["close"].pct_change(slow_window)                # [新增]
    cross = fast_mom - slow_mom                                   # [新增]
    return pd.DataFrame(                                          # [新增]
        {"momentum_cross": cross}, index=df.index                 # [新增]
    )                                                             # [新增]


# ---- 精选特征4: autocorrelation_1 (1阶自相关系数) ----        # [新增]
@register_feature(                                                # [新增]
    group="时间序列",                                              # [新增]
    level="level6_transforms",                                    # [新增]
    description="收益率的1阶自相关系数，正值表示趋势性，负值表示均值回归",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["autocorrelation_1"],                           # [新增]
)                                                                 # [新增]
def compute_autocorrelation_1(                                    # [新增]
    df, features_df=None, window=20, **kwargs                     # [新增]
):                                                                # [新增]
    """计算滚动自相关系数。

    衡量序列的短期记忆性：正值表示趋势性（上涨后更可能上涨），
    负值表示均值回归倾向。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小

    Returns
    -------
    pd.DataFrame
        包含 autocorrelation_1 列
    """                                                           # [新增]
    from features.feature_engineering_enhanced import (            # [新增]
        _rolling_autocorr_numba,                                  # [新增]
    )                                                             # [新增]
    returns = df["close"].pct_change().values.astype(np.float64)  # [新增]
    result = _rolling_autocorr_numba(returns, window)             # [新增]
    return pd.DataFrame(                                          # [新增]
        {"autocorrelation_1": result}, index=df.index             # [新增]
    )                                                             # [新增]


# ---- 精选特征5: vwap_std (VWAP标准差) ----                    # [新增]
@register_feature(                                                # [新增]
    group="成交量",                                                # [新增]
    level="level5_cross",                                         # [新增]
    description="VWAP的滚动标准差，衡量成交价格离散度，高值表示市场分歧大",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["vwap_std"],                                    # [新增]
)                                                                 # [新增]
def compute_vwap_std(df, features_df=None, window=20, **kwargs):  # [新增]
    """计算VWAP的滚动标准差。

    标准差小 → 成交集中，市场共识强；标准差大 → 成交分散，
    市场分歧大，可能预示反转。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: high, low, close, volume
    window : int
        滚动窗口大小

    Returns
    -------
    pd.DataFrame
        包含 vwap_std 列
    """                                                           # [新增]
    high = df["high"].values.astype(np.float64)                   # [新增]
    low = df["low"].values.astype(np.float64)                     # [新增]
    close = df["close"].values.astype(np.float64)                 # [新增]
    volume = df["volume"].values.astype(np.float64)               # [新增]
                                                                  # [新增]
    typical_price = (high + low + close) / 3.0                    # [新增]
    min_p = max(3, window // 3)                                   # [新增]
                                                                  # [新增]
    tp_vol = typical_price * volume                               # [新增]
    vol_sum = pd.Series(volume).rolling(                          # [新增]
        window, min_periods=min_p                                 # [新增]
    ).sum().values                                                # [新增]
    tp_vol_sum = pd.Series(tp_vol).rolling(                       # [新增]
        window, min_periods=min_p                                 # [新增]
    ).sum().values                                                # [新增]
    vwap = tp_vol_sum / (vol_sum + 1e-12)                         # [新增]
                                                                  # [新增]
    weighted_sq = (typical_price - vwap) ** 2 * volume            # [新增]
    weighted_sq_sum = pd.Series(weighted_sq).rolling(             # [新增]
        window, min_periods=min_p                                 # [新增]
    ).sum().values                                                # [新增]
    variance = weighted_sq_sum / (vol_sum + 1e-12)                # [新增]
    vwap_std = np.sqrt(np.maximum(variance, 0))                   # [新增]
                                                                  # [新增]
    return pd.DataFrame(                                          # [新增]
        {"vwap_std": vwap_std}, index=df.index                    # [新增]
    )                                                             # [新增]


# ---- 精选特征6: tick_imbalance_proxy (Tick不平衡代理) ----     # [新增]
@register_feature(                                                # [新增]
    group="微观结构",                                              # [新增]
    level="level4_micro",                                         # [新增]
    description="价格变动方向的不平衡代理，正值表示买方主导，负值表示卖方主导",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["tick_imbalance_proxy"],                        # [新增]
)                                                                 # [新增]
def compute_tick_imbalance_proxy(                                 # [新增]
    df, features_df=None, window=10, **kwargs                     # [新增]
):                                                                # [新增]
    """计算tick不平衡代理指标。

    用分钟K线的价格变动方向模拟tick不平衡：统计窗口内上涨K线
    数量与下跌K线数量的差值比例。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        统计窗口大小

    Returns
    -------
    pd.DataFrame
        包含 tick_imbalance_proxy 列
    """                                                           # [新增]
    close = df["close"].values.astype(np.float64)                 # [新增]
                                                                  # [新增]
    price_diff = np.diff(close, prepend=np.nan)                   # [新增]
    up_ticks = (price_diff > 0).astype(float)                     # [新增]
    down_ticks = (price_diff < 0).astype(float)                   # [新增]
                                                                  # [新增]
    min_p = max(3, window // 2)                                   # [新增]
    up_sum = pd.Series(up_ticks).rolling(                         # [新增]
        window, min_periods=min_p                                 # [新增]
    ).sum().values                                                # [新增]
    down_sum = pd.Series(down_ticks).rolling(                     # [新增]
        window, min_periods=min_p                                 # [新增]
    ).sum().values                                                # [新增]
                                                                  # [新增]
    total = up_sum + down_sum                                     # [新增]
    imbalance = np.where(total > 0, (up_sum - down_sum) / total, np.nan)  # [新增]
                                                                  # [新增]
    return pd.DataFrame(                                          # [新增]
        {"tick_imbalance_proxy": imbalance}, index=df.index       # [新增]
    )                                                             # [新增]


# ---- 精选特征7: volatility_of_volatility (波动率的波动率) ---- # [新增]
@register_feature(                                                # [新增]
    group="波动率",                                                # [新增]
    level="level5_cross",                                         # [新增]
    description="波动率的波动率，高值表示市场不稳定期",           # [新增]
    depends_on=[],                                                # [新增]
    output_names=["volatility_of_volatility"],                    # [新增]
)                                                                 # [新增]
def compute_volatility_of_volatility(                             # [新增]
    df, features_df=None, vol_window=20, vov_window=20, **kwargs  # [新增]
):                                                                # [新增]
    """计算波动率的波动率。

    当vov高时，市场处于不稳定期，容易出现趋势转折或异常行情；
    当vov低时，市场波动稳定，适合趋势跟踪策略。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    vol_window : int
        波动率计算窗口
    vov_window : int
        波动率的波动率计算窗口

    Returns
    -------
    pd.DataFrame
        包含 volatility_of_volatility 列
    """                                                           # [新增]
    returns = df["close"].pct_change()                            # [新增]
    volatility = returns.rolling(                                 # [新增]
        vol_window, min_periods=max(5, vol_window // 3)           # [新增]
    ).std()                                                       # [新增]
    vov = volatility.rolling(                                     # [新增]
        vov_window, min_periods=max(5, vov_window // 3)           # [新增]
    ).std()                                                       # [新增]
                                                                  # [新增]
    return pd.DataFrame(                                          # [新增]
        {"volatility_of_volatility": vov}, index=df.index         # [新增]
    )                                                             # [新增]


# ---- 精选特征8: trend_strength_ratio (趋势强度比率) ----       # [新增]
@register_feature(                                                # [新增]
    group="趋势指标",                                              # [新增]
    level="level5_cross",                                         # [新增]
    description="短周期趋势强度与长周期趋势强度的比率，经tanh归一化到[-1,1]",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["trend_strength_ratio"],                        # [新增]
)                                                                 # [新增]
def compute_trend_strength_ratio(                                 # [新增]
    df, features_df=None, short_window=10, long_window=30, **kwargs  # [新增]
):                                                                # [新增]
    """计算趋势强度比率。

    短期斜率 / |长期斜率| 的比值反映短期动量的相对强弱，
    通过tanh压缩到[-1,1]区间。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    short_window : int
        短周期窗口
    long_window : int
        长周期窗口

    Returns
    -------
    pd.DataFrame
        包含 trend_strength_ratio 列
    """                                                           # [新增]
    from features.feature_engineering_enhanced import (            # [新增]
        _rolling_slope,                                           # [新增]
    )                                                             # [新增]
    close = df["close"].values.astype(np.float64)                 # [新增]
                                                                  # [新增]
    short_slope = _rolling_slope(close, short_window)             # [新增]
    long_slope = _rolling_slope(close, long_window)               # [新增]
                                                                  # [新增]
    ratio = short_slope / (np.abs(long_slope) + 1e-8)             # [新增]
    ratio_norm = np.tanh(ratio * 10)                              # [新增]
                                                                  # [新增]
    return pd.DataFrame(                                          # [新增]
        {"trend_strength_ratio": ratio_norm}, index=df.index      # [新增]
    )                                                             # [新增]


# ---- 精选特征9: volume_profile_skew (成交量分布偏度) ----      # [新增]
@_njit_custom                                                     # [新增]
def _volume_profile_skew_numba(high, low, close, volume, window, bins):  # [新增]
    """Numba加速的成交量分布偏度计算"""                           # [新增]
    n = len(close)                                                # [新增]
    result = np.full(n, np.nan)                                   # [新增]
                                                                  # [新增]
    for i in range(window, n):                                    # [新增]
        w_high = high[i - window: i]                              # [新增]
        w_low = low[i - window: i]                                # [新增]
        w_close = close[i - window: i]                            # [新增]
        w_vol = volume[i - window: i]                             # [新增]
                                                                  # [新增]
        price_min = np.min(w_low)                                 # [新增]
        price_max = np.max(w_high)                                # [新增]
        if price_max - price_min < 1e-8:                          # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        bin_edges = np.linspace(price_min, price_max, bins + 1)   # [新增]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2        # [新增]
        vol_by_price = np.zeros(bins)                             # [新增]
                                                                  # [新增]
        for j in range(window):                                   # [新增]
            price = w_close[j]                                    # [新增]
            vol = w_vol[j]                                        # [新增]
            for k in range(bins):                                 # [新增]
                if bin_edges[k] <= price < bin_edges[k + 1]:      # [新增]
                    vol_by_price[k] += vol                        # [新增]
                    break                                         # [新增]
                                                                  # [新增]
        total_vol = np.sum(vol_by_price)                          # [新增]
        if total_vol < 1e-8:                                      # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        weighted_price = np.sum(bin_centers * vol_by_price) / total_vol  # [新增]
        mid_price = (price_min + price_max) / 2                   # [新增]
        price_range = price_max - price_min                       # [新增]
                                                                  # [新增]
        skew = (weighted_price - mid_price) / (price_range / 2)   # [新增]
        result[i] = skew                                          # [新增]
                                                                  # [新增]
    return result                                                 # [新增]


@register_feature(                                                # [新增]
    group="成交量",                                                # [新增]
    level="level6_transforms",                                    # [新增]
    description="成交量分布偏度，正偏表示成交偏向高价区（阻力），负偏表示成交偏向低价区（支撑）",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["volume_profile_skew"],                         # [新增]
)                                                                 # [新增]
def compute_volume_profile_skew(                                  # [新增]
    df, features_df=None, window=20, bins=10, **kwargs            # [新增]
):                                                                # [新增]
    """计算成交量分布偏度。

    正偏表示成交偏向高价区，可能形成上涨阻力；
    负偏表示成交偏向低价区，可能形成下跌支撑。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: high, low, close, volume
    window : int
        滚动窗口大小
    bins : int
        价格区间数量

    Returns
    -------
    pd.DataFrame
        包含 volume_profile_skew 列
    """                                                           # [新增]
    high = df["high"].values.astype(np.float64)                   # [新增]
    low = df["low"].values.astype(np.float64)                     # [新增]
    close = df["close"].values.astype(np.float64)                 # [新增]
    volume = df["volume"].values.astype(np.float64)               # [新增]
                                                                  # [新增]
    result = _volume_profile_skew_numba(                          # [新增]
        high, low, close, volume, window, bins                    # [新增]
    )                                                             # [新增]
    return pd.DataFrame(                                          # [新增]
        {"volume_profile_skew": result}, index=df.index           # [新增]
    )                                                             # [新增]


# ---- 精选特征10: hurst_exponent_approx (Hurst指数近似) ----    # [新增]
@_njit_custom                                                     # [新增]
def _hurst_exponent_numba(returns, min_window, max_window):       # [新增]
    """Numba加速的Hurst指数近似计算"""                            # [新增]
    n = len(returns)                                              # [新增]
    if n < max_window * 2:                                        # [新增]
        return np.nan                                             # [新增]
                                                                  # [新增]
    upper = min(max_window, n // 2)                               # [新增]
    if upper <= min_window:                                       # [新增]
        return np.nan                                             # [新增]
    lags = np.arange(min_window, upper)                           # [新增]
    if len(lags) < 2:                                             # [新增]
        return np.nan                                             # [新增]
                                                                  # [新增]
    tau = np.zeros(len(lags))                                     # [新增]
                                                                  # [新增]
    for idx in range(len(lags)):                                  # [新增]
        lag = lags[idx]                                           # [新增]
        if lag >= n:                                              # [新增]
            tau[idx] = np.nan                                     # [新增]
            continue                                              # [新增]
                                                                  # [新增]
        diff_sum = 0.0                                            # [新增]
        count = 0                                                 # [新增]
        for i in range(lag, n):                                   # [新增]
            if not np.isnan(returns[i]) and not np.isnan(returns[i - lag]):  # [新增]
                diff = returns[i] - returns[i - lag]              # [新增]
                diff_sum += diff * diff                           # [新增]
                count += 1                                        # [新增]
                                                                  # [新增]
        if count > 0:                                             # [新增]
            tau[idx] = np.sqrt(diff_sum / count)                  # [新增]
        else:                                                     # [新增]
            tau[idx] = np.nan                                     # [新增]
                                                                  # [新增]
    valid_lags = []                                               # [新增]
    valid_tau = []                                                # [新增]
    for i in range(len(lags)):                                    # [新增]
        if not np.isnan(tau[i]) and tau[i] > 0:                   # [新增]
            valid_lags.append(lags[i])                            # [新增]
            valid_tau.append(tau[i])                              # [新增]
                                                                  # [新增]
    if len(valid_lags) < 2:                                       # [新增]
        return np.nan                                             # [新增]
                                                                  # [新增]
    log_lags = np.log(np.array(valid_lags, dtype=np.float64))     # [新增]
    log_tau = np.log(np.array(valid_tau, dtype=np.float64))       # [新增]
                                                                  # [新增]
    n_valid = len(log_lags)                                       # [新增]
    mean_x = np.mean(log_lags)                                    # [新增]
    mean_y = np.mean(log_tau)                                     # [新增]
                                                                  # [新增]
    cov = 0.0                                                     # [新增]
    var_x = 0.0                                                   # [新增]
    for i in range(n_valid):                                      # [新增]
        cov += (log_lags[i] - mean_x) * (log_tau[i] - mean_y)    # [新增]
        var_x += (log_lags[i] - mean_x) ** 2                     # [新增]
                                                                  # [新增]
    if var_x < 1e-12:                                             # [新增]
        return np.nan                                             # [新增]
                                                                  # [新增]
    hurst = cov / var_x                                           # [新增]
    return hurst                                                  # [新增]


@register_feature(                                                # [新增]
    group="时间序列",                                              # [新增]
    level="level6_transforms",                                    # [新增]
    description="Hurst指数近似值，>0.5表示趋势性，=0.5表示随机游走，<0.5表示均值回归",  # [新增]
    depends_on=[],                                                # [新增]
    output_names=["hurst_exponent_approx"],                       # [新增]
)                                                                 # [新增]
def compute_hurst_exponent_approx(                                # [新增]
    df, features_df=None,                                         # [新增]
    min_window=10, max_window=50, min_periods=100, **kwargs       # [新增]
):                                                                # [新增]
    """计算Hurst指数近似值。

    H > 0.5: 趋势性序列（长记忆性）
    H = 0.5: 随机游走
    H < 0.5: 均值回归（负记忆性）

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    min_window : int
        最小滞后窗口
    max_window : int
        最大滞后窗口
    min_periods : int
        最小所需数据量

    Returns
    -------
    pd.DataFrame
        包含 hurst_exponent_approx 列
    """                                                           # [新增]
    close = df["close"].values.astype(np.float64)                 # [新增]
                                                                  # [新增]
    returns = np.zeros(len(close))                                # [新增]
    returns[0] = np.nan                                           # [新增]
    for i in range(1, len(close)):                                # [新增]
        if close[i - 1] > 0 and not np.isnan(close[i - 1]):      # [新增]
            returns[i] = (close[i] - close[i - 1]) / close[i - 1]  # [新增]
        else:                                                     # [新增]
            returns[i] = np.nan                                   # [新增]
                                                                  # [新增]
    result = np.full(len(close), np.nan)                          # [新增]
                                                                  # [新增]
    for i in range(min_periods, len(close)):                      # [新增]
        window_returns = returns[i - min_periods: i]              # [新增]
        hurst = _hurst_exponent_numba(                            # [新增]
            window_returns, min_window, max_window                # [新增]
        )                                                         # [新增]
        result[i] = hurst                                         # [新增]
                                                                  # [新增]
    return pd.DataFrame(                                          # [新增]
        {"hurst_exponent_approx": result}, index=df.index         # [新增]
    )                                                             # [新增]
