"""
商品期货机器学习量化模型 - 自定义特征
Custom features registered via @register_feature decorator.

依据《hfml特征工程增强报告-56项目挖掘-最终可执行版.md》集成的9个特征因子，
加上2个原有示例特征，共11个注册特征。

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
