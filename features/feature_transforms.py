"""
商品期货机器学习量化模型 - 深度特征变换模块
Deep feature transformations for maximizing information extraction from existing data.

在现有数据边界内挖掘深度特征，不新增数据源，只提升信息密度。

特征变换类别:
1. 非线性变换 — 捕捉极端值和消除异方差
2. 跨周期比率 — 利用不同窗口模拟多周期对比
3. 特征变化率与加速度 — 指标的一阶/二阶导数
4. 条件特征 — 基于市场状态的动态特征
5. 特征交互项 — 技术指标与量/仓的交互效应
"""

import numpy as np
import pandas as pd

from features.rolling_numba import rolling_mean, rolling_std, rsi as nb_rsi


# --- 改进点 2.4: 深度变换 pct_change()*100 ---

def pct_change_100(series: pd.Series) -> pd.Series:  # [新增]
    return series.pct_change() * 100.0  # [新增]


def compute_nonlinear_transforms(df, features=None):
    """
    对关键特征进行非线性变换。

    对每个特征生成四种变换:
    - _squared: 平方项（捕捉极端值，放大强信号）
    - _log: 对数变换（降低异方差，压缩极端值）
    - _rank: 滚动百分位排名（反映相对位置，消除量纲）
    - _zscore: Z-score标准化（标准化后的偏离度）

    Parameters
    ----------
    df : pd.DataFrame
        特征矩阵（已计算好的特征DataFrame）
    features : list of str, optional
        要变换的特征列表。默认使用 ['rsi_14', 'macd_hist', 'atr', 'vol_ratio']

    Returns
    -------
    pd.DataFrame
        非线性变换后的新特征
    """
    if features is None:
        features = ["rsi_14", "macd_hist", "atr", "vol_ratio"]

    result = {}

    for feat in features:
        if feat not in df.columns:
            continue

        vals = df[feat]

        # 平方项（保留符号方向）
        result[f"{feat}_squared"] = vals ** 2

        # 对数变换（保留符号）
        result[f"{feat}_log"] = np.log1p(vals.abs()) * np.sign(vals)

        # 滚动百分位排名（最近60周期内的相对位置）
        result[f"{feat}_rank"] = vals.rolling(60, min_periods=20).rank(pct=True)

        # Z-score标准化
        roll_mean = vals.rolling(20, min_periods=5).mean()
        roll_std = vals.rolling(20, min_periods=5).std()
        result[f"{feat}_zscore"] = (vals - roll_mean) / (roll_std + 1e-8)

    return pd.DataFrame(result, index=df.index)


def compute_cross_period_ratios(df):
    """
    计算跨周期比率特征。

    在同一K线周期内，用不同窗口长度计算同一指标来模拟跨周期对比，
    捕捉短期vs长期的相对强弱关系。

    输出特征:
    - rsi_fast_slow_ratio: 快速RSI(6) / 慢速RSI(24) 比率
    - volatility_ratio_fast_slow: 短期波动率(5) / 长期波动率(20) 比率
    - ma_cross_ratio: 快速MA(5) / 慢速MA(20) 比率

    Parameters
    ----------
    df : pd.DataFrame
        原始K线数据，需包含 close 列

    Returns
    -------
    pd.DataFrame
        跨周期比率特征
    """
    result = {}
    close = df["close"].values.astype(np.float64)

    # 快速RSI vs 慢速RSI
    rsi_fast = nb_rsi(close, 6)
    rsi_slow = nb_rsi(close, 24)
    result["rsi_fast_slow_ratio"] = rsi_fast / (rsi_slow + 1e-8)

    # 短期波动率 / 长期波动率
    returns = np.empty_like(close)
    returns[0] = np.nan
    returns[1:] = close[1:] / close[:-1] - 1.0
    vol_short = rolling_std(returns, 5)
    vol_long = rolling_std(returns, 20)
    result["volatility_ratio_fast_slow"] = vol_short / (vol_long + 1e-8)

    # 快速MA / 慢速MA
    ma_fast = rolling_mean(close, 5)
    ma_slow = rolling_mean(close, 20)
    result["ma_cross_ratio"] = ma_fast / (ma_slow + 1e-8)

    return pd.DataFrame(result, index=df.index)


def compute_feature_velocity(df, features=None):
    """
    计算特征的变化率（一阶导数）与加速度（二阶导数）。

    不仅看指标当前值，更关注指标的变化速度和变化方向的变化，
    这对趋势反转的提前预警特别有价值。

    对每个特征生成:
    - _velocity: 3周期差分（一阶导数，变化率）
    - _acceleration: 3周期差分的差分（二阶导数，加速度）
    - _direction_consistency: 5周期内变化方向的一致性（-1~+1）

    Parameters
    ----------
    df : pd.DataFrame
        特征矩阵
    features : list of str, optional
        要计算的特征列表。默认使用 ['rsi_14', 'macd_hist', 'boll_width']

    Returns
    -------
    pd.DataFrame
        特征的变化率和加速度
    """
    if features is None:
        features = ["rsi_14", "macd_hist", "boll_width"]

    result = {}

    for feat in features:
        if feat not in df.columns:
            continue

        vals = df[feat]

        # 一阶导数（3周期变化率）
        velocity = vals.diff(3) / 3.0
        result[f"{feat}_velocity"] = velocity

        # 二阶导数（加速度）
        result[f"{feat}_acceleration"] = velocity.diff(3) / 3.0

        # 变化方向一致性
        diffs = vals.diff()
        direction_sum = diffs.rolling(5, min_periods=2).apply(
            lambda x: np.nansum(np.sign(x)) / len(x), raw=True
        )
        result[f"{feat}_direction_consistency"] = direction_sum

    return pd.DataFrame(result, index=df.index)


def compute_contextual_features(df):
    """
    计算基于市场状态的条件特征。

    同一指标在不同市场状态下意义不同（如RSI在趋势市和震荡市的含义不同）。
    通过将指标与市场状态交叉，生成条件感知的特征。

    输出特征:
    - rsi_high_vol: 高波动环境下的RSI值（低波动时为NaN）
    - rsi_low_vol: 低波动环境下的RSI值（高波动时为NaN）
    - momentum_in_trend: 趋势市中的动量值
    - momentum_in_range: 震荡市中的动量值

    Parameters
    ----------
    df : pd.DataFrame
        特征矩阵，需包含 close 列

    Returns
    -------
    pd.DataFrame
        条件特征
    """
    result = {}
    close = df["close"]

    # 判断市场状态
    returns = close.pct_change()
    volatility = returns.rolling(20, min_periods=5).std()
    vol_median = volatility.expanding(min_periods=20).median()

    high_vol = volatility > vol_median
    low_vol = ~high_vol

    # RSI在不同波动率环境下
    if "rsi_14" in df.columns:
        result["rsi_high_vol"] = df["rsi_14"].where(high_vol, 0.0)
        result["rsi_low_vol"] = df["rsi_14"].where(low_vol, 0.0)

    # 动量在趋势/震荡环境下
    ma_20 = close.rolling(20, min_periods=5).mean()
    trend_strength = (close - ma_20).abs() / (ma_20 + 1e-8)
    is_trending = trend_strength > trend_strength.rolling(
        60, min_periods=20
    ).median()

    momentum = returns.rolling(5, min_periods=2).sum()
    result["momentum_in_trend"] = momentum.where(is_trending, 0.0)
    result["momentum_in_range"] = momentum.where(~is_trending, 0.0)

    return pd.DataFrame(result, index=df.index)


def compute_feature_interactions(df):
    """
    计算特征间的交互效应。

    捕捉技术指标与成交量/持仓量之间的乘法交互关系，
    这些交互效应往往比单独的指标更有预测价值。

    输出特征:
    - rsi_volume_interaction: RSI × 量比（超买/超卖 + 放量 = 强信号）
    - momentum_vol_interaction: MACD柱 / ATR（波动率标准化后的动量）
    - oi_price_alignment: 持仓变化方向 × 价格变化方向（+1=同向=趋势延续, -1=反向=可能反转）
    - oi_price_magnitude: |持仓变化| / |价格变化|（资金流与价格变动的比例）

    Parameters
    ----------
    df : pd.DataFrame
        特征矩阵，可能包含 rsi_14, vol_ratio_5, macd_hist, atr,
        oi_change, close 等列

    Returns
    -------
    pd.DataFrame
        交互特征
    """
    result = {}

    # RSI × 量比
    if "rsi_14" in df.columns and "vol_ratio_5" in df.columns:
        result["rsi_volume_interaction"] = df["rsi_14"] * df["vol_ratio_5"]

    # 动量 / 波动率
    if "macd_hist" in df.columns and "atr" in df.columns:
        result["momentum_vol_interaction"] = df["macd_hist"] / (df["atr"] + 1e-8)

    # 持仓变化 × 价格变化方向
    if "oi_change" in df.columns and "close" in df.columns:
        returns = df["close"].pct_change()
        result["oi_price_alignment"] = (
            np.sign(df["oi_change"]) * np.sign(returns)
        )
        # 限制比值上界，防止价格变化极小时产生极端值
        max_oi_price_ratio = 100.0
        result["oi_price_magnitude"] = (
            df["oi_change"].abs() / (returns.abs() + 1e-8)
        ).clip(upper=max_oi_price_ratio)

    return pd.DataFrame(result, index=df.index)


def compute_all_transforms(df, raw_df=None):
    """
    计算所有深度特征变换。

    Parameters
    ----------
    df : pd.DataFrame
        已计算好的特征矩阵（含基础特征）
    raw_df : pd.DataFrame, optional
        原始K线数据（用于跨周期比率计算）。若不提供则从df推断。

    Returns
    -------
    pd.DataFrame
        所有深度变换特征的合并结果
    """
    frames = []

    # 1. 非线性变换
    frames.append(compute_nonlinear_transforms(df))

    # 2. 跨周期比率（需要原始close）
    source = raw_df if raw_df is not None and "close" in raw_df.columns else df
    if "close" in source.columns:
        frames.append(compute_cross_period_ratios(source))

    # 3. 特征变化率与加速度
    frames.append(compute_feature_velocity(df))

    # 4. 条件特征
    if "close" in df.columns or (raw_df is not None and "close" in raw_df.columns):
        ctx_source = df if "close" in df.columns else raw_df
        frames.append(compute_contextual_features(
            pd.concat([ctx_source[["close"]], df], axis=1)
            if "close" not in df.columns else df
        ))

    # 5. 特征交互项
    frames.append(compute_feature_interactions(df))

    # 合并所有变换（跳过空DataFrame）
    non_empty = [f for f in frames if len(f.columns) > 0]
    if non_empty:
        return pd.concat(non_empty, axis=1)
    return pd.DataFrame(index=df.index)
