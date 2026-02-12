"""
商品期货机器学习量化模型 - 自定义特征示例
Example custom features demonstrating the @register_feature decorator.

每个函数通过 @register_feature 装饰器自动注册到全局 FeatureRegistry，
添加后无需修改任何下游代码（特征筛选、模型训练、回测等）。

新增特征步骤:
    1. 在本文件中定义计算函数，并用 @register_feature 装饰
    2. 完成！系统自动发现、计算、参与筛选和训练
"""

import numpy as np                                            # [新增]
import pandas as pd                                           # [新增]

from features.feature_registry import register_feature        # [新增]


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
