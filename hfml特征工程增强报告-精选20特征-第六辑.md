# hfml特征工程增强报告-精选20特征-第六辑（单品种期货专用）

**报告时间**：2026-02-25  
**文档版本**：v6.1-clean  
**适用范围**：单品种商品期货（1m/5m/15m），输入为 `OHLCV + open_interest（持仓量）`

---

## 一、整理说明（本次合并结果）

本文件由多轮对话稿合并整理，已完成以下处理：

1. 删除重复章节（原稿 + 续完 + 剩余补充的重复内容）。
2. 统一为**唯一“最终20特征”口径**，避免同名特征多版本冲突。
3. 保留可选扩展特征 2 个，不计入 20 个正式名额。
4. 统一特征命名、编号、优先级和依赖字段表述。

---

## 二、最终20特征（唯一口径）

> 说明：以下 20 个为第六辑正式特征，编号固定；同名函数建议一处定义、唯一输出。

| 序号 | 特征名 | 类别 | 优先级 | 依赖列 |
| --- | --- | --- | --- | --- |
| 1 | `open_interest_velocity` | 持仓量深度 | P0 | `open_interest` |
| 2 | `open_interest_acceleration` | 持仓量深度 | P0 | `open_interest` |
| 3 | `long_short_imbalance` | 多空博弈 | P0 | `close, open_interest, volume` |
| 4 | `new_position_ratio` | 开平仓压力 | P1 | `open_interest, volume` |
| 5 | `liquidation_pressure` | 开平仓压力 | P1 | `close, open_interest` |
| 6 | `hedging_ratio_proxy` | 套保比率 | P1 | `open_interest, volume` |
| 7 | `speculation_index` | 投机度 | P1 | `volume, open_interest` |
| 8 | `volume_oi_correlation` | 持仓量深度 | P1 | `volume, open_interest` |
| 9 | `oi_seasonal_pattern` | 持仓量结构 | P2 | `open_interest` |
| 10 | `rollover_activity` | 展期行为 | P1 | `volume, open_interest` |
| 11 | `contract_rolling_pressure` | 主力合约切换 | P1 | `close, volume, open_interest` |
| 12 | `oi_price_regime` | 持仓量结构 | P1 | `close, open_interest` |
| 13 | `volume_oi_ratio_zscore` | 成交量深度 | P1 | `volume, open_interest` |
| 14 | `oi_price_divergence_strength` | 持仓量结构 | P1 | `close, open_interest` |
| 15 | `oi_extreme_ratio` | 持仓量结构 | P2 | `open_interest` |
| 16 | `volume_breakout` | 成交量深度 | P1 | `volume` |
| 17 | `volume_stability` | 成交量深度 | P2 | `volume` |
| 18 | `price_conviction` | 价格形态 | P1 | `high, low, close` |
| 19 | `oi_momentum_ratio` | 持仓量结构 | P1 | `open_interest` |
| 20 | `volume_price_efficiency` | 量价关系 | P1 | `close, volume` |

---

## 二点一、代码一一对应（已验证）

> 对应来源：`见【九、特征因子代码】`。  
> 验证口径：仅使用当前与历史数据（无 `shift(-k)`、无未来窗口、无 `center=True`）。

| 序号 | 文档特征名 | 代码函数名 | 默认参数 | 核心实现（代码口径） | 未来函数检查 |
| --- | --- | --- | --- | --- | --- |
| 1 | `open_interest_velocity` | `compute_open_interest_velocity` | `window=5` | `(oi_t - oi_{t-window}) / window` | 通过 |
| 2 | `open_interest_acceleration` | `compute_open_interest_acceleration` | `window=5` | `velocity_t - velocity_{t-window}` 再除 `window` | 通过 |
| 3 | `long_short_imbalance` | `compute_long_short_imbalance` | `window=10` | `sign(close.diff) * (oi.diff/oi_roll_mean) * (vol/vol_roll_mean)` | 通过 |
| 4 | `new_position_ratio` | `compute_new_position_ratio` | `window=5` | `abs(oi.diff)/(volume+eps)` 后滚动均值 | 通过 |
| 5 | `liquidation_pressure` | `compute_liquidation_pressure` | `window=10` | 仅在 `price_change*oi_change<0` 时取 `|pct_close|*|pct_oi|` 并平滑 | 通过 |
| 6 | `hedging_ratio_proxy` | `compute_hedging_ratio_proxy` | `window=20` | `oi/(volume+eps)` 后滚动均值 | 通过 |
| 7 | `speculation_index` | `compute_speculation_index` | `window=20` | `volume/(oi+eps)` 后滚动均值 | 通过 |
| 8 | `volume_oi_correlation` | `compute_volume_oi_correlation` | `window=20` | `rolling_corr(volume, oi)` | 通过 |
| 9 | `oi_seasonal_pattern` | `compute_oi_seasonal_pattern` | `window=20`（`lookback_window=4*window`） | 滚动窗口内“当前值历史分位” | 通过 |
| 10 | `rollover_activity` | `compute_rollover_activity` | `window=20` | `(volume/volume_roll_mean) * clip((oi_{t-window}-oi_t)/oi_{t-window},0,∞)` | 通过 |
| 11 | `contract_rolling_pressure` | `compute_contract_rolling_pressure` | `window=10` | `sign(close_t-close_{t-window}) * oi_decline * vol_surge`，并加阈值掩码 | 通过 |
| 12 | `oi_price_regime` | `compute_oi_price_regime` | `window=5` | 按 `price_trend` 与 `oi_trend` 四象限编码 `1~4`（其余 `0`） | 通过 |
| 13 | `volume_oi_ratio_zscore` | `compute_volume_oi_ratio_zscore` | `window=20` | `zscore(volume/oi)`（滚动均值与滚动标准差） | 通过 |
| 14 | `oi_price_divergence_strength` | `compute_oi_price_divergence_strength` | `window=5` | 若价格与持仓方向相反，取 `|price_ret|*|oi_ret|` | 通过 |
| 15 | `oi_extreme_ratio` | `compute_oi_extreme_ratio` | `window=100` | `(oi-rolling_min)/(rolling_max-rolling_min+eps)` | 通过 |
| 16 | `volume_breakout` | `compute_volume_breakout` | `window=20` | `volume/volume_roll_mean - 1` | 通过 |
| 17 | `volume_stability` | `compute_volume_stability` | `window=20` | `rolling_mean(volume)/(rolling_std(volume)+eps)` | 通过 |
| 18 | `price_conviction` | `compute_price_conviction` | 无 | `abs(2*((close-low)/(high-low))-1)`；`high==low` 置 NaN | 通过 |
| 19 | `oi_momentum_ratio` | `compute_oi_momentum_ratio` | `fast=5, slow=20` | `MA_fast(oi)/MA_slow(oi)`（`fast>slow` 自动互换） | 通过 |
| 20 | `volume_price_efficiency` | `compute_volume_price_efficiency` | `window=20, scale_factor=10000` | `abs(close_t-close_{t-window}) / rolling_sum(volume) * scale_factor` | 通过 |

## 三、替换关系（已固化）

以下旧候选已由本版替换并冻结，不再作为第六辑正式项：

- `delivery_month_effect` → `oi_seasonal_pattern`
- `carry_cost_proxy / calendar_spread / calendar_spread_velocity / calendar_spread_zscore / roll_yield / term_structure_*` → `oi_price_regime`（单品种无跨合约输入时的统一替代）
- `funding_rate_proxy` → `volume_oi_ratio_zscore`

> 原则：本版仅保留**单品种可直接计算**且不依赖其他合约/外部数据的特征。

---

## 四、实现规范（统一约束）

### 4.1 输入字段要求

- 必备：`open, high, low, close, volume, open_interest`
- 索引：建议 `DatetimeIndex`
- 缺失值：特征函数需容忍 NaN，窗口前段允许返回 NaN

### 4.2 输出规范

- 每个特征函数返回 `pd.DataFrame`，列名与 `output_names` 一致。
- 正式20特征默认每个函数**单列输出**，避免同名函数在不同版本出现多列冲突。
- 扩展列（如 zscore、binary）若保留，建议单独命名为“扩展函数”，不占20正式名额。

### 4.3 数值与稳定性

- 除零统一使用 `+1e-12`
- 趋势判定阈值建议 `1e-8`
- 价格/持仓变化涉及比例时需对分母做有效性检查

---

## 五、Numba辅助函数（建议保留）

建议在 `features/feature_engineering_enhanced.py` 统一维护以下底层函数：

- `_rolling_velocity_numba(arr, window)`
- `_rolling_acceleration_numba(arr, window)`
- `_rolling_corr_numba(x, y, window)`
- `_zscore_numba(arr, window)`

用途分别对应：速度、加速度、滚动相关、滚动标准化。  
这些函数已在历史稿中多次复用，保留一份即可。

---

## 六、特征分组与默认参数

| 分组 | 特征 | 默认参数 |
| --- | --- | --- |
| 持仓量深度 | 1,2,8 | `window=5/5/20` |
| 多空博弈 | 3 | `window=10` |
| 开平仓压力 | 4,5 | `window=5/10` |
| 套保/投机 | 6,7 | `window=20/20` |
| 持仓量结构 | 9,12,14,15,19 | `window=20/5/5/100; fast=5, slow=20` |
| 展期/换月 | 10,11 | `window=20/10` |
| 成交量深度 | 13,16,17 | `window=20` |
| 价格形态 | 18 | 无窗口 |
| 量价关系 | 20 | `window=20, scale_factor=10000` |

---

## 七、可选扩展特征（不占20名额）

以下可作为实验特征，不纳入正式20：

1. `oi_mean_reversion`（持仓量均值回归）
2. `volume_price_divergence`（量价背离）

建议在实验阶段单独开关，不默认进入生产训练集。

---

## 八、落地集成步骤（建议）

1. 在 `features/feature_engineering_enhanced.py` 保留一套辅助函数定义。  
2. 为上述 20 个特征各保留一个 `compute_*` 实现，避免重名重复定义。  
3. 通过 `@register_feature` 统一注册，`output_names` 与实际返回列严格一致。  
4. 在特征加载/组装流程中，仅启用本页“最终20特征”清单。  
5. 运行全流程后检查：列是否齐全、是否存在重复列名、NaN比例是否符合预期。

## 九、特征因子代码

```
"""自定义特征实现：第六辑单品种期货20因子（含2个可选扩展）。"""

import numpy as np
import pandas as pd

from features.feature_registry import register_feature


EPSILON = 1e-12


def _min_periods(window: int, floor_value: int = 3) -> int:
    window = max(int(window), 1)
    return min(window, max(int(floor_value), window // 2))


def _safe_window(window: int, default: int = 5) -> int:
    try:
        w = int(window)
    except (TypeError, ValueError):
        return max(int(default), 1)
    return max(w, 1)


def _rolling_percentile_last(window_values: np.ndarray) -> float:
    valid_values = window_values[np.isfinite(window_values)]
    if valid_values.size == 0:
        return np.nan
    current_value = window_values[-1]
    if not np.isfinite(current_value):
        return np.nan
    return float(np.sum(valid_values < current_value) / valid_values.size)


@register_feature(
    group="持仓量深度",
    level="level4_micro",
    description="持仓量变化速度，正值表示资金流入加速",
    depends_on=[],
    output_names=["open_interest_velocity"],
)
def compute_open_interest_velocity(df, features_df=None, window: int = 5, **kwargs):
    window = _safe_window(window, 5)
    open_interest = df["open_interest"].astype(float)
    velocity = (open_interest - open_interest.shift(window)) / float(window)
    return pd.DataFrame({"open_interest_velocity": velocity}, index=df.index)


@register_feature(
    group="持仓量深度",
    level="level4_micro",
    description="持仓量加速度，正值表示资金流入加速增加",
    depends_on=[],
    output_names=["open_interest_acceleration"],
)
def compute_open_interest_acceleration(df, features_df=None, window: int = 5, **kwargs):
    window = _safe_window(window, 5)
    open_interest = df["open_interest"].astype(float)
    velocity = (open_interest - open_interest.shift(window)) / float(window)
    acceleration = (velocity - velocity.shift(window)) / float(window)
    return pd.DataFrame({"open_interest_acceleration": acceleration}, index=df.index)


@register_feature(
    group="多空博弈",
    level="level4_micro",
    description="多空不平衡度，正值表示多头主导，负值表示空头主导",
    depends_on=[],
    output_names=["long_short_imbalance"],
)
def compute_long_short_imbalance(df, features_df=None, window: int = 10, **kwargs):
    window = _safe_window(window, 10)
    close = df["close"].astype(float)
    open_interest = df["open_interest"].astype(float)
    volume = df["volume"].astype(float)

    price_sign = np.sign(close.diff()).fillna(0.0)
    oi_baseline = open_interest.rolling(window, min_periods=_min_periods(window)).mean()
    oi_ratio = open_interest.diff() / (oi_baseline + EPSILON)
    volume_baseline = volume.rolling(window, min_periods=_min_periods(window)).mean()
    volume_ratio = volume / (volume_baseline + EPSILON)
    imbalance = price_sign * oi_ratio * volume_ratio
    return pd.DataFrame({"long_short_imbalance": imbalance}, index=df.index)


@register_feature(
    group="开平仓压力",
    level="level4_micro",
    description="新增仓位占比，高值表示开仓为主",
    depends_on=[],
    output_names=["new_position_ratio"],
)
def compute_new_position_ratio(df, features_df=None, window: int = 5, **kwargs):
    window = _safe_window(window, 5)
    open_interest = df["open_interest"].astype(float)
    volume = df["volume"].astype(float)
    raw_ratio = open_interest.diff().abs() / (volume + EPSILON)
    smoothed_ratio = raw_ratio.rolling(window, min_periods=_min_periods(window)).mean()
    return pd.DataFrame({"new_position_ratio": smoothed_ratio}, index=df.index)


@register_feature(
    group="开平仓压力",
    level="level4_micro",
    description="平仓压力指标，高值表示获利/止损平仓",
    depends_on=[],
    output_names=["liquidation_pressure"],
)
def compute_liquidation_pressure(df, features_df=None, window: int = 10, **kwargs):
    window = _safe_window(window, 10)
    close = df["close"].astype(float)
    open_interest = df["open_interest"].astype(float)

    price_change = close.diff()
    oi_change = open_interest.diff()
    opposite_direction = (price_change * oi_change) < 0
    price_return_abs = close.pct_change().abs()
    oi_return_abs = open_interest.pct_change().abs()
    pressure_raw = (price_return_abs * oi_return_abs).where(opposite_direction)
    pressure_smoothed = pressure_raw.rolling(window, min_periods=_min_periods(window)).mean()
    return pd.DataFrame({"liquidation_pressure": pressure_smoothed}, index=df.index)


@register_feature(
    group="套保比率",
    level="level4_micro",
    description="套保需求代理，高值表示套保盘主导",
    depends_on=[],
    output_names=["hedging_ratio_proxy"],
)
def compute_hedging_ratio_proxy(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    open_interest = df["open_interest"].astype(float)
    volume = df["volume"].astype(float)
    raw_ratio = open_interest / (volume + EPSILON)
    smoothed_ratio = raw_ratio.rolling(window, min_periods=_min_periods(window, 5)).mean()
    return pd.DataFrame({"hedging_ratio_proxy": smoothed_ratio}, index=df.index)


@register_feature(
    group="投机度",
    level="level4_micro",
    description="投机活跃度，高值表示投机盘主导",
    depends_on=[],
    output_names=["speculation_index"],
)
def compute_speculation_index(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    volume = df["volume"].astype(float)
    open_interest = df["open_interest"].astype(float)
    raw_ratio = volume / (open_interest + EPSILON)
    smoothed_ratio = raw_ratio.rolling(window, min_periods=_min_periods(window, 5)).mean()
    return pd.DataFrame({"speculation_index": smoothed_ratio}, index=df.index)


@register_feature(
    group="持仓量深度",
    level="level4_micro",
    description="成交量与持仓量相关性，正相关表示增仓放量",
    depends_on=[],
    output_names=["volume_oi_correlation"],
)
def compute_volume_oi_correlation(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    volume = df["volume"].astype(float)
    open_interest = df["open_interest"].astype(float)
    correlation = volume.rolling(window, min_periods=_min_periods(window)).corr(open_interest)
    return pd.DataFrame({"volume_oi_correlation": correlation}, index=df.index)


@register_feature(
    group="持仓量结构",
    level="level4_micro",
    description="持仓量季节性模式（滚动分位），仅使用历史窗口",
    depends_on=[],
    output_names=["oi_seasonal_pattern"],
)
def compute_oi_seasonal_pattern(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    open_interest = df["open_interest"].astype(float)
    lookback_window = _safe_window(kwargs.get("lookback_window", window * 4), window * 4)
    lookback_window = max(lookback_window, window + 1)
    min_samples = min(lookback_window, max(10, window * 2))
    pattern = open_interest.rolling(
        lookback_window,
        min_periods=min_samples,
    ).apply(_rolling_percentile_last, raw=True)
    return pd.DataFrame({"oi_seasonal_pattern": pattern}, index=df.index)


@register_feature(
    group="展期行为",
    level="level5_cross",
    description="移仓换月活动强度",
    depends_on=[],
    output_names=["rollover_activity"],
)
def compute_rollover_activity(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    volume = df["volume"].astype(float)
    open_interest = df["open_interest"].astype(float)

    volume_baseline = volume.rolling(window, min_periods=_min_periods(window)).mean()
    volume_ratio = volume / (volume_baseline + EPSILON)
    oi_decline = ((open_interest.shift(window) - open_interest) / (open_interest.shift(window) + EPSILON)).clip(lower=0.0)
    rollover = volume_ratio * oi_decline
    return pd.DataFrame({"rollover_activity": rollover}, index=df.index)


@register_feature(
    group="主力合约切换",
    level="level5_cross",
    description="换月压力，正值表示多头平仓压力",
    depends_on=[],
    output_names=["contract_rolling_pressure"],
)
def compute_contract_rolling_pressure(df, features_df=None, window: int = 10, **kwargs):
    window = _safe_window(window, 10)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    open_interest = df["open_interest"].astype(float)

    current_volume_ma = volume.rolling(window, min_periods=_min_periods(window)).mean()
    previous_volume_ma = current_volume_ma.shift(window)
    volume_surge = current_volume_ma / (previous_volume_ma + EPSILON)

    oi_decline = ((open_interest.shift(window) - open_interest) / (open_interest.shift(window) + EPSILON)).clip(lower=0.0)
    price_direction = np.sign(close - close.shift(window))
    pressure = price_direction * oi_decline * volume_surge
    valid_mask = (oi_decline > 0.05) & (volume_surge > 1.2)
    pressure = pressure.where(valid_mask)
    return pd.DataFrame({"contract_rolling_pressure": pressure}, index=df.index)


@register_feature(
    group="持仓量结构",
    level="level4_micro",
    description="持仓量-价格联合状态（1-4，0表示震荡）",
    depends_on=[],
    output_names=["oi_price_regime"],
)
def compute_oi_price_regime(df, features_df=None, window: int = 5, **kwargs):
    window = _safe_window(window, 5)
    close = df["close"].astype(float)
    open_interest = df["open_interest"].astype(float)

    price_trend = close - close.shift(window)
    oi_trend = open_interest - open_interest.shift(window)
    regime = pd.Series(np.nan, index=df.index, dtype=float)

    valid = price_trend.notna() & oi_trend.notna()
    regime.loc[valid] = 0.0
    regime.loc[(price_trend > 1e-8) & (oi_trend > 1e-8)] = 1.0
    regime.loc[(price_trend > 1e-8) & (oi_trend < -1e-8)] = 2.0
    regime.loc[(price_trend < -1e-8) & (oi_trend > 1e-8)] = 3.0
    regime.loc[(price_trend < -1e-8) & (oi_trend < -1e-8)] = 4.0
    return pd.DataFrame({"oi_price_regime": regime}, index=df.index)


@register_feature(
    group="成交量深度",
    level="level4_micro",
    description="量仓比Z-Score，衡量异常交易活动",
    depends_on=[],
    output_names=["volume_oi_ratio_zscore"],
)
def compute_volume_oi_ratio_zscore(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    volume = df["volume"].astype(float)
    open_interest = df["open_interest"].astype(float)
    ratio = volume / (open_interest + EPSILON)
    ratio_mean = ratio.rolling(window, min_periods=_min_periods(window, 5)).mean()
    ratio_std = ratio.rolling(window, min_periods=_min_periods(window, 5)).std()
    zscore = (ratio - ratio_mean) / (ratio_std + EPSILON)
    return pd.DataFrame({"volume_oi_ratio_zscore": zscore}, index=df.index)


@register_feature(
    group="持仓量结构",
    level="level4_micro",
    description="价量背离强度，高值表示趋势可能反转",
    depends_on=[],
    output_names=["oi_price_divergence_strength"],
)
def compute_oi_price_divergence_strength(df, features_df=None, window: int = 5, **kwargs):
    window = _safe_window(window, 5)
    close = df["close"].astype(float)
    open_interest = df["open_interest"].astype(float)

    price_return = close.pct_change(window)
    oi_return = open_interest.pct_change(window)
    opposite_direction = (np.sign(price_return) * np.sign(oi_return)) < 0
    divergence = (price_return.abs() * oi_return.abs()).where(opposite_direction)
    return pd.DataFrame({"oi_price_divergence_strength": divergence}, index=df.index)


@register_feature(
    group="持仓量结构",
    level="level4_micro",
    description="持仓量在历史区间的位置，0-1之间",
    depends_on=[],
    output_names=["oi_extreme_ratio"],
)
def compute_oi_extreme_ratio(df, features_df=None, window: int = 100, **kwargs):
    window = _safe_window(window, 100)
    open_interest = df["open_interest"].astype(float)
    oi_low = open_interest.rolling(window, min_periods=_min_periods(window, 10)).min()
    oi_high = open_interest.rolling(window, min_periods=_min_periods(window, 10)).max()
    ratio = (open_interest - oi_low) / (oi_high - oi_low + EPSILON)
    return pd.DataFrame({"oi_extreme_ratio": ratio}, index=df.index)


@register_feature(
    group="成交量深度",
    level="level4_micro",
    description="成交量突破强度，>0表示放量",
    depends_on=[],
    output_names=["volume_breakout"],
)
def compute_volume_breakout(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    volume = df["volume"].astype(float)
    volume_ma = volume.rolling(window, min_periods=_min_periods(window, 5)).mean()
    breakout = volume / (volume_ma + EPSILON) - 1.0
    return pd.DataFrame({"volume_breakout": breakout}, index=df.index)


@register_feature(
    group="成交量深度",
    level="level4_micro",
    description="成交量稳定性，高值表示成交量稳定",
    depends_on=[],
    output_names=["volume_stability"],
)
def compute_volume_stability(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    volume = df["volume"].astype(float)
    volume_mean = volume.rolling(window, min_periods=_min_periods(window, 5)).mean()
    volume_std = volume.rolling(window, min_periods=_min_periods(window, 5)).std()
    stability = volume_mean / (volume_std + EPSILON)
    return pd.DataFrame({"volume_stability": stability}, index=df.index)


@register_feature(
    group="价格形态",
    level="level1_price",
    description="价格信念度，高值表示一方主导，低值表示分歧",
    depends_on=[],
    output_names=["price_conviction"],
)
def compute_price_conviction(df, features_df=None, **kwargs):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    candle_range = high - low
    valid_range = candle_range > EPSILON
    position = pd.Series(np.nan, index=df.index, dtype=float)
    position.loc[valid_range] = ((close - low) / candle_range).loc[valid_range]
    conviction = (2.0 * position - 1.0).abs()
    return pd.DataFrame({"price_conviction": conviction}, index=df.index)


@register_feature(
    group="持仓量结构",
    level="level4_micro",
    description="持仓量动量比率，快慢周期比",
    depends_on=[],
    output_names=["oi_momentum_ratio"],
)
def compute_oi_momentum_ratio(df, features_df=None, fast: int = 5, slow: int = 20, **kwargs):
    fast = _safe_window(fast, 5)
    slow = _safe_window(slow, 20)
    if fast > slow:
        fast, slow = slow, fast
    open_interest = df["open_interest"].astype(float)
    fast_ma = open_interest.rolling(fast, min_periods=_min_periods(fast)).mean()
    slow_ma = open_interest.rolling(slow, min_periods=_min_periods(slow, 5)).mean()
    ratio = fast_ma / (slow_ma + EPSILON)
    return pd.DataFrame({"oi_momentum_ratio": ratio}, index=df.index)


@register_feature(
    group="量价关系",
    level="level4_micro",
    description="量价效率，单位成交量推动的价格变化",
    depends_on=[],
    output_names=["volume_price_efficiency"],
)
def compute_volume_price_efficiency(
    df,
    features_df=None,
    window: int = 20,
    scale_factor: float = 10000.0,
    **kwargs,
):
    window = _safe_window(window, 20)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    price_change = (close - close.shift(window)).abs()
    volume_sum = volume.rolling(window, min_periods=_min_periods(window, 5)).sum()
    efficiency = price_change / (volume_sum + EPSILON) * float(scale_factor)
    return pd.DataFrame({"volume_price_efficiency": efficiency}, index=df.index)


@register_feature(
    group="持仓量结构",
    level="level4_micro",
    description="持仓量均值回归信号（可选扩展）",
    depends_on=[],
    output_names=["oi_mean_reversion"],
)
def compute_oi_mean_reversion(df, features_df=None, window: int = 20, **kwargs):
    window = _safe_window(window, 20)
    open_interest = df["open_interest"].astype(float)
    oi_mean = open_interest.rolling(window, min_periods=_min_periods(window, 5)).mean()
    deviation = (open_interest - oi_mean) / (oi_mean + EPSILON)
    signal = -np.sign(deviation) * (deviation.abs() ** 2)
    return pd.DataFrame({"oi_mean_reversion": signal}, index=df.index)


@register_feature(
    group="量价关系",
    level="level4_micro",
    description="量价背离信号（可选扩展）",
    depends_on=[],
    output_names=["volume_price_divergence"],
)
def compute_volume_price_divergence(df, features_df=None, window: int = 5, **kwargs):
    window = _safe_window(window, 5)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    price_sign = np.sign(close - close.shift(window))
    current_volume_mean = volume.rolling(window, min_periods=_min_periods(window)).mean()
    previous_volume_mean = current_volume_mean.shift(window)
    volume_ratio = current_volume_mean / (previous_volume_mean + EPSILON)

    divergence = pd.Series(np.nan, index=df.index, dtype=float)
    divergence.loc[(price_sign > 0) & (volume_ratio < 0.8)] = -volume_ratio[(price_sign > 0) & (volume_ratio < 0.8)]
    divergence.loc[(price_sign < 0) & (volume_ratio > 1.2)] = volume_ratio[(price_sign < 0) & (volume_ratio > 1.2)]
    return pd.DataFrame({"volume_price_divergence": divergence}, index=df.index)

```

**结论**：第六辑现已整理为可执行的单一版本，正式特征固定为 20 个，可选扩展 2 个，适配单品种 `OHLCV + open_interest` 数据流。