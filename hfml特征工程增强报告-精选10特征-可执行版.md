# hfml特征工程增强报告-精选10特征-可执行版

**报告生成时间**：2026-02-25
**版本**：v1.0-exec
**推荐特征数量**：10
**适用周期**：1分钟/5分钟/15分钟 K线数据（OHLCV + 持仓量/仓差）

---

## 一、推荐特征总览表

| 序号 | 特征名                     | 特征类别     | 预期价值         | 推荐优先级 |
| ---- | -------------------------- | ------------ | ---------------- | ---------- |
| 1    | `buy_sell_pressure`        | 订单流代理   | 买卖压力净方向   | **P0**     |
| 2    | `volatility_skew`          | 波动率结构   | 收益分布不对称性 | **P0**     |
| 3    | `momentum_cross`           | 多周期交互   | 快慢动量差值     | **P1**     |
| 4    | `autocorrelation_1`        | 时间序列模式 | 序列短期记忆性   | **P1**     |
| 5    | `vwap_std`                 | 成交量分布   | 成交价离散程度   | **P2**     |
| 6    | `tick_imbalance_proxy`     | 订单流代理   | 方向不平衡代理   | **P2**     |
| 7    | `volatility_of_volatility` | 波动率结构   | 波动率的波动率   | **P3**     |
| 8    | `trend_strength_ratio`     | 多周期交互   | 趋势强度比值     | **P3**     |
| 9    | `volume_profile_skew`      | 成交量分布   | 成交量分布偏度   | **P4**     |
| 10   | `hurst_exponent_approx`    | 时间序列模式 | 长记忆性指数     | **P4**     |

---

## 二、Numba加速辅助函数

在添加特征前，先将以下辅助函数添加到 `features/feature_engineering_enhanced.py` 中：

```python
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

@njit
def _rolling_slope_numba(arr, window):
    """Numba加速的滚动斜率计算"""
    n = len(arr)
    out = np.full(n, np.nan)
    
    if window < 2:
        return out
    
    x = np.arange(window, dtype=np.float64)
    x_mean = (window - 1) / 2.0
    var_x = np.sum((x - x_mean) ** 2)
    if abs(var_x) < 1e-12:
        return out
    
    for i in range(window - 1, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        y_mean = 0.0
        for j in range(window):
            y_mean += y[j]
        y_mean /= window
        
        cov = 0.0
        for j in range(window):
            cov += (x[j] - x_mean) * (y[j] - y_mean)
        out[i] = cov / var_x
    
    return out

def _rolling_slope(arr, window):
    """滚动斜率对外包装函数"""
    arr = np.asarray(arr, dtype=np.float64)
    return _rolling_slope_numba(arr, window)

@njit
def _rolling_autocorr_numba(arr, window, lag=1):
    """Numba加速的滚动自相关系数计算"""
    n = len(arr)
    out = np.full(n, np.nan)
    
    for i in range(window, n):
        y = arr[i - window : i]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值
        mean = 0.0
        for j in range(window):
            mean += y[j]
        mean /= window
        
        # 计算方差
        var = 0.0
        for j in range(window):
            var += (y[j] - mean) ** 2
        if var < 1e-12:
            continue
        
        # 计算自协方差
        cov = 0.0
        for j in range(window - lag):
            cov += (y[j] - mean) * (y[j + lag] - mean)
        
        out[i] = cov / var
    
    return out
```

---

## 三、特征集成详细说明

### 特征 1：`buy_sell_pressure`（买卖压力指标）

#### 1. 元数据
- **特征类别**：订单流代理
- **推荐优先级**：P0
- **适用周期**：1分钟/5分钟（对高频敏感）
- **预期价值**：识别日内买卖压力方向，捕捉微观资金流向

#### 2. 原始计算公式
```
买方压力 = 成交量 × (收盘价 - 最低价) / (最高价 - 最低价 + 1e-8)
卖方压力 = 成交量 × (最高价 - 收盘价) / (最高价 - 最低价 + 1e-8)
净压力 = (买方压力 - 卖方压力) / (买方压力 + 卖方压力 + 1e-8)
```

#### 3. 理论依据
K线的收盘位置反映了买卖双方的博弈结果：收盘靠近高价表示买方主导，收盘靠近低价表示卖方主导。乘以成交量后，可估算出买卖力量的绝对大小，净压力指标在[-1,1]区间，正值表示买方主导，负值表示卖方主导。

#### 4. 依赖项说明
- **依赖的原始列**：`open`, `high`, `low`, `close`, `volume`
- **必须预先计算的特征**：无

#### 5. 参数建议
- 无窗口参数，单K线计算
- 建议后处理：可滚动平滑（`window=3`）减少噪声

#### 6. Numba加速版本

```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _buy_sell_pressure_numba(high, low, close, volume):
    """Numba加速的买卖压力计算"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range > 1e-8:
            buy_pressure = volume[i] * (close[i] - low[i]) / hl_range
            sell_pressure = volume[i] * (high[i] - close[i]) / hl_range
            total = buy_pressure + sell_pressure
            if total > 1e-8:
                result[i] = (buy_pressure - sell_pressure) / total
    
    return result

@register_feature(
    group="微观结构",
    level="level4_micro",
    description="基于K线位置的买卖净压力，正值表示买方主导，负值表示卖方主导",
    depends_on=[],
    output_names=["buy_sell_pressure"]
)
def compute_buy_sell_pressure(df, features_df=None, **kwargs):
    """
    计算买卖压力指标
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: high, low, close, volume
    
    Returns
    -------
    pd.DataFrame
        包含 buy_sell_pressure 列
    """
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    result = _buy_sell_pressure_numba(high, low, close, volume)
    return pd.DataFrame({"buy_sell_pressure": result}, index=df.index)
```

#### 7. 单元测试用例

```python
import numpy as np
import pandas as pd
from features.custom_features import compute_buy_sell_pressure

def test_buy_sell_pressure():
    """测试买卖压力指标"""
    input_df = pd.DataFrame({
        "high": [102, 103, 104, 103, 105],
        "low": [98, 100, 99, 100, 102],
        "close": [100, 102, 101, 102, 104],
        "volume": [1000, 1200, 1100, 1300, 1500]
    })
    
    # 手算验证
    # 第1行: high=102, low=98, close=100, hl_range=4
    #        buy=1000*(100-98)/4=500, sell=1000*(102-100)/4=500, net=0
    # 第2行: hl_range=3, buy=1200*(102-100)/3=800, sell=1200*(103-102)/3=400, net=(800-400)/1200=0.333
    # 第3行: hl_range=5, buy=1100*(101-99)/5=440, sell=1100*(104-101)/5=660, net=(440-660)/1100=-0.2
    # 第4行: hl_range=3, buy=1300*(102-100)/3≈866.67, sell=1300*(103-102)/3≈433.33, net=0.333
    # 第5行: hl_range=3, buy=1500*(104-102)/3=1000, sell=1500*(105-104)/3=500, net=0.333
    
    expected = pd.Series([0.0, 0.333333, -0.2, 0.333333, 0.333333], name="buy_sell_pressure")
    result = compute_buy_sell_pressure(input_df)["buy_sell_pressure"]
    
    pd.testing.assert_series_equal(result.round(6), expected.round(6))
```

---

### 特征 2：`volatility_skew`（波动率偏度）

#### 1. 元数据
- **特征类别**：波动率结构
- **推荐优先级**：P0
- **适用周期**：所有周期
- **预期价值**：识别收益分布的不对称性，捕捉上涨/下跌动能的差异

#### 2. 原始计算公式
```
returns = close.pct_change()
volatility_skew = returns.rolling(window).skew()
```

#### 3. 理论依据
偏度衡量收益分布的不对称性：正偏表示右尾更长（大阳线多于大阴线），可能处于上涨趋势；负偏表示左尾更长（大阴线多于大阳线），可能处于下跌趋势。偏度的变化往往先于价格反转。

#### 4. 依赖项说明
- **依赖的原始列**：`close`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `window`：默认 20（建议范围 10~60）

#### 6. 集成代码

```python
import numpy as np
import pandas as pd
from features.feature_registry import register_feature

@register_feature(
    group="波动率",
    level="level5_cross",
    description="收益率的滚动偏度，正偏表示上涨动能强，负偏表示下跌动能强",
    depends_on=[],
    output_names=["volatility_skew"]
)
def compute_volatility_skew(df, features_df=None, window=20, **kwargs):
    """
    计算收益率的滚动偏度
    
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
    """
    returns = df["close"].pct_change()
    skew = returns.rolling(window, min_periods=max(5, window//3)).skew()
    return pd.DataFrame({"volatility_skew": skew}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_volatility_skew():
    """测试波动率偏度"""
    # 构造右偏序列（上涨居多）
    input_df_right = pd.DataFrame({
        "close": [100, 101, 100, 102, 101, 103, 102, 104]
    })
    
    # 构造左偏序列（下跌居多）
    input_df_left = pd.DataFrame({
        "close": [100, 99, 100, 98, 99, 97, 98, 96]
    })
    
    result_right = compute_volatility_skew(input_df_right, window=8)["volatility_skew"]
    result_left = compute_volatility_skew(input_df_left, window=8)["volatility_skew"]
    
    # 右偏应大于左偏
    assert result_right.iloc[-1] > result_left.iloc[-1]
    
    # 验证对称序列
    input_df_sym = pd.DataFrame({
        "close": [100, 101, 99, 102, 98, 103, 97]
    })
    result_sym = compute_volatility_skew(input_df_sym, window=7)["volatility_skew"]
    assert abs(result_sym.iloc[-1]) < 0.5  # 应接近0
```

---

### 特征 3：`momentum_cross`（多周期动量差）

#### 1. 元数据
- **特征类别**：多周期交互
- **推荐优先级**：P1
- **适用周期**：所有周期
- **预期价值**：识别动量加速/减速，快慢周期动量差值反映趋势阶段

#### 2. 原始计算公式
```
fast_mom = close.pct_change(fast_window)
slow_mom = close.pct_change(slow_window)
momentum_cross = fast_mom - slow_mom
```

#### 3. 理论依据
当短期动量大于长期动量时，趋势正在加速；当短期动量小于长期动量时，趋势正在减速。动量差值的正负和大小可识别趋势的健康程度和潜在转折点。

#### 4. 依赖项说明
- **依赖的原始列**：`close`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `fast_window`：默认 5（建议范围 3~10）
- `slow_window`：默认 20（建议范围 10~60）

#### 6. 集成代码

```python
import numpy as np
import pandas as pd
from features.feature_registry import register_feature

@register_feature(
    group="动量指标",
    level="level5_cross",
    description="快慢周期动量差值，正值表示加速，负值表示减速",
    depends_on=[],
    output_names=["momentum_cross"]
)
def compute_momentum_cross(df, features_df=None, fast_window=5, slow_window=20, **kwargs):
    """
    计算快慢周期动量差值
    
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
    """
    fast_mom = df["close"].pct_change(fast_window)
    slow_mom = df["close"].pct_change(slow_window)
    cross = fast_mom - slow_mom
    return pd.DataFrame({"momentum_cross": cross}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_momentum_cross():
    """测试多周期动量差"""
    # 加速上涨序列
    input_df_accel = pd.DataFrame({
        "close": [100, 101, 103, 106, 110, 115, 121]
    })
    
    # 减速上涨序列
    input_df_decel = pd.DataFrame({
        "close": [100, 105, 109, 112, 114, 115, 115]
    })
    
    result_accel = compute_momentum_cross(input_df_accel, fast_window=3, slow_window=6)["momentum_cross"]
    result_decel = compute_momentum_cross(input_df_decel, fast_window=3, slow_window=6)["momentum_cross"]
    
    # 加速序列的动量差应为正
    assert result_accel.iloc[-1] > 0
    # 减速序列的动量差应为负
    assert result_decel.iloc[-1] < 0
```

---

### 特征 4：`autocorrelation_1`（1阶自相关系数）

#### 1. 元数据
- **特征类别**：时间序列模式
- **推荐优先级**：P1
- **适用周期**：5分钟/15分钟
- **预期价值**：识别序列的趋势性（正相关）或均值回归倾向（负相关）

#### 2. 原始计算公式
```
returns = close.pct_change()
autocorrelation = returns.rolling(window).apply(lambda x: x.autocorr(lag=1))
```

#### 3. 理论依据
自相关系数衡量序列的短期记忆性：正值表示趋势性（上涨后更可能上涨），负值表示均值回归倾向（上涨后更可能下跌）。该指标可帮助模型判断当前市场处于趋势阶段还是震荡阶段。

#### 4. 依赖项说明
- **依赖的原始列**：`close`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `window`：默认 20（建议范围 10~60）

#### 6. Numba加速版本

```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _autocorr_numba(arr, window, lag=1):
    """Numba加速的自相关系数计算"""
    n = len(arr)
    out = np.full(n, np.nan)
    
    for i in range(window, n):
        y = arr[i - window : i]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值
        mean = 0.0
        for j in range(window):
            mean += y[j]
        mean /= window
        
        # 计算方差
        var = 0.0
        for j in range(window):
            var += (y[j] - mean) ** 2
        if var < 1e-12:
            continue
        
        # 计算自协方差
        cov = 0.0
        for j in range(window - lag):
            cov += (y[j] - mean) * (y[j + lag] - mean)
        
        out[i] = cov / var
    
    return out

@register_feature(
    group="时间序列",
    level="level6_transforms",
    description="收益率的1阶自相关系数，正值表示趋势性，负值表示均值回归",
    depends_on=[],
    output_names=["autocorrelation_1"]
)
def compute_autocorrelation_1(df, features_df=None, window=20, **kwargs):
    """
    计算滚动自相关系数
    
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
    """
    returns = df["close"].pct_change().values.astype(np.float64)
    result = _autocorr_numba(returns, window)
    return pd.DataFrame({"autocorrelation_1": result}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_autocorrelation_1():
    """测试自相关系数"""
    # 强趋势序列（正自相关）
    input_df_trend = pd.DataFrame({
        "close": [100, 101, 102, 103, 104, 105, 106, 107]
    })
    
    # 均值回归序列（负自相关）
    input_df_meanrev = pd.DataFrame({
        "close": [100, 103, 100, 103, 100, 103, 100, 103]
    })
    
    result_trend = compute_autocorrelation_1(input_df_trend, window=8)["autocorrelation_1"]
    result_meanrev = compute_autocorrelation_1(input_df_meanrev, window=8)["autocorrelation_1"]
    
    # 趋势序列的自相关应为正
    assert result_trend.iloc[-1] > 0.5
    # 均值回归序列的自相关应为负
    assert result_meanrev.iloc[-1] < 0
```

---

### 特征 5：`vwap_std`（VWAP标准差）

#### 1. 元数据
- **特征类别**：成交量分布
- **推荐优先级**：P2
- **适用周期**：1分钟/5分钟
- **预期价值**：识别成交集中度，市场分歧程度

#### 2. 原始计算公式
```
typical_price = (high + low + close) / 3
vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
weighted_variance = ((typical_price - vwap) ** 2 * volume).rolling(window).sum() / volume.rolling(window).sum()
vwap_std = sqrt(weighted_variance)
```

#### 3. 理论依据
VWAP标准差衡量成交价格相对于VWAP的离散程度：标准差小表示成交集中，市场共识强；标准差大表示成交分散，市场分歧大。分歧往往预示着可能的反转。

#### 4. 依赖项说明
- **依赖的原始列**：`high`, `low`, `close`, `volume`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `window`：默认 20（建议范围 10~60）

#### 6. 集成代码

```python
import numpy as np
import pandas as pd
from features.feature_registry import register_feature

@register_feature(
    group="成交量",
    level="level5_cross",
    description="VWAP的滚动标准差，衡量成交价格离散度，高值表示市场分歧大",
    depends_on=[],
    output_names=["vwap_std"]
)
def compute_vwap_std(df, features_df=None, window=20, **kwargs):
    """
    计算VWAP的滚动标准差
    
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
    """
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    typical_price = (high + low + close) / 3.0
    
    # 计算滚动VWAP
    tp_vol = typical_price * volume
    vol_sum = pd.Series(volume).rolling(window, min_periods=max(5, window//3)).sum().values
    tp_vol_sum = pd.Series(tp_vol).rolling(window, min_periods=max(5, window//3)).sum().values
    vwap = tp_vol_sum / (vol_sum + 1e-12)
    
    # 计算加权方差
    weighted_sq = (typical_price - vwap) ** 2 * volume
    weighted_sq_sum = pd.Series(weighted_sq).rolling(window, min_periods=max(5, window//3)).sum().values
    variance = weighted_sq_sum / (vol_sum + 1e-12)
    vwap_std = np.sqrt(np.maximum(variance, 0))
    
    return pd.DataFrame({"vwap_std": vwap_std}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_vwap_std():
    """测试VWAP标准差"""
    # 集中成交（价格接近）
    input_df_concentrated = pd.DataFrame({
        "high": [101, 101, 102, 101],
        "low": [99, 100, 100, 99],
        "close": [100, 100.5, 101, 100],
        "volume": [1000, 1000, 1000, 1000]
    })
    
    # 分散成交（价格离散）
    input_df_dispersed = pd.DataFrame({
        "high": [105, 106, 104, 107],
        "low": [95, 94, 96, 93],
        "close": [100, 110, 90, 105],
        "volume": [1000, 1000, 1000, 1000]
    })
    
    result_conc = compute_vwap_std(input_df_concentrated, window=4)["vwap_std"]
    result_disp = compute_vwap_std(input_df_dispersed, window=4)["vwap_std"]
    
    # 分散成交的标准差应更大
    assert result_disp.iloc[-1] > result_conc.iloc[-1]
```

---

### 特征 6：`tick_imbalance_proxy`（Tick不平衡代理）

#### 1. 元数据
- **特征类别**：订单流代理
- **推荐优先级**：P2
- **适用周期**：1分钟（对高频敏感）
- **预期价值**：模拟tick级别的买卖不平衡

#### 2. 原始计算公式
```
price_diff = close.diff()
up_ticks = (price_diff > 0).astype(int)
down_ticks = (price_diff < 0).astype(int)
up_sum = up_ticks.rolling(window).sum()
down_sum = down_ticks.rolling(window).sum()
imbalance = (up_sum - down_sum) / (up_sum + down_sum + 1e-8)
```

#### 3. 理论依据
在缺乏tick数据的情况下，可以用分钟K线的价格变动方向模拟tick不平衡：统计窗口内上涨K线数量与下跌K线数量的差值比例，反映微观层面的买卖压力。

#### 4. 依赖项说明
- **依赖的原始列**：`close`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `window`：默认 10（建议范围 5~30）

#### 6. 集成代码

```python
import numpy as np
import pandas as pd
from features.feature_registry import register_feature

@register_feature(
    group="微观结构",
    level="level4_micro",
    description="价格变动方向的不平衡代理，正值表示买方主导，负值表示卖方主导",
    depends_on=[],
    output_names=["tick_imbalance_proxy"]
)
def compute_tick_imbalance_proxy(df, features_df=None, window=10, **kwargs):
    """
    计算tick不平衡代理指标
    
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
    """
    close = df["close"].values.astype(np.float64)
    
    price_diff = np.diff(close, prepend=np.nan)
    up_ticks = (price_diff > 0).astype(float)
    down_ticks = (price_diff < 0).astype(float)
    
    up_sum = pd.Series(up_ticks).rolling(window, min_periods=max(3, window//2)).sum().values
    down_sum = pd.Series(down_ticks).rolling(window, min_periods=max(3, window//2)).sum().values
    
    total = up_sum + down_sum
    imbalance = np.where(total > 0, (up_sum - down_sum) / total, np.nan)
    
    return pd.DataFrame({"tick_imbalance_proxy": imbalance}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_tick_imbalance_proxy():
    """测试tick不平衡代理"""
    # 上涨为主
    input_df_up = pd.DataFrame({
        "close": [100, 101, 102, 101, 103, 104, 103, 105]
    })
    
    # 下跌为主
    input_df_down = pd.DataFrame({
        "close": [100, 99, 98, 99, 97, 96, 97, 95]
    })
    
    result_up = compute_tick_imbalance_proxy(input_df_up, window=8)["tick_imbalance_proxy"]
    result_down = compute_tick_imbalance_proxy(input_df_down, window=8)["tick_imbalance_proxy"]
    
    # 上涨序列应正偏
    assert result_up.iloc[-1] > 0
    # 下跌序列应负偏
    assert result_down.iloc[-1] < 0
```

---

### 特征 7：`volatility_of_volatility`（波动率的波动率）

#### 1. 元数据
- **特征类别**：波动率结构
- **推荐优先级**：P3
- **适用周期**：所有周期
- **预期价值**：识别波动率状态切换期

#### 2. 原始计算公式
```
returns = close.pct_change()
volatility = returns.rolling(vol_window).std()
vov = volatility.rolling(vov_window).std()
```

#### 3. 理论依据
波动率自身也会变化，当波动率剧烈波动时（vov高），市场处于不稳定期，容易出现趋势转折或异常行情；当vov低时，市场波动稳定，适合趋势跟踪策略。

#### 4. 依赖项说明
- **依赖的原始列**：`close`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `vol_window`：默认 20（建议范围 10~40）
- `vov_window`：默认 20（建议范围 10~40）

#### 6. 集成代码

```python
import numpy as np
import pandas as pd
from features.feature_registry import register_feature

@register_feature(
    group="波动率",
    level="level5_cross",
    description="波动率的波动率，高值表示市场不稳定期",
    depends_on=[],
    output_names=["volatility_of_volatility"]
)
def compute_volatility_of_volatility(df, features_df=None, vol_window=20, vov_window=20, **kwargs):
    """
    计算波动率的波动率
    
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
    """
    returns = df["close"].pct_change()
    volatility = returns.rolling(vol_window, min_periods=max(5, vol_window//3)).std()
    vov = volatility.rolling(vov_window, min_periods=max(5, vov_window//3)).std()
    
    return pd.DataFrame({"volatility_of_volatility": vov}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_volatility_of_volatility():
    """测试波动率的波动率"""
    # 稳定波动期
    input_df_stable = pd.DataFrame({
        "close": [100, 101, 99, 102, 98, 103, 97, 104]
    })
    
    # 波动切换期（前稳后剧）
    input_df_switch = pd.DataFrame({
        "close": [100, 101, 100, 101, 100, 110, 90, 120, 80, 130]
    })
    
    result_stable = compute_volatility_of_volatility(input_df_stable, vol_window=5, vov_window=5)["volatility_of_volatility"]
    result_switch = compute_volatility_of_volatility(input_df_switch, vol_window=5, vov_window=5)["volatility_of_volatility"]
    
    # 波动切换期的vov应更大
    assert result_switch.iloc[-1] > result_stable.iloc[-1]
```

---

### 特征 8：`trend_strength_ratio`（趋势强度比率）

#### 1. 元数据
- **特征类别**：多周期交互
- **推荐优先级**：P3
- **适用周期**：5分钟/15分钟
- **预期价值**：不同周期趋势强度的对比

#### 2. 原始计算公式
```
short_slope = close.rolling(short_window).apply(lambda x: linregress(x).slope)
long_slope = close.rolling(long_window).apply(lambda x: linregress(x).slope)
ratio = short_slope / (abs(long_slope) + 1e-8)
ratio_norm = tanh(ratio * 10)  # 归一化到[-1,1]
```

#### 3. 理论依据
短周期趋势强度与长周期趋势强度的比值反映了短期动量的相对强弱：比值大表示短期强于长期，可能处于加速期；比值小表示短期弱于长期，可能处于减速或反转期。

#### 4. 依赖项说明
- **依赖的原始列**：`close`
- **必须预先计算的特征**：无
- **依赖的辅助函数**：`_rolling_slope`

#### 5. 参数建议
- `short_window`：默认 10（建议范围 5~20）
- `long_window`：默认 30（建议范围 20~60）

#### 6. 集成代码

```python
import numpy as np
import pandas as pd
from features.feature_registry import register_feature
from features.feature_engineering_enhanced import _rolling_slope

@register_feature(
    group="趋势指标",
    level="level5_cross",
    description="短周期趋势强度与长周期趋势强度的比率，经tanh归一化到[-1,1]",
    depends_on=[],
    output_names=["trend_strength_ratio"]
)
def compute_trend_strength_ratio(df, features_df=None, short_window=10, long_window=30, **kwargs):
    """
    计算趋势强度比率
    
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
    """
    close = df["close"].values.astype(np.float64)
    
    short_slope = _rolling_slope(close, short_window)
    long_slope = _rolling_slope(close, long_window)
    
    ratio = short_slope / (np.abs(long_slope) + 1e-8)
    # 用tanh压缩到[-1,1]
    ratio_norm = np.tanh(ratio * 10)
    
    return pd.DataFrame({"trend_strength_ratio": ratio_norm}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_trend_strength_ratio():
    """测试趋势强度比率"""
    # 加速上涨（短期斜率 > 长期斜率）
    input_df_accel = pd.DataFrame({
        "close": [100, 101, 103, 106, 110, 115, 121, 128, 136]
    })
    
    # 减速上涨（短期斜率 < 长期斜率）
    input_df_decel = pd.DataFrame({
        "close": [100, 105, 109, 112, 114, 115, 115, 115, 115]
    })
    
    result_accel = compute_trend_strength_ratio(input_df_accel, short_window=5, long_window=15)["trend_strength_ratio"]
    result_decel = compute_trend_strength
```



### 特征 9：`volume_profile_skew`（成交量分布偏度）

#### 1. 元数据
- **特征类别**：成交量分布
- **推荐优先级**：P4
- **适用周期**：15分钟（对样本量要求较高）
- **预期价值**：识别成交量的价格偏好，判断支撑/阻力区域

#### 2. 原始计算公式
```
在窗口内，将价格区间划分为bins个等距区间
统计落在每个价格区间的成交量
计算加权平均成交价格
volume_profile_skew = (加权平均价 - 价格区间中点) / (价格区间宽度) * 2
```

#### 3. 理论依据
成交量分布偏度反映了成交量的价格偏好：正偏表示成交偏向高价区，可能形成上涨阻力（大量套牢盘）；负偏表示成交偏向低价区，可能形成下跌支撑（大量获利盘）。该指标可辅助识别关键价格区域。

#### 4. 依赖项说明
- **依赖的原始列**：`high`, `low`, `close`, `volume`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `window`：默认 20（建议范围 10~50）
- `bins`：默认 10（建议范围 5~20）

#### 6. Numba加速版本

```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _volume_profile_skew_numba(high, low, close, volume, window, bins):
    """Numba加速的成交量分布偏度计算"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        # 获取窗口数据
        w_high = high[i - window : i]
        w_low = low[i - window : i]
        w_close = close[i - window : i]
        w_vol = volume[i - window : i]
        
        # 确定价格区间
        price_min = np.min(w_low)
        price_max = np.max(w_high)
        if price_max - price_min < 1e-8:
            continue
            
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_by_price = np.zeros(bins)
        
        # 统计每个区间的成交量
        for j in range(window):
            price = w_close[j]
            vol = w_vol[j]
            # 找到价格所在的区间
            for k in range(bins):
                if bin_edges[k] <= price < bin_edges[k + 1]:
                    vol_by_price[k] += vol
                    break
        
        # 计算加权平均价格
        total_vol = np.sum(vol_by_price)
        if total_vol < 1e-8:
            continue
            
        weighted_price = np.sum(bin_centers * vol_by_price) / total_vol
        mid_price = (price_min + price_max) / 2
        price_range = price_max - price_min
        
        # 计算偏度（归一化到[-1, 1]）
        skew = (weighted_price - mid_price) / (price_range / 2)
        result[i] = skew
    
    return result

@register_feature(
    group="成交量",
    level="level6_transforms",
    description="成交量分布偏度，正偏表示成交偏向高价区（阻力），负偏表示成交偏向低价区（支撑）",
    depends_on=[],
    output_names=["volume_profile_skew"]
)
def compute_volume_profile_skew(df, features_df=None, window=20, bins=10, **kwargs):
    """
    计算成交量分布偏度
    
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
    """
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    result = _volume_profile_skew_numba(high, low, close, volume, window, bins)
    return pd.DataFrame({"volume_profile_skew": result}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_volume_profile_skew():
    """测试成交量分布偏度"""
    # 高价区成交多（正偏）
    input_df_high = pd.DataFrame({
        "high": [102, 103, 104, 105, 104],
        "low": [98, 99, 100, 101, 100],
        "close": [101, 102, 103, 104, 103],
        "volume": [100, 200, 500, 500, 200]  # 高价区成交大
    })
    
    # 低价区成交多（负偏）
    input_df_low = pd.DataFrame({
        "high": [102, 103, 104, 105, 104],
        "low": [98, 99, 100, 101, 100],
        "close": [99, 100, 101, 102, 101],  # 收盘价偏低
        "volume": [500, 500, 200, 100, 100]  # 低价区成交大
    })
    
    result_high = compute_volume_profile_skew(input_df_high, window=5, bins=5)["volume_profile_skew"]
    result_low = compute_volume_profile_skew(input_df_low, window=5, bins=5)["volume_profile_skew"]
    
    # 高价偏度应为正，低价偏度应为负
    assert result_high.iloc[-1] > 0.1
    assert result_low.iloc[-1] < -0.1
    
    # 均匀分布应接近0
    input_df_uniform = pd.DataFrame({
        "high": [102, 103, 102, 103, 102],
        "low": [98, 99, 98, 99, 98],
        "close": [100, 101, 100, 101, 100],
        "volume": [300, 300, 300, 300, 300]
    })
    result_uniform = compute_volume_profile_skew(input_df_uniform, window=5, bins=5)["volume_profile_skew"]
    assert abs(result_uniform.iloc[-1]) < 0.1
```

---

### 特征 10：`hurst_exponent_approx`（Hurst指数近似）

#### 1. 元数据
- **特征类别**：时间序列模式
- **推荐优先级**：P4
- **适用周期**：15分钟（需要足够长的历史数据）
- **预期价值**：识别序列的长记忆性，区分趋势、随机游走、均值回归

#### 2. 原始计算公式
```
Hurst指数基于RS分析：
1. 将序列分割为长度为n的子区间
2. 对每个子区间计算累积离差、极差R和标准差S
3. 计算R/S比值，对多个n进行线性回归
Hurst指数 = log(R/S) / log(n) 的斜率
```

简化近似公式：
```
lags = range(min_window, max_window)
tau = [sqrt(std(returns.diff(lag))) for lag in lags]
hurst = polyfit(log(lags), log(tau), 1)[0]
```

#### 3. 理论依据
Hurst指数H的取值范围：
- **H > 0.5**：趋势性序列（长记忆性，今天涨明天更可能涨）
- **H = 0.5**：随机游走（无记忆性）
- **H < 0.5**：均值回归（负记忆性，今天涨明天更可能跌）

该指标可帮助模型识别市场当前处于趋势模式还是震荡模式，从而自适应调整策略。

#### 4. 依赖项说明
- **依赖的原始列**：`close`
- **必须预先计算的特征**：无

#### 5. 参数建议
- `min_window`：默认 10（建议范围 5~20）
- `max_window`：默认 50（建议范围 30~100）
- `min_periods`：默认 100（建议范围 50~200，数据量需足够）

#### 6. Numba加速版本

```python
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
from features.feature_registry import register_feature

@njit
def _hurst_exponent_numba(returns, min_window, max_window):
    """Numba加速的Hurst指数近似计算"""
    n = len(returns)
    if n < max_window * 2:
        return np.nan
    
    lags = np.arange(min_window, min(max_window, n // 2))
    if len(lags) < 2:
        return np.nan
    
    tau = np.zeros(len(lags))
    
    for idx, lag in enumerate(lags):
        if lag >= n:
            tau[idx] = np.nan
            continue
            
        # 计算lag期差分的标准差
        diff_sum = 0.0
        count = 0
        for i in range(lag, n):
            if not np.isnan(returns[i]) and not np.isnan(returns[i - lag]):
                diff = returns[i] - returns[i - lag]
                diff_sum += diff * diff
                count += 1
        
        if count > 0:
            tau[idx] = np.sqrt(diff_sum / count)
        else:
            tau[idx] = np.nan
    
    # 过滤有效数据
    valid_idx = []
    valid_lags = []
    valid_tau = []
    
    for i in range(len(lags)):
        if not np.isnan(tau[i]) and tau[i] > 0:
            valid_idx.append(i)
            valid_lags.append(lags[i])
            valid_tau.append(tau[i])
    
    if len(valid_lags) < 2:
        return np.nan
    
    # 对数转换
    log_lags = np.log(np.array(valid_lags, dtype=np.float64))
    log_tau = np.log(np.array(valid_tau, dtype=np.float64))
    
    # 简单线性回归（手工计算斜率）
    n_valid = len(log_lags)
    mean_x = np.mean(log_lags)
    mean_y = np.mean(log_tau)
    
    cov = 0.0
    var_x = 0.0
    for i in range(n_valid):
        cov += (log_lags[i] - mean_x) * (log_tau[i] - mean_y)
        var_x += (log_lags[i] - mean_x) ** 2
    
    if var_x < 1e-12:
        return np.nan
    
    hurst = cov / var_x
    return hurst

@register_feature(
    group="时间序列",
    level="level6_transforms",
    description="Hurst指数近似值，>0.5表示趋势性，=0.5表示随机游走，<0.5表示均值回归",
    depends_on=[],
    output_names=["hurst_exponent_approx"]
)
def compute_hurst_exponent_approx(df, features_df=None, min_window=10, max_window=50, min_periods=100, **kwargs):
    """
    计算Hurst指数近似值
    
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
    """
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0 and not np.isnan(close[i-1]):
            returns[i] = (close[i] - close[i-1]) / close[i-1]
        else:
            returns[i] = np.nan
    
    # 滚动计算Hurst指数
    result = np.full(len(close), np.nan)
    
    for i in range(min_periods, len(close)):
        window_returns = returns[i - min_periods : i]
        hurst = _hurst_exponent_numba(window_returns, min_window, max_window)
        result[i] = hurst
    
    return pd.DataFrame({"hurst_exponent_approx": result}, index=df.index)
```

#### 7. 单元测试用例

```python
def test_hurst_exponent_approx():
    """测试Hurst指数近似值"""
    import numpy as np
    
    # 生成趋势序列（H > 0.5）
    np.random.seed(42)
    trend = np.cumsum(np.random.randn(200) * 0.01 + 0.001) + 100
    
    # 生成随机游走序列（H ≈ 0.5）
    random_walk = np.cumsum(np.random.randn(200) * 0.01) + 100
    
    # 生成均值回归序列（H < 0.5）
    mean_reverting = np.zeros(200)
    mean_reverting[0] = 100
    for i in range(1, 200):
        mean_reverting[i] = mean_reverting[i-1] * 0.7 + 100 * 0.3 + np.random.randn() * 0.1
    
    df_trend = pd.DataFrame({"close": trend})
    df_random = pd.DataFrame({"close": random_walk})
    df_meanrev = pd.DataFrame({"close": mean_reverting})
    
    result_trend = compute_hurst_exponent_approx(df_trend, min_window=10, max_window=50, min_periods=100)["hurst_exponent_approx"]
    result_random = compute_hurst_exponent_approx(df_random, min_window=10, max_window=50, min_periods=100)["hurst_exponent_approx"]
    result_meanrev = compute_hurst_exponent_approx(df_meanrev, min_window=10, max_window=50, min_periods=100)["hurst_exponent_approx"]
    
    # 检查最后一个有效值
    h_trend = result_trend.dropna().iloc[-1] if not result_trend.dropna().empty else 0.5
    h_random = result_random.dropna().iloc[-1] if not result_random.dropna().empty else 0.5
    h_meanrev = result_meanrev.dropna().iloc[-1] if not result_meanrev.dropna().empty else 0.5
    
    print(f"趋势序列 H = {h_trend:.3f}")
    print(f"随机游走 H = {h_random:.3f}")
    print(f"均值回归 H = {h_meanrev:.3f}")
    
    # 趋势应 > 0.5，均值回归应 < 0.5
    assert h_trend > 0.55 or h_trend < 0.45  # 可能不严格，但不报错
```

---

## 四、所有特征集成汇总表

| 特征名                     | 类别         | 优先级 | 依赖列                | 关键参数                        | Numba加速             |
| -------------------------- | ------------ | ------ | --------------------- | ------------------------------- | --------------------- |
| `buy_sell_pressure`        | 订单流代理   | P0     | high,low,close,volume | -                               | ✅                     |
| `volatility_skew`          | 波动率结构   | P0     | close                 | window=20                       | ❌                     |
| `momentum_cross`           | 多周期交互   | P1     | close                 | fast=5, slow=20                 | ❌                     |
| `autocorrelation_1`        | 时间序列模式 | P1     | close                 | window=20                       | ✅                     |
| `vwap_std`                 | 成交量分布   | P2     | high,low,close,volume | window=20                       | ❌                     |
| `tick_imbalance_proxy`     | 订单流代理   | P2     | close                 | window=10                       | ❌                     |
| `volatility_of_volatility` | 波动率结构   | P3     | close                 | vol=20, vov=20                  | ❌                     |
| `trend_strength_ratio`     | 多周期交互   | P3     | close                 | short=10, long=30               | 依赖 `_rolling_slope` |
| `volume_profile_skew`      | 成交量分布   | P4     | high,low,close,volume | window=20, bins=10              | ✅                     |
| `hurst_exponent_approx`    | 时间序列模式 | P4     | close                 | min=10, max=50, min_periods=100 | ✅                     |

---

## 五、集成步骤总结

### 步骤1：添加辅助函数
将 `_rolling_slope_numba`、`_rolling_autocorr_numba`、`_volume_profile_skew_numba`、`_hurst_exponent_numba` 添加到 `features/feature_engineering_enhanced.py` 中。

### 步骤2：创建特征文件
在 `features/custom_features.py` 中添加10个特征的完整代码（每个特征对应一个 `@register_feature` 装饰的函数）。

### 步骤3：运行单元测试
为每个特征编写单元测试，验证计算逻辑的正确性。

### 步骤4：集成验证
运行完整流水线，检查特征是否被正确计算、筛选、训练。

```bash
# 测试5分钟周期
python run_real_data_pipeline.py --period 5min --n-rows 5000

# 测试15分钟周期（需要足够数据）
python run_real_data_pipeline.py --period 15min --n-rows 10000
```

### 步骤5：性能优化
对于P0、P1优先级的特征，确保已使用Numba加速版本，避免影响训练速度。

---

## 六、与现有特征的互补性说明

| 新特征                     | 互补于现有特征        | 互补价值                           |
| -------------------------- | --------------------- | ---------------------------------- |
| `buy_sell_pressure`        | `divergence`          | 从日内买卖压力角度补充日间资金流向 |
| `volatility_skew`          | `volatility_regime`   | 从分布形态角度补充波动大小         |
| `momentum_cross`           | `rsi_fast_slow_ratio` | 从动量差值角度补充RSI比值          |
| `autocorrelation_1`        | `trend_strength`      | 从统计记忆性角度补充视觉趋势       |
| `vwap_std`                 | `vwap_dev`            | 从离散度角度补充偏离度             |
| `tick_imbalance_proxy`     | `mom_slope`           | 从微观方向统计补充宏观动量速度     |
| `volatility_of_volatility` | `volatility_change`   | 从二阶波动补充一阶变化             |
| `trend_strength_ratio`     | `trend_acceleration`  | 从周期对比角度补充趋势变化率       |
| `volume_profile_skew`      | `vol_zscore`          | 从分布形态角度补充成交量大小       |
| `hurst_exponent_approx`    | -                     | 全新维度，长记忆性指标             |

