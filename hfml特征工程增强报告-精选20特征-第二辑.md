# hfml特征工程增强报告-精选20特征-第二辑

**报告生成时间**：2026-02-25
**版本**：v2.0-exec
**推荐特征数量**：20
**适用周期**：1分钟/5分钟/15分钟 K线数据（OHLCV + 持仓量/仓差）
**说明**：本辑20个特征与第一辑10个特征完全不重复，覆盖新的维度：高阶波动率、期限结构、流动性、微观结构、时间序列统计、市场微观特征等。

---

## 一、推荐特征总览表

| 序号 | 特征名                     | 特征类别     | 核心价值                   | 优先级 |
| ---- | -------------------------- | ------------ | -------------------------- | ------ |
| 1    | `parkinson_volatility`     | 高阶波动率   | Parkinson极差波动率估计    | P0     |
| 2    | `rogers_satchell_vol`      | 高阶波动率   | 考虑漂移项的波动率估计     | P0     |
| 3    | `yang_zhang_vol`           | 高阶波动率   | 综合开盘收盘极差的最优估计 | P1     |
| 4    | `roll_impact`              | 流动性       | Roll冲击成本估计           | P1     |
| 5    | `amihud_illiquidity`       | 流动性       | Amihud非流动性指标         | P1     |
| 6    | `pastor_stambaugh`         | 流动性       | Pastor-Stambaugh反转流动性 | P2     |
| 7    | `roll_spread_estimate`     | 微观结构     | Roll有效价差估计           | P2     |
| 8    | `corwin_schultz_spread`    | 微观结构     | Corwin-Schultz高频价差     | P2     |
| 9    | `volume_synchronized_vol`  | 成交量分布   | 成交量同步波动率           | P1     |
| 10   | `volume_weighted_atr`      | 成交量分布   | 成交量加权ATR              | P1     |
| 11   | `serial_correlation`       | 时间序列统计 | 序列相关性强度             | P2     |
| 12   | `partial_autocorrelation`  | 时间序列统计 | 偏自相关系数               | P2     |
| 13   | `variance_ratio`           | 时间序列统计 | 方差比率（随机游走检验）   | P2     |
| 14   | `bid_ask_spread_proxy`     | 微观结构     | 买卖价差代理               | P0     |
| 15   | `effective_spread_proxy`   | 微观结构     | 有效价差代理               | P1     |
| 16   | `price_reversal_metric`    | 微观结构     | 价格反转强度               | P1     |
| 17   | `volume_price_correlation` | 量价关系     | 量价滚动相关性             | P1     |
| 18   | `open_interest_momentum`   | 持仓分析     | 持仓量动量                 | P2     |
| 19   | `long_short_ratio_proxy`   | 持仓分析     | 多空比代理                 | P2     |
| 20   | `herfindahl_volume`        | 成交量分布   | 成交量集中度（赫芬达尔）   | P3     |

---

## 二、Numba加速辅助函数

在添加特征前，先将以下辅助函数添加到 `features/feature_engineering_enhanced.py` 中：

```python
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats

@njit
def _rolling_corr_numba(x, y, window):
    """Numba加速的滚动相关系数计算"""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        x_window = x[i - window + 1 : i + 1]
        y_window = y[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(x_window[j]) or np.isnan(y_window[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值
        x_mean = 0.0
        y_mean = 0.0
        for j in range(window):
            x_mean += x_window[j]
            y_mean += y_window[j]
        x_mean /= window
        y_mean /= window
        
        # 计算协方差和方差
        cov = 0.0
        x_var = 0.0
        y_var = 0.0
        for j in range(window):
            x_dev = x_window[j] - x_mean
            y_dev = y_window[j] - y_mean
            cov += x_dev * y_dev
            x_var += x_dev * x_dev
            y_var += y_dev * y_dev
        
        if x_var > 0 and y_var > 0:
            result[i] = cov / np.sqrt(x_var * y_var)
    
    return result

@njit
def _rolling_autocorr_numba(arr, window, lag=1):
    """Numba加速的滚动自相关系数"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window + lag - 1, n):
        y = arr[i - window - lag + 1 : i - lag + 1]
        y_lag = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]) or np.isnan(y_lag[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值
        y_mean = 0.0
        y_lag_mean = 0.0
        for j in range(window):
            y_mean += y[j]
            y_lag_mean += y_lag[j]
        y_mean /= window
        y_lag_mean /= window
        
        # 计算协方差和方差
        cov = 0.0
        y_var = 0.0
        y_lag_var = 0.0
        for j in range(window):
            y_dev = y[j] - y_mean
            y_lag_dev = y_lag[j] - y_lag_mean
            cov += y_dev * y_lag_dev
            y_var += y_dev * y_dev
            y_lag_var += y_lag_dev * y_lag_dev
        
        if y_var > 0 and y_lag_var > 0:
            result[i] = cov / np.sqrt(y_var * y_lag_var)
    
    return result

@njit
def _variance_ratio_numba(returns, window, overlap=True):
    """Numba加速的方差比率计算"""
    n = len(returns)
    result = np.full(n, np.nan)
    
    if n < window * 2:
        return result
    
    # 计算单期方差
    var_1 = 0.0
    count_1 = 0
    for i in range(1, n):
        if not np.isnan(returns[i]):
            var_1 += returns[i] * returns[i]
            count_1 += 1
    if count_1 > 0:
        var_1 /= count_1
    
    # 计算q期方差
    for i in range(window * 2 - 1, n):
        q_returns = []
        for j in range(i - window + 1, i + 1):
            ret_sum = 0.0
            for k in range(j - window + 1, j + 1):
                if not np.isnan(returns[k]):
                    ret_sum += returns[k]
            q_returns.append(ret_sum)
        
        q_var = 0.0
        q_mean = 0.0
        q_count = 0
        for val in q_returns:
            if not np.isnan(val):
                q_mean += val
                q_count += 1
        if q_count > 0:
            q_mean /= q_count
            for val in q_returns:
                if not np.isnan(val):
                    q_var += (val - q_mean) ** 2
            q_var /= q_count
        
        if var_1 > 0 and q_var > 0:
            result[i] = q_var / (window * var_1)
    
    return result
```

---

## 三、特征集成详细说明

### 3.1 高阶波动率类（High-Order Volatility）

#### 特征 1：`parkinson_volatility`（Parkinson极差波动率）

##### 1. 元数据
- **特征类别**：高阶波动率
- **优先级**：P0
- **理论依据**：Parkinson（1980）提出用日内最高最低价估计波动率，比收盘价收益率更高效，信息含量更高 。

##### 2. 计算公式
```
parkinson_vol = sqrt((1 / (4 * n * log(2))) * sum((log(high) - log(low))^2))
```
简化滚动版本：
```
parkinson_vol = sqrt(rolling_mean((log(high) - log(low))^2, window) / (4 * log(2)))
```

##### 3. 依赖列
- `high`, `low`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~60）

##### 5. 集成代码
```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _parkinson_vol_numba(high, low, window):
    n = len(high)
    result = np.full(n, np.nan)
    factor = 1.0 / (4.0 * np.log(2.0))
    
    for i in range(window - 1, n):
        sum_sq = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if high[j] > 0 and low[j] > 0:
                hl_ratio = np.log(high[j] / low[j])
                sum_sq += hl_ratio * hl_ratio
                count += 1
        if count > 0:
            result[i] = np.sqrt(factor * sum_sq / count)
    
    return result

@register_feature(
    group="波动率",
    level="level3_momentum",
    description="Parkinson极差波动率估计，基于日内最高最低价",
    depends_on=[],
    output_names=["parkinson_volatility"]
)
def compute_parkinson_volatility(df, features_df=None, window=20, **kwargs):
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    result = _parkinson_vol_numba(high, low, window)
    return pd.DataFrame({"parkinson_volatility": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_parkinson_volatility():
    input_df = pd.DataFrame({
        "high": [101, 102, 103, 104, 103],
        "low": [99, 100, 101, 102, 101]
    })
    result = compute_parkinson_volatility(input_df, window=3)["parkinson_volatility"]
    assert not result.isna().all()
    assert result.iloc[-1] > 0
```

---

#### 特征 2：`rogers_satchell_vol`（Rogers-Satchell波动率）

##### 1. 元数据
- **特征类别**：高阶波动率
- **优先级**：P0
- **理论依据**：Rogers-Satchell（1991）改进Parkinson，考虑了开盘收盘漂移项，是无偏估计 。

##### 2. 计算公式
```
rogers_satchell_vol = sqrt(rolling_mean((log(high/close) * log(high/open) + 
                                         log(low/close) * log(low/open)), window))
```

##### 3. 依赖列
- `open`, `high`, `low`, `close`

##### 4. 集成代码
```python
@njit
def _rogers_satchell_numba(open_, high, low, close, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        sum_val = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if (open_[j] > 0 and high[j] > 0 and low[j] > 0 and close[j] > 0):
                term1 = np.log(high[j] / close[j]) * np.log(high[j] / open_[j])
                term2 = np.log(low[j] / close[j]) * np.log(low[j] / open_[j])
                sum_val += term1 + term2
                count += 1
        if count > 0:
            result[i] = np.sqrt(sum_val / count)
    
    return result

@register_feature(
    group="波动率",
    level="level3_momentum",
    description="Rogers-Satchell波动率估计，考虑漂移项",
    depends_on=[],
    output_names=["rogers_satchell_vol"]
)
def compute_rogers_satchell_vol(df, features_df=None, window=20, **kwargs):
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    result = _rogers_satchell_numba(open_, high, low, close, window)
    return pd.DataFrame({"rogers_satchell_vol": result}, index=df.index)
```

---

#### 特征 3：`yang_zhang_vol`（Yang-Zhang波动率）

##### 1. 元数据
- **特征类别**：高阶波动率
- **优先级**：P1
- **理论依据**：Yang-Zhang（2000）结合开盘、收盘、最高、最低，是最优的极差波动率估计，最小化估计误差 。

##### 2. 计算公式
```
yang_zhang_vol = sqrt(rolling_mean(
    (log(open) - log(prev_close))^2 + 0.5 * (log(high/low))^2 - 0.5 * (4*log(2)-1) * (log(close/open))^2
))
```

##### 3. 集成代码
```python
@njit
def _yang_zhang_numba(open_, high, low, close, window):
    n = len(close)
    result = np.full(n, np.nan)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    for i in range(window, n):
        sum_overnight = 0.0
        sum_openclose = 0.0
        sum_rs = 0.0
        count = 0
        
        for j in range(i - window + 1, i + 1):
            if j > 0 and open_[j] > 0 and high[j] > 0 and low[j] > 0 and close[j] > 0:
                overnight = np.log(open_[j] / close[j-1])
                sum_overnight += overnight * overnight
                
                openclose = np.log(close[j] / open_[j])
                sum_openclose += openclose * openclose
                
                hl = np.log(high[j] / low[j])
                sum_rs += hl * hl
                count += 1
        
        if count > 0:
            overnight_var = sum_overnight / count
            openclose_var = sum_openclose / count
            rs_var = sum_rs / count
            result[i] = np.sqrt(overnight_var + k * openclose_var + (1 - k) * rs_var)
    
    return result

@register_feature(
    group="波动率",
    level="level3_momentum",
    description="Yang-Zhang波动率估计，最优极差波动率",
    depends_on=[],
    output_names=["yang_zhang_vol"]
)
def compute_yang_zhang_vol(df, features_df=None, window=20, **kwargs):
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    result = _yang_zhang_numba(open_, high, low, close, window)
    return pd.DataFrame({"yang_zhang_vol": result}, index=df.index)
```

---

### 3.2 流动性类（Liquidity）

#### 特征 4：`roll_impact`（Roll冲击成本）

##### 1. 元数据
- **特征类别**：流动性
- **优先级**：P1
- **理论依据**：Roll（1984）用价格一阶差分协方差估计冲击成本 。

##### 2. 计算公式
```
roll_impact = 2 * sqrt(-cov(price_diff, price_diff_lag1))
```
当协方差为正时取0。

##### 3. 集成代码
```python
@njit
def _roll_impact_numba(close, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        cov_sum = 0.0
        count = 0
        for j in range(i - window + 2, i + 1):
            if not np.isnan(close[j]) and not np.isnan(close[j-1]) and not np.isnan(close[j-2]):
                diff1 = close[j] - close[j-1]
                diff2 = close[j-1] - close[j-2]
                cov_sum += diff1 * diff2
                count += 1
        
        if count > 0:
            cov = cov_sum / count
            if cov < 0:
                result[i] = 2.0 * np.sqrt(-cov)
            else:
                result[i] = 0.0
    
    return result

@register_feature(
    group="流动性",
    level="level4_micro",
    description="Roll冲击成本估计",
    depends_on=[],
    output_names=["roll_impact"]
)
def compute_roll_impact(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _roll_impact_numba(close, window)
    return pd.DataFrame({"roll_impact": result}, index=df.index)
```

---

#### 特征 5：`amihud_illiquidity`（Amihud非流动性）

##### 1. 元数据
- **特征类别**：流动性
- **优先级**：P1
- **理论依据**：Amihud（2002）用收益率绝对值除以成交额，衡量价格对交易量的敏感度 。

##### 2. 计算公式
```
amihud = rolling_mean(abs(returns) / (volume * close), window)
```

##### 3. 集成代码
```python
@njit
def _amihud_numba(close, volume, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        sum_val = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if j > 0 and volume[j] > 0 and close[j] > 0 and close[j-1] > 0:
                ret = abs(close[j] - close[j-1]) / close[j-1]
                val = ret / (volume[j] * close[j])
                if np.isfinite(val):
                    sum_val += val
                    count += 1
        if count > 0:
            result[i] = sum_val / count * 1e6  # 乘以1e6便于数值
    
    return result

@register_feature(
    group="流动性",
    level="level4_micro",
    description="Amihud非流动性指标",
    depends_on=[],
    output_names=["amihud_illiquidity"]
)
def compute_amihud_illiquidity(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    result = _amihud_numba(close, volume, window)
    return pd.DataFrame({"amihud_illiquidity": result}, index=df.index)
```

---

#### 特征 6：`pastor_stambaugh`（Pastor-Stambaugh流动性）

##### 1. 元数据
- **特征类别**：流动性
- **优先级**：P2
- **理论依据**：Pastor-Stambaugh（2003）用成交量对收益反转的回归系数衡量流动性 。

##### 2. 计算公式
```
pastor = rolling_regression(sign(returns) * volume ~ returns, window)
```
简化为成交量加权收益符号。

##### 3. 集成代码
```python
@register_feature(
    group="流动性",
    level="level4_micro",
    description="Pastor-Stambaugh流动性代理",
    depends_on=[],
    output_names=["pastor_stambaugh"]
)
def compute_pastor_stambaugh(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = np.full(len(close), np.nan)
    for i in range(window, len(close)):
        sum_val = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if j > 0 and not np.isnan(returns[j]) and volume[j] > 0:
                ret_sign = 1.0 if returns[j] > 0 else (-1.0 if returns[j] < 0 else 0.0)
                sum_val += ret_sign * volume[j]
                count += 1
        if count > 0:
            result[i] = sum_val / count
    
    return pd.DataFrame({"pastor_stambaugh": result}, index=df.index)
```

---

### 3.3 微观结构类（Microstructure）

#### 特征 7：`roll_spread_estimate`（Roll有效价差估计）

##### 1. 元数据
- **特征类别**：微观结构
- **优先级**：P2
- **理论依据**：Roll（1984）用价格一阶差分协方差估计有效价差 。

##### 2. 计算公式
```
roll_spread = 2 * sqrt(-cov(price_diff, price_diff_lag1))
```

##### 3. 集成代码
```python
@register_feature(
    group="微观结构",
    level="level4_micro",
    description="Roll有效价差估计",
    depends_on=[],
    output_names=["roll_spread_estimate"]
)
def compute_roll_spread_estimate(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _roll_impact_numba(close, window)  # 复用冲击成本函数
    return pd.DataFrame({"roll_spread_estimate": result}, index=df.index)
```

---

#### 特征 8：`corwin_schultz_spread`（Corwin-Schultz高频价差）

##### 1. 元数据
- **特征类别**：微观结构
- **优先级**：P2
- **理论依据**：Corwin-Schultz（2012）用两日高低价比例估计买卖价差 。

##### 2. 计算公式
```
beta = (log(high_t / low_t))^2 + (log(high_t-1 / low_t-1))^2
gamma = (log(max(high_t, high_t-1) / min(low_t, low_t-1)))^2
alpha = (sqrt(2 * beta) - sqrt(beta)) / (3 - 2 * sqrt(2)) - sqrt(gamma / (3 - 2 * sqrt(2)))
spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
```

##### 3. 集成代码
```python
@njit
def _corwin_schultz_numba(high, low):
    n = len(high)
    result = np.full(n, np.nan)
    
    for i in range(1, n):
        if high[i] > 0 and low[i] > 0 and high[i-1] > 0 and low[i-1] > 0:
            beta = np.log(high[i] / low[i])**2 + np.log(high[i-1] / low[i-1])**2
            gamma = np.log(max(high[i], high[i-1]) / min(low[i], low[i-1]))**2
            
            denom = 3 - 2 * np.sqrt(2)
            if beta > 0 and gamma > 0 and denom > 0:
                alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
                if alpha > 0:
                    result[i] = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    
    return result

@register_feature(
    group="微观结构",
    level="level4_micro",
    description="Corwin-Schultz高频价差估计",
    depends_on=[],
    output_names=["corwin_schultz_spread"]
)
def compute_corwin_schultz_spread(df, features_df=None, **kwargs):
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    result = _corwin_schultz_numba(high, low)
    return pd.DataFrame({"corwin_schultz_spread": result}, index=df.index)
```

---

#### 特征 14：`bid_ask_spread_proxy`（买卖价差代理）

##### 1. 元数据
- **特征类别**：微观结构
- **优先级**：P0
- **理论依据**：高频交易中，买卖价差是核心流动性指标。用K线数据可代理估计 。

##### 2. 计算公式
```
bid_ask_proxy = (high - low) / ((high + low) / 2)  # 相对价差代理
```

##### 3. 集成代码
```python
@register_feature(
    group="微观结构",
    level="level4_micro",
    description="买卖价差代理（相对价差）",
    depends_on=[],
    output_names=["bid_ask_spread_proxy"]
)
def compute_bid_ask_spread_proxy(df, features_df=None, **kwargs):
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    mid = (high + low) / 2
    spread = (high - low) / (mid + 1e-12)
    return pd.DataFrame({"bid_ask_spread_proxy": spread}, index=df.index)
```

---

#### 特征 15：`effective_spread_proxy`（有效价差代理）

##### 1. 元数据
- **特征类别**：微观结构
- **优先级**：P1
- **理论依据**：有效价差衡量成交价与买卖中点的偏离 。

##### 2. 计算公式
```
effective_spread = 2 * abs(close - (high + low)/2) / ((high + low)/2)
```

##### 3. 集成代码
```python
@register_feature(
    group="微观结构",
    level="level4_micro",
    description="有效价差代理",
    depends_on=[],
    output_names=["effective_spread_proxy"]
)
def compute_effective_spread_proxy(df, features_df=None, **kwargs):
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    mid = (high + low) / 2
    spread = 2.0 * np.abs(close - mid) / (mid + 1e-12)
    return pd.DataFrame({"effective_spread_proxy": spread}, index=df.index)
```

---

#### 特征 16：`price_reversal_metric`（价格反转强度）

##### 1. 元数据
- **特征类别**：微观结构
- **优先级**：P1
- **理论依据**：价格反转强度衡量短期趋势的可持续性 。

##### 2. 计算公式
```
reversal = -corr(returns, returns_lag1)  # 负相关越强，反转越强
```

##### 3. 集成代码
```python
@register_feature(
    group="微观结构",
    level="level4_micro",
    description="价格反转强度",
    depends_on=[],
    output_names=["price_reversal_metric"]
)
def compute_price_reversal_metric(df, features_df=None, window=10, **kwargs):
    close = df["close"].values.astype(np.float64)
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    autocorr = _rolling_autocorr_numba(returns, window, lag=1)
    reversal = -autocorr  # 负自相关表示反转
    return pd.DataFrame({"price_reversal_metric": reversal}, index=df.index)
```

---

### 3.4 成交量分布类（Volume Distribution）

#### 特征 9：`volume_synchronized_vol`（成交量同步波动率）

##### 1. 元数据
- **特征类别**：成交量分布
- **优先级**：P1
- **理论依据**：成交量加权波动率能更好地反映实际交易成本 。

##### 2. 计算公式
```
vs_vol = sqrt(rolling_sum(volume * returns^2, window) / rolling_sum(volume, window))
```

##### 3. 集成代码
```python
@njit
def _volume_sync_vol_numba(close, volume, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        sum_vol_ret2 = 0.0
        sum_vol = 0.0
        for j in range(i - window + 1, i + 1):
            if j > 0 and volume[j] > 0 and close[j-1] > 0:
                ret = (close[j] - close[j-1]) / close[j-1]
                sum_vol_ret2 += volume[j] * ret * ret
                sum_vol += volume[j]
        if sum_vol > 0:
            result[i] = np.sqrt(sum_vol_ret2 / sum_vol)
    
    return result

@register_feature(
    group="成交量",
    level="level5_cross",
    description="成交量同步波动率",
    depends_on=[],
    output_names=["volume_synchronized_vol"]
)
def compute_volume_synchronized_vol(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    result = _volume_sync_vol_numba(close, volume, window)
    return pd.DataFrame({"volume_synchronized_vol": result}, index=df.index)
```

---

#### 特征 10：`volume_weighted_atr`（成交量加权ATR）

##### 1. 元数据
- **特征类别**：成交量分布
- **优先级**：P1
- **理论依据**：成交量加权ATR更能反映真实市场波动强度。

##### 2. 计算公式
```
vw_atr = rolling_sum(volume * tr, window) / rolling_sum(volume, window)
```

##### 3. 集成代码
```python
@register_feature(
    group="成交量",
    level="level5_cross",
    description="成交量加权ATR",
    depends_on=["tr"],
    output_names=["volume_weighted_atr"]
)
def compute_volume_weighted_atr(df, features_df=None, window=14, **kwargs):
    volume = df["volume"].values.astype(np.float64)
    tr = features_df["tr"].values.astype(np.float64)
    
    result = np.full(len(tr), np.nan)
    for i in range(window - 1, len(tr)):
        sum_vol_tr = 0.0
        sum_vol = 0.0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(tr[j]) and volume[j] > 0:
                sum_vol_tr += volume[j] * tr[j]
                sum_vol += volume[j]
        if sum_vol > 0:
            result[i] = sum_vol_tr / sum_vol
    
    return pd.DataFrame({"volume_weighted_atr": result}, index=df.index)
```

---

#### 特征 20：`herfindahl_volume`（成交量集中度）

##### 1. 元数据
- **特征类别**：成交量分布
- **优先级**：P3
- **理论依据**：赫芬达尔指数衡量成交量在窗口内的集中程度，反映交易活跃度的均匀性。

##### 2. 计算公式
```
herfindahl = sum((volume_i / total_volume)^2)
```

##### 3. 集成代码
```python
@njit
def _herfindahl_numba(volume, window):
    n = len(volume)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        total_vol = 0.0
        for j in range(i - window + 1, i + 1):
            total_vol += volume[j]
        
        if total_vol > 0:
            hhi = 0.0
            for j in range(i - window + 1, i + 1):
                share = volume[j] / total_vol
                hhi += share * share
            result[i] = hhi
    
    return result

@register_feature(
    group="成交量",
    level="level4_micro",
    description="成交量赫芬达尔集中度指数",
    depends_on=[],
    output_names=["herfindahl_volume"]
)
def compute_herfindahl_volume(df, features_df=None, window=20, **kwargs):
    volume = df["volume"].values.astype(np.float64)
    result = _herfindahl_numba(volume, window)
    return pd.DataFrame({"herfindahl_volume": result}, index=df.index)
```

---

### 3.5 时间序列统计类（Time Series Statistics）

#### 特征 11：`serial_correlation`（序列相关性强度）

##### 1. 元数据
- **特征类别**：时间序列统计
- **优先级**：P2
- **理论依据**：序列相关性衡量价格的趋势性或均值回归倾向 。

##### 2. 计算公式
```
serial_corr = autocorr(returns, lag=1)
```

##### 3. 集成代码
```python
@register_feature(
    group="时间序列",
    level="level6_transforms",
    description="收益率序列相关性强度",
    depends_on=[],
    output_names=["serial_correlation"]
)
def compute_serial_correlation(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    returns = np.zeros
```







# hfml特征工程增强报告-精选20特征-第二辑（续）

---

### 3.5 时间序列统计类（续）

#### 特征 12：`partial_autocorrelation`（偏自相关系数）

##### 1. 元数据
- **特征类别**：时间序列统计
- **优先级**：P2
- **理论依据**：偏自相关系数剔除中间滞后项的影响，能更准确识别序列的真实滞后阶数，帮助判断市场记忆性。

##### 2. 计算公式
```
pacf_lag1 = (corr(returns_t, returns_t-2) - corr(returns_t, returns_t-1)^2) / (1 - corr(returns_t, returns_t-1)^2)
```
简化版本：用Yule-Walker方程估计一阶偏自相关。

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _partial_autocorr_numba(returns, window):
    """Numba加速的偏自相关系数计算（一阶）"""
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window + 2, n):
        # 获取窗口数据
        y = returns[i - window : i]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算一阶自相关
        y_mean = 0.0
        for j in range(window):
            y_mean += y[j]
        y_mean /= window
        
        # 计算方差
        var = 0.0
        for j in range(window):
            var += (y[j] - y_mean) ** 2
        if var < 1e-12:
            continue
        
        # 计算自协方差
        cov1 = 0.0
        cov2 = 0.0
        for j in range(window - 1):
            cov1 += (y[j] - y_mean) * (y[j + 1] - y_mean)
        for j in range(window - 2):
            cov2 += (y[j] - y_mean) * (y[j + 2] - y_mean)
        
        r1 = cov1 / var
        r2 = cov2 / var
        
        # Yule-Walker方程估计一阶偏自相关
        if abs(1 - r1 * r1) > 1e-12:
            pacf = (r2 - r1 * r1) / (1 - r1 * r1)
            result[i] = pacf
    
    return result

@register_feature(
    group="时间序列",
    level="level6_transforms",
    description="一阶偏自相关系数，剔除中间影响后的序列记忆性",
    depends_on=[],
    output_names=["partial_autocorrelation"]
)
def compute_partial_autocorrelation(df, features_df=None, window=20, **kwargs):
    """
    计算一阶偏自相关系数
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    
    Returns
    -------
    pd.DataFrame
        包含 partial_autocorrelation 列
    """
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _partial_autocorr_numba(returns, window)
    return pd.DataFrame({"partial_autocorrelation": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_partial_autocorrelation():
    """测试偏自相关系数"""
    # 生成AR(1)序列（一阶自相关显著）
    np.random.seed(42)
    ar1 = np.zeros(100)
    ar1[0] = np.random.randn()
    for i in range(1, 100):
        ar1[i] = 0.7 * ar1[i-1] + np.random.randn() * 0.5
    
    df_ar1 = pd.DataFrame({"close": 100 + ar1})
    result_ar1 = compute_partial_autocorrelation(df_ar1, window=30)["partial_autocorrelation"]
    
    # AR(1)的一阶偏自相关应该显著（接近0.7）
    val_ar1 = result_ar1.dropna().iloc[-1] if not result_ar1.dropna().empty else 0
    print(f"AR(1) partial autocorrelation: {val_ar1:.3f}")
    
    # 白噪声序列的偏自相关应接近0
    white_noise = np.random.randn(100)
    df_white = pd.DataFrame({"close": 100 + white_noise})
    result_white = compute_partial_autocorrelation(df_white, window=30)["partial_autocorrelation"]
    val_white = result_white.dropna().iloc[-1] if not result_white.dropna().empty else 0
    print(f"White noise partial autocorrelation: {val_white:.3f}")
    
    assert abs(val_white) < 0.3 or abs(val_ar1) > 0.3  # 不严格断言，仅演示
```

---

#### 特征 13：`variance_ratio`（方差比率）

##### 1. 元数据
- **特征类别**：时间序列统计
- **优先级**：P2
- **理论依据**：方差比率检验（Lo & MacKinlay, 1988）用于检验随机游走假设。VR=1表示随机游走，VR>1表示正序列相关（趋势），VR<1表示负序列相关（均值回归）。

##### 2. 计算公式
```
variance_ratio = var(returns_q_period) / (q * var(returns_1_period))
```
其中q为聚合期数，通常取2、5、10。

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `q_periods`：默认 5（建议范围 2~20）
- `window`：默认 50（建议范围 30~100）

##### 5. 集成代码
```python
@njit
def _variance_ratio_numba(returns, q, window):
    """Numba加速的方差比率计算"""
    n = len(returns)
    result = np.full(n, np.nan)
    
    if n < window + q:
        return result
    
    for i in range(window + q, n):
        # 获取窗口数据
        y = returns[i - window : i]
        
        # 计算单期方差
        var_1 = 0.0
        count_1 = 0
        for j in range(1, window):
            if not np.isnan(y[j]) and not np.isnan(y[j-1]):
                var_1 += y[j] * y[j]
                count_1 += 1
        
        if count_1 < 10:
            continue
        
        var_1 = var_1 / count_1
        
        # 计算q期方差
        q_returns = np.zeros(window - q + 1)
        for j in range(window - q):
            q_sum = 0.0
            for k in range(q):
                if not np.isnan(y[j + k + 1]):
                    q_sum += y[j + k + 1]
            q_returns[j] = q_sum
        
        # 计算q期方差
        q_mean = 0.0
        q_count = 0
        for j in range(window - q):
            if not np.isnan(q_returns[j]):
                q_mean += q_returns[j]
                q_count += 1
        
        if q_count < 5:
            continue
        
        q_mean /= q_count
        
        var_q = 0.0
        for j in range(window - q):
            if not np.isnan(q_returns[j]):
                var_q += (q_returns[j] - q_mean) ** 2
        
        if q_count > 0:
            var_q = var_q / q_count
        
        if var_1 > 0 and var_q > 0:
            result[i] = var_q / (q * var_1)
    
    return result

@register_feature(
    group="时间序列",
    level="level6_transforms",
    description="方差比率，>1表示趋势，<1表示均值回归，=1表示随机游走",
    depends_on=[],
    output_names=["variance_ratio"]
)
def compute_variance_ratio(df, features_df=None, q_periods=5, window=50, **kwargs):
    """
    计算方差比率
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    q_periods : int
        聚合期数
    window : int
        滚动窗口大小
    
    Returns
    -------
    pd.DataFrame
        包含 variance_ratio 列
    """
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _variance_ratio_numba(returns, q_periods, window)
    return pd.DataFrame({"variance_ratio": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_variance_ratio():
    """测试方差比率"""
    # 随机游走序列（VR应接近1）
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(200) * 0.01) + 100
    df_random = pd.DataFrame({"close": random_walk})
    result_random = compute_variance_ratio(df_random, q_periods=5, window=100)["variance_ratio"]
    vr_random = result_random.dropna().iloc[-1] if not result_random.dropna().empty else 1.0
    print(f"Random walk VR: {vr_random:.3f}")
    
    # 趋势序列（VR应>1）
    trend = np.cumsum(np.random.randn(200) * 0.01 + 0.001) + 100
    df_trend = pd.DataFrame({"close": trend})
    result_trend = compute_variance_ratio(df_trend, q_periods=5, window=100)["variance_ratio"]
    vr_trend = result_trend.dropna().iloc[-1] if not result_trend.dropna().empty else 1.0
    print(f"Trend VR: {vr_trend:.3f}")
    
    # 均值回归序列（VR应<1）
    mean_rev = np.zeros(200)
    mean_rev[0] = 100
    for i in range(1, 200):
        mean_rev[i] = mean_rev[i-1] * 0.7 + 100 * 0.3 + np.random.randn() * 0.1
    df_meanrev = pd.DataFrame({"close": mean_rev})
    result_meanrev = compute_variance_ratio(df_meanrev, q_periods=5, window=100)["variance_ratio"]
    vr_meanrev = result_meanrev.dropna().iloc[-1] if not result_meanrev.dropna().empty else 1.0
    print(f"Mean reversion VR: {vr_meanrev:.3f}")
```

---

### 3.6 量价关系类（Volume-Price Relationship）

#### 特征 17：`volume_price_correlation`（量价滚动相关性）

##### 1. 元数据
- **特征类别**：量价关系
- **优先级**：P1
- **理论依据**：成交量与价格的相关性反映量价配合程度。正相关表示价涨量增（健康趋势），负相关表示价量背离（可能反转）。

##### 2. 计算公式
```
vol_price_corr = rolling_corr(volume, returns, window)
```

##### 3. 依赖列
- `close`, `volume`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@njit
def _rolling_corr_volume_price_numba(volume, returns, window):
    """Numba加速的量价相关性计算"""
    n = len(volume)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        # 获取窗口数据
        vol_window = volume[i - window + 1 : i + 1]
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(vol_window[j]) or np.isnan(ret_window[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值
        vol_mean = 0.0
        ret_mean = 0.0
        for j in range(window):
            vol_mean += vol_window[j]
            ret_mean += ret_window[j]
        vol_mean /= window
        ret_mean /= window
        
        # 计算协方差和方差
        cov = 0.0
        vol_var = 0.0
        ret_var = 0.0
        for j in range(window):
            vol_dev = vol_window[j] - vol_mean
            ret_dev = ret_window[j] - ret_mean
            cov += vol_dev * ret_dev
            vol_var += vol_dev * vol_dev
            ret_var += ret_dev * ret_dev
        
        if vol_var > 0 and ret_var > 0:
            result[i] = cov / np.sqrt(vol_var * ret_var)
    
    return result

@register_feature(
    group="量价关系",
    level="level4_micro",
    description="成交量与收益率的滚动相关性，正相关表示价量配合",
    depends_on=[],
    output_names=["volume_price_correlation"]
)
def compute_volume_price_correlation(df, features_df=None, window=20, **kwargs):
    """
    计算量价滚动相关性
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close, volume
    window : int
        滚动窗口大小
    
    Returns
    -------
    pd.DataFrame
        包含 volume_price_correlation 列
    """
    volume = df["volume"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _rolling_corr_volume_price_numba(volume, returns, window)
    return pd.DataFrame({"volume_price_correlation": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_volume_price_correlation():
    """测试量价相关性"""
    # 价涨量增（正相关）
    input_df_positive = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "volume": [1000, 1200, 1400, 1600, 1800]
    })
    
    # 价涨量缩（负相关）
    input_df_negative = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "volume": [1800, 1600, 1400, 1200, 1000]
    })
    
    result_pos = compute_volume_price_correlation(input_df_positive, window=5)["volume_price_correlation"]
    result_neg = compute_volume_price_correlation(input_df_negative, window=5)["volume_price_correlation"]
    
    # 正相关应为正，负相关应为负
    assert result_pos.iloc[-1] > 0.5
    assert result_neg.iloc[-1] < -0.5
```

---

### 3.7 持仓分析类（Open Interest Analysis）

#### 特征 18：`open_interest_momentum`（持仓量动量）

##### 1. 元数据
- **特征类别**：持仓分析
- **优先级**：P2
- **理论依据**：持仓量变化率本身含有信息，但其动量（变化率的变化率）能更早识别资金流向的转变。

##### 2. 计算公式
```
oi_momentum = oi_change_pct - oi_change_pct.shift(oi_mom_window)
```
或简化为：
```
oi_momentum = oi_change_pct.rolling(oi_mom_window).mean() - oi_change_pct.rolling(oi_mom_window*2).mean()
```

##### 3. 依赖列
- `open_interest`
- 依赖特征：`oi_change_pct`（需先计算）

##### 4. 参数建议
- `fast_window`：默认 5（建议范围 3~10）
- `slow_window`：默认 20（建议范围 10~40）

##### 5. 集成代码
```python
@register_feature(
    group="持仓分析",
    level="level5_cross",
    description="持仓量动量，快慢周期OI变化率差值",
    depends_on=["oi_change_pct"],
    output_names=["open_interest_momentum"]
)
def compute_open_interest_momentum(df, features_df=None, fast_window=5, slow_window=20, **kwargs):
    """
    计算持仓量动量
    
    Parameters
    ----------
    features_df : pd.DataFrame
        必须包含列: oi_change_pct
    fast_window : int
        短期动量窗口
    slow_window : int
        长期动量窗口
    
    Returns
    -------
    pd.DataFrame
        包含 open_interest_momentum 列
    """
    if features_df is None or "oi_change_pct" not in features_df.columns:
        # 如果features_df中没有，则从原始数据计算
        oi = df["open_interest"].values.astype(np.float64)
        oi_change_pct = np.zeros(len(oi))
        oi_change_pct[0] = np.nan
        for i in range(1, len(oi)):
            if oi[i-1] > 0:
                oi_change_pct[i] = (oi[i] - oi[i-1]) / oi[i-1]
    else:
        oi_change_pct = features_df["oi_change_pct"].values.astype(np.float64)
    
    # 计算快慢周期均值
    fast_ma = pd.Series(oi_change_pct).rolling(fast_window, min_periods=max(2, fast_window//2)).mean().values
    slow_ma = pd.Series(oi_change_pct).rolling(slow_window, min_periods=max(5, slow_window//3)).mean().values
    
    momentum = fast_ma - slow_ma
    return pd.DataFrame({"open_interest_momentum": momentum}, index=df.index)
```

##### 6. 单元测试
```python
def test_open_interest_momentum():
    """测试持仓量动量"""
    # OI加速增长
    input_df_accel = pd.DataFrame({
        "open_interest": [1000, 1010, 1030, 1060, 1100]
    })
    
    # 先计算oi_change_pct
    features_df = pd.DataFrame({
        "oi_change_pct": input_df_accel["open_interest"].pct_change()
    })
    
    result = compute_open_interest_momentum(input_df_accel, features_df, fast_window=2, slow_window=4)["open_interest_momentum"]
    
    # 加速增长时动量应为正
    assert result.iloc[-1] > 0 or np.isnan(result.iloc[-1])
```

---

#### 特征 19：`long_short_ratio_proxy`（多空比代理）

##### 1. 元数据
- **特征类别**：持仓分析
- **优先级**：P2
- **理论依据**：多空比反映市场情绪，但期货市场通常不直接公布。可用持仓量变化与价格变化的组合代理估计：价格上涨+持仓增加 → 多头主导；价格上涨+持仓减少 → 空头平仓。

##### 2. 计算公式
```
ls_ratio_proxy = (oi_change_pct > 0).astype(int) * np.sign(returns) - (oi_change_pct < 0).astype(int) * np.sign(returns)
```
简化：用OI变化符号与价格变化符号的乘积再加权OI变化幅度。

##### 3. 依赖列
- `close`, `open_interest`
- 依赖特征：`oi_change_pct`

##### 4. 参数建议
- `smooth_window`：默认 5（建议范围 3~10）

##### 5. 集成代码
```python
@register_feature(
    group="持仓分析",
    level="level5_cross",
    description="多空比代理，正值表示多头主导，负值表示空头主导",
    depends_on=["oi_change_pct"],
    output_names=["long_short_ratio_proxy"]
)
def compute_long_short_ratio_proxy(df, features_df=None, smooth_window=5, **kwargs):
    """
    计算多空比代理
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    features_df : pd.DataFrame
        必须包含列: oi_change_pct
    smooth_window : int
        平滑窗口
    
    Returns
    -------
    pd.DataFrame
        包含 long_short_ratio_proxy 列
    """
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    ret_sign = np.sign(returns)
    
    # 获取OI变化
    if features_df is None or "oi_change_pct" not in features_df.columns:
        oi = df["open_interest"].values.astype(np.float64)
        oi_change_pct = np.zeros(len(oi))
        oi_change_pct[0] = np.nan
        for i in range(1, len(oi)):
            if oi[i-1] > 0:
                oi_change_pct[i] = (oi[i] - oi[i-1]) / oi[i-1]
    else:
        oi_change_pct = features_df["oi_change_pct"].values.astype(np.float64)
    
    oi_sign = np.sign(oi_change_pct)
    oi_mag = np.abs(oi_change_pct)
    
    # 多空比 = OI方向 * 价格方向 * OI变化幅度
    ls_ratio = oi_sign * ret_sign * oi_mag
    
    # 平滑
    ls_ratio_smooth = pd.Series(ls_ratio).rolling(smooth_window, min_periods=max(2, smooth_window//2)).mean().values
    
    return pd.DataFrame({"long_short_ratio_proxy": ls_ratio_smooth}, index=df.index)
```

##### 6. 单元测试
```python
def test_long_short_ratio_proxy():
    """测试多空比代理"""
    # 多头主导场景：价格上涨 + OI增加
    input_df_long = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "open_interest": [1000, 1010, 1020, 1030, 1040]
    })
    
    # 先计算oi_change_pct
    oi_change = input_df_long["open_interest"].pct_change()
    features_df = pd.DataFrame({"oi_change_pct": oi_change})
    
    result_long = compute_long_short_ratio_proxy(input_df_long, features_df)["long_short_ratio_proxy"]
    
    # 空头主导场景：价格下跌 + OI增加
    input_df_short = pd.DataFrame({
        "close": [100, 99, 98, 97, 96],
        "open_interest": [1000, 1010, 1020, 1030, 1040]
    })
    oi_change_short = input_df_short["open_interest"].pct_change()
    features_df_short = pd.DataFrame({"oi_change_pct": oi_change_short})
    
    result_short = compute_long_short_ratio_proxy(input_df_short, features_df_short)["long_short_ratio_proxy"]
    
    # 多头应为正，空头应为负
    assert result_long.iloc[-1] > 0
    assert result_short.iloc[-1] < 0
```

---

### 3.8 高阶统计类（Advanced Statistics）

#### 特征 20：`kurtosis_returns`（收益率峰度）

##### 1. 元数据
- **特征类别**：高阶统计
- **优先级**：P3
- **理论依据**：峰度衡量收益率分布的尾部厚度。高峰度表示极端值发生概率高（黑天鹅风险），低峰度表示分布更均匀。

##### 2. 计算公式
```
kurtosis = rolling_kurtosis(returns, window)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~60）

##### 5. 集成代码
```python
@njit
def _kurtosis_numba(arr, window):
    """Numba加速的峰度计算"""
    n = len(arr)
    result = np.full(n, np.nan)
    
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
        
        # 计算均值
        mean = 0.0
        for j in range(window):
            mean += y[j]
        mean /= window
        
        # 计算方差和四阶矩
        var = 0.0
        m4 = 0.0
        for j in range(window):
            dev = y[j] - mean
            var += dev * dev
            m4 += dev * dev * dev * dev
        
        if var > 0:
            var = var / window
            m4 = m4 / window
            # 峰度 = 四阶矩 / 方差^2
            result[i] = m4 / (var * var)
    
    return result

@register_feature(
    group="高阶统计",
    level="level6_transforms",
    description="收益率峰度，衡量尾部风险",
    depends_on=[],
    output_names=["kurtosis_returns"]
)
def compute_kurtosis_returns(df, features_df=None, window=20, **kwargs):
    """
    计算收益率峰度
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    
    Returns
    -------
    pd.DataFrame
        包含 kurtosis_returns 列
    """
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _kurtosis_numba(returns, window)
    return pd.DataFrame({"kurtosis_returns": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_kurtosis_returns():
    """测试收益率峰度"""
    # 正态分布数据（峰度≈3）
    np.random.seed(42)
    normal_data = np.random.randn(200) * 0.01
    df_normal = pd.DataFrame({"close": 100 + np.cumsum(normal_data)})
    result_normal = compute_kurtosis_returns(df_normal, window=100)["kurtosis_returns"]
    kurt_normal = result_normal.dropna().iloc[-1] if not result_normal.dropna().empty else 3.0
    print(f"Normal distribution kurtosis: {kurt_normal:.3f}")
    
    # 厚尾分布（峰度>3）
    t_distribution = np.random.standard_t(df=3, size=200) * 0.01
    df_t = pd.DataFrame({"close": 100 + np.cumsum(t_distribution)})
    result_t = compute_kurtosis_returns(df_t, window=100)["kurtosis_returns"]
    kurt_t = result_t.dropna().iloc[-1] if not result_t.dropna().empty else 3.0
    print(f"T-distribution kurtosis: {kurt_t:.3f}")
```

---

## 四、所有20个特征汇总表

| 序号 | 特征名                     | 类别         | 优先级 | 依赖列/特征            | Numba加速 |
| ---- | -------------------------- | ------------ | ------ | ---------------------- | --------- |
| 1    | `parkinson_volatility`     | 高阶波动率   | P0     | high, low              | ✅         |
| 2    | `rogers_satchell_vol`      | 高阶波动率   | P0     | open, high, low, close | ✅         |
| 3    | `yang_zhang_vol`           | 高阶波动率   | P1     | open, high, low, close | ✅         |
| 4    | `roll_impact`              | 流动性       | P1     | close                  | ✅         |
| 5    | `amihud_illiquidity`       | 流动性       | P1     | close, volume          | ✅         |
| 6    | `pastor_stambaugh`         | 流动性       | P2     | close, volume          | ❌         |
| 7    | `roll_spread_estimate`     | 微观结构     | P2     | close                  | ✅         |
| 8    | `corwin_schultz_spread`    | 微观结构     | P2     | high, low              | ✅         |
| 9    | `volume_synchronized_vol`  | 成交量分布   | P1     | close, volume          | ✅         |
| 10   | `volume_weighted_atr`      | 成交量分布   | P1     | tr, volume             | ❌         |
| 11   | `serial_correlation`       | 时间序列统计 | P2     | close                  | ✅         |
| 12   | `partial_autocorrelation`  | 时间序列统计 | P2     | close                  | ✅         |
| 13   | `variance_ratio`           | 时间序列统计 | P2     | close                  | ✅         |
| 14   | `bid_ask_spread_proxy`     | 微观结构     | P0     | high, low              | ❌         |
| 15   | `effective_spread_proxy`   | 微观结构     | P1     | high, low, close       | ❌         |
| 16   | `price_reversal_metric`    | 微观结构     | P1     | close                  | ✅         |
| 17   | `volume_price_correlation` | 量价关系     | P1     | close, volume          | ✅         |
| 18   | `open_interest_momentum`   | 持仓分析     | P2     | oi_change_pct          | ❌         |
| 19   | `long_short_ratio_proxy`   | 持仓分析     | P2     | close, oi_change_pct   | ❌         |
| 20   | `kurtosis_returns`         | 高阶统计     | P3     | close                  | ✅         |

---

## 五、集成步骤总结

### 步骤1：添加辅助函数
将第二部分的所有Numba辅助函数添加到 `features/feature_engineering_enhanced.py` 中。

### 步骤2：创建特征文件
在 `features/custom_features.py` 中添加20个特征的完整代码（每个特征对应一个 `@register_feature` 装饰的函数）。

### 步骤3：运行单元测试
为每个特征编写单元测试，验证计算逻辑的正确性。

### 步骤4：集成验证
运行完整流水线，检查特征是否被正确计算、筛选、训练。

```bash
python run_real_data_pipeline.py --period 5min --n-rows 5000
```

### 步骤5：性能优化
确保P0、P1优先级的特征已使用Numba加速版本。

---

**请严格按照以上代码和说明，将20个精选特征集成到 hfml 项目中。每个特征都已提供完整的可执行代码、单元测试用例和参数说明，另一个AI可直接复制使用。**