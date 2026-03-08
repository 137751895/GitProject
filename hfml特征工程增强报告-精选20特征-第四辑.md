# hfml特征工程增强报告-精选20特征-第四辑

**报告生成时间**：2026-02-25
**版本**：v4.0-exec
**推荐特征数量**：20
**适用周期**：1分钟/5分钟/15分钟 K线数据（OHLCV + 持仓量/仓差）
**说明**：本辑20个特征与前二辑、第三辑完全不重复，覆盖新的维度：谱分析、小波变换、极值理论、copula依赖、机器学习衍生特征、市场微观结构深度、行为金融指标、日历效应、高阶统计、复杂网络等。

---

## 一、推荐特征总览表

| 序号 | 特征名                       | 特征类别     | 核心价值         | 优先级 |
| ---- | ---------------------------- | ------------ | ---------------- | ------ |
| 1    | `spectral_ratio`             | 谱分析       | 市场周期强度     | P2     |
| 2    | `dominant_frequency`         | 谱分析       | 主导周期         | P2     |
| 3    | `wavelet_energy`             | 小波变换     | 多尺度能量分布   | P2     |
| 4    | `wavelet_entropy`            | 小波变换     | 多尺度复杂度     | P2     |
| 5    | `extreme_value_index`        | 极值理论     | 尾部风险指数     | P1     |
| 6    | `tail_dependence`            | 极值理论     | 尾部相关性       | P2     |
| 7    | `copula_dependence`          | copula依赖   | 非线性相关性     | P2     |
| 8    | `rank_correlation`           | copula依赖   | 秩相关性         | P1     |
| 9    | `ml_derived_volatility`      | 机器学习衍生 | 模型预测波动率   | P1     |
| 10   | `ml_derived_trend`           | 机器学习衍生 | 模型预测趋势     | P1     |
| 11   | `order_book_imbalance_proxy` | 微观结构深度 | 订单簿不平衡代理 | P1     |
| 12   | `depth_pressure`             | 微观结构深度 | 深度压力         | P1     |
| 13   | `herding_behavior`           | 行为金融     | 羊群行为指标     | P2     |
| 14   | `overreaction_score`         | 行为金融     | 过度反应得分     | P2     |
| 15   | `calendar_effect`            | 日历效应     | 时段效应         | P1     |
| 16   | `seasonality_strength`       | 日历效应     | 季节性强度       | P2     |
| 17   | `higher_order_cumulant`      | 高阶统计     | 高阶累积量       | P2     |
| 18   | `z_score_of_z_scores`        | 高阶统计     | 极端值检测       | P1     |
| 19   | `network_centrality`         | 复杂网络     | 市场中心度       | P3     |
| 20   | `community_strength`         | 复杂网络     | 市场群落强度     | P3     |

---

## 二、Numba加速辅助函数

在添加特征前，先将以下辅助函数添加到 `features/feature_engineering_enhanced.py` 中：

```python
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats, signal
import warnings

@njit
def _spectral_ratio_numba(arr, window, low_freq=0.05, high_freq=0.5):
    """Numba加速的谱能量比计算（简化版FFT）"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window * 2, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 去均值
        y_mean = 0.0
        for j in range(window):
            y_mean += y[j]
        y_mean /= window
        
        y_detrend = np.zeros(window)
        for j in range(window):
            y_detrend[j] = y[j] - y_mean
        
        # 简化FFT（用自相关替代）
        # 计算自相关
        acf = np.zeros(window // 2)
        for lag in range(1, window // 2):
            acf_sum = 0.0
            for j in range(window - lag):
                acf_sum += y_detrend[j] * y_detrend[j + lag]
            if lag > 0:
                acf[lag] = acf_sum / (window - lag)
        
        # 估计谱能量（用自相关的FFT替代）
        low_energy = 0.0
        high_energy = 0.0
        for freq_idx in range(1, len(acf)):
            freq = freq_idx / window
            if low_freq <= freq <= high_freq:
                high_energy += acf[freq_idx] ** 2
            else:
                low_energy += acf[freq_idx] ** 2
        
        if low_energy + high_energy > 0:
            result[i] = high_energy / (low_energy + high_energy)
    
    return result

@njit
def _wavelet_energy_numba(arr, window, scales=5):
    """Numba加速的小波能量计算（用Haar小波近似）"""
    n = len(arr)
    result = np.full((n, scales), np.nan)
    
    for i in range(window * 2, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 多尺度Haar小波变换
        for scale in range(scales):
            scale_size = 2 ** (scale + 1)
            if scale_size > window:
                continue
            
            n_coeff = window // scale_size
            energy = 0.0
            
            for j in range(n_coeff):
                start = j * scale_size
                mid = start + scale_size // 2
                end = start + scale_size
                
                if end <= window:
                    # Haar小波系数（近似+细节）
                    sum1 = 0.0
                    sum2 = 0.0
                    for k in range(start, mid):
                        sum1 += y[k]
                    for k in range(mid, end):
                        sum2 += y[k]
                    
                    approx = (sum1 + sum2) / scale_size
                    detail = (sum1 - sum2) / scale_size
                    energy += detail * detail
            
            result[i, scale] = energy / n_coeff if n_coeff > 0 else 0.0
    
    return result

@njit
def _extreme_value_index_numba(arr, window, threshold_percentile=95):
    """Numba加速的极值指数计算"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算阈值
        sorted_y = np.sort(y)
        threshold_idx = int(window * threshold_percentile / 100)
        if threshold_idx >= window:
            threshold_idx = window - 1
        threshold = sorted_y[threshold_idx]
        
        # 计算超过阈值的极值
        exceedances = []
        for j in range(window):
            if y[j] > threshold:
                exceedances.append(y[j] - threshold)
        
        if len(exceedances) > 5:
            # 拟合广义帕累托分布参数（简化版）
            exceedances = np.array(exceedances)
            mean_exceed = 0.0
            for val in exceedances:
                mean_exceed += val
            mean_exceed /= len(exceedances)
            
            # 极值指数 = 1 / shape参数（简化估计）
            if mean_exceed > 0:
                var_exceed = 0.0
                for val in exceedances:
                    var_exceed += (val - mean_exceed) ** 2
                var_exceed /= len(exceedances)
                
                if var_exceed > 0:
                    shape = 0.5 * (1 - (mean_exceed ** 2) / var_exceed)
                    if shape > 0:
                        result[i] = 1.0 / shape
    
    return result

@njit
def _tail_dependence_numba(returns1, returns2, window, quantile=0.1):
    """Numba加速的尾部相关性计算"""
    n = len(returns1)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        r1 = returns1[i - window + 1 : i + 1]
        r2 = returns2[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(r1[j]) or np.isnan(r2[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算分位数阈值
        sorted_r1 = np.sort(r1)
        sorted_r2 = np.sort(r2)
        thresh_idx = int(window * quantile)
        
        if thresh_idx >= window:
            thresh_idx = window - 1
        
        r1_low = sorted_r1[thresh_idx]
        r1_high = sorted_r1[window - thresh_idx - 1]
        r2_low = sorted_r2[thresh_idx]
        r2_high = sorted_r2[window - thresh_idx - 1]
        
        # 下尾相关性
        count_both_low = 0
        count_r1_low = 0
        for j in range(window):
            if r1[j] <= r1_low:
                count_r1_low += 1
                if r2[j] <= r2_low:
                    count_both_low += 1
        
        # 上尾相关性
        count_both_high = 0
        count_r1_high = 0
        for j in range(window):
            if r1[j] >= r1_high:
                count_r1_high += 1
                if r2[j] >= r2_high:
                    count_both_high += 1
        
        lower_tail = count_both_low / count_r1_low if count_r1_low > 0 else 0
        upper_tail = count_both_high / count_r1_high if count_r1_high > 0 else 0
        
        result[i] = (lower_tail + upper_tail) / 2
    
    return result

@njit
def _rank_correlation_numba(x, y, window):
    """Numba加速的秩相关性（Spearman）计算"""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
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
        
        # 计算秩
        x_rank = np.zeros(window)
        y_rank = np.zeros(window)
        
        for j in range(window):
            # x的秩
            rank = 1
            for k in range(window):
                if x_window[k] < x_window[j]:
                    rank += 1
                elif x_window[k] == x_window[j] and k < j:
                    rank += 1
            x_rank[j] = rank
            
            # y的秩
            rank = 1
            for k in range(window):
                if y_window[k] < y_window[j]:
                    rank += 1
                elif y_window[k] == y_window[j] and k < j:
                    rank += 1
            y_rank[j] = rank
        
        # 计算秩相关系数
        rank_mean = (window + 1) / 2
        cov = 0.0
        var_x = 0.0
        var_y = 0.0
        
        for j in range(window):
            dev_x = x_rank[j] - rank_mean
            dev_y = y_rank[j] - rank_mean
            cov += dev_x * dev_y
            var_x += dev_x * dev_x
            var_y += dev_y * dev_y
        
        if var_x > 0 and var_y > 0:
            result[i] = cov / np.sqrt(var_x * var_y)
    
    return result

@njit
def _z_score_of_z_scores_numba(arr, window):
    """Numba加速的Z-Score的Z-Score（极端值检测）"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    # 计算原始Z-Score
    z_scores = np.zeros(n)
    for i in range(window, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值和标准差
        mean = 0.0
        for j in range(window):
            mean += y[j]
        mean /= window
        
        std = 0.0
        for j in range(window):
            std += (y[j] - mean) ** 2
        std = np.sqrt(std / window)
        
        if std > 0:
            z_scores[i] = (arr[i] - mean) / std
    
    # 对Z-Scores再计算Z-Score
    for i in range(window * 2, n):
        z_window = z_scores[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(z_window[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        mean_z = 0.0
        for j in range(window):
            mean_z += z_window[j]
        mean_z /= window
        
        std_z = 0.0
        for j in range(window):
            std_z += (z_window[j] - mean_z) ** 2
        std_z = np.sqrt(std_z / window)
        
        if std_z > 0:
            result[i] = (z_scores[i] - mean_z) / std_z
    
    return result
```

---

## 三、特征集成详细说明

### 3.1 谱分析类（Spectral Analysis）

#### 特征 1：`spectral_ratio`（谱能量比）

##### 1. 元数据
- **特征类别**：谱分析
- **优先级**：P2
- **理论依据**：不同频率的谱能量比反映市场周期结构，高频能量占比高表示噪声大，低频能量占比高表示趋势强。

##### 2. 计算公式
```
spectral_ratio = E_high / (E_low + E_high)
```
其中E_high和E_low分别为高频和低频段的谱能量。

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `low_freq`：默认 0.05（低频阈值）
- `high_freq`：默认 0.5（高频阈值）

##### 5. 集成代码
```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@register_feature(
    group="谱分析",
    level="level6_transforms",
    description="谱能量比，高频占比高表示噪声大",
    depends_on=[],
    output_names=["spectral_ratio"]
)
def compute_spectral_ratio(df, features_df=None, window=100, low_freq=0.05, high_freq=0.5, **kwargs):
    """
    计算谱能量比
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    low_freq : float
        低频阈值
    high_freq : float
        高频阈值
    
    Returns
    -------
    pd.DataFrame
        包含 spectral_ratio 列
    """
    close = df["close"].values.astype(np.float64)
    result = _spectral_ratio_numba(close, window, low_freq, high_freq)
    return pd.DataFrame({"spectral_ratio": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_spectral_ratio():
    """测试谱能量比"""
    # 强趋势序列（低频为主）
    trend = np.cumsum(np.ones(200) * 0.1) + 100
    df_trend = pd.DataFrame({"close": trend})
    result_trend = compute_spectral_ratio(df_trend, window=100)["spectral_ratio"]
    
    # 高频噪声序列
    np.random.seed(42)
    noise = 100 + np.random.randn(200) * 2
    df_noise = pd.DataFrame({"close": noise})
    result_noise = compute_spectral_ratio(df_noise, window=100)["spectral_ratio"]
    
    ratio_trend = result_trend.dropna().iloc[-1] if not result_trend.dropna().empty else 0.5
    ratio_noise = result_noise.dropna().iloc[-1] if not result_noise.dropna().empty else 0.5
    
    print(f"Trend ratio: {ratio_trend:.3f}, Noise ratio: {ratio_noise:.3f}")
    assert ratio_trend < ratio_noise
```

---

#### 特征 2：`dominant_frequency`（主导频率）

##### 1. 元数据
- **特征类别**：谱分析
- **优先级**：P2
- **理论依据**：主导频率对应市场的主要周期，可用于识别波段周期长度。

##### 2. 计算公式
```
dominant_freq = argmax(spectrum)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）

##### 5. 集成代码
```python
@njit
def _dominant_frequency_numba(arr, window):
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window * 2, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 去均值
        y_mean = 0.0
        for j in range(window):
            y_mean += y[j]
        y_mean /= window
        
        y_detrend = np.zeros(window)
        for j in range(window):
            y_detrend[j] = y[j] - y_mean
        
        # 计算自相关
        acf = np.zeros(window // 2)
        for lag in range(1, window // 2):
            acf_sum = 0.0
            for j in range(window - lag):
                acf_sum += y_detrend[j] * y_detrend[j + lag]
            if lag > 0:
                acf[lag] = acf_sum / (window - lag)
        
        # 找到最大自相关的滞后
        max_acf = -1.0
        max_lag = 0
        for lag in range(1, len(acf)):
            if acf[lag] > max_acf:
                max_acf = acf[lag]
                max_lag = lag
        
        if max_lag > 0:
            result[i] = 1.0 / max_lag
    
    return result

@register_feature(
    group="谱分析",
    level="level6_transforms",
    description="主导频率，对应市场主要周期",
    depends_on=[],
    output_names=["dominant_frequency"]
)
def compute_dominant_frequency(df, features_df=None, window=100, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _dominant_frequency_numba(close, window)
    return pd.DataFrame({"dominant_frequency": result}, index=df.index)
```

---

### 3.2 小波变换类（Wavelet Transform）

#### 特征 3：`wavelet_energy`（小波能量）

##### 1. 元数据
- **特征类别**：小波变换
- **优先级**：P2
- **理论依据**：小波能量在多尺度的分布反映市场在不同时间尺度上的活动强度。

##### 2. 计算公式
```
wavelet_energy_scale_k = sum(d_k^2)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 128（建议范围 64~256）
- `scales`：默认 5（尺度数量）

##### 5. 集成代码
```python
@register_feature(
    group="小波变换",
    level="level6_transforms",
    description="小波能量（多尺度），返回5个尺度的能量",
    depends_on=[],
    output_names=["wavelet_energy_s1", "wavelet_energy_s2", "wavelet_energy_s3", 
                  "wavelet_energy_s4", "wavelet_energy_s5"]
)
def compute_wavelet_energy(df, features_df=None, window=128, scales=5, **kwargs):
    """
    计算小波能量（多尺度）
    
    Returns
    -------
    pd.DataFrame
        包含5个尺度的能量列
    """
    close = df["close"].values.astype(np.float64)
    energy_matrix = _wavelet_energy_numba(close, window, scales)
    
    result = pd.DataFrame(index=df.index)
    for s in range(scales):
        result[f"wavelet_energy_s{s+1}"] = energy_matrix[:, s]
    
    return result
```

---

#### 特征 4：`wavelet_entropy`（小波熵）

##### 1. 元数据
- **特征类别**：小波变换
- **优先级**：P2
- **理论依据**：小波熵衡量能量在多尺度的分布均匀性，高熵表示多尺度都活跃，低熵表示能量集中在特定尺度。

##### 2. 计算公式
```
wavelet_entropy = -sum(p_i * log(p_i))
```
其中p_i = energy_i / total_energy

##### 3. 依赖列
- `close`
- 依赖特征：`wavelet_energy_s1` ~ `wavelet_energy_s5`

##### 4. 集成代码
```python
@register_feature(
    group="小波变换",
    level="level6_transforms",
    description="小波熵，衡量能量多尺度分布均匀性",
    depends_on=["wavelet_energy_s1", "wavelet_energy_s2", "wavelet_energy_s3",
                "wavelet_energy_s4", "wavelet_energy_s5"],
    output_names=["wavelet_entropy"]
)
def compute_wavelet_entropy(df, features_df=None, **kwargs):
    if features_df is None:
        return pd.DataFrame({"wavelet_entropy": np.nan}, index=df.index)
    
    # 获取各尺度能量
    energy_cols = [f"wavelet_energy_s{i}" for i in range(1, 6)]
    energy = np.zeros((len(df), 5))
    
    for i, col in enumerate(energy_cols):
        if col in features_df.columns:
            energy[:, i] = features_df[col].values.astype(np.float64)
        else:
            energy[:, i] = np.nan
    
    # 计算小波熵
    result = np.full(len(df), np.nan)
    
    for i in range(len(df)):
        total = 0.0
        for j in range(5):
            if not np.isnan(energy[i, j]):
                total += energy[i, j]
        
        if total > 0:
            entropy = 0.0
            for j in range(5):
                if not np.isnan(energy[i, j]) and energy[i, j] > 0:
                    p = energy[i, j] / total
                    entropy -= p * np.log(p + 1e-12)
            result[i] = entropy / np.log(5)  # 归一化
    
    return pd.DataFrame({"wavelet_entropy": result}, index=df.index)
```

---

### 3.3 极值理论类（Extreme Value Theory）

#### 特征 5：`extreme_value_index`（极值指数）

##### 1. 元数据
- **特征类别**：极值理论
- **优先级**：P1
- **理论依据**：极值指数衡量尾部风险的强度，高值表示极端事件发生概率高。

##### 2. 计算公式
基于广义帕累托分布的形状参数：
```
ξ = 1 / EV_index
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `threshold_percentile`：默认 95（阈值分位数）

##### 5. 集成代码
```python
@register_feature(
    group="极值理论",
    level="level6_transforms",
    description="极值指数，衡量尾部风险强度",
    depends_on=[],
    output_names=["extreme_value_index"]
)
def compute_extreme_value_index(df, features_df=None, window=100, threshold_percentile=95, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _extreme_value_index_numba(np.abs(returns), window, threshold_percentile)
    return pd.DataFrame({"extreme_value_index": result}, index=df.index)
```

---

#### 特征 6：`tail_dependence`（尾部相关性）

##### 1. 元数据
- **特征类别**：极值理论
- **优先级**：P2
- **理论依据**：尾部相关性衡量极端行情下品种间的联动性，高值表示危机时同涨同跌。

##### 2. 计算公式
```
tail_dep = P(Y > y_q | X > x_q)
```

##### 3. 依赖列
- `close`（当前品种）
- 需要外部基准数据

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `quantile`：默认 0.1（尾部阈值）

##### 5. 集成代码
```python
@register_feature(
    group="极值理论",
    level="level6_transforms",
    description="尾部相关性，极端行情下联动性",
    depends_on=[],
    output_names=["tail_dependence"]
)
def compute_tail_dependence(df, features_df=None, window=100, quantile=0.1, benchmark_col=None, **kwargs):
    """
    计算尾部相关性
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close 和 benchmark_col
    benchmark_col : str
        基准列名
    """
    if benchmark_col is None or benchmark_col not in df.columns:
        return pd.DataFrame({"tail_dependence": np.nan}, index=df.index)
    
    close = df["close"].values.astype(np.float64)
    bench = df[benchmark_col].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    bench_returns = np.zeros(len(bench))
    returns[0] = np.nan
    bench_returns[0] = np.nan
    
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
        if bench[i-1] > 0:
            bench_returns[i] = (bench[i] - bench[i-1]) / bench[i-1]
    
    result = _tail_dependence_numba(returns, bench_returns, window, quantile)
    return pd.DataFrame({"tail_dependence": result}, index=df.index)
```

---

### 3.4 Copula依赖类（Copula Dependence）

#### 特征 7：`copula_dependence`（Copula相关性）

##### 1. 元数据
- **特征类别**：copula依赖
- **优先级**：P2
- **理论依据**：Copula能捕捉非线性相关性，比线性相关系数更全面。

##### 2. 计算公式
基于Kendall's tau转换为高斯copula参数：
```
copula_dep = sin(π * τ / 2)
```

##### 3. 依赖列
- `close`（当前品种）
- 需要外部基准数据

##### 4. 参数建议
- `window`：默认 60（建议范围 30~120）

##### 5. 集成代码
```python
@njit
def _kendalls_tau_numba(x, y, window):
    """Numba加速的Kendall's tau计算"""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
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
        
        concordant = 0
        discordant = 0
        
        for j in range(window):
            for k in range(j + 1, window):
                x_diff = x_window[j] - x_window[k]
                y_diff = y_window[j] - y_window[k]
                
                if x_diff * y_diff > 0:
                    concordant += 1
                elif x_diff * y_diff < 0:
                    discordant += 1
        
        total = concordant + discordant
        if total > 0:
            result[i] = (concordant - discordant) / total
    
    return result

@register_feature(
    group="Copula依赖",
    level="level6_transforms",
    description="Copula相关性（基于Kendall's tau）",
    depends_on=[],
    output_names=["copula_dependence"]
)
def compute_copula_dependence(df, features_df=None, window=60, benchmark_col=None, **kwargs):
    if benchmark_col is None or benchmark_col not in df.columns:
        return pd.DataFrame({"copula_dependence": np.nan}, index=df.index)
    
    close = df["close"].values.astype(np.float64)
    bench = df[benchmark_col].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    bench_returns = np.zeros(len(bench))
    returns[0] = np.nan
    bench_returns[0] = np.nan
    
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
        if bench[i-1] > 0:
            bench_returns[i] = (bench[i] - bench[i-1]) / bench[i-1]
    
    tau = _kendalls_tau_numba(returns, bench_returns, window)
    # 转换为高斯copula参数
    copula = np.sin(np.pi * tau / 2)
    
    return pd.DataFrame({"copula_dependence": copula}, index=df.index)
```

---

#### 特征 8：`rank_correlation`（秩相关性）

##### 1. 元数据
- **特征类别**：copula依赖
- **优先级**：P1
- **理论依据**：秩相关性（Spearman）对异常值稳健，能捕捉单调非线性关系。

##### 2. 计算公式
```
rank_corr = corr(rank(x), rank(y))
```

##### 3. 依赖列
- `close`（当前品种）
- 需要外部基准数据或内部特征

##### 4. 参数建议
- `window`：默认 60（建议范围 30~120）

##### 5. 集成代码
```python
@register_feature(
    group="Copula依赖",
    level="level6_transforms",
    description="秩相关性（Spearman）",
    depends_on=[],
    output_names=["rank_correlation"]
)
def compute_rank_correlation(df, features_df=None, window=60, col1=None, col2=None, **kwargs):
    """
    计算两个序列的秩相关性
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: col1 和 col2
    col1, col2 : str
        要计算相关性的列名
    """
    if col1 is None or col2 is None or col1 not in df.columns or col2 not in df.columns:
        return pd.DataFrame({"rank_correlation": np.nan}, index=df.index)
    
    x = df[col1].values.astype(np.float64)
    y = df[col2].values.astype(np.float64)
    
    result = _rank_correlation_numba(x, y, window)
    return pd.DataFrame({"rank_correlation": result}, index=df.index)
```



### 3.5 机器学习衍生类（ML-Derived Features）（续）

#### 特征 9：`ml_derived_volatility`（模型预测波动率）

##### 1. 元数据
- **特征类别**：机器学习衍生
- **优先级**：P1
- **理论依据**：用简单模型预测的波动率可作为特征，捕捉非线性模式。

##### 2. 计算公式
用GARCH(1,1)简化版：
```
sigma_t^2 = ω + α * r_{t-1}^2 + β * sigma_{t-1}^2
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）

##### 5. 集成代码
```python
@njit
def _garch11_numba(returns, window, omega=0.01, alpha=0.1, beta=0.85):
    n = len(returns)
    result = np.full(n, np.nan)
    sigma2 = np.zeros(n)
    
    # 初始方差
    var_sum = 0.0
    count = 0
    for i in range(1, window):
        if not np.isnan(returns[i]):
            var_sum += returns[i] ** 2
            count += 1
    if count > 0:
        sigma2[window-1] = var_sum / count
    
    for i in range(window, n):
        if not np.isnan(returns[i-1]):
            sigma2[i] = (omega + 
                        alpha * returns[i-1] ** 2 + 
                        beta * sigma2[i-1])
            result[i] = np.sqrt(sigma2[i])
    
    return result

@register_feature(
    group="机器学习衍生",
    level="level6_transforms",
    description="GARCH(1,1)预测波动率",
    depends_on=[],
    output_names=["ml_derived_volatility"]
)
def compute_ml_derived_volatility(df, features_df=None, window=50, omega=0.01, alpha=0.1, beta=0.85, **kwargs):
    """
    计算GARCH(1,1)预测波动率
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    omega, alpha, beta : float
        GARCH模型参数
    
    Returns
    -------
    pd.DataFrame
        包含 ml_derived_volatility 列
    """
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _garch11_numba(returns, window, omega, alpha, beta)
    return pd.DataFrame({"ml_derived_volatility": result}, index=df.index)
```

---

#### 特征 10：`ml_derived_trend`（模型预测趋势）

##### 1. 元数据
- **特征类别**：机器学习衍生
- **优先级**：P1
- **理论依据**：用简单线性模型预测的趋势方向，可作为特征增强信号。

##### 2. 计算公式
用滚动线性回归的预测值：
```
trend_pred = slope * (window) + intercept
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@njit
def _linear_trend_prediction_numba(close, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        y = close[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 线性回归
        x = np.arange(window, dtype=np.float64)
        x_mean = (window - 1) / 2.0
        
        # 计算斜率
        y_mean = 0.0
        for j in range(window):
            y_mean += y[j]
        y_mean /= window
        
        cov = 0.0
        var_x = 0.0
        for j in range(window):
            cov += (j - x_mean) * (y[j] - y_mean)
            var_x += (j - x_mean) ** 2
        
        if var_x > 0:
            slope = cov / var_x
            intercept = y_mean - slope * x_mean
            
            # 预测下一个值
            result[i] = slope * window + intercept
    
    return result

@register_feature(
    group="机器学习衍生",
    level="level6_transforms",
    description="线性模型预测趋势",
    depends_on=[],
    output_names=["ml_derived_trend"]
)
def compute_ml_derived_trend(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _linear_trend_prediction_numba(close, window)
    return pd.DataFrame({"ml_derived_trend": result}, index=df.index)
```

---

### 3.6 微观结构深度类（Microstructure Depth）

#### 特征 11：`order_book_imbalance_proxy`（订单簿不平衡代理）

##### 1. 元数据
- **特征类别**：微观结构深度
- **优先级**：P1
- **理论依据**：用价格变动和成交量模拟订单簿不平衡，反映买卖压力。

##### 2. 计算公式
```
obi_proxy = (volume * price_change) / (high - low + 1e-8)
```

##### 3. 依赖列
- `open`, `high`, `low`, `close`, `volume`

##### 4. 参数建议
- `window`：默认 10（建议范围 5~30）

##### 5. 集成代码
```python
@njit
def _order_book_imbalance_proxy_numba(open_, high, low, close, volume, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        sum_obi = 0.0
        count = 0
        
        for j in range(i - window + 1, i + 1):
            if j > 0 and high[j] > low[j] and volume[j] > 0:
                price_change = close[j] - open_[j]
                price_range = high[j] - low[j]
                if price_range > 0:
                    obi = volume[j] * price_change / price_range
                    sum_obi += obi
                    count += 1
        
        if count > 0:
            result[i] = sum_obi / count
    
    return result

@register_feature(
    group="微观结构深度",
    level="level4_micro",
    description="订单簿不平衡代理，正值表示买方压力",
    depends_on=[],
    output_names=["order_book_imbalance_proxy"]
)
def compute_order_book_imbalance_proxy(df, features_df=None, window=10, **kwargs):
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    result = _order_book_imbalance_proxy_numba(open_, high, low, close, volume, window)
    return pd.DataFrame({"order_book_imbalance_proxy": result}, index=df.index)
```

---

#### 特征 12：`depth_pressure`（深度压力）

##### 1. 元数据
- **特征类别**：微观结构深度
- **优先级**：P1
- **理论依据**：深度压力衡量价格变动所需的成交量，反映市场深度。

##### 2. 计算公式
```
depth_pressure = abs(price_change) / (volume + 1e-8)
```

##### 3. 依赖列
- `close`, `volume`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@njit
def _depth_pressure_numba(close, volume, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        sum_pressure = 0.0
        sum_vol = 0.0
        count = 0
        
        for j in range(i - window + 1, i + 1):
            if j > 0 and volume[j] > 0:
                price_change = abs(close[j] - close[j-1])
                pressure = price_change / volume[j]
                if np.isfinite(pressure):
                    sum_pressure += pressure
                    count += 1
        
        if count > 0:
            result[i] = sum_pressure / count
    
    return result

@register_feature(
    group="微观结构深度",
    level="level4_micro",
    description="深度压力，单位成交量引起的价格变动",
    depends_on=[],
    output_names=["depth_pressure"]
)
def compute_depth_pressure(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    result = _depth_pressure_numba(close, volume, window)
    return pd.DataFrame({"depth_pressure": result}, index=df.index)
```

---

### 3.7 行为金融类（Behavioral Finance）

#### 特征 13：`herding_behavior`（羊群行为指标）

##### 1. 元数据
- **特征类别**：行为金融
- **优先级**：P2
- **理论依据**：羊群行为指标用收益率的截面分散度衡量，低分散度表示羊群效应强。

##### 2. 计算公式
```
herding = -log(CSAD / expected_CSAD)
```
简化版用收益率的截面标准差。

##### 3. 依赖列
- `close`
- 需要多个品种数据，此处用滚动窗口内的收益率分布

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@njit
def _herding_behavior_numba(returns, window):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) > 5:
            # 计算截面标准差
            ret_mean = 0.0
            for val in valid_returns:
                ret_mean += val
            ret_mean /= len(valid_returns)
            
            csad = 0.0
            for val in valid_returns:
                csad += abs(val - ret_mean)
            csad /= len(valid_returns)
            
            # 预期CSAD（用市场收益率近似）
            market_ret = ret_mean  # 用均值作为市场收益率
            
            # 计算预期CSAD（简化模型）
            expected_csad = 0.0
            for j in range(len(valid_returns)):
                expected_csad += abs(valid_returns[j] - market_ret)
            expected_csad /= len(valid_returns)
            
            if expected_csad > 0:
                # 羊群指标 = 1 - CSAD/expected_CSAD，值越大羊群效应越强
                result[i] = 1 - csad / expected_csad
    
    return result

@register_feature(
    group="行为金融",
    level="level6_transforms",
    description="羊群行为指标，高值表示羊群效应强",
    depends_on=[],
    output_names=["herding_behavior"]
)
def compute_herding_behavior(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _herding_behavior_numba(returns, window)
    return pd.DataFrame({"herding_behavior": result}, index=df.index)
```

---

#### 特征 14：`overreaction_score`（过度反应得分）

##### 1. 元数据
- **特征类别**：行为金融
- **优先级**：P2
- **理论依据**：过度反应指标用价格对极端收益后的反转强度衡量。

##### 2. 计算公式
```
overreaction = abs(return_t) * (return_t * return_t+1 < 0)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）
- `extreme_threshold`：默认 2.0（极端收益阈值，单位标准差）

##### 5. 集成代码
```python
@njit
def _overreaction_score_numba(returns, window, extreme_threshold=2.0):
    n = len(returns)
    result = np.full(n, np.nan)
    
    # 计算滚动均值和标准差
    for i in range(window, n - 1):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) > 5:
            # 计算均值和标准差
            mean = 0.0
            for val in valid_returns:
                mean += val
            mean /= len(valid_returns)
            
            std = 0.0
            for val in valid_returns:
                std += (val - mean) ** 2
            std = np.sqrt(std / len(valid_returns))
            
            if std > 0:
                # 当前收益是否极端
                current_ret = returns[i]
                if abs(current_ret - mean) > extreme_threshold * std:
                    # 检查下一期是否反转
                    if i + 1 < n and not np.isnan(returns[i+1]):
                        if current_ret * returns[i+1] < 0:
                            # 反转强度
                            result[i] = abs(current_ret) / std
    
    return result

@register_feature(
    group="行为金融",
    level="level6_transforms",
    description="过度反应得分，高值表示可能过度反应",
    depends_on=[],
    output_names=["overreaction_score"]
)
def compute_overreaction_score(df, features_df=None, window=20, extreme_threshold=2.0, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _overreaction_score_numba(returns, window, extreme_threshold)
    return pd.DataFrame({"overreaction_score": result}, index=df.index)
```

---

### 3.8 日历效应类（Calendar Effects）

#### 特征 15：`calendar_effect`（时段效应）

##### 1. 元数据
- **特征类别**：日历效应
- **优先级**：P1
- **理论依据**：不同交易时段（早盘、午盘、收盘）有不同统计特征。

##### 2. 计算公式
```
calendar_effect = indicator(时段) * volatility_ratio
```

##### 3. 依赖列
- `close`, `datetime`（索引）

##### 4. 参数建议
- `session_hours`：根据品种定义

##### 5. 集成代码
```python
@register_feature(
    group="日历效应",
    level="level1_price",
    description="时段效应指标",
    depends_on=[],
    output_names=["morning_session", "afternoon_session", "night_session", "session_vol_ratio"]
)
def compute_calendar_effect(df, features_df=None, **kwargs):
    """
    计算时段效应特征
    
    Returns
    -------
    pd.DataFrame
        morning_session: 早盘标识
        afternoon_session: 午盘标识
        night_session: 夜盘标识
        session_vol_ratio: 时段波动率比率
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取时间索引
    if isinstance(df.index, pd.DatetimeIndex):
        dt_idx = df.index
    else:
        dt_idx = pd.to_datetime(df.index)
    
    # 时段标识（根据中国商品期货市场）
    hour = dt_idx.hour
    
    # 早盘：9:00-10:15, 10:30-11:30
    morning = ((hour == 9) | (hour == 10) | (hour == 11))
    # 午盘：13:30-15:00
    afternoon = (hour == 13) | (hour == 14) | ((hour == 15) & (dt_idx.minute <= 0))
    # 夜盘：21:00-23:00（或次日凌晨）
    night = (hour >= 21) | (hour <= 2)
    
    result["morning_session"] = morning.astype(float)
    result["afternoon_session"] = afternoon.astype(float)
    result["night_session"] = night.astype(float)
    
    # 时段波动率比率
    returns = df["close"].pct_change()
    vol_full = returns.rolling(100, min_periods=50).std()
    
    # 计算各时段波动率
    morning_returns = returns.copy()
    morning_returns[~morning] = np.nan
    morning_vol = morning_returns.rolling(50, min_periods=20).std()
    
    afternoon_returns = returns.copy()
    afternoon_returns[~afternoon] = np.nan
    afternoon_vol = afternoon_returns.rolling(50, min_periods=20).std()
    
    night_returns = returns.copy()
    night_returns[~night] = np.nan
    night_vol = night_returns.rolling(50, min_periods=20).std()
    
    # 合并时段波动率比率
    session_vol = pd.DataFrame(index=df.index)
    session_vol["morning_ratio"] = morning_vol / vol_full
    session_vol["afternoon_ratio"] = afternoon_vol / vol_full
    session_vol["night_ratio"] = night_vol / vol_full
    
    # 取当前时段对应的比率
    result["session_vol_ratio"] = np.where(morning, session_vol["morning_ratio"],
                                  np.where(afternoon, session_vol["afternoon_ratio"],
                                  np.where(night, session_vol["night_ratio"], np.nan)))
    
    return result
```

---

#### 特征 16：`seasonality_strength`（季节性强度）

##### 1. 元数据
- **特征类别**：日历效应
- **优先级**：P2
- **理论依据**：月度、周度等季节性模式的强度。

##### 2. 计算公式
```
seasonality = abs(mean_return_by_period) / std_return
```

##### 3. 依赖列
- `close`, `datetime`

##### 4. 参数建议
- `period`：默认 "month"（可选 "weekday", "hour"）

##### 5. 集成代码
```python
@register_feature(
    group="日历效应",
    level="level5_cross",
    description="季节性强度",
    depends_on=[],
    output_names=["month_seasonality", "weekday_seasonality", "hour_seasonality"]
)
def compute_seasonality_strength(df, features_df=None, **kwargs):
    """
    计算多尺度季节性强度
    
    Returns
    -------
    pd.DataFrame
        包含月度、周度、小时季节性强度
    """
    result = pd.DataFrame(index=df.index)
    
    # 获取时间索引
    if isinstance(df.index, pd.DatetimeIndex):
        dt_idx = df.index
    else:
        dt_idx = pd.to_datetime(df.index)
    
    returns = df["close"].pct_change().dropna()
    
    # 计算各周期均值收益率和标准差
    for period, name in [("M", "month"), ("W", "weekday"), ("H", "hour")]:
        # 计算周期均值
        period_means = returns.groupby(dt_idx[returns.index].to_period(period)).mean()
        period_stds = returns.groupby(dt_idx[returns.index].to_period(period)).std()
        
        # 映射回原索引
        period_idx = dt_idx[returns.index].to_period(period)
        mapped_means = period_idx.map(period_means)
        mapped_stds = period_idx.map(period_stds)
        
        # 季节性强度 = |均值| / 标准差
        strength = np.abs(mapped_means) / (mapped_stds + 1e-8)
        
        # 填充到完整索引
        full_strength = pd.Series(index=df.index, dtype=float)
        full_strength[returns.index] = strength
        
        result[f"{name}_seasonality"] = full_strength
    
    return result
```

---

### 3.9 高阶统计类（Higher-order Statistics）

#### 特征 17：`higher_order_cumulant`（高阶累积量）

##### 1. 元数据
- **特征类别**：高阶统计
- **优先级**：P2
- **理论依据**：三阶、四阶累积量比矩更独立，能捕捉更多分布信息。

##### 2. 计算公式
```
c3 = E[(x-μ)^3]
c4 = E[(x-μ)^4] - 3(E[(x-μ)^2])^2
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）

##### 5. 集成代码
```python
@njit
def _higher_order_cumulant_numba(returns, window):
    n = len(returns)
    c3_result = np.full(n, np.nan)
    c4_result = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) > 10:
            # 计算均值
            mean = 0.0
            for val in valid_returns:
                mean += val
            mean /= len(valid_returns)
            
            # 计算二阶、三阶、四阶中心矩
            m2 = 0.0
            m3 = 0.0
            m4 = 0.0
            for val in valid_returns:
                dev = val - mean
                m2 += dev * dev
                m3 += dev * dev * dev
                m4 += dev * dev * dev * dev
            
            m2 /= len(valid_returns)
            m3 /= len(valid_returns)
            m4 /= len(valid_returns)
            
            # 累积量
            c3_result[i] = m3
            c4_result[i] = m4 - 3 * m2 * m2
    
    return c3_result, c4_result

@register_feature(
    group="高阶统计",
    level="level6_transforms",
    description="高阶累积量（三阶、四阶）",
    depends_on=[],
    output_names=["cumulant_3", "cumulant_4"]
)
def compute_higher_order_cumulant(df, features_df=None, window=50, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    c3, c4 = _higher_order_cumulant_numba(returns, window)
    
    result = pd.DataFrame(index=df.index)
    result["cumulant_3"] = c3
    result["cumulant_4"] = c4
    return result
```

---

#### 特征 18：`z_score_of_z_scores`（Z分数的Z分数）

##### 1. 元数据
- **特征类别**：高阶统计
- **优先级**：P1
- **理论依据**：对Z分数再求Z分数，能检测出极端异常值。

##### 2. 计算公式
```
z2 = (z - mean(z)) / std(z)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）

##### 5. 集成代码
```python
@register_feature(
    group="高阶统计",
    level="level6_transforms",
    description="Z分数的Z分数，检测极端异常值",
    depends_on=[],
    output_names=["z_score_of_z_scores"]
)
def compute_z_score_of_z_scores(df, features_df=None, window=50, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _z_score_of_z_scores_numba(returns, window)
    return pd.DataFrame({"z_score_of_z_scores": result}, index=df.index)
```

---

### 3.10 复杂网络类（Complex Network）

#### 特征 19：`network_centrality`（网络中心度）

##### 1. 元数据
- **特征类别**：复杂网络
- **优先级**：P3
- **理论依据**：在多品种市场中，计算品种在相关性网络中的中心度，反映其市场影响力。

##### 2. 计算公式
基于滚动相关性矩阵的特征向量中心度：
```
centrality = eigenvector_centrality(corr_matrix)
```

##### 3. 依赖列
- `close`（需要多个品种数据）

##### 4. 参数建议
- `window`：默认 60（建议范围 30~120）
- `symbols`：需要其他品种的列名列表

##### 5. 集成代码
```python
@njit
def _eigenvector_centrality_numba(corr_matrix, n_iter=100):
    """简化的特征向量中心度计算（幂迭代法）"""
    n = corr_matrix.shape[0]
    centrality = np.ones(n) / np.sqrt(n)
    
    for _ in range(n_iter):
        # 乘以相关矩阵
        new_centrality = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if not np.isnan(corr_matrix[i, j]):
                    new_centrality[i] += corr_matrix[i, j] * centrality[j]
        
        # 归一化
        norm = 0.0
        for i in range(n):
            norm += new_centrality[i] ** 2
        norm = np.sqrt(norm)
        
        if norm > 0:
            for i in range(n):
                new_centrality[i] /= norm
        
        # 检查收敛
        diff = 0.0
        for i in range(n):
            diff += (new_centrality[i] - centrality[i]) ** 2
        
        centrality = new_centrality
        if diff < 1e-6:
            break
    
    return centrality

@register_feature(
    group="复杂网络",
    level="level6_transforms",
    description="网络中心度（需多品种数据）",
    depends_on=[],
    output_names=["network_centrality"]
)
def compute_network_centrality(df, features_df=None, window=60, symbol_cols=None, **kwargs):
    """
    计算网络中心度
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含多个品种的收盘价列
    symbol_cols : list
        品种列名列表
    """
    if symbol_cols is None or len(symbol_cols) < 3:
        return pd.DataFrame({"network_centrality": np.nan}, index=df.index)
    
    n_symbols = len(symbol_cols)
    n_periods = len(df)
    result = np.full(n_periods, np.nan)
    
    # 计算各品种收益率
    returns_dict = {}
    for col in symbol_cols:
        close = df[col].values.astype(np.float64)
        ret = np.zeros(n_periods)
        ret[0] = np.nan
        for i in range(1, n_periods):
            if close[i-1] > 0:
                ret[i] = (close[i] - close[i-1]) / close[i-1]
        returns_dict[col] = ret
    
    # 滚动计算中心度
    for i in range(window, n_periods):
        # 构建相关矩阵
        corr_matrix = np.zeros((n_symbols, n_symbols))
        
        for s1 in range(n_symbols):
            for s2 in range(s1, n_symbols):
                ret1 = returns_dict[symbol_cols[s1]][i - window + 1 : i + 1]
                ret2 = returns_dict[symbol_cols[s2]][i - window + 1 : i + 1]
                
                # 检查NaN
                valid_idx = []
                for j in range(window):
                    if not np.isnan(ret1[j]) and not np.isnan(ret2[j]):
                        valid_idx.append(j)
                
                if len(valid_idx) > window // 2:
                    # 计算相关系数
                    ret1_valid = np.array([ret1[j] for j in valid_idx])
                    ret2_valid = np.array([ret2[j] for j in valid_idx])
                    
                    mean1 = 0.0
                    mean2 = 0.0
                    for val in ret1_valid:
                        mean1 += val
                    for val in ret2_valid:
                        mean2 += val
                    mean1 /= len(ret1_valid)
                    mean2 /= len(ret2_valid)
                    
                    cov = 0.0
                    var1 = 0.0
                    var2 = 0.0
                    for j in range(len(ret1_valid)):
                        dev1 = ret1_valid[j] - mean1
                        dev2 = ret2_valid[j] - mean2
                        cov += dev1 * dev2
                        var1 += dev1 * dev1
                        var2 += dev2 * dev2
                    
                    if var1 > 0 and var2 > 0:
                        corr = cov / np.sqrt(var1 * var2)
                        corr_matrix[s1, s2] = corr
                        corr_matrix[s2, s1] = corr
        
        # 计算中心度
        centrality = _eigenvector_centrality_numba(corr_matrix)
        
        # 取当前品种的中心度（假设当前品种是第一个）
        if len(centrality) > 0:
            result[i] = centrality[0]
    
    return pd.DataFrame({"network_centrality": result}, index=df.index)
```

---

#### 特征 20：`community_strength`（群落强度）

##### 1. 元数据
- **特征类别**：复杂网络
- **优先级**：P3
- **理论依据**：群落强度衡量品种与其所属板块的一致性。

##### 2. 计算公式
```
community_strength = mean(correlation_with_same_sector) - mean(correlation_with_other_sectors)
```

##### 3. 依赖列
- `close`（需要多个品种数据及板块信息）

##### 4. 参数建议
- `window`：默认 60（建议范围 30~120）
- `sector_map`：品种到板块的映射字典

##### 5. 集成代码
```python
@register_feature(
    group="复杂网络",
    level="level6_transforms",
    description="群落强度，与同板块相关性",
    depends_on=[],
    output_names=["community_strength"]
)
def compute_community_strength(df, features_df=None, window=60, symbol_cols=None, sector_map=None, **kwargs):
    """
    计算群落强度
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含多个品种的收盘价列
    symbol_cols : list
        品种列名列表
    sector_map : dict
        品种到板块的映射，如 {"ag": "金属", "cu": "金属", "rb": "黑色", ...}
    """
    if symbol_cols is None or len(symbol_cols) < 3 or sector_map is None:
        return pd.DataFrame({"community_strength": np.nan}, index=df.index)
    
    n_symbols = len(symbol_cols)
    n_periods = len(df)
    result = np.full(n_periods, np.nan)
    
    # 获取当前品种的板块
    current_symbol = symbol_cols[0]  # 假设当前品种是第一个
    current_sector = sector_map.get(current_symbol, None)
    
    if current_sector is None:
        return pd.DataFrame({"community_strength": np.nan}, index=df.index)
    
    # 计算各品种收益率
    returns_dict = {}
    for col in symbol_cols:
        close = df[col].values.astype(np.float64)
        ret = np.zeros(n_periods)
        ret[0] = np.nan
        for i in range(1, n_periods):
            if close[i-1] > 0:
                ret[i] = (close[i] - close[i-1]) / close[i-1]
        returns_dict[col] = ret
    
    for i in range(window, n_periods):
        same_sector_corrs = []
        other_sector_corrs = []
        
        for col in symbol_cols:
            if col == current_symbol:
                continue
            
            ret_current = returns_dict[current_symbol][i - window + 1 : i + 1]
            ret_other = returns_dict[col][i - window + 1 : i + 1]
            
            # 检查NaN
            valid_idx = []
            for j in range(window):
                if not np.isnan(ret_current[j]) and not np.isnan(ret_other[j]):
                    valid_idx.append(j)
            
            if len(valid_idx) > window // 2:
                # 计算相关系数
                ret_c_valid = np.array([ret_current[j] for j in valid_idx])
                ret_o_valid = np.array([ret_other[j] for j in valid_idx])
                
                mean_c = 0.0
                mean_o = 0.0
                for val in ret_c_valid:
                    mean_c += val
                for val in ret_o_valid:
                    mean_o += val
                mean_c /= len(ret_c_valid)
                mean_o /= len(ret_o_valid)
                
                cov = 0.0
                var_c = 0.0
                var_o = 0.0
                for j in range(len(ret_c_valid
```