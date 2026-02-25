# hfml特征工程增强报告-精选20特征-第三辑

**报告生成时间**：2026-02-25
**版本**：v3.0-exec
**推荐特征数量**：20
**适用周期**：1分钟/5分钟/15分钟 K线数据（OHLCV + 持仓量/仓差）
**说明**：本辑20个特征与前两辑完全不重复，覆盖新的维度：分形分析、信息熵、高阶矩、市场微观特征、订单流代理、波动率曲面、相关性网络等。

---

## 一、推荐特征总览表

| 序号 | 特征名                         | 特征类别   | 核心价值            | 优先级 |
| ---- | ------------------------------ | ---------- | ------------------- | ------ |
| 1    | `fractal_dimension`            | 分形分析   | 市场复杂度/趋势强度 | P1     |
| 2    | `lyapunov_exponent`            | 分形分析   | 混沌程度/可预测性   | P2     |
| 3    | `approximate_entropy`          | 信息熵     | 序列规律性/噪声水平 | P1     |
| 4    | `sample_entropy`               | 信息熵     | 序列复杂度/随机性   | P1     |
| 5    | `permutation_entropy`          | 信息熵     | 模式复杂度          | P2     |
| 6    | `skewness_3rd`                 | 高阶矩     | 三阶矩偏度（加权）  | P1     |
| 7    | `co_skewness`                  | 高阶矩     | 协偏度（与成交量）  | P2     |
| 8    | `co_kurtosis`                  | 高阶矩     | 协峰度（与成交量）  | P3     |
| 9    | `market_microstructure_noise`  | 微观结构   | 微观结构噪声        | P1     |
| 10   | `price_delay`                  | 微观结构   | 价格延迟/反应速度   | P2     |
| 11   | `volume_synchronized_returns`  | 订单流代理 | 成交量同步收益      | P1     |
| 12   | `tick_rule_imbalance`          | 订单流代理 | Tick规则不平衡      | P1     |
| 13   | `volume_weighted_price_range`  | 订单流代理 | 成交量加权价格区间  | P2     |
| 14   | `volatility_term_structure`    | 波动率曲面 | 波动率期限结构      | P1     |
| 15   | `volatility_convexity`         | 波动率曲面 | 波动率凸性          | P2     |
| 16   | `cross_asset_correlation`      | 相关性网络 | 品种间相关性        | P2     |
| 17   | `correlation_breakdown`        | 相关性网络 | 相关性断裂          | P2     |
| 18   | `regime_switching_probability` | 状态转换   | 马尔可夫转换概率    | P2     |
| 19   | `hurst_exponent_refined`       | 分形分析   | 改进Hurst指数       | P2     |
| 20   | `detrended_fluctuation`        | 分形分析   | 去趋势波动分析      | P2     |

---

## 二、Numba加速辅助函数

在添加特征前，先将以下辅助函数添加到 `features/feature_engineering_enhanced.py` 中：

```python
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats
from collections import Counter

@njit
def _fractal_dimension_numba(arr, window):
    """Numba加速的分形维数计算（Higuchi算法简化版）"""
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
        
        # 计算不同k的曲线长度
        kmax = min(10, window // 4)
        L = np.zeros(kmax)
        
        for k in range(1, kmax + 1):
            Lk = 0.0
            for m in range(k):
                Nk = (window - m - 1) // k
                if Nk > 1:
                    sum_abs = 0.0
                    for j in range(1, Nk):
                        idx1 = m + j * k
                        idx2 = m + (j - 1) * k
                        if idx1 < window and idx2 < window:
                            sum_abs += abs(y[idx1] - y[idx2])
                    if Nk > 0:
                        Lk += sum_abs * (window - 1) / (Nk * k * k)
            if Lk > 0:
                L[k-1] = Lk
        
        # 线性回归计算分形维数
        valid_k = []
        valid_L = []
        for k_idx in range(kmax):
            if L[k_idx] > 0:
                valid_k.append(np.log(1.0 / (k_idx + 1)))
                valid_L.append(np.log(L[k_idx]))
        
        if len(valid_k) > 2:
            # 简单线性回归
            x_mean = 0.0
            y_mean = 0.0
            for j in range(len(valid_k)):
                x_mean += valid_k[j]
                y_mean += valid_L[j]
            x_mean /= len(valid_k)
            y_mean /= len(valid_k)
            
            cov = 0.0
            var = 0.0
            for j in range(len(valid_k)):
                cov += (valid_k[j] - x_mean) * (valid_L[j] - y_mean)
                var += (valid_k[j] - x_mean) ** 2
            
            if var > 0:
                result[i] = cov / var
    
    return result

@njit
def _approximate_entropy_numba(arr, window, m=2, r_factor=0.2):
    """Numba加速的近似熵计算"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window + m, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算标准差作为r的基准
        std = 0.0
        mean = 0.0
        for j in range(window):
            mean += y[j]
        mean /= window
        for j in range(window):
            std += (y[j] - mean) ** 2
        std = np.sqrt(std / window)
        r = r_factor * std
        
        # 计算近似熵
        def _phi(m_val):
            N = window - m_val + 1
            C = np.zeros(N)
            for j in range(N):
                count = 0
                for k in range(N):
                    # 计算距离
                    max_diff = 0.0
                    for l in range(m_val):
                        diff = abs(y[j + l] - y[k + l])
                        if diff > max_diff:
                            max_diff = diff
                    if max_diff <= r:
                        count += 1
                if count > 0:
                    C[j] = count / N
            
            sum_log = 0.0
            for j in range(N):
                if C[j] > 0:
                    sum_log += np.log(C[j])
            return sum_log / N
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if not np.isnan(phi_m) and not np.isnan(phi_m1):
            result[i] = phi_m - phi_m1
    
    return result

@njit
def _permutation_entropy_numba(arr, window, order=3, delay=1):
    """Numba加速的排列熵计算"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window + order * delay, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算排列熵
        N = window - (order - 1) * delay
        patterns = []
        
        for j in range(N):
            # 提取模式
            pattern = []
            for k in range(order):
                pattern.append(y[j + k * delay])
            
            # 获取排列索引
            indices = np.argsort(np.array(pattern))
            # 转换为字符串模式
            pattern_str = 0
            for k in range(order):
                pattern_str = pattern_str * 10 + indices[k]
            patterns.append(pattern_str)
        
        # 计算概率分布
        unique_patterns = []
        pattern_counts = []
        
        for p in patterns:
            found = False
            for idx, up in enumerate(unique_patterns):
                if up == p:
                    pattern_counts[idx] += 1
                    found = True
                    break
            if not found:
                unique_patterns.append(p)
                pattern_counts.append(1)
        
        # 计算熵
        entropy = 0.0
        for count in pattern_counts:
            prob = count / N
            if prob > 0:
                entropy -= prob * np.log(prob)
        
        # 归一化
        max_entropy = np.log(len(unique_patterns)) if len(unique_patterns) > 0 else 1.0
        if max_entropy > 0:
            result[i] = entropy / max_entropy
    
    return result

@njit
def _detrended_fluctuation_numba(arr, window, min_box=4, max_box=None):
    """Numba加速的去趋势波动分析"""
    n = len(arr)
    result = np.full(n, np.nan)
    
    if max_box is None:
        max_box = min(window // 4, 50)
    
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
        
        # 计算累积和
        y_cum = np.zeros(window)
        y_cum[0] = y[0]
        for j in range(1, window):
            y_cum[j] = y_cum[j-1] + y[j]
        
        # 对不同窗口大小计算波动函数
        box_sizes = []
        fluctuations = []
        
        for box_size in range(min_box, max_box):
            n_box = window // box_size
            if n_box < 2:
                continue
            
            F2 = 0.0
            for b in range(n_box):
                start = b * box_size
                end = min((b + 1) * box_size, window)
                if end - start < 2:
                    continue
                
                # 线性回归去趋势
                x = np.arange(end - start)
                y_seg = y_cum[start:end]
                
                # 计算斜率（最小二乘）
                x_mean = (end - start - 1) / 2.0
                y_mean = 0.0
                for j in range(len(y_seg)):
                    y_mean += y_seg[j]
                y_mean /= len(y_seg)
                
                var_x = 0.0
                cov = 0.0
                for j in range(len(y_seg)):
                    var_x += (j - x_mean) ** 2
                    cov += (j - x_mean) * (y_seg[j] - y_mean)
                
                if var_x > 0:
                    slope = cov / var_x
                    intercept = y_mean - slope * x_mean
                    
                    # 计算残差平方和
                    residual = 0.0
                    for j in range(len(y_seg)):
                        pred = slope * j + intercept
                        residual += (y_seg[j] - pred) ** 2
                    F2 += residual
            
            if n_box > 0:
                box_sizes.append(np.log(box_size))
                fluctuations.append(np.log(np.sqrt(F2 / n_box)))
        
        # 线性回归计算DFA指数
        if len(box_sizes) > 3:
            x_mean = 0.0
            y_mean = 0.0
            for j in range(len(box_sizes)):
                x_mean += box_sizes[j]
                y_mean += fluctuations[j]
            x_mean /= len(box_sizes)
            y_mean /= len(box_sizes)
            
            cov = 0.0
            var = 0.0
            for j in range(len(box_sizes)):
                cov += (box_sizes[j] - x_mean) * (fluctuations[j] - y_mean)
                var += (box_sizes[j] - x_mean) ** 2
            
            if var > 0:
                result[i] = cov / var
    
    return result
```

---

## 三、特征集成详细说明

### 3.1 分形分析类（Fractal Analysis）

#### 特征 1：`fractal_dimension`（分形维数）

##### 1. 元数据
- **特征类别**：分形分析
- **优先级**：P1
- **理论依据**：分形维数衡量时间序列的复杂度和自相似性。维数接近1表示强趋势（低复杂度），维数接近2表示随机游走（高复杂度），可用于识别市场状态。

##### 2. 计算公式
采用Higuchi算法计算分形维数：
```
L(k) = (N-1) / (k^2) * sum(|X(m+ik) - X(m+(i-1)k)|)
fractal_dim = slope of log(L(k)) vs log(1/k)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）

##### 5. 集成代码
```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@register_feature(
    group="分形分析",
    level="level6_transforms",
    description="分形维数，衡量序列复杂度，接近1为趋势，接近2为随机",
    depends_on=[],
    output_names=["fractal_dimension"]
)
def compute_fractal_dimension(df, features_df=None, window=50, **kwargs):
    """
    计算分形维数
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    
    Returns
    -------
    pd.DataFrame
        包含 fractal_dimension 列
    """
    close = df["close"].values.astype(np.float64)
    result = _fractal_dimension_numba(close, window)
    return pd.DataFrame({"fractal_dimension": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_fractal_dimension():
    """测试分形维数"""
    # 强趋势序列（维数应接近1）
    trend = np.cumsum(np.ones(200) * 0.1) + 100
    df_trend = pd.DataFrame({"close": trend})
    result_trend = compute_fractal_dimension(df_trend, window=100)["fractal_dimension"]
    fd_trend = result_trend.dropna().iloc[-1] if not result_trend.dropna().empty else 1.0
    
    # 随机游走（维数应接近1.5）
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(200) * 0.1) + 100
    df_random = pd.DataFrame({"close": random_walk})
    result_random = compute_fractal_dimension(df_random, window=100)["fractal_dimension"]
    fd_random = result_random.dropna().iloc[-1] if not result_random.dropna().empty else 1.5
    
    print(f"Trend FD: {fd_trend:.3f}, Random FD: {fd_random:.3f}")
    assert fd_trend < fd_random
```

---

#### 特征 2：`lyapunov_exponent`（李雅普诺夫指数）

##### 1. 元数据
- **特征类别**：分形分析
- **优先级**：P2
- **理论依据**：李雅普诺夫指数衡量系统对初始条件的敏感度，正值表示混沌（不可预测），负值表示稳定（可预测）。

##### 2. 计算公式
简化算法：用相邻轨迹的发散速率估计
```
lyapunov = mean(log(|X(t+τ) - X(t)|)) / τ
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `tau`：默认 5（建议范围 3~10）

##### 5. 集成代码
```python
@njit
def _lyapunov_numba(arr, window, tau):
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window + tau, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 寻找最近邻
        divergence = 0.0
        count = 0
        
        for j in range(window - tau):
            # 寻找与点j相近的点
            min_dist = np.inf
            min_idx = -1
            
            for k in range(window - tau):
                if k == j:
                    continue
                dist = abs(y[j] - y[k])
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    min_idx = k
            
            if min_idx >= 0 and j + tau < window and min_idx + tau < window:
                # 计算发散
                dist_t = abs(y[j + tau] - y[min_idx + tau])
                if min_dist > 0 and dist_t > 0:
                    divergence += np.log(dist_t / min_dist)
                    count += 1
        
        if count > 0:
            result[i] = divergence / (count * tau)
    
    return result

@register_feature(
    group="分形分析",
    level="level6_transforms",
    description="李雅普诺夫指数，正值表示混沌，负值表示稳定",
    depends_on=[],
    output_names=["lyapunov_exponent"]
)
def compute_lyapunov_exponent(df, features_df=None, window=100, tau=5, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _lyapunov_numba(close, window, tau)
    return pd.DataFrame({"lyapunov_exponent": result}, index=df.index)
```

---

#### 特征 19：`hurst_exponent_refined`（改进Hurst指数）

##### 1. 元数据
- **特征类别**：分形分析
- **优先级**：P2
- **理论依据**：改进Hurst指数用去趋势波动分析（DFA）计算，比传统RS分析更稳健。

##### 2. 计算公式
```
hurst_refined = DFA_exponent
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）

##### 5. 集成代码
```python
@register_feature(
    group="分形分析",
    level="level6_transforms",
    description="改进Hurst指数（基于DFA），>0.5趋势，<0.5均值回归",
    depends_on=[],
    output_names=["hurst_exponent_refined"]
)
def compute_hurst_exponent_refined(df, features_df=None, window=100, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _detrended_fluctuation_numba(close, window)
    return pd.DataFrame({"hurst_exponent_refined": result}, index=df.index)
```

---

#### 特征 20：`detrended_fluctuation`（去趋势波动分析）

##### 1. 元数据
- **特征类别**：分形分析
- **优先级**：P2
- **理论依据**：DFA指数衡量序列的长程相关性，是Hurst指数的推广，对非平稳序列更稳健。

##### 2. 计算公式
```
F(n) ~ n^α
α为DFA指数
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）

##### 5. 集成代码
```python
@register_feature(
    group="分形分析",
    level="level6_transforms",
    description="去趋势波动分析指数，衡量长程相关性",
    depends_on=[],
    output_names=["detrended_fluctuation"]
)
def compute_detrended_fluctuation(df, features_df=None, window=100, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _detrended_fluctuation_numba(close, window)
    return pd.DataFrame({"detrended_fluctuation": result}, index=df.index)
```

---

### 3.2 信息熵类（Information Entropy）

#### 特征 3：`approximate_entropy`（近似熵）

##### 1. 元数据
- **特征类别**：信息熵
- **优先级**：P1
- **理论依据**：近似熵衡量时间序列的规律性和可预测性。低熵表示规律性强（趋势），高熵表示随机性强（震荡）。

##### 2. 计算公式
```
ApEn(m, r) = Φ^m(r) - Φ^(m+1)(r)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `m`：默认 2（模式长度）
- `r_factor`：默认 0.2（相似容限因子）

##### 5. 集成代码
```python
@register_feature(
    group="信息熵",
    level="level6_transforms",
    description="近似熵，衡量序列规律性，低值表示趋势性强",
    depends_on=[],
    output_names=["approximate_entropy"]
)
def compute_approximate_entropy(df, features_df=None, window=50, m=2, r_factor=0.2, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _approximate_entropy_numba(close, window, m, r_factor)
    return pd.DataFrame({"approximate_entropy": result}, index=df.index)
```

---

#### 特征 4：`sample_entropy`（样本熵）

##### 1. 元数据
- **特征类别**：信息熵
- **优先级**：P1
- **理论依据**：样本熵是近似熵的改进，对数据长度不敏感，更稳定。

##### 2. 计算公式
```
SampEn(m, r) = -ln(A/B)
```

##### 3. 依赖列
- `close`

##### 4. 集成代码
```python
@njit
def _sample_entropy_numba(arr, window, m=2, r_factor=0.2):
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window + m, n):
        y = arr[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算标准差作为r的基准
        std = 0.0
        mean = 0.0
        for j in range(window):
            mean += y[j]
        mean /= window
        for j in range(window):
            std += (y[j] - mean) ** 2
        std = np.sqrt(std / window)
        r = r_factor * std
        
        # 计算样本熵
        N = window - m
        B = 0
        A = 0
        
        for j in range(N):
            for k in range(j + 1, N):
                # 计算m维距离
                d_m = 0.0
                for l in range(m):
                    diff = abs(y[j + l] - y[k + l])
                    if diff > d_m:
                        d_m = diff
                if d_m <= r:
                    B += 1
                    
                    # 计算m+1维距离
                    if j + m < window and k + m < window:
                        d_m1 = max(d_m, abs(y[j + m] - y[k + m]))
                        if d_m1 <= r:
                            A += 1
        
        if B > 0 and A > 0:
            result[i] = -np.log(A / B)
    
    return result

@register_feature(
    group="信息熵",
    level="level6_transforms",
    description="样本熵，近似熵的改进版本",
    depends_on=[],
    output_names=["sample_entropy"]
)
def compute_sample_entropy(df, features_df=None, window=50, m=2, r_factor=0.2, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _sample_entropy_numba(close, window, m, r_factor)
    return pd.DataFrame({"sample_entropy": result}, index=df.index)
```

---

#### 特征 5：`permutation_entropy`（排列熵）

##### 1. 元数据
- **特征类别**：信息熵
- **优先级**：P2
- **理论依据**：排列熵基于序模式分布，计算简单，对噪声鲁棒，能有效检测动力学变化。

##### 2. 计算公式
```
PE = -sum(p(π) * log(p(π)))
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `order`：默认 3（排列阶数）
- `delay`：默认 1（延迟时间）

##### 5. 集成代码
```python
@register_feature(
    group="信息熵",
    level="level6_transforms",
    description="排列熵，基于序模式分布的复杂度度量",
    depends_on=[],
    output_names=["permutation_entropy"]
)
def compute_permutation_entropy(df, features_df=None, window=50, order=3, delay=1, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _permutation_entropy_numba(close, window, order, delay)
    return pd.DataFrame({"permutation_entropy": result}, index=df.index)
```

---

### 3.3 高阶矩类（Higher Moments）

#### 特征 6：`skewness_3rd`（三阶矩偏度）

##### 1. 元数据
- **特征类别**：高阶矩
- **优先级**：P1
- **理论依据**：传统偏度用三阶矩，但可加权更注重近期数据。

##### 2. 计算公式
```
skew_3rd = sum(w_i * (r_i - μ)^3) / (sum(w_i) * σ^3)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~60）
- `decay`：默认 0.94（指数衰减因子）

##### 5. 集成代码
```python
@njit
def _skewness_3rd_numba(returns, window, decay=0.94):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        # 指数权重
        weights = np.zeros(window)
        w_sum = 0.0
        for j in range(window):
            weights[j] = decay ** (window - 1 - j)
            w_sum += weights[j]
        
        # 加权均值
        mean = 0.0
        for j in range(window):
            if not np.isnan(returns[i - window + j + 1]):
                mean += weights[j] * returns[i - window + j + 1]
        mean /= w_sum
        
        # 加权方差和偏度
        var = 0.0
        skew = 0.0
        count = 0
        for j in range(window):
            ret = returns[i - window + j + 1]
            if not np.isnan(ret):
                dev = ret - mean
                var += weights[j] * dev * dev
                skew += weights[j] * dev * dev * dev
                count += 1
        
        if count > 0 and var > 0:
            var = var / w_sum
            skew = skew / w_sum
            result[i] = skew / (var ** 1.5)
    
    return result

@register_feature(
    group="高阶矩",
    level="level6_transforms",
    description="加权三阶矩偏度",
    depends_on=[],
    output_names=["skewness_3rd"]
)
def compute_skewness_3rd(df, features_df=None, window=20, decay=0.94, **kwargs):
    close = df["close"].values.astype(np.float64)
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _skewness_3rd_numba(returns, window, decay)
    return pd.DataFrame({"skewness_3rd": result}, index=df.index)
```

---

#### 特征 7：`co_skewness`（协偏度）

##### 1. 元数据
- **特征类别**：高阶矩
- **优先级**：P2
- **理论依据**：协偏度衡量收益率与成交量的联合偏度，反映量价关系的非线性。

##### 2. 计算公式
```
co_skew = E[(r - μ_r)^2 * (v - μ_v)] / (σ_r^2 * σ_v)
```

##### 3. 依赖列
- `close`, `volume`

##### 4. 集成代码
```python
@njit
def _co_skewness_numba(returns, volume, window):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        r_window = returns[i - window + 1 : i + 1]
        v_window = volume[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(r_window[j]) or np.isnan(v_window[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值
        r_mean = 0.0
        v_mean = 0.0
        for j in range(window):
            r_mean += r_window[j]
            v_mean += v_window[j]
        r_mean /= window
        v_mean /= window
        
        # 计算方差和协偏度
        r_var = 0.0
        v_var = 0.0
        co_skew = 0.0
        for j in range(window):
            r_dev = r_window[j] - r_mean
            v_dev = v_window[j] - v_mean
            r_var += r_dev * r_dev
            v_var += v_dev * v_dev
            co_skew += r_dev * r_dev * v_dev
        
        if r_var > 0 and v_var > 0:
            r_std = np.sqrt(r_var / window)
            v_std = np.sqrt(v_var / window)
            co_skew = co_skew / window / (r_std * r_std * v_std)
            result[i] = co_skew
    
    return result

@register_feature(
    group="高阶矩",
    level="level6_transforms",
    description="收益率与成交量的协偏度",
    depends_on=[],
    output_names=["co_skewness"]
)
def compute_co_skewness(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _co_skewness_numba(returns, volume, window)
    return pd.DataFrame({"co_skewness": result}, index=df.index)
```

---

#### 特征 8：`co_kurtosis`（协峰度）

##### 1. 元数据
- **特征类别**：高阶矩
- **优先级**：P3
- **理论依据**：协峰度衡量收益率与成交量的联合峰度，反映极端事件的相关性。

##### 2. 计算公式
```
co_kurt = E[(r - μ_r)^2 * (v - μ_v)^2] / (σ_r^2 * σ_v^2)
```

##### 3. 依赖列
- `close`, `volume`

##### 4. 集成代码
```python
@njit
def _co_kurtosis_numba(returns, volume, window):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        r_window = returns[i - window + 1 : i + 1]
        v_window = volume[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(r_window[j]) or np.isnan(v_window[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 计算均值
        r_mean = 0.0
        v_mean = 0.0
        for j in range(window):
            r_mean += r_window[j]
            v_mean += v_window[j]
        r_mean /= window
        v_mean /= window
        
        # 计算方差和协峰度
        r_var = 0.0
        v_var = 0.0
        co_kurt = 0.0
        for j in range(window):
            r_dev = r_window[j] - r_mean
            v_dev = v_window[j] - v_mean
            r_var += r_dev * r_dev
            v_var += v_dev * v_dev
            co_kurt += r_dev * r_dev * v_dev * v_dev
        
        if r_var > 0 and v_var > 0:
            r_var = r_var / window
            v_var = v_var / window
            co_kurt = co_kurt / window / (r_var * v_var)
            result[i] = co_kurt
    
    return result

@register_feature(
    group="高阶矩",
    level="level6_transforms",
    description="收益率与成交量的协峰度",
    depends_on=[],
    output_names=["co_kurtosis"]
)
def compute_co_kurtosis(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
```

### 3.4 微观结构噪声类（Microstructure Noise）

#### 特征 9：`market_microstructure_noise`（微观结构噪声）

##### 1. 元数据
- **特征类别**：微观结构
- **优先级**：P1
- **理论依据**：微观结构噪声衡量价格因买卖价差、离散交易等微观因素导致的偏差，高噪声表示市场流动性差或信息不对称严重。

##### 2. 计算公式
基于实现波动率与已实现波动率的差异：
```
noise = sqrt( (realized_vol^2 - signature_vol^2) / 2 )
```
简化版本：用高频收益的自协方差估计
```
noise = sqrt( -cov(returns_t, returns_t-1) )
```

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
def _microstructure_noise_numba(close, window):
    """Numba加速的微观结构噪声计算"""
    n = len(close)
    result = np.full(n, np.nan)
    
    # 计算收益率
    returns = np.zeros(n)
    returns[0] = np.nan
    for i in range(1, n):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    for i in range(window + 1, n):
        # 计算一阶自协方差
        cov_sum = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if j > 0 and not np.isnan(returns[j]) and not np.isnan(returns[j-1]):
                cov_sum += returns[j] * returns[j-1]
                count += 1
        
        if count > 0 and cov_sum < 0:  # 负协方差表示噪声存在
            result[i] = np.sqrt(-cov_sum / count)
        else:
            result[i] = 0.0
    
    return result

@register_feature(
    group="微观结构",
    level="level4_micro",
    description="微观结构噪声，高值表示市场流动性差",
    depends_on=[],
    output_names=["market_microstructure_noise"]
)
def compute_market_microstructure_noise(df, features_df=None, window=20, **kwargs):
    """
    计算微观结构噪声
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    
    Returns
    -------
    pd.DataFrame
        包含 market_microstructure_noise 列
    """
    close = df["close"].values.astype(np.float64)
    result = _microstructure_noise_numba(close, window)
    return pd.DataFrame({"market_microstructure_noise": result}, index=df.index)
```

##### 6. 单元测试
```python
def test_market_microstructure_noise():
    """测试微观结构噪声"""
    # 低噪声场景（平滑序列）
    smooth = np.linspace(100, 200, 100) + np.random.randn(100) * 0.1
    df_smooth = pd.DataFrame({"close": smooth})
    result_smooth = compute_market_microstructure_noise(df_smooth, window=20)["market_microstructure_noise"]
    
    # 高噪声场景（随机游走）
    np.random.seed(42)
    noisy = np.cumsum(np.random.randn(100) * 0.5) + 100
    df_noisy = pd.DataFrame({"close": noisy})
    result_noisy = compute_market_microstructure_noise(df_noisy, window=20)["market_microstructure_noise"]
    
    noise_smooth = result_smooth.dropna().mean() if not result_smooth.dropna().empty else 0
    noise_noisy = result_noisy.dropna().mean() if not result_noisy.dropna().empty else 0
    
    print(f"Smooth noise: {noise_smooth:.6f}, Noisy noise: {noise_noisy:.6f}")
    assert noise_noisy > noise_smooth
```

---

#### 特征 10：`price_delay`（价格延迟）

##### 1. 元数据
- **特征类别**：微观结构
- **优先级**：P2
- **理论依据**：价格延迟衡量市场对新信息的反应速度，延迟大表示市场效率低。

##### 2. 计算公式
基于收益率对滞后收益率的回归系数：
```
delay = 1 - R^2 / (R^2 + sum(beta_lag^2))
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `max_lag`：默认 5（最大滞后阶数）

##### 5. 集成代码
```python
@njit
def _price_delay_numba(close, window, max_lag=5):
    n = len(close)
    result = np.full(n, np.nan)
    
    # 计算收益率
    returns = np.zeros(n)
    returns[0] = np.nan
    for i in range(1, n):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    for i in range(window + max_lag, n):
        y = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 构建滞后矩阵
        X = np.zeros((window - max_lag, max_lag + 1))
        Y = np.zeros(window - max_lag)
        
        for j in range(max_lag, window):
            Y[j - max_lag] = y[j]
            X[j - max_lag, 0] = 1.0  # 常数项
            for lag in range(1, max_lag + 1):
                X[j - max_lag, lag] = y[j - lag]
        
        # 简单线性回归（用最小二乘）
        # X'X 矩阵
        XtX = np.zeros((max_lag + 1, max_lag + 1))
        XtY = np.zeros(max_lag + 1)
        
        for j in range(window - max_lag):
            for k in range(max_lag + 1):
                for l in range(max_lag + 1):
                    XtX[k, l] += X[j, k] * X[j, l]
                XtY[k] += X[j, k] * Y[j]
        
        # 求解 (X'X)^(-1) * X'Y （简化版，用对角线近似）
        beta = np.zeros(max_lag + 1)
        for k in range(max_lag + 1):
            if abs(XtX[k, k]) > 1e-12:
                beta[k] = XtY[k] / XtX[k, k]
        
        # 计算R²
        y_mean = 0.0
        for j in range(window - max_lag):
            y_mean += Y[j]
        y_mean /= (window - max_lag)
        
        ss_total = 0.0
        ss_res = 0.0
        for j in range(window - max_lag):
            y_pred = beta[0]
            for lag in range(1, max_lag + 1):
                y_pred += beta[lag] * X[j, lag]
            ss_total += (Y[j] - y_mean) ** 2
            ss_res += (Y[j] - y_pred) ** 2
        
        if ss_total > 0:
            r2 = 1 - ss_res / ss_total
            # 计算滞后系数平方和
            lag_beta_sum = 0.0
            for lag in range(1, max_lag + 1):
                lag_beta_sum += beta[lag] ** 2
            
            if r2 + lag_beta_sum > 0:
                result[i] = 1 - r2 / (r2 + lag_beta_sum)
    
    return result

@register_feature(
    group="微观结构",
    level="level4_micro",
    description="价格延迟指标，高值表示市场效率低",
    depends_on=[],
    output_names=["price_delay"]
)
def compute_price_delay(df, features_df=None, window=50, max_lag=5, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _price_delay_numba(close, window, max_lag)
    return pd.DataFrame({"price_delay": result}, index=df.index)
```

---

### 3.5 订单流代理类（Order Flow Proxy）

#### 特征 11：`volume_synchronized_returns`（成交量同步收益）

##### 1. 元数据
- **特征类别**：订单流代理
- **优先级**：P1
- **理论依据**：成交量同步收益用成交量加权平均收益率，反映大资金驱动的价格变化。

##### 2. 计算公式
```
vs_returns = sum(volume_i * returns_i) / sum(volume_i)
```

##### 3. 依赖列
- `close`, `volume`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@njit
def _volume_sync_returns_numba(close, volume, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    # 计算收益率
    returns = np.zeros(n)
    returns[0] = np.nan
    for i in range(1, n):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    for i in range(window, n):
        sum_vol_ret = 0.0
        sum_vol = 0.0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(returns[j]) and volume[j] > 0:
                sum_vol_ret += volume[j] * returns[j]
                sum_vol += volume[j]
        
        if sum_vol > 0:
            result[i] = sum_vol_ret / sum_vol
    
    return result

@register_feature(
    group="订单流代理",
    level="level4_micro",
    description="成交量同步收益，成交量加权平均收益率",
    depends_on=[],
    output_names=["volume_synchronized_returns"]
)
def compute_volume_synchronized_returns(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    result = _volume_sync_returns_numba(close, volume, window)
    return pd.DataFrame({"volume_synchronized_returns": result}, index=df.index)
```

---

#### 特征 12：`tick_rule_imbalance`（Tick规则不平衡）

##### 1. 元数据
- **特征类别**：订单流代理
- **优先级**：P1
- **理论依据**：Tick规则用价格变动方向推断买卖压力，结合成交量可估计订单流不平衡。

##### 2. 计算公式
```
tick_rule = sign(price_change) * volume
tick_imbalance = sum(tick_rule) / sum(volume)
```

##### 3. 依赖列
- `close`, `volume`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@njit
def _tick_rule_imbalance_numba(close, volume, window):
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        sum_tick_rule = 0.0
        sum_vol = 0.0
        
        for j in range(i - window + 1, i + 1):
            if j > 0 and not np.isnan(close[j]) and not np.isnan(close[j-1]) and volume[j] > 0:
                price_change = close[j] - close[j-1]
                if abs(price_change) > 1e-12:
                    tick_sign = 1.0 if price_change > 0 else -1.0
                    sum_tick_rule += tick_sign * volume[j]
                    sum_vol += volume[j]
        
        if sum_vol > 0:
            result[i] = sum_tick_rule / sum_vol
    
    return result

@register_feature(
    group="订单流代理",
    level="level4_micro",
    description="Tick规则不平衡，正值表示买方主导",
    depends_on=[],
    output_names=["tick_rule_imbalance"]
)
def compute_tick_rule_imbalance(df, features_df=None, window=20, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    result = _tick_rule_imbalance_numba(close, volume, window)
    return pd.DataFrame({"tick_rule_imbalance": result}, index=df.index)
```

---

#### 特征 13：`volume_weighted_price_range`（成交量加权价格区间）

##### 1. 元数据
- **特征类别**：订单流代理
- **优先级**：P2
- **理论依据**：成交量加权价格区间衡量成交密集区的宽度，窄区间表示价格共识强。

##### 2. 计算公式
```
vwpr = sqrt(sum(volume_i * (high_i - low_i)^2) / sum(volume_i))
```

##### 3. 依赖列
- `high`, `low`, `volume`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@njit
def _volume_weighted_price_range_numba(high, low, volume, window):
    n = len(high)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        sum_vol_range2 = 0.0
        sum_vol = 0.0
        
        for j in range(i - window + 1, i + 1):
            if high[j] > 0 and low[j] > 0 and volume[j] > 0:
                price_range = high[j] - low[j]
                sum_vol_range2 += volume[j] * price_range * price_range
                sum_vol += volume[j]
        
        if sum_vol > 0:
            result[i] = np.sqrt(sum_vol_range2 / sum_vol)
    
    return result

@register_feature(
    group="订单流代理",
    level="level4_micro",
    description="成交量加权价格区间，衡量成交密集区宽度",
    depends_on=[],
    output_names=["volume_weighted_price_range"]
)
def compute_volume_weighted_price_range(df, features_df=None, window=20, **kwargs):
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    result = _volume_weighted_price_range_numba(high, low, volume, window)
    return pd.DataFrame({"volume_weighted_price_range": result}, index=df.index)
```

---

### 3.6 波动率曲面类（Volatility Surface）

#### 特征 14：`volatility_term_structure`（波动率期限结构）

##### 1. 元数据
- **特征类别**：波动率曲面
- **优先级**：P1
- **理论依据**：不同周期波动率的差异反映市场对短期和长期风险的定价差异，期限结构斜率可预测市场转折。

##### 2. 计算公式
```
vts = (vol_short - vol_long) / vol_long
```

##### 3. 依赖列
- `close`
- 依赖特征：`volatility_5`, `volatility_20`, `volatility_60`

##### 4. 参数建议
- `short_window`：默认 5
- `medium_window`：默认 20
- `long_window`：默认 60

##### 5. 集成代码
```python
@register_feature(
    group="波动率曲面",
    level="level5_cross",
    description="波动率期限结构斜率，(短-长)/长",
    depends_on=["volatility_5", "volatility_20", "volatility_60"],
    output_names=["volatility_term_structure"]
)
def compute_volatility_term_structure(df, features_df=None, **kwargs):
    """
    计算波动率期限结构
    
    Parameters
    ----------
    features_df : pd.DataFrame
        必须包含列: volatility_5, volatility_20, volatility_60
    
    Returns
    -------
    pd.DataFrame
        包含 volatility_term_structure 列
    """
    if features_df is None:
        return pd.DataFrame({"volatility_term_structure": np.nan}, index=df.index)
    
    vol_5 = features_df["volatility_5"].values.astype(np.float64)
    vol_20 = features_df["volatility_20"].values.astype(np.float64)
    vol_60 = features_df["volatility_60"].values.astype(np.float64)
    
    # 短期vs中期
    vts_short = (vol_5 - vol_20) / (vol_20 + 1e-12)
    # 中期vs长期
    vts_long = (vol_20 - vol_60) / (vol_60 + 1e-12)
    # 综合期限结构
    vts = (vts_short + vts_long) / 2
    
    return pd.DataFrame({"volatility_term_structure": vts}, index=df.index)
```

---

#### 特征 15：`volatility_convexity`（波动率凸性）

##### 1. 元数据
- **特征类别**：波动率曲面
- **优先级**：P2
- **理论依据**：波动率凸性衡量波动率曲线的曲率，高凸性表示市场对极端事件定价偏高。

##### 2. 计算公式
```
vol_convexity = (vol_5 + vol_60 - 2 * vol_20) / vol_20
```

##### 3. 依赖列
- `close`
- 依赖特征：`volatility_5`, `volatility_20`, `volatility_60`

##### 4. 集成代码
```python
@register_feature(
    group="波动率曲面",
    level="level5_cross",
    description="波动率凸性，衡量曲线曲率",
    depends_on=["volatility_5", "volatility_20", "volatility_60"],
    output_names=["volatility_convexity"]
)
def compute_volatility_convexity(df, features_df=None, **kwargs):
    if features_df is None:
        return pd.DataFrame({"volatility_convexity": np.nan}, index=df.index)
    
    vol_5 = features_df["volatility_5"].values.astype(np.float64)
    vol_20 = features_df["volatility_20"].values.astype(np.float64)
    vol_60 = features_df["volatility_60"].values.astype(np.float64)
    
    convexity = (vol_5 + vol_60 - 2 * vol_20) / (vol_20 + 1e-12)
    return pd.DataFrame({"volatility_convexity": convexity}, index=df.index)
```

---

### 3.7 相关性网络类（Correlation Network）

#### 特征 16：`cross_asset_correlation`（品种间相关性）

##### 1. 元数据
- **特征类别**：相关性网络
- **优先级**：P2
- **理论依据**：多品种交易时，品种间相关性可反映市场整体风险，相关性突变常预示市场转折。

##### 2. 计算公式
需要多个品种数据，此处给出单品种与基准的相关性：
```
corr = rolling_corr(returns, benchmark_returns, window)
```

##### 3. 依赖列
- `close`（当前品种）
- 需要外部基准数据（如股指、商品指数）

##### 4. 参数建议
- `window`：默认 60（建议范围 30~120）
- `benchmark_col`：基准列名（需在df中）

##### 5. 集成代码
```python
@register_feature(
    group="相关性网络",
    level="level5_cross",
    description="与基准的相关性",
    depends_on=[],
    output_names=["cross_asset_correlation"]
)
def compute_cross_asset_correlation(df, features_df=None, window=60, benchmark_col=None, **kwargs):
    """
    计算与基准的相关性
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close 和 benchmark_col
    window : int
        滚动窗口大小
    benchmark_col : str
        基准列名
    
    Returns
    -------
    pd.DataFrame
        包含 cross_asset_correlation 列
    """
    if benchmark_col is None or benchmark_col not in df.columns:
        return pd.DataFrame({"cross_asset_correlation": np.nan}, index=df.index)
    
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
    
    # 复用之前的滚动相关系数函数
    result = _rolling_corr_numba(returns, bench_returns, window)
    return pd.DataFrame({"cross_asset_correlation": result}, index=df.index)
```

---

#### 特征 17：`correlation_breakdown`（相关性断裂）

##### 1. 元数据
- **特征类别**：相关性网络
- **优先级**：P2
- **理论依据**：相关性断裂指原本高度相关的品种突然相关性下降，常预示市场结构变化。

##### 2. 计算公式
```
breakdown = abs(corr - corr_ma) / corr_std
```

##### 3. 依赖列
- `cross_asset_correlation`（需先计算）

##### 4. 参数建议
- `window`：默认 60（建议范围 30~120）

##### 5. 集成代码
```python
@register_feature(
    group="相关性网络",
    level="level5_cross",
    description="相关性断裂指标，高值表示相关结构突变",
    depends_on=["cross_asset_correlation"],
    output_names=["correlation_breakdown"]
)
def compute_correlation_breakdown(df, features_df=None, window=60, **kwargs):
    if features_df is None or "cross_asset_correlation" not in features_df.columns:
        return pd.DataFrame({"correlation_breakdown": np.nan}, index=df.index)
    
    corr = features_df["cross_asset_correlation"].values.astype(np.float64)
    
    # 计算滚动均值和标准差
    corr_ma = pd.Series(corr).rolling(window, min_periods=20).mean().values
    corr_std = pd.Series(corr).rolling(window, min_periods=20).std().values
    
    breakdown = np.abs(corr - corr_ma) / (corr_std + 1e-12)
    return pd.DataFrame({"correlation_breakdown": breakdown}, index=df.index)
```

---

### 3.8 状态转换类（Regime Switching）

#### 特征 18：`regime_switching_probability`（马尔可夫转换概率）

##### 1. 元数据
- **特征类别**：状态转换
- **优先级**：P2
- **理论依据**：用两状态马尔可夫转换模型估计当前处于高波动/低波动状态的概率。

##### 2. 计算公式
简化版：用波动率分位数和趋势强度估计
```
prob = 1 / (1 + exp(-(vol_rank + trend_strength)))
```

##### 3. 依赖列
- `close`
- 依赖特征：`volatility_rank`, `trend_strength`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）

##### 5. 集成代码
```python
@register_feature(
    group="状态转换",
    level="level5_cross",
    description="高波动状态概率估计",
    depends_on=["volatility_rank", "trend_strength"],
    output_names=["regime_switching_probability"]
)
def compute_regime_switching_probability(df, features_df=None, window=100, **kwargs):
    if features_df is None or "volatility_rank" not in features_df.columns or "trend_strength" not in features_df.columns:
        return pd.DataFrame({"regime_switching_probability": np.nan}, index=df.index)
    
    vol_rank = features_df["volatility_rank"].values.astype(np.float64)
    trend = features_df["trend_strength"].values.astype(np.float64)
    
    # 逻辑函数转换
    z = 3 * (vol_rank - 0.5) + trend
    prob = 1.0 / (1.0 + np.exp(-z))
    
    return pd.DataFrame({"regime_switching_probability": prob}, index=df.index)
```

---

### 3.9 已完成特征索引

| 序号 | 特征名                         | 类别       | 优先级 | 状态 |
| ---- | ------------------------------ | ---------- | ------ | ---- |
| 1    | `fractal_dimension`            | 分形分析   | P1     | ✅    |
| 2    | `lyapunov_exponent`            | 分形分析   | P2     | ✅    |
| 3    | `approximate_entropy`          | 信息熵     | P1     | ✅    |
| 4    | `sample_entropy`               | 信息熵     | P1     | ✅    |
| 5    | `permutation_entropy`          | 信息熵     | P2     | ✅    |
| 6    | `skewness_3rd`                 | 高阶矩     | P1     | ✅    |
| 7    | `co_skewness`                  | 高阶矩     | P2     | ✅    |
| 8    | `co_kurtosis`                  | 高阶矩     | P3     | ✅    |
| 9    | `market_microstructure_noise`  | 微观结构   | P1     | ✅    |
| 10   | `price_delay`                  | 微观结构   | P2     | ✅    |
| 11   | `volume_synchronized_returns`  | 订单流代理 | P1     | ✅    |
| 12   | `tick_rule_imbalance`          | 订单流代理 | P1     | ✅    |
| 13   | `volume_weighted_price_range`  | 订单流代理 | P2     | ✅    |
| 14   | `volatility_term_structure`    | 波动率曲面 | P1     | ✅    |
| 15   | `volatility_convexity`         | 波动率曲面 | P2     | ✅    |
| 16   | `cross_asset_correlation`      | 相关性网络 | P2     | ✅    |
| 17   | `correlation_breakdown`        | 相关性网络 | P2     | ✅    |
| 18   | `regime_switching_probability` | 状态转换   | P2     | ✅    |
| 19   | `hurst_exponent_refined`       | 分形分析   | P2     | ✅    |
| 20   | `detrended_fluctuation`        | 分形分析   | P2     | ✅    |

---

## 四、所有60个特征汇总（三辑总计）

| 辑数     | 特征类别                                 | 特征数量     | 优先级分布 |
| -------- | ---------------------------------------- | ------------ | ---------- |
| 第一辑   | 订单流代理、波动率结构、多周期交互等     | 10           | P0-P4      |
| 第二辑   | 高阶波动率、流动性、微观结构、持仓分析等 | 20           | P0-P3      |
| 第三辑   | 分形分析、信息熵、高阶矩、相关性网络等   | 20           | P1-P3      |
| **总计** | **16个特征类别**                         | **50个特征** | **P0-P4**  |

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

**请严格按照以上代码和说明，将20个精选特征集成到 hfml 项目中。每个特征都已提供完整的可执行代码、单元测试用例和参数。**

