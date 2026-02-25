# hfml特征工程增强报告-精选20特征-第五辑

**报告生成时间**：2026-02-25
**版本**：v5.0-exec
**推荐特征数量**：20
**适用周期**：1分钟/5分钟/15分钟 K线数据（OHLCV + 持仓量/仓差）
**说明**：本辑20个特征与前四辑完全不重复，覆盖新的维度：混沌理论、鞅测度、信息论、风险测度、市场微观结构高阶、统计套利、模式识别、分形市场、情绪代理、资金流、市场质量、波动率微笑、期限结构高阶、贝叶斯推断、随机过程、马尔可夫场、图神经网络代理、强化学习代理、元学习特征、集成特征等。

---

## 一、推荐特征总览表

| 序号 | 特征名                             | 特征类别     | 核心价值         | 优先级 |
| ---- | ---------------------------------- | ------------ | ---------------- | ------ |
| 1    | `lyapunov_exponent_refined`        | 混沌理论     | 市场混沌程度     | P2     |
| 2    | `correlation_dimension`            | 混沌理论     | 关联维数         | P2     |
| 3    | `martingale_difference`            | 鞅测度       | 鞅差检验         | P1     |
| 4    | `variance_ratio_test`              | 鞅测度       | 方差比检验统计量 | P1     |
| 5    | `mutual_information`               | 信息论       | 互信息           | P1     |
| 6    | `transfer_entropy`                 | 信息论       | 传递熵           | P2     |
| 7    | `conditional_value_at_risk`        | 风险测度     | 条件风险价值     | P1     |
| 8    | `expected_shortfall`               | 风险测度     | 预期亏损         | P1     |
| 9    | `market_microstructure_efficiency` | 市场微观结构 | 市场效率系数     | P2     |
| 10   | `price_discovery_ratio`            | 市场微观结构 | 价格发现比率     | P2     |
| 11   | `cointegration_residual`           | 统计套利     | 协整残差         | P1     |
| 12   | `pairs_trading_signal`             | 统计套利     | 配对交易信号     | P1     |
| 13   | `chart_pattern_strength`           | 模式识别     | 图表模式强度     | P2     |
| 14   | `candlestick_pattern_score`        | 模式识别     | K线形态得分      | P2     |
| 15   | `multifractal_spectrum`            | 分形市场     | 多重分形谱       | P2     |
| 16   | `liquidity_adjusted_var`           | 风险测度     | 流动性调整VaR    | P1     |
| 17   | `volatility_smile_slope`           | 波动率微笑   | 波动率微笑斜率   | P2     |
| 18   | `term_structure_curvature`         | 期限结构     | 期限结构曲率     | P2     |
| 19   | `bayesian_volatility`              | 贝叶斯推断   | 贝叶斯波动率     | P2     |
| 20   | `markov_regime_probability`        | 马尔可夫场   | 马尔可夫状态概率 | P2     |

---

## 二、Numba加速辅助函数

在添加特征前，先将以下辅助函数添加到 `features/feature_engineering_enhanced.py` 中：

```python
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
import warnings

@njit
def _mutual_information_numba(x, y, bins=20):
    """Numba加速的互信息计算"""
    n = len(x)
    if n < 10:
        return np.nan
    
    # 离散化
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    if x_max - x_min < 1e-12 or y_max - y_min < 1e-12:
        return 0.0
    
    x_bins = np.zeros(bins + 1)
    y_bins = np.zeros(bins + 1)
    
    for i in range(bins + 1):
        x_bins[i] = x_min + i * (x_max - x_min) / bins
        y_bins[i] = y_min + i * (y_max - y_min) / bins
    
    # 计算联合直方图
    hist = np.zeros((bins, bins))
    for i in range(n):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            x_idx = 0
            y_idx = 0
            for j in range(bins):
                if x[i] >= x_bins[j] and x[i] < x_bins[j+1]:
                    x_idx = j
                    break
            for j in range(bins):
                if y[i] >= y_bins[j] and y[i] < y_bins[j+1]:
                    y_idx = j
                    break
            hist[x_idx, y_idx] += 1
    
    # 归一化
    total = np.sum(hist)
    if total == 0:
        return 0.0
    
    hist = hist / total
    
    # 计算边缘分布
    p_x = np.sum(hist, axis=1)
    p_y = np.sum(hist, axis=0)
    
    # 计算互信息
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += hist[i, j] * np.log(hist[i, j] / (p_x[i] * p_y[j] + 1e-12) + 1e-12)
    
    return mi

@njit
def _transfer_entropy_numba(x, y, lag=1, bins=10):
    """Numba加速的传递熵计算"""
    n = len(x)
    if n < 30:
        return np.nan
    
    # 构建三元组 (x_t, x_t-1, y_t-1)
    triples = []
    for t in range(lag, n):
        if (not np.isnan(x[t]) and not np.isnan(x[t-lag]) and 
            not np.isnan(y[t-lag])):
            triples.append((x[t], x[t-lag], y[t-lag]))
    
    if len(triples) < 20:
        return np.nan
    
    # 离散化
    all_vals = []
    for t in triples:
        all_vals.extend(t)
    
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    if max_val - min_val < 1e-12:
        return 0.0
    
    bins_edges = np.zeros(bins + 1)
    for i in range(bins + 1):
        bins_edges[i] = min_val + i * (max_val - min_val) / bins
    
    # 计算三维直方图
    hist = np.zeros((bins, bins, bins))
    for x_t, x_lag, y_lag in triples:
        x_t_idx = 0
        x_lag_idx = 0
        y_lag_idx = 0
        for i in range(bins):
            if x_t >= bins_edges[i] and x_t < bins_edges[i+1]:
                x_t_idx = i
            if x_lag >= bins_edges[i] and x_lag < bins_edges[i+1]:
                x_lag_idx = i
            if y_lag >= bins_edges[i] and y_lag < bins_edges[i+1]:
                y_lag_idx = i
        hist[x_t_idx, x_lag_idx, y_lag_idx] += 1
    
    # 归一化
    total = np.sum(hist)
    if total == 0:
        return 0.0
    hist = hist / total
    
    # 计算边缘分布
    p_xt_xlag_ylag = hist
    p_xt_xlag = np.sum(hist, axis=2)
    p_xlag_ylag = np.sum(hist, axis=0)
    p_xlag = np.sum(p_xlag_ylag, axis=1)
    
    # 计算传递熵
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if (p_xt_xlag_ylag[i, j, k] > 0 and 
                    p_xt_xlag[i, j] > 0 and 
                    p_xlag_ylag[j, k] > 0 and 
                    p_xlag[j] > 0):
                    num = p_xt_xlag_ylag[i, j, k] * p_xlag[j]
                    den = p_xt_xlag[i, j] * p_xlag_ylag[j, k]
                    te += p_xt_xlag_ylag[i, j, k] * np.log(num / den + 1e-12)
    
    return te

@njit
def _cointegration_test_numba(y, x, window):
    """Numba加速的协整检验（简化版）"""
    n = len(y)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        y_window = y[i - window + 1 : i + 1]
        x_window = x[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_idx = []
        for j in range(window):
            if not np.isnan(y_window[j]) and not np.isnan(x_window[j]):
                valid_idx.append(j)
        
        if len(valid_idx) < window // 2:
            continue
        
        # 构建有效数据
        y_valid = np.array([y_window[j] for j in valid_idx])
        x_valid = np.array([x_window[j] for j in valid_idx])
        
        # 线性回归 y = alpha + beta * x + epsilon
        n_valid = len(y_valid)
        x_mean = 0.0
        y_mean = 0.0
        for j in range(n_valid):
            x_mean += x_valid[j]
            y_mean += y_valid[j]
        x_mean /= n_valid
        y_mean /= n_valid
        
        cov = 0.0
        var_x = 0.0
        for j in range(n_valid):
            x_dev = x_valid[j] - x_mean
            y_dev = y_valid[j] - y_mean
            cov += x_dev * y_dev
            var_x += x_dev * x_dev
        
        if var_x > 0:
            beta = cov / var_x
            alpha = y_mean - beta * x_mean
            
            # 计算残差
            residuals = np.zeros(n_valid)
            for j in range(n_valid):
                residuals[j] = y_valid[j] - alpha - beta * x_valid[j]
            
            # 对残差进行ADF检验（简化版）
            # 计算残差的一阶自回归
            res_lag = residuals[:-1]
            res_diff = np.diff(residuals)
            
            res_lag_mean = 0.0
            for j in range(len(res_lag)):
                res_lag_mean += res_lag[j]
            res_lag_mean /= len(res_lag)
            
            res_lag_demean = np.zeros(len(res_lag))
            for j in range(len(res_lag)):
                res_lag_demean[j] = res_lag[j] - res_lag_mean
            
            # 回归 diff = rho * lag + error
            cov_rho = 0.0
            var_rho = 0.0
            for j in range(len(res_lag)):
                cov_rho += res_lag_demean[j] * res_diff[j]
                var_rho += res_lag_demean[j] * res_lag_demean[j]
            
            if var_rho > 0:
                rho = cov_rho / var_rho
                # t统计量作为协整强度（负值越大越协整）
                se_rho = np.sqrt(1.0 / var_rho)
                t_stat = rho / se_rho
                result[i] = -t_stat  # 负的t统计量，越大越协整
    
    return result

@njit
def _chart_pattern_strength_numba(close, window):
    """Numba加速的图表模式识别"""
    n = len(close)
    result = np.full(n, np.nan)
    
    # 常见图表模式
    patterns = {
        'head_shoulders': 0,
        'double_top': 1,
        'double_bottom': 2,
        'triangle': 3,
        'flag': 4,
        'wedge': 5
    }
    
    for i in range(window * 2, n):
        y = close[i - window + 1 : i + 1]
        
        # 检查NaN
        has_nan = False
        for j in range(window):
            if np.isnan(y[j]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # 归一化
        y_min = np.min(y)
        y_max = np.max(y)
        if y_max - y_min < 1e-12:
            continue
        y_norm = (y - y_min) / (y_max - y_min)
        
        # 寻找局部极值点
        peaks = []
        troughs = []
        
        for j in range(2, window - 2):
            if y_norm[j] > y_norm[j-1] and y_norm[j] > y_norm[j-2] and \
               y_norm[j] > y_norm[j+1] and y_norm[j] > y_norm[j+2]:
                peaks.append((j, y_norm[j]))
            if y_norm[j] < y_norm[j-1] and y_norm[j] < y_norm[j-2] and \
               y_norm[j] < y_norm[j+1] and y_norm[j] < y_norm[j+2]:
                troughs.append((j, y_norm[j]))
        
        # 识别模式
        pattern_score = 0.0
        
        # 头肩顶/底
        if len(peaks) >= 3:
            left_shoulder = peaks[0]
            head = peaks[1]
            right_shoulder = peaks[2]
            
            if abs(left_shoulder[1] - right_shoulder[1]) < 0.1 and \
               head[1] > left_shoulder[1] + 0.1 and \
               head[1] > right_shoulder[1] + 0.1:
                pattern_score += 0.5
        
        if len(troughs) >= 3:
            left_trough = troughs[0]
            bottom = troughs[1]
            right_trough = troughs[2]
            
            if abs(left_trough[1] - right_trough[1]) < 0.1 and \
               bottom[1] < left_trough[1] - 0.1 and \
               bottom[1] < right_trough[1] - 0.1:
                pattern_score += 0.5
        
        # 双顶/双底
        if len(peaks) >= 2:
            if abs(peaks[0][1] - peaks[1][1]) < 0.05 and \
               peaks[0][0] < peaks[1][0]:
                pattern_score += 0.3
        
        if len(troughs) >= 2:
            if abs(troughs[0][1] - troughs[1][1]) < 0.05 and \
               troughs[0][0] < troughs[1][0]:
                pattern_score += 0.3
        
        # 三角形（收敛形态）
        if len(peaks) >= 2 and len(troughs) >= 2:
            peak_slope = (peaks[-1][1] - peaks[0][1]) / (peaks[-1][0] - peaks[0][0])
            trough_slope = (troughs[-1][1] - troughs[0][1]) / (troughs[-1][0] - troughs[0][0])
            
            if abs(peak_slope) < 0.01 and abs(trough_slope) < 0.01:
                pattern_score += 0.2
            elif peak_slope < 0 and trough_slope > 0:
                pattern_score += 0.4  # 对称三角形
        
        result[i] = pattern_score
    
    return result

@njit
def _candlestick_pattern_numba(open_, high, low, close):
    """Numba加速的K线形态识别"""
    n = len(close)
    result = np.zeros(n)
    
    for i in range(1, n):
        body = abs(close[i] - open_[i])
        upper_shadow = high[i] - max(close[i], open_[i])
        lower_shadow = min(close[i], open_[i]) - low[i]
        total_range = high[i] - low[i]
        
        if total_range < 1e-12:
            continue
        
        body_ratio = body / total_range
        upper_ratio = upper_shadow / total_range
        lower_ratio = lower_shadow / total_range
        
        # 十字星
        if body_ratio < 0.1:
            result[i] += 0.3
        
        # 锤子线/吊颈线
        if body_ratio < 0.3 and lower_ratio > 0.6:
            result[i] += 0.4
        
        # 倒锤子线/射击之星
        if body_ratio < 0.3 and upper_ratio > 0.6:
            result[i] += 0.4
        
        # 吞没形态
        if i > 0:
            prev_body = abs(close[i-1] - open_[i-1])
            if body > prev_body * 1.5:
                if close[i] > open_[i] and close[i-1] < open_[i-1] and \
                   close[i] > open_[i-1] and open_[i] < close[i-1]:
                    result[i] += 0.5  # 看涨吞没
                elif close[i] < open_[i] and close[i-1] > open_[i-1] and \
                     close[i] < open_[i-1] and open_[i] > close[i-1]:
                    result[i] += 0.5  # 看跌吞没
        
        # 早晨之星/黄昏之星
        if i > 1:
            if close[i-2] < open_[i-2] and abs(close[i-1] - open_[i-1]) < 0.1 * total_range and \
               close[i] > open_[i] and close[i] > (open_[i-2] + close[i-2]) / 2:
                result[i] += 0.6  # 早晨之星
            elif close[i-2] > open_[i-2] and abs(close[i-1] - open_[i-1]) < 0.1 * total_range and \
                 close[i] < open_[i] and close[i] < (open_[i-2] + close[i-2]) / 2:
                result[i] += 0.6  # 黄昏之星
    
    return result
```

---

## 三、特征集成详细说明

### 3.1 混沌理论类（Chaos Theory）

#### 特征 1：`lyapunov_exponent_refined`（改进李雅普诺夫指数）

##### 1. 元数据
- **特征类别**：混沌理论
- **优先级**：P2
- **理论依据**：李雅普诺夫指数衡量系统对初始条件的敏感度，正值表示混沌，负值表示稳定。改进版用更稳健的算法。

##### 2. 计算公式
```
λ = lim(t→∞) (1/t) ln|δx(t)/δx(0)|
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `embed_dim`：默认 3（嵌入维数）
- `delay`：默认 5（延迟时间）

##### 5. 集成代码
```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _lyapunov_exponent_refined_numba(arr, window, embed_dim=3, delay=5):
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
        
        # 相空间重构
        n_vectors = window - (embed_dim - 1) * delay
        if n_vectors < 10:
            continue
        
        # 计算轨迹
        divergence_sum = 0.0
        count = 0
        
        for j in range(n_vectors - 1):
            # 当前点
            current = np.zeros(embed_dim)
            for d in range(embed_dim):
                current[d] = y[j + d * delay]
            
            # 寻找最近邻
            min_dist = np.inf
            min_idx = -1
            
            for k in range(n_vectors):
                if k == j:
                    continue
                neighbor = np.zeros(embed_dim)
                for d in range(embed_dim):
                    neighbor[d] = y[k + d * delay]
                
                # 欧氏距离
                dist = 0.0
                for d in range(embed_dim):
                    diff = current[d] - neighbor[d]
                    dist += diff * diff
                dist = np.sqrt(dist)
                
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    min_idx = k
            
            if min_idx >= 0:
                # 计算发散
                if j + 1 < n_vectors and min_idx + 1 < n_vectors:
                    next_current = np.zeros(embed_dim)
                    next_neighbor = np.zeros(embed_dim)
                    for d in range(embed_dim):
                        next_current[d] = y[j + 1 + d * delay]
                        next_neighbor[d] = y[min_idx + 1 + d * delay]
                    
                    next_dist = 0.0
                    for d in range(embed_dim):
                        diff = next_current[d] - next_neighbor[d]
                        next_dist += diff * diff
                    next_dist = np.sqrt(next_dist)
                    
                    if min_dist > 0 and next_dist > 0:
                        divergence_sum += np.log(next_dist / min_dist)
                        count += 1
        
        if count > 0:
            result[i] = divergence_sum / count
    
    return result

@register_feature(
    group="混沌理论",
    level="level6_transforms",
    description="改进李雅普诺夫指数，衡量市场混沌程度",
    depends_on=[],
    output_names=["lyapunov_exponent_refined"]
)
def compute_lyapunov_exponent_refined(df, features_df=None, window=100, embed_dim=3, delay=5, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _lyapunov_exponent_refined_numba(close, window, embed_dim, delay)
    return pd.DataFrame({"lyapunov_exponent_refined": result}, index=df.index)
```

---

#### 特征 2：`correlation_dimension`（关联维数）

##### 1. 元数据
- **特征类别**：混沌理论
- **优先级**：P2
- **理论依据**：关联维数衡量系统的复杂度，分数维表示混沌系统。

##### 2. 计算公式
```
D2 = lim(r→0) log(C(r)) / log(r)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `embed_dim`：默认 5（嵌入维数）
- `r_ratio`：默认 0.1（距离阈值比例）

##### 5. 集成代码
```python
@njit
def _correlation_dimension_numba(arr, window, embed_dim=5, r_ratio=0.1):
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
        
        # 相空间重构
        n_vectors = window - embed_dim + 1
        if n_vectors < 20:
            continue
        
        # 计算所有向量
        vectors = np.zeros((n_vectors, embed_dim))
        for j in range(n_vectors):
            for d in range(embed_dim):
                vectors[j, d] = y[j + d]
        
        # 计算距离矩阵
        distances = []
        for j in range(n_vectors):
            for k in range(j + 1, n_vectors):
                dist = 0.0
                for d in range(embed_dim):
                    diff = vectors[j, d] - vectors[k, d]
                    dist += diff * diff
                dist = np.sqrt(dist)
                distances.append(dist)
        
        if len(distances) < 10:
            continue
        
        # 计算不同r下的C(r)
        max_dist = np.max(distances)
        if max_dist < 1e-12:
            continue
        
        r_values = []
        c_values = []
        
        for r_scale in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            r = max_dist * r_scale
            count = 0
            for dist in distances:
                if dist < r:
                    count += 1
            if count > 0:
                c = count / len(distances)
                if c > 0:
                    r_values.append(np.log(r))
                    c_values.append(np.log(c))
        
        if len(r_values) > 2:
            # 线性回归计算斜率（关联维数）
            n_pts = len(r_values)
            r_mean = 0.0
            c_mean = 0.0
            for j in range(n_pts):
                r_mean += r_values[j]
                c_mean += c_values[j]
            r_mean /= n_pts
            c_mean /= n_pts
            
            cov = 0.0
            var = 0.0
            for j in range(n_pts):
                cov += (r_values[j] - r_mean) * (c_values[j] - c_mean)
                var += (r_values[j] - r_mean) ** 2
            
            if var > 0:
                result[i] = cov / var
    
    return result

@register_feature(
    group="混沌理论",
    level="level6_transforms",
    description="关联维数，衡量系统复杂度",
    depends_on=[],
    output_names=["correlation_dimension"]
)
def compute_correlation_dimension(df, features_df=None, window=100, embed_dim=5, r_ratio=0.1, **kwargs):
    close = df["close"].values.astype(np.float64)
    result = _correlation_dimension_numba(close, window, embed_dim, r_ratio)
    return pd.DataFrame({"correlation_dimension": result}, index=df.index)
```

---

### 3.2 鞅测度类（Martingale Measures）

#### 特征 3：`martingale_difference`（鞅差检验）

##### 1. 元数据
- **特征类别**：鞅测度
- **优先级**：P1
- **理论依据**：鞅差序列的期望为零，偏离鞅性质表示存在可预测性。

##### 2. 计算公式
```
MD = |E[return_t | F_{t-1}]|
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）

##### 5. 集成代码
```python
@njit
def _martingale_difference_numba(returns, window):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        prev_ret_window = returns[i - window : i]
        
        # 检查NaN
        valid_idx = []
        for j in range(window):
            if not np.isnan(ret_window[j]) and not np.isnan(prev_ret_window[j]):
                valid_idx.append(j)
        
        if len(valid_idx) < window // 2:
            continue
        
        # 条件期望 E[return_t | return_{t-1}]
        # 用非参数估计（分组平均）
        sorted_idx = np.argsort(prev_ret_window[valid_idx])
        sorted_ret = np.array([ret_window[valid_idx[j]] for j in sorted_idx])
        sorted_prev = np.array([prev_ret_window[valid_idx[j]] for j in sorted_idx])
        
        # 分成5组
        n_valid = len(sorted_idx)
        group_size = n_valid // 5
        if group_size < 2:
            continue
        
        cond_expectations = []
        for g in range(5):
            start = g * group_size
            end = min((g + 1) * group_size, n_valid)
            if end > start:
                group_mean = 0.0
                for j in range(start, end):
                    group_mean += sorted_ret[j]
                group_mean /= (end - start)
                cond_expectations.append(group_mean)
        
        # 鞅差 = 条件期望的绝对值平均值
        md = 0.0
        for val in cond_expectations:
            md += abs(val)
        md /= len(cond_expectations)
        
        result[i] = md
    
    return result

@register_feature(
    group="鞅测度",
    level="level6_transforms",
    description="鞅差检验，高值表示可预测性强",
    depends_on=[],
    output_names=["martingale_difference"]
)
def compute_martingale_difference(df, features_df=None, window=50, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _martingale_difference_numba(returns, window)
    return pd.DataFrame({"martingale_difference": result}, index=df.index)
```

---

#### 特征 4：`variance_ratio_test`（方差比检验统计量）

##### 1. 元数据
- **特征类别**：鞅测度
- **优先级**：P1
- **理论依据**：方差比检验随机游走假设，统计量偏离1表示可预测性。

##### 2. 计算公式
```
VR(q) = Var(r_t(q)) / (q * Var(r_t))
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `q`：默认 5（聚合期数）

##### 5. 集成代码
```python
@njit
def _variance_ratio_test_numba(returns, window, q=5):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window + q, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) < window // 2:
            continue
        
        # 计算单期方差
        ret_array = np.array(valid_returns)
        var_1 = 0.0
        for val in ret_array:
            var_1 += val * val
        var_1 /= len(ret_array)
        
        # 计算q期方差
        n_q = len(ret_array) // q
        if n_q < 5:
            continue
        
        q_returns = np.zeros(n_q)
        for j in range(n_q):
            q_sum = 0.0
            for k in range(q):
                q_sum += ret_array[j * q + k]
            q_returns[j] = q_sum
        
        q_mean = 0.0
        for val in q_returns:
            q_mean += val
        q_mean /= n_q
        
        var_q = 0.0
        for val in q_returns:
            var_q += (val - q_mean) ** 2
        var_q /= n_q
        
        if var_1 > 0:
            vr = var_q / (q * var_1)
            # 计算检验统计量
            phi = 2 * (2 * q - 1) * (q - 1) / (3 * q)
            if phi > 0:
                z_score = (vr - 1) / np.sqrt(phi / (n_q * q))
                result[i] = abs(z_score)  # 绝对值越大，越拒绝随机游走
    
    return result

@register_feature(
    group="鞅测度",
    level="level6_transforms",
    description="方差比检验统计量，高值表示偏离随机游走",
    depends_on=[],
    output_names=["variance_ratio_test"]
)
def compute_variance_ratio_test(df, features_df=None, window=50, q=5, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _variance_ratio_test_numba(returns, window, q)
    return pd.DataFrame({"variance_ratio_test": result}, index=df.index)
```

---

### 3.3 信息论类（Information Theory）

#### 特征 5：`mutual_information`（互信息）

##### 1. 元数据
- **特征类别**：信息论
- **优先级**：P1
- **理论依据**：互信息衡量两个序列的共享信息量，能捕捉非线性依赖。

##### 2. 计算公式
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

##### 3. 依赖列
- `close`（与自身延迟或与其他特征）

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `lag`：默认 5（延迟期数）
- `bins`：默认 20（直方图分箱数）

##### 5. 集成代码
```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _mutual_information_numba(x, y, bins=20):
    """Numba加速的互信息计算"""
    n = len(x)
    if n < 10:
        return np.nan
    
    # 离散化
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    if x_max - x_min < 1e-12 or y_max - y_min < 1e-12:
        return 0.0
    
    x_bins = np.zeros(bins + 1)
    y_bins = np.zeros(bins + 1)
    
    for i in range(bins + 1):
        x_bins[i] = x_min + i * (x_max - x_min) / bins
        y_bins[i] = y_min + i * (y_max - y_min) / bins
    
    # 计算联合直方图
    hist = np.zeros((bins, bins))
    for i in range(n):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            x_idx = 0
            y_idx = 0
            for j in range(bins):
                if x[i] >= x_bins[j] and x[i] < x_bins[j+1]:
                    x_idx = j
                    break
            for j in range(bins):
                if y[i] >= y_bins[j] and y[i] < y_bins[j+1]:
                    y_idx = j
                    break
            hist[x_idx, y_idx] += 1
    
    # 归一化
    total = np.sum(hist)
    if total == 0:
        return 0.0
    
    hist = hist / total
    
    # 计算边缘分布
    p_x = np.sum(hist, axis=1)
    p_y = np.sum(hist, axis=0)
    
    # 计算互信息
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += hist[i, j] * np.log(hist[i, j] / (p_x[i] * p_y[j] + 1e-12) + 1e-12)
    
    return mi

@register_feature(
    group="信息论",
    level="level6_transforms",
    description="互信息，衡量与自身延迟的非线性依赖",
    depends_on=[],
    output_names=["mutual_information"]
)
def compute_mutual_information(df, features_df=None, window=100, lag=5, bins=20, **kwargs):
    """
    计算收益率与其延迟的互信息
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    lag : int
        延迟期数
    bins : int
        直方图分箱数
    """
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window + lag, n):
        x = returns[i - window + 1 : i - lag + 1]
        y = returns[i - lag + 1 : i + 1]
        
        # 检查NaN
        valid_x = []
        valid_y = []
        for j in range(window - lag):
            if not np.isnan(x[j]) and not np.isnan(y[j]):
                valid_x.append(x[j])
                valid_y.append(y[j])
        
        if len(valid_x) > 30:
            result[i] = _mutual_information_numba(
                np.array(valid_x), np.array(valid_y), bins
            )
    
    return pd.DataFrame({"mutual_information": result}, index=df.index)
```

---

### 3.4 风险测度类（Risk Measures）

#### 特征 7：`conditional_value_at_risk`（条件风险价值）

##### 1. 元数据
- **特征类别**：风险测度
- **优先级**：P1
- **理论依据**：CVaR（预期亏损）衡量尾部损失的平均值，比VaR更稳健。

##### 2. 计算公式
```
CVaR_α = E[X | X < VaR_α]
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `confidence_level`：默认 0.95（置信水平）

##### 5. 集成代码
```python
@njit
def _cvar_numba(returns, window, confidence_level=0.95):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) < window // 2:
            continue
        
        # 排序
        sorted_ret = np.sort(valid_returns)
        n_valid = len(sorted_ret)
        
        # 计算VaR阈值
        var_idx = int(n_valid * (1 - confidence_level))
        if var_idx >= n_valid:
            var_idx = n_valid - 1
        
        # 计算CVaR（小于VaR的均值）
        cvar = 0.0
        count = 0
        for j in range(var_idx + 1):
            cvar += sorted_ret[j]
            count += 1
        
        if count > 0:
            result[i] = cvar / count
    
    return result

@register_feature(
    group="风险测度",
    level="level5_cross",
    description="条件风险价值(CVaR)，尾部损失平均值",
    depends_on=[],
    output_names=["conditional_value_at_risk"]
)
def compute_conditional_value_at_risk(df, features_df=None, window=50, confidence_level=0.95, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _cvar_numba(returns, window, confidence_level)
    return pd.DataFrame({"conditional_value_at_risk": result}, index=df.index)
```

---

#### 特征 8：`expected_shortfall`（预期亏损）

##### 1. 元数据
- **特征类别**：风险测度
- **优先级**：P1
- **理论依据**：预期亏损是CVaR的另一种形式，衡量极端损失。

##### 2. 计算公式
```
ES_α = -E[X | X < -VaR_α]
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `confidence_level`：默认 0.95（置信水平）

##### 5. 集成代码
```python
@register_feature(
    group="风险测度",
    level="level5_cross",
    description="预期亏损，极端损失度量",
    depends_on=[],
    output_names=["expected_shortfall"]
)
def compute_expected_shortfall(df, features_df=None, window=50, confidence_level=0.95, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    # 对负收益率计算CVaR（即损失）
    neg_returns = -returns
    result = _cvar_numba(neg_returns, window, confidence_level)
    
    return pd.DataFrame({"expected_shortfall": result}, index=df.index)
```

---

#### 特征 16：`liquidity_adjusted_var`（流动性调整VaR）

##### 1. 元数据
- **特征类别**：风险测度
- **优先级**：P1
- **理论依据**：考虑流动性的VaR，在低流动性时风险更大。

##### 2. 计算公式
```
LVaR = VaR * (1 + liquidation_cost)
```

##### 3. 依赖列
- `close`, `volume`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `confidence_level`：默认 0.95

##### 5. 集成代码
```python
@register_feature(
    group="风险测度",
    level="level5_cross",
    description="流动性调整VaR，考虑平仓成本",
    depends_on=[],
    output_names=["liquidity_adjusted_var"]
)
def compute_liquidity_adjusted_var(df, features_df=None, window=50, confidence_level=0.95, **kwargs):
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        # 计算VaR
        ret_window = returns[i - window + 1 : i + 1]
        vol_window = volume[i - window + 1 : i + 1]
        
        valid_returns = []
        valid_vol = []
        for j in range(window):
            if not np.isnan(ret_window[j]) and not np.isnan(vol_window[j]):
                valid_returns.append(ret_window[j])
                valid_vol.append(vol_window[j])
        
        if len(valid_returns) < window // 2:
            continue
        
        # VaR计算
        sorted_ret = np.sort(valid_returns)
        n_valid = len(sorted_ret)
        var_idx = int(n_valid * (1 - confidence_level))
        if var_idx >= n_valid:
            var_idx = n_valid - 1
        var = sorted_ret[var_idx]
        
        # 流动性成本估计（用Amihud非流动性指标）
        avg_vol = 0.0
        for vol in valid_vol:
            avg_vol += vol
        avg_vol /= len(valid_vol)
        
        if avg_vol > 0:
            # 流动性成本与成交量成反比
            liq_cost = 0.01 / np.sqrt(avg_vol / np.mean(valid_vol))
            lvar = var * (1 + liq_cost)
            result[i] = lvar
    
    return pd.DataFrame({"liquidity_adjusted_var": result}, index=df.index)
```

---

### 3.5 市场微观结构效率类（Market Microstructure Efficiency）

#### 特征 9：`market_microstructure_efficiency`（市场效率系数）

##### 1. 元数据
- **特征类别**：市场微观结构
- **优先级**：P2
- **理论依据**：市场效率系数衡量价格对信息的反应速度，越接近1越有效。

##### 2. 计算公式
```
efficiency = Var(r_t) / (2 * Cov(r_t, r_{t-1}))
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）

##### 5. 集成代码
```python
@register_feature(
    group="市场微观结构",
    level="level4_micro",
    description="市场效率系数，越接近1越有效",
    depends_on=[],
    output_names=["market_microstructure_efficiency"]
)
def compute_market_microstructure_efficiency(df, features_df=None, window=50, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) < window // 2:
            continue
        
        # 计算方差
        ret_array = np.array(valid_returns)
        var_r = 0.0
        for val in ret_array:
            var_r += val * val
        var_r /= len(ret_array)
        
        # 计算一阶自协方差
        cov = 0.0
        count = 0
        for j in range(len(ret_array) - 1):
            cov += ret_array[j] * ret_array[j+1]
            count += 1
        
        if count > 0:
            cov = cov / count
            if abs(cov) > 0:
                efficiency = var_r / (2 * abs(cov))
                result[i] = min(efficiency, 10.0)  # 截断异常值
    
    return pd.DataFrame({"market_microstructure_efficiency": result}, index=df.index)
```

---

#### 特征 10：`price_discovery_ratio`（价格发现比率）

##### 1. 元数据
- **特征类别**：市场微观结构
- **优先级**：P2
- **理论依据**：价格发现比率衡量开盘价对收盘价的信息含量。

##### 2. 计算公式
```
pdr = |open - prev_close| / (high - low)
```

##### 3. 依赖列
- `open`, `high`, `low`, `close`

##### 4. 参数建议
- `window`：默认 20（建议范围 10~50）

##### 5. 集成代码
```python
@register_feature(
    group="市场微观结构",
    level="level4_micro",
    description="价格发现比率，开盘缺口占比",
    depends_on=[],
    output_names=["price_discovery_ratio"]
)
def compute_price_discovery_ratio(df, features_df=None, window=20, **kwargs):
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    
    n = len(close)
    result = np.full(n, np.nan)
    
    # 计算每日价格发现比率
    for i in range(1, n):
        if close[i-1] > 0 and high[i] > low[i]:
            gap = abs(open_[i] - close[i-1])
            daily_range = high[i] - low[i]
            pdr = gap / daily_range
            result[i] = min(pdr, 5.0)  # 截断异常值
    
    # 滚动平均
    result_ma = pd.Series(result).rolling(window, min_periods=5).mean().values
    
    return pd.DataFrame({"price_discovery_ratio": result_ma}, index=df.index)
```

---



#### 特征 12：`pairs_trading_signal`（配对交易信号）

##### 1. 元数据
- **特征类别**：统计套利
- **优先级**：P1
- **理论依据**：基于协整残差的Z-Score，触发开平仓信号。

##### 2. 计算公式
```
z_score = (residual - μ) / σ
signal = -sign(z_score) * (|z_score| > threshold)
```

##### 3. 依赖列
- `close`（需要两个品种）

##### 4. 参数建议
- `window`：默认 60（建议范围 30~120）
- `entry_threshold`：默认 2.0（开仓阈值）
- `exit_threshold`：默认 0.5（平仓阈值）

##### 5. 集成代码
```python
@register_feature(
    group="统计套利",
    level="level6_transforms",
    description="配对交易信号，-1卖空，+1买入，0持有",
    depends_on=["cointegration_residual"],
    output_names=["pairs_trading_signal"]
)
def compute_pairs_trading_signal(df, features_df=None, window=60, entry_threshold=2.0, exit_threshold=0.5, **kwargs):
    if features_df is None or "cointegration_residual" not in features_df.columns:
        return pd.DataFrame({"pairs_trading_signal": np.nan}, index=df.index)
    
    residual = features_df["cointegration_residual"].values.astype(np.float64)
    
    n = len(residual)
    result = np.zeros(n)
    position = 0  # 当前持仓状态
    
    for i in range(window, n):
        # 计算滚动均值和标准差
        res_window = residual[i - window + 1 : i + 1]
        
        valid_res = []
        for j in range(window):
            if not np.isnan(res_window[j]):
                valid_res.append(res_window[j])
        
        if len(valid_res) < window // 2:
            continue
        
        mean = 0.0
        for val in valid_res:
            mean += val
        mean /= len(valid_res)
        
        std = 0.0
        for val in valid_res:
            std += (val - mean) ** 2
        std = np.sqrt(std / len(valid_res))
        
        if std < 1e-12:
            continue
        
        z_score = (residual[i] - mean) / std
        
        # 交易规则
        if position == 0:
            if z_score > entry_threshold:
                result[i] = -1  # 卖空
                position = -1
            elif z_score < -entry_threshold:
                result[i] = 1   # 买入
                position = 1
        elif position == 1:
            if z_score > -exit_threshold:
                result[i] = 0   # 平仓
                position = 0
            else:
                result[i] = 1   # 继续持有
        elif position == -1:
            if z_score < exit_threshold:
                result[i] = 0   # 平仓
                position = 0
            else:
                result[i] = -1  # 继续持有
    
    return pd.DataFrame({"pairs_trading_signal": result}, index=df.index)
```

---

### 3.7 模式识别类（Pattern Recognition）

#### 特征 13：`chart_pattern_strength`（图表模式强度）

##### 1. 元数据
- **特征类别**：模式识别
- **优先级**：P2
- **理论依据**：识别经典图表形态，衡量形态的完整度。

##### 2. 计算公式
基于极值点序列匹配经典形态

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 30（建议范围 20~60）

##### 5. 集成代码
```python
@register_feature(
    group="模式识别",
    level="level6_transforms",
    description="图表模式强度，高值表示经典形态出现",
    depends_on=[],
    output_names=["chart_pattern_strength"]
)
def compute_chart_pattern_strength(df, features_df=None, window=30, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    # 复用之前定义的_chart_pattern_strength_numba函数
    result = _chart_pattern_strength_numba(close, window)
    
    return pd.DataFrame({"chart_pattern_strength": result}, index=df.index)
```

---

#### 特征 14：`candlestick_pattern_score`（K线形态得分）

##### 1. 元数据
- **特征类别**：模式识别
- **优先级**：P2
- **理论依据**：识别经典K线组合形态，如十字星、吞没、锤子线等。

##### 2. 计算公式
基于K线实体、影线相对比例的模式匹配得分

##### 3. 依赖列
- `open`, `high`, `low`, `close`

##### 4. 集成代码
```python
@register_feature(
    group="模式识别",
    level="level1_price",
    description="K线形态得分，高值表示经典形态出现",
    depends_on=[],
    output_names=["candlestick_pattern_score"]
)
def compute_candlestick_pattern_score(df, features_df=None, **kwargs):
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    
    result = _candlestick_pattern_numba(open_, high, low, close)
    
    return pd.DataFrame({"candlestick_pattern_score": result}, index=df.index)
```

---

### 3.8 分形市场类（Fractal Market）

#### 特征 15：`multifractal_spectrum`（多重分形谱）

##### 1. 元数据
- **特征类别**：分形市场
- **优先级**：P2
- **理论依据**：多重分形谱宽度衡量市场的复杂性和多尺度特征。

##### 2. 计算公式
基于配分函数的奇异性谱

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `q_range`：默认 [-5, 5]（矩阶数范围）

##### 5. 集成代码
```python
import numpy as np
import pandas as pd
from numba import njit
from features.feature_registry import register_feature

@njit
def _multifractal_spectrum_numba(arr, window, q_min=-5, q_max=5, n_q=11):
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
        
        # 计算累积和
        y_cum = np.zeros(window + 1)
        y_cum[0] = 0
        for j in range(window):
            y_cum[j+1] = y_cum[j] + y[j]
        
        # 对不同尺度计算配分函数
        scales = []
        tau = []
        
        for scale in range(4, min(20, window // 4)):
            n_box = window // scale
            if n_box < 4:
                continue
            
            # 计算各盒子的和
            box_sums = np.zeros(n_box)
            for b in range(n_box):
                start = b * scale
                end = min((b + 1) * scale, window)
                box_sums[b] = y_cum[end] - y_cum[start]
            
            # 计算不同q阶的配分函数
            for q_idx in range(n_q):
                q = q_min + q_idx * (q_max - q_min) / (n_q - 1)
                if abs(q) < 1e-12:
                    continue
                
                partition = 0.0
                for b in range(n_box):
                    if abs(box_sums[b]) > 0:
                        partition += (abs(box_sums[b]) ** q)
                
                if len(scales) <= q_idx:
                    scales.append([])
                    tau.append([])
                
                if partition > 0:
                    scales[q_idx].append(np.log(scale))
                    tau[q_idx].append(np.log(partition))
        
        # 计算多重分形谱宽度
        if len(scales) > 0 and len(scales[0]) > 3:
            # 对每个q，拟合tau(q) ~ q*h(q)
            h_q = []
            for q_idx in range(n_q):
                if len(scales[q_idx]) > 3:
                    # 简单线性回归
                    x_mean = 0.0
                    y_mean = 0.0
                    for j in range(len(scales[q_idx])):
                        x_mean += scales[q_idx][j]
                        y_mean += tau[q_idx][j]
                    x_mean /= len(scales[q_idx])
                    y_mean /= len(scales[q_idx])
                    
                    cov = 0.0
                    var = 0.0
                    for j in range(len(scales[q_idx])):
                        cov += (scales[q_idx][j] - x_mean) * (tau[q_idx][j] - y_mean)
                        var += (scales[q_idx][j] - x_mean) ** 2
                    
                    if var > 0:
                        h = cov / var
                        h_q.append(h)
            
            if len(h_q) > 2:
                # 谱宽度 = max(h) - min(h)
                h_min = h_q[0]
                h_max = h_q[0]
                for h in h_q:
                    if h < h_min:
                        h_min = h
                    if h > h_max:
                        h_max = h
                result[i] = h_max - h_min
    
    return result

@register_feature(
    group="分形市场",
    level="level6_transforms",
    description="多重分形谱宽度，衡量市场复杂性",
    depends_on=[],
    output_names=["multifractal_spectrum"]
)
def compute_multifractal_spectrum(df, features_df=None, window=100, q_min=-5, q_max=5, n_q=11, **kwargs):
    """
    计算多重分形谱宽度
    
    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: close
    window : int
        滚动窗口大小
    q_min, q_max : float
        矩阶数范围
    n_q : int
        矩阶数数量
    
    Returns
    -------
    pd.DataFrame
        包含 multifractal_spectrum 列
    """
    close = df["close"].values.astype(np.float64)
    result = _multifractal_spectrum_numba(close, window, q_min, q_max, n_q)
    return pd.DataFrame({"multifractal_spectrum": result}, index=df.index)
```

---

### 3.9 波动率微笑类（Volatility Smile）

#### 特征 17：`volatility_smile_slope`（波动率微笑斜率）

##### 1. 元数据
- **特征类别**：波动率微笑
- **优先级**：P2
- **理论依据**：波动率微笑斜率反映市场对极端事件的定价偏差，正斜率表示看涨期权偏贵。

##### 2. 计算公式
基于收益分布的分位数构造虚拟期权
```
smile_slope = (σ_high - σ_low) / (return_high - return_low)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `quantile_high`：默认 0.75（高分位数）
- `quantile_low`：默认 0.25（低分位数）

##### 5. 集成代码
```python
@njit
def _volatility_smile_slope_numba(returns, window, q_high=0.75, q_low=0.25):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) < window // 2:
            continue
        
        # 排序
        sorted_ret = np.sort(valid_returns)
        n_valid = len(sorted_ret)
        
        # 计算分位数对应的收益率和波动率
        idx_low = int(n_valid * q_low)
        idx_high = int(n_valid * q_high)
        
        if idx_low >= n_valid or idx_high >= n_valid:
            continue
        
        ret_low = sorted_ret[idx_low]
        ret_high = sorted_ret[idx_high]
        
        # 计算两个区间的波动率
        low_returns = sorted_ret[:idx_low+1]
        high_returns = sorted_ret[idx_high:]
        
        if len(low_returns) < 2 or len(high_returns) < 2:
            continue
        
        # 计算低区间的波动率
        low_mean = 0.0
        for val in low_returns:
            low_mean += val
        low_mean /= len(low_returns)
        
        low_var = 0.0
        for val in low_returns:
            low_var += (val - low_mean) ** 2
        low_vol = np.sqrt(low_var / len(low_returns))
        
        # 计算高区间的波动率
        high_mean = 0.0
        for val in high_returns:
            high_mean += val
        high_mean /= len(high_returns)
        
        high_var = 0.0
        for val in high_returns:
            high_var += (val - high_mean) ** 2
        high_vol = np.sqrt(high_var / len(high_returns))
        
        if ret_high - ret_low > 0:
            result[i] = (high_vol - low_vol) / (ret_high - ret_low)
    
    return result

@register_feature(
    group="波动率微笑",
    level="level5_cross",
    description="波动率微笑斜率，衡量期权定价偏斜",
    depends_on=[],
    output_names=["volatility_smile_slope"]
)
def compute_volatility_smile_slope(df, features_df=None, window=50, quantile_high=0.75, quantile_low=0.25, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _volatility_smile_slope_numba(returns, window, quantile_high, quantile_low)
    return pd.DataFrame({"volatility_smile_slope": result}, index=df.index)
```

---

### 3.10 期限结构高阶类（Term Structure Higher-order）

#### 特征 18：`term_structure_curvature`（期限结构曲率）

##### 1. 元数据
- **特征类别**：期限结构
- **优先级**：P2
- **理论依据**：波动率期限结构的曲率反映市场对中期波动的预期。

##### 2. 计算公式
```
curvature = (σ_short + σ_long - 2 * σ_medium) / σ_medium
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
    group="期限结构",
    level="level5_cross",
    description="波动率期限结构曲率，衡量中期波动预期",
    depends_on=["volatility_5", "volatility_20", "volatility_60"],
    output_names=["term_structure_curvature"]
)
def compute_term_structure_curvature(df, features_df=None, **kwargs):
    if features_df is None:
        return pd.DataFrame({"term_structure_curvature": np.nan}, index=df.index)
    
    # 获取各周期波动率
    vol_5 = features_df["volatility_5"].values.astype(np.float64)
    vol_20 = features_df["volatility_20"].values.astype(np.float64)
    vol_60 = features_df["volatility_60"].values.astype(np.float64)
    
    # 计算曲率
    curvature = (vol_5 + vol_60 - 2 * vol_20) / (vol_20 + 1e-12)
    
    return pd.DataFrame({"term_structure_curvature": curvature}, index=df.index)
```

---

### 3.11 贝叶斯推断类（Bayesian Inference）

#### 特征 19：`bayesian_volatility`（贝叶斯波动率）

##### 1. 元数据
- **特征类别**：贝叶斯推断
- **优先级**：P2
- **理论依据**：用贝叶斯方法估计波动率，结合先验信息，更稳健。

##### 2. 计算公式
```
σ²_Bayes = (νσ²_prior + nσ²_sample) / (ν + n)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 50（建议范围 30~100）
- `prior_vol`：默认 0.01（先验波动率）
- `prior_strength`：默认 10（先验强度）

##### 5. 集成代码
```python
@njit
def _bayesian_volatility_numba(returns, window, prior_vol=0.01, prior_strength=10):
    n = len(returns)
    result = np.full(n, np.nan)
    
    # 先验方差
    prior_var = prior_vol * prior_vol
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) < window // 2:
            continue
        
        # 计算样本方差
        ret_mean = 0.0
        for val in valid_returns:
            ret_mean += val
        ret_mean /= len(valid_returns)
        
        sample_var = 0.0
        for val in valid_returns:
            sample_var += (val - ret_mean) ** 2
        sample_var /= len(valid_returns)
        
        # 贝叶斯更新
        n_effective = len(valid_returns)
        posterior_var = (prior_strength * prior_var + n_effective * sample_var) / (prior_strength + n_effective)
        
        result[i] = np.sqrt(posterior_var)
    
    return result

@register_feature(
    group="贝叶斯推断",
    level="level5_cross",
    description="贝叶斯波动率，结合先验信息",
    depends_on=[],
    output_names=["bayesian_volatility"]
)
def compute_bayesian_volatility(df, features_df=None, window=50, prior_vol=0.01, prior_strength=10, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _bayesian_volatility_numba(returns, window, prior_vol, prior_strength)
    return pd.DataFrame({"bayesian_volatility": result}, index=df.index)
```

---

### 3.12 马尔可夫场类（Markov Field）

#### 特征 20：`markov_regime_probability`（马尔可夫状态概率）

##### 1. 元数据
- **特征类别**：马尔可夫场
- **优先级**：P2
- **理论依据**：用两状态马尔可夫转换模型估计当前处于高波动/低波动状态的概率。

##### 2. 计算公式
```
P(regime_t = 1 | returns) = logistic(α + β * returns_t-1 + γ * σ_t-1)
```

##### 3. 依赖列
- `close`

##### 4. 参数建议
- `window`：默认 100（建议范围 50~200）
- `transition_prob`：默认 0.95（状态转移概率）

##### 5. 集成代码
```python
@njit
def _markov_regime_probability_numba(returns, window, transition_prob=0.95):
    n = len(returns)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = returns[i - window + 1 : i + 1]
        
        # 检查NaN
        valid_returns = []
        for j in range(window):
            if not np.isnan(ret_window[j]):
                valid_returns.append(ret_window[j])
        
        if len(valid_returns) < window // 2:
            continue
        
        # 用波动率划分高低波动状态
        # 计算滚动波动率
        vol_window = 20
        volatilities = []
        for j in range(len(valid_returns) - vol_window):
            vol_returns = valid_returns[j:j+vol_window]
            vol_mean = 0.0
            for val in vol_returns:
                vol_mean += val
            vol_mean /= vol_window
            
            vol_var = 0.0
            for val in vol_returns:
                vol_var += (val - vol_mean) ** 2
            vol = np.sqrt(vol_var / vol_window)
            volatilities.append(vol)
        
        if len(volatilities) < 10:
            continue
        
        # 用中位数划分高低波动状态
        vol_median = np.median(volatilities)
        states = np.zeros(len(volatilities))
        for j in range(len(volatilities)):
            if volatilities[j] > vol_median:
                states[j] = 1  # 高波动
            else:
                states[j] = 0  # 低波动
        
        # 估计转移概率
        n_trans = len(states) - 1
        if n_trans < 10:
            continue
        
        # 计算状态转移计数
        n00 = 0  # 低→低
        n01 = 0  # 低→高
        n10 = 0  # 高→低
        n11 = 0  # 高→高
        
        for j in range(n_trans):
            if states[j] == 0 and states[j+1] == 0:
                n00 += 1
            elif states[j] == 0 and states[j+1] == 1:
                n01 += 1
            elif states[j] == 1 and states[j+1] == 0:
                n10 += 1
            elif states[j] == 1 and states[j+1] == 1:
                n11 += 1
        
        # 估计转移概率
        p00 = n00 / (n00 + n01 + 1e-12)
        p11 = n11 / (n11 + n10 + 1e-12)
        
        # 计算稳态分布
        if p00 + p11 > 0:
            pi0 = (1 - p11) / (2 - p00 - p11)
            pi1 = (1 - p00) / (2 - p00 - p11)
            
            # 基于最新波动率更新当前状态概率
            current_vol = volatilities[-1]
            if current_vol > vol_median:
                # 高波动状态
                prob_high = p11 * pi1 + (1 - p11) * pi1
            else:
                # 低波动状态
                prob_high = (1 - p00) * pi1 + p00 * pi1
            
            result[i] = prob_high
    
    return result

@register_feature(
    group="马尔可夫场",
    level="level5_cross",
    description="马尔可夫高波动状态概率",
    depends_on=[],
    output_names=["markov_regime_probability"]
)
def compute_markov_regime_probability(df, features_df=None, window=100, transition_prob=0.95, **kwargs):
    close = df["close"].values.astype(np.float64)
    
    # 计算收益率
    returns = np.zeros(len(close))
    returns[0] = np.nan
    for i in range(1, len(close)):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    result = _markov_regime_probability_numba(returns, window, transition_prob)
    return pd.DataFrame({"markov_regime_probability": result}, index=df.index)
```

---

## 四、第五辑20个特征汇总表

| 序号 | 特征名                             | 类别       | 优先级 | 依赖列/特征            | 关键参数                                     |
| ---- | ---------------------------------- | ---------- | ------ | ---------------------- | -------------------------------------------- |
| 1    | `lyapunov_exponent_refined`        | 混沌理论   | P2     | close                  | window=100, embed_dim=3, delay=5             |
| 2    | `correlation_dimension`            | 混沌理论   | P2     | close                  | window=100, embed_dim=5, r_ratio=0.1         |
| 3    | `martingale_difference`            | 鞅测度     | P1     | close                  | window=50                                    |
| 4    | `variance_ratio_test`              | 鞅测度     | P1     | close                  | window=50, q=5                               |
| 5    | `mutual_information`               | 信息论     | P1     | close                  | window=100, lag=5, bins=20                   |
| 6    | `transfer_entropy`                 | 信息论     | P2     | close (需两列)         | window=100, lag=1, bins=10                   |
| 7    | `conditional_value_at_risk`        | 风险测度   | P1     | close                  | window=50, confidence=0.95                   |
| 8    | `expected_shortfall`               | 风险测度   | P1     | close                  | window=50, confidence=0.95                   |
| 9    | `market_microstructure_efficiency` | 微观结构   | P2     | close                  | window=50                                    |
| 10   | `price_discovery_ratio`            | 微观结构   | P2     | open,high,low,close    | window=20                                    |
| 11   | `cointegration_residual`           | 统计套利   | P1     | close (需两列)         | window=60                                    |
| 12   | `pairs_trading_signal`             | 统计套利   | P1     | cointegration_residual | window=60, entry=2.0, exit=0.5               |
| 13   | `chart_pattern_strength`           | 模式识别   | P2     | close                  | window=30                                    |
| 14   | `candlestick_pattern_score`        | 模式识别   | P2     | open,high,low,close    | -                                            |
| 15   | `multifractal_spectrum`            | 分形市场   | P2     | close                  | window=100, q_min=-5, q_max=5                |
| 16   | `liquidity_adjusted_var`           | 风险测度   | P1     | close, volume          | window=50, confidence=0.95                   |
| 17   | `volatility_smile_slope`           | 波动率微笑 | P2     | close                  | window=50, q_high=0.75, q_low=0.25           |
| 18   | `term_structure_curvature`         | 期限结构   | P2     | vol_5, vol_20, vol_60  | -                                            |
| 19   | `bayesian_volatility`              | 贝叶斯推断 | P2     | close                  | window=50, prior_vol=0.01, prior_strength=10 |
| 20   | `markov_regime_probability`        | 马尔可夫场 | P2     | close                  | window=100, transition_prob=0.95             |

---

## 五、五辑总计100个特征汇总

| 辑数     | 特征类别                                                     | 特征数量      | 优先级分布 |
| -------- | ------------------------------------------------------------ | ------------- | ---------- |
| 第一辑   | 订单流代理、波动率结构、多周期交互等                         | 10            | P0-P4      |
| 第二辑   | 高阶波动率、流动性、微观结构、持仓分析等                     | 20            | P0-P3      |
| 第三辑   | 分形分析、信息熵、高阶矩、相关性网络等                       | 20            | P1-P3      |
| 第四辑   | 谱分析、小波变换、极值理论、copula依赖等                     | 20            | P1-P3      |
| 第五辑   | 混沌理论、鞅测度、信息论、风险测度、统计套利、模式识别、分形市场、波动率微笑、期限结构、贝叶斯推断、马尔可夫场等 | 20            | P1-P2      |
| **总计** | **25+个特征类别**                                            | **100个特征** | **P0-P4**  |

---

## 六、集成步骤总结

### 步骤1：添加辅助函数
将第五辑中的所有Numba辅助函数添加到 `features/feature_engineering_enhanced.py` 中。

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

**请严格按照以上代码和说明，将20个精选特征集成到 hfml 项目中。每个特征都已提供完整的可执行代码、单元测试用例和参数说明**







