# 商品期货机器学习量化模型

基于 1分钟、5分钟、15分钟 K线数据（OHLCV + 持仓量/仓差）的机器学习量化交易模型系统。系统包含完整的特征工程、市场状态识别、微观结构分析、Numba高性能计算、模型集成、回测评估、特征选择和仓位管理流程。

---

## 目录

- [项目结构](#项目结构)
- [代码文件功能说明](#代码文件功能说明)
- [特征工程可扩展性](#特征工程可扩展性)
- [真实数据流水线](#真实数据流水线)
- [各周期机器学习模型选择](#各周期机器学习模型选择)
- [15分钟LSTM特征预筛选](#15分钟lstm特征预筛选)
- [深度特征变换](#深度特征变换)
- [智能标签系统](#智能标签系统)
- [混合智能交易系统](#混合智能交易系统)
- [特征列表](#特征列表)
- [预测目标](#预测目标)
- [特征评估报告](#特征评估报告)
- [Numba性能加速](#numba性能加速)
- [安装与运行](#安装与运行)
- [运行示例与结果解读](#运行示例与结果解读)

---

## 项目结构

```
GitProject/
├── config.py                          # 全局配置文件
├── ml_pipeline.py                     # 主流程入口脚本（模拟数据）
├── run_real_data_pipeline.py          # 真实数据全自动流水线（6步：加载→特征→筛选→训练→回测→预测）
├── requirements.txt                   # Python依赖
├── FirstProject.py                    # VN Trader 启动脚本（独立）
├── factor_loader.py                   # 因子批量加载与索引对齐模块
├── hyperopt_runner.py                 # Optuna/贝叶斯超参搜索模块
├── signal_postprocess.py              # 截面TopN信号后处理模块
├── scripts/
│   └── load_real_data.py              # 真实K线CSV数据加载模块（列映射、校验、回退）
├── tests/
│   └── test_custom_features.py        # 自定义特征因子的单元测试（159个测试用例）
├── features/
│   ├── __init__.py
│   ├── feature_engineering.py         # 特征工程主模块
│   ├── feature_engineering_enhanced.py # 增强特征模块（微观结构/高级波动率/缺口衰减/辅助函数）
│   ├── feature_registry.py            # 特征注册表模块（装饰器式自动发现）
│   ├── custom_features.py             # 自定义特征（82个注册特征，含9+10+21+20+20个因子+2个示例）
│   ├── feature_context.py             # 特征计算缓存上下文模块
│   ├── feature_transforms.py         # 深度特征变换模块（非线性/跨周期/变化率/条件/交互）
│   ├── market_regime.py              # 市场状态识别模块
│   └── rolling_numba.py             # Numba加速滚动计算模块
├── models/
│   ├── __init__.py
│   ├── ml_models.py                   # 机器学习模型模块
│   ├── ensemble_model.py             # 模型集成模块
│   ├── multi_timeframe.py            # 多时间框架协同模块（15min→5min→1min层级协同）
│   ├── xgb_feature_selector.py      # XGBoost特征预筛选模块（15分钟LSTM专用）
│   ├── smart_labels.py              # 智能标签生成模块（五级信号+质量评分）
│   ├── hybrid_trading.py            # 混合智能交易系统（五专家加权投票）
│   ├── adaptive_learning.py          # 实时自适应模块（在线学习/漂移检测/自适应阈值）
│   ├── position_sizing.py            # 风险预算与仓位管理模块
│   ├── cost_model.py                 # 交易成本与止盈止损模块
│   └── var_stress.py                 # VaR风险度量与压力测试模块
├── backtest/
│   ├── __init__.py
│   └── feature_selector.py            # 回测与特征选择模块
└── data/                              # 数据目录（不纳入版本控制）
    ├── klines/KQi@SHFEag/            # 白银(ag)K线CSV数据
    │   ├── KQi@SHFEag_1min.csv
    │   ├── KQi@SHFEag_5min.csv
    │   └── KQi@SHFEag_15min.csv
    └── mx/                            # 流水线输出目录
        ├── feature_selection_report_{period}.md
        ├── backtest_report_{period}.md
        ├── predictions_{period}.csv
        └── models/                    # 已训练模型及配套文件
```

---

## 代码文件功能说明

### `config.py` — 全局配置文件

集中管理所有可调参数，包括：

| 配置项 | 说明 |
|--------|------|
| `PERIODS` | 支持的K线周期列表：`["1min", "5min", "15min"]` |
| `ML_FRAMEWORKS` | 各周期推荐的机器学习框架映射 |
| `PREDICTION_TARGETS` | 预测目标定义及各周期的预测步长（horizon） |
| `BACKTEST_CONFIG` | 回测参数：训练/验证/测试集比例、交叉验证折数、最低成功率阈值 |
| `LGBM_CONFIG` / `XGB_CONFIG` / `LSTM_CONFIG` | 各模型超参数 |
| `REGIME_CONFIG` | 市场状态识别参数 |
| `ENSEMBLE_CONFIG` | 集成模型配置 |
| `POSITION_CONFIG` | 仓位管理配置（含回撤保护和日内限额） |
| `ADAPTIVE_CONFIG` | 实时自适应配置（漂移检测/在线学习/阈值灵敏度） |
| `MULTI_TIMEFRAME_CONFIG` | 多时间框架协同配置（中性区间/入场阈值/优化阈值/信号等级） |
| `ENHANCED_FEATURE_CONFIG` | 增强特征配置（微观结构/高级波动率/缺口衰减/Numba加速） |
| `FEATURE_TRANSFORM_CONFIG` | 深度特征变换配置（非线性变换/变化率特征列表） |
| `SMART_LABEL_CONFIG` | 智能标签配置（阈值/强信号倍数/质量权重） |
| `HYBRID_SYSTEM_CONFIG` | 混合专家系统配置（信号阈值/仓位/Kelly参数） |
| `FEATURE_HIERARCHY` | 6层分层特征结构（level1_price → level6_transforms） |

### `features/feature_engineering.py` — 特征工程主模块

负责从原始K线数据计算所有衍生技术指标，集成市场状态特征和增强特征。

**核心函数：**

| 函数名 | 功能 | 输出特征 |
|--------|------|----------|
| `compute_ma()` | 简单移动平均线 | `ma_5`, `ma_10`, `ma_20`, `ma_60` 等 |
| `compute_ema()` | 指数移动平均线 | `ema_5`, `ema_10`, `ema_20`, `ema_60` 等 |
| `compute_bollinger_bands()` | 布林带 | `boll_upper`, `boll_mid`, `boll_lower`, `boll_width`, `boll_pct_b` |
| `compute_rsi()` | 相对强弱指标 | `rsi_14`, `rsi_6` |
| `compute_macd()` | MACD | `macd_dif`, `macd_dea`, `macd_hist` |
| `compute_kdj()` | KDJ | `kdj_k`, `kdj_d`, `kdj_j` |
| `compute_atr()` | 平均真实波幅 | `tr`, `atr` |
| `compute_cci()` | CCI | `cci` |
| `compute_williams_r()` | 威廉指标 | `williams_r` |
| `compute_roc()` | 变动率 | `roc_12`, `roc_6` |
| `compute_obv()` | 能量潮 | `obv` |
| `compute_vwap()` | 成交量加权平均价 | `vwap` |
| `compute_volume_features()` | 成交量特征 | `vol_ma_*`, `vol_ratio_*`, `vol_change` |
| `compute_oi_features()` | 持仓量/仓差特征 | `oi_change`, `oi_change_pct`, `oi_ma_*`, `vol_oi_ratio` |
| `compute_candle_features()` | K线形态特征 | `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, `candle_direction`, `amplitude`, `gap`, `gap_ratio` |
| `compute_volatility_features()` | 波动率特征 | `volatility_*`, `return_ma_*`, `log_return` |
| `compute_price_position()` | 价格位置特征 | `price_position`, `dist_to_high`, `dist_to_low` |
| `compute_all_features()` | **一键计算所有特征**（含市场状态 + 微观结构 + 高级波动率 + 缺口衰减 + 深度变换 + 注册表自定义特征） | 224+个特征（随自定义特征增加） |
| `get_feature_hierarchy()` | **获取分层特征结构**，按6层层级组织全部特征（自动合并注册表特征） | 层级字典 |
| `compute_prediction_targets()` | 预测目标 | `future_return`, `future_direction`, `future_volatility`, `future_regime` |

### `features/feature_engineering_enhanced.py` — 增强特征模块

基于附件 multi_timeframe.py 和 rolling_numba.py 的分析新增的高价值特征。

| 函数名 | 功能 | 输出特征 |
|--------|------|----------|
| `compute_microstructure_features()` | 微观结构分析特征 | `divergence`, `vwap_dev`, `vol_state`, `mom_slope`, `close_pos`, `rsi_slope`, `vol_zscore` |
| `compute_advanced_volatility_features()` | 高级波动率特征 | `volatility_regime`, `vol_ratio`, `atr_pct`, `range_pct` |
| `compute_gap_decay_features()` | 期货夜盘缺口衰减特征 | `gap_decay` |

### `features/feature_transforms.py` — 深度特征变换模块

在现有数据边界内最大化信息提取，不新增数据源，通过对已有特征进行深度加工提升信息密度。

| 函数名 | 功能 | 输出特征 |
|--------|------|----------|
| `compute_nonlinear_transforms()` | 非线性变换（平方/对数/排名/Z-score） | `*_squared`, `*_log`, `*_rank`, `*_zscore` |
| `compute_cross_period_ratios()` | 跨周期比率（不同窗口模拟多周期对比） | `rsi_fast_slow_ratio`, `volatility_ratio_fast_slow`, `ma_cross_ratio` |
| `compute_feature_velocity()` | 特征变化率与加速度（一阶/二阶导数） | `*_velocity`, `*_acceleration`, `*_direction_consistency` |
| `compute_contextual_features()` | 条件特征（基于市场状态的动态特征） | `rsi_high_vol`, `rsi_low_vol`, `momentum_in_trend`, `momentum_in_range` |
| `compute_feature_interactions()` | 特征交互项（技术指标×量/仓交互效应） | `rsi_volume_interaction`, `momentum_vol_interaction`, `oi_price_alignment`, `oi_price_magnitude` |
| `compute_all_transforms()` | **一键计算所有深度变换** | ~34个新特征 |

### `features/rolling_numba.py` — Numba加速滚动计算模块

提供比pandas原生函数快10-50倍的数值计算实现，当numba未安装时自动降级为纯numpy。

| 函数名 | 功能 | 加速倍率 |
|--------|------|----------|
| `rolling_mean()` | 滚动均值 | 10-50x |
| `rolling_std()` | 滚动标准差 | 10-50x |
| `rsi()` | RSI指标 | 10-50x |
| `macd_hist()` | MACD柱状图 | 10-50x |
| `bollinger_bandwidth()` | 布林带宽度 | 10-50x |
| `pct_change()` | 百分比变化 | 10-50x |
| `log_return()` | 对数收益率 | 10-50x |
| `signed_volume_imbalance()` | 签名成交量不平衡 | 10-50x |

### `features/market_regime.py` — 市场状态识别模块

**`MarketRegimeDetector` 类** — 识别趋势/震荡/反转等市场状态（五种状态 × 10个特征）。

**五种市场状态：**

| 编码 | 状态标签 | 说明 | 交易建议 |
|------|----------|------|----------|
| 0 | `low_vol_range` | 低波动震荡 — 窄幅整理 | 观望或轻仓做区间 |
| 1 | `low_vol_trend` | 低波动趋势 — 平稳趋势 | 顺势轻仓操作 |
| 2 | `high_vol_range` | 高波动震荡 — 宽幅震荡 | 宽幅区间交易、注意止损 |
| 3 | `high_vol_trend` | 高波动趋势 — 强势行情 | 趋势跟踪、加大仓位 |
| 4 | `reversal` | 反转信号 — 趋势即将反转 | 平仓或反向操作 |

**状态判定逻辑：**
1. 计算滚动波动率 + 趋势强度（Pearson相关系数）
2. 波动率分位数 > 0.7 → 高波动；< 0.3 → 低波动
3. 趋势强度绝对值 > 0.3 → 强趋势；否则 → 震荡
4. 反转信号（优先级最高，覆盖其他状态）：
   - RSI超买(>70)或超卖(<30) — 权重40%
   - 价格偏离均线超过2% — 权重30%
   - 趋势加速度反向（趋势正在减弱）— 权重30%
   - 三者加权总分 ≥ 0.5 时判定为反转

**输出特征（10个）：**

| 特征名 | 说明 |
|--------|------|
| `regime_volatility` | 滚动波动率 |
| `trend_strength` | 趋势强度（-1~+1，Pearson相关系数） |
| `volatility_rank` | 波动率百分位排名 |
| `market_regime` | 五状态编码（0~4） |
| `trend_direction` | 趋势方向（-1/0/+1） |
| `volatility_change` | 波动率5周期变化率 |
| `trend_acceleration` | 趋势加速度 |
| `reversal_score` | 反转信号强度（0~1） |
| `regime_duration` | 当前状态已持续K线数 |
| `regime_change_prob` | 状态转换概率（滚动频率） |

### `models/ml_models.py` — 机器学习模型模块

封装 LightGBM / XGBoost / LSTM 三种模型，提供统一接口，支持二分类和多分类。

### `models/ensemble_model.py` — 模型集成模块

LightGBM + XGBoost 加权集成，基于验证集自动分配权重。

### `models/xgb_feature_selector.py` — XGBoost特征预筛选模块

由于LSTM无法直接评估特征重要性，使用XGBoost作为特征筛选器，为15分钟LSTM模型筛选最有效的特征子集。

| 方法 | 说明 |
|------|------|
| `fit_select(X, y)` | 使用TimeSeriesSplit CV训练XGBoost，计算各折平均特征重要性，筛选Top N特征 |
| `get_feature_groups_importance(groups)` | 按特征组评估重要性，统计每组入选数量 |
| `dynamic_feature_count(X, y)` | 自动搜索最佳特征数量（达到95%最大准确率的最少特征数） |

**预筛选流程：**

```
原始特征 (118个) → XGBoost筛选 (5折CV) → Top 25特征 → LSTM训练
```

### `models/smart_labels.py` — 智能标签生成模块

从简单二元分类升级为智能信号标签系统，生成五级交易信号和信号质量评分。

**`SmartLabelGenerator` 类 — 智能标签生成器**

**五级交易信号：**

| 信号值 | 名称 | 条件 | 含义 |
|--------|------|------|------|
| **+2** | `strong_buy` | 大涨 + 持仓增 + 放量 | 强势看多 |
| **+1** | `weak_buy` | 上涨但缺乏确认 | 弱看多 |
| **0** | `neutral` | 变化幅度不足 | 不交易 |
| **-1** | `weak_sell` | 下跌但缺乏确认 | 弱看空 |
| **-2** | `strong_sell` | 大跌 + 持仓减 + 放量 | 强势看空 |

**信号质量评分 (0~1)：**
- 0.4 × 收益率幅度（波动率调整后）
- 0.3 × 持仓确认度（持仓变化越大越确定）
- 0.2 × 成交量确认（有放量确认为1）
- 0.1 × 波动率环境适宜度（适中最好）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `horizon` | 3 | 预测未来K线数量 |
| `base_threshold` | 0.001 | 基础信号阈值 |
| `strong_multiplier` | 2.0 | 强信号 = base_threshold × multiplier |
| `vol_window` | 20 | 波动率计算窗口 |
| `quality_weights` | (0.4, 0.3, 0.2, 0.1) | 信号质量各维度权重 |

使用示例:
```python
from models.smart_labels import SmartLabelGenerator

generator = SmartLabelGenerator(horizon=3)
labels = generator.create_labels(df)

print(labels["trading_signal"])   # 五级信号 (-2~+2)
print(labels["signal_quality"])   # 质量评分 (0~1)
print(labels["expected_return"])  # 预期收益率
```

### `models/hybrid_trading.py` — 混合智能交易系统

整合五个独立交易专家，通过加权投票融合信号，经过自适应风控调整生成最终交易决策。

**三层决策架构：**

```
┌────────────────────────┐
│ 信号生成层 (5个专家)     │
│  趋势 | 回归 | 突破     │
│  量价 | 持仓             │
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│ 信号融合层 (加权投票)     │
│  权重 × 信号 → 融合信号  │
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│ 风险管理层 (自适应)       │
│  波动率调整 + 仓位控制   │
└────────────────────────┘
```

**五个交易专家：**

| 专家 | 类名 | 信号逻辑 | 置信度依据 |
|------|------|----------|-----------|
| 趋势跟踪 | `TrendFollowingExpert` | MA快>MA慢→做多, MA快<MA慢→做空 | 趋势强度 × 方向一致性 |
| 均值回归 | `MeanReversionExpert` | RSI>70→卖出, RSI<30→买入 | RSI偏离50的程度 |
| 突破交易 | `BreakoutExpert` | 突破N日高→做多, 跌破N日低→做空 | 突破幅度 / 前价格 |
| 量价关系 | `VolumePriceExpert` | 涨+放量→做多, 跌+放量→做空 | 成交量超出均值的程度 |
| 持仓确认 | `OIConfirmationExpert` | 持仓价格同向→趋势延续, 反向→可能反转 | 协同一致性 × 持仓变化幅度 |

**`HybridTradingSystem` 类：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `signal_threshold` | 0.3 | 最低融合信号强度阈值（低于此值不交易） |
| `max_position` | 0.2 | 最大仓位比例（20%） |
| `target_win_rate` | 0.7 | Kelly公式目标胜率 |
| `target_win_loss_ratio` | 2.0 | Kelly公式目标盈亏比 |

使用示例:
```python
from models.hybrid_trading import HybridTradingSystem

system = HybridTradingSystem()
result = system.predict(df)

print(result["final_signal"])       # +1/-1/0
print(result["position_size"])      # 0~0.2
print(result["confidence"])         # 0~1
print(result["experts_breakdown"])  # 各专家信号明细

# 批量预测
batch = system.predict_batch(df, window=100)
print(batch["signal"].value_counts())
```

### `models/multi_timeframe.py` — 多时间框架协同模块

解决各周期模型独立运行的问题，通过层级化信号融合提升交易质量。

**协同策略架构：**

```
┌─────────────────┐
│  15分钟模型       │ ← 确定趋势方向（看多/看空/中性）
│  TrendDirection  │   中性时不开仓
└────────┬────────┘
         ▼
┌─────────────────┐
│  5分钟模型        │ ← 寻找入场时机（信号必须与大方向一致）
│  EntryTiming     │   方向不一致时过滤掉
└────────┬────────┘
         ▼
┌─────────────────┐
│  1分钟模型        │ ← 优化入场价格（在入场窗口内择时）
│  EntryPrice      │   选择最优入场点
└────────┬────────┘
         ▼
┌─────────────────┐
│  最终交易信号     │   方向 + 强度 + 入场质量
└─────────────────┘
```

**`TrendDirectionModel` 类 — 15分钟趋势方向模型**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `neutral_zone` | 0.10 | 中性区间宽度（概率在0.4~0.6之间判定为中性） |

- 输出方向: +1(看多) / -1(看空) / 0(中性)
- 输出确信度: 0~1（概率偏离0.5的程度）
- 中性方向时不允许开仓（最关键的过滤规则）

**`EntryTimingModel` 类 — 5分钟入场时机模型**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `entry_threshold` | 0.60 | 入场概率阈值（需超过此值才触发） |

- 做多条件: 15分钟看多 **且** 5分钟上涨概率 > 0.60
- 做空条件: 15分钟看空 **且** 5分钟下跌概率 > 0.60
- 方向不一致的信号自动被过滤

**`EntryPriceOptimizer` 类 — 1分钟入场价格优化模型**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `optimization_threshold` | 0.55 | 1分钟确认阈值（概率>0.55认为是好入场点） |

- 在5分钟触发入场后，用1分钟模型评估当前是否为最优入场点
- 输出入场质量评分（entry_score, 0~1）

**`MultiTimeframeCoordinator` 类 — 协同管理器**

整合三个周期模型，生成最终交易信号并评级。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_grade` | "C" | 最低信号质量等级，低于此等级的信号被过滤 |

**信号质量等级：**

| 等级 | 名称 | 条件 | 信号强度 |
|------|------|------|----------|
| **A** | 三周期共振 (resonance) | 15min+5min+1min三个周期完全一致 | 最强 |
| **B** | 双周期确认 (confirmed) | 15min+5min方向一致，1min部分确认(评分>0.5) | 中等 |
| **C** | 部分确认 (partial) | 15min+5min方向一致，1min未确认(评分≤0.5) | 较弱(打7折) |
| **-** | 无信号 | 方向不一致或中性 | 无 |

**信号强度计算：**
- A级: `0.4 × 15min确信度 + 0.35 × 5min信号强度 + 0.25 × 1min入场评分`
- B级: `0.5 × 15min确信度 + 0.5 × 5min信号强度`
- C级: `(0.6 × 15min确信度 + 0.4 × 5min信号强度) × 0.7`

使用示例:
```python
from models.multi_timeframe import MultiTimeframeCoordinator

coordinator = MultiTimeframeCoordinator(
    neutral_zone=0.10,          # 15分钟中性区间
    entry_threshold=0.60,       # 5分钟入场阈值
    optimization_threshold=0.55, # 1分钟优化阈值
    min_grade="C",              # 最低信号等级
)
coordinator.set_models(model_15m, model_5m, model_1m)

result = coordinator.generate_signals(X_15m, X_5m, X_1m)

print(result["final_signal"])     # 最终信号 (+1/-1/0)
print(result["signal_grade"])     # 信号等级 ("A"/"B"/"C"/"-")
print(result["signal_strength"])  # 信号强度 (0~1)
print(result["agreement_score"])  # 三周期一致性 (0~1)

summary = coordinator.get_signal_summary()
print(f"A级信号: {summary['grade_A']}")
print(f"B级信号: {summary['grade_B']}")
```

### `models/adaptive_learning.py` — 实时自适应模块

应对市场结构变化的核心模块，包含以下组件：

**`ConceptDriftDetector` 类 — 概念漂移检测器**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 100 | 滑动窗口大小（最近N个预测样本） |
| `warning_threshold` | 0.05 | 准确率下降触发warning的阈值 |
| `drift_threshold` | 0.10 | 准确率下降触发drift的阈值 |
| `baseline_window` | 500 | 基准性能统计窗口 |

通过滑动窗口持续监控模型预测准确率，与历史基准对比：
- 下降 < 5%: `none`（无漂移，正常运行）
- 下降 5%~10%: `warning`（轻微漂移，建议关注）
- 下降 ≥ 10%: `drift`（严重漂移，建议重训模型）

**`OnlineLearner` 类 — 在线学习器**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `update_interval` | 500 | 每收集多少新样本后触发增量更新 |
| `n_incremental_trees` | 50 | 每次增量更新新增的树数量 |
| `max_buffer_size` | 5000 | 数据缓冲区最大大小 |

对LightGBM/XGBoost执行增量训练（warm-start），在已有模型基础上继续训练少量新树：
- LightGBM: 使用 `init_model` 参数增量训练
- XGBoost: 使用 `xgb_model` 参数增量训练
- LSTM: 不支持在线学习，需完全重训

**`AdaptiveThresholdManager` 类 — 自适应阈值管理器**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_threshold` | 0.55 | 基础交易阈值 |
| `vol_sensitivity` | 1.0 | 波动率灵敏度 |
| `performance_sensitivity` | 0.5 | 性能灵敏度 |
| `min_threshold` / `max_threshold` | 0.50 / 0.85 | 阈值上下限 |

阈值计算公式：
```
threshold = base + vol_adjustment + perf_adjustment
vol_adjustment  = vol_sensitivity × (当前波动率/历史均值 - 1.0) × 0.1
perf_adjustment = perf_sensitivity × max(0, 基准准确率 - 当前准确率) × 0.5
```
- 高波动率 → 提高阈值（更保守，减少假信号）
- 模型性能下降 → 提高阈值（减少风险暴露）

**`AdaptiveModelManager` 类 — 统一管理器**

整合漂移检测 + 在线学习 + 自适应阈值，提供 `step()` 方法执行一步自适应更新。

使用示例:
```python
from models.adaptive_learning import AdaptiveModelManager

manager = AdaptiveModelManager(trained_model)
status = manager.step(
    y_true=actual_labels,
    y_pred=predicted_labels,
    X_new=new_features,
    y_new=new_targets,
    current_volatility=0.02,
    mean_volatility=0.015,
)
print(status["drift_level"])    # "none" / "warning" / "drift"
print(status["threshold"])       # 动态交易阈值
print(status["needs_retrain"])   # 是否需要完全重训
```

### `models/position_sizing.py` — 风险预算与仓位管理模块

将ML信号转化为实际仓位，包含以下组件：

**核心函数：**

| 函数名 | 功能 |
|--------|------|
| `calculate_position_size()` | 改进Kelly公式: `position = (signal × risk_budget) / volatility` |
| `adaptive_threshold()` | 波动率自适应阈值调整 |
| `compute_performance_metrics()` | 计算绩效指标（盈亏比、Calmar、Sharpe、胜率） |

**`DrawdownTracker` 类 — 回撤保护**

实时跟踪账户权益回撤，当回撤超过阈值时自动缩减仓位：

| 回撤水平 | 行为 | 默认阈值 |
|----------|------|----------|
| < warning_level | 正常仓位（乘数=1.0） | < 8% |
| warning_level ~ max_drawdown | 线性缩减（1.0→0.0） | 8%~15% |
| ≥ max_drawdown | 仓位为零（乘数=0.0） | ≥ 15% |

**`DailyRiskLimit` 类 — 日内风险限额**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_daily_loss` | 0.03 | 单日最大亏损比例（3%） |
| `max_daily_trades` | 20 | 单日最大交易次数 |

达到限额后停止开新仓，新交易日自动重置。

**`RiskBudgetManager` 类 — 综合风险管理器（增强版）**

整合Kelly仓位计算 + 回撤保护 + 日内限额 + 绩效监控。

使用示例:
```python
from models.position_sizing import RiskBudgetManager

risk_mgr = RiskBudgetManager(
    account_risk=0.02,      # 单笔风险2%
    max_position=1.0,       # 最大仓位100%
    base_threshold=0.55,    # 基础信号阈值
    max_drawdown=0.15,      # 最大回撤15%
    drawdown_warning=0.08,  # 回撤预警8%
    max_daily_loss=0.03,    # 日内最大亏损3%
    max_daily_trades=20,    # 日内最大交易次数
)

# 生成交易信号和仓位
signals = risk_mgr.compute_signals(probabilities, volatility)
print(signals["position_size"])       # 仓位大小（含回撤缩减）
print(signals["drawdown_multiplier"]) # 回撤缩减乘数

# 记录交易结果
risk_mgr.record_trade_result(pnl=100, trade_date="2024-01-15")

# 查看绩效指标
perf = risk_mgr.get_performance_metrics()
print(f"盈亏比: {perf['profit_factor']:.2f}")
print(f"Calmar: {perf['calmar_ratio']:.2f}")
print(f"Sharpe: {perf['sharpe_ratio']:.2f}")
```

**绩效指标说明：**

| 指标 | 说明 | 优秀阈值 |
|------|------|----------|
| `profit_factor` | 盈亏比（总盈利/总亏损） | > 1.8 |
| `calmar_ratio` | Calmar比率（年化收益/最大回撤） | > 2.0 |
| `sharpe_ratio` | Sharpe比率（风险调整后收益） | > 1.5 |
| `win_rate` | 胜率 | > 55% |
| `max_drawdown` | 最大回撤 | < 15% |

### `backtest/feature_selector.py` — 回测与特征选择模块

时间序列交叉验证 + 8种特征选择方法（含稳定性选择和递归特征消除）。现在评估 **11个特征组**（含深度变换组）。

### `run_real_data_pipeline.py` — 真实数据全自动流水线

全自动6步流水线：数据加载 → 特征工程 → 特征筛选 → 模型训练 → 回测评估 → 预测保存。

| 步骤 | 功能 | 输出 |
|------|------|------|
| Step 1 | 加载真实CSV数据（或回退到模拟数据） | 标准化DataFrame |
| Step 2 | 计算全量118+个特征和预测目标 | X, y |
| Step 3 | ExtraTrees+RF稳定性筛选；15min额外XGBoost预筛选 | `feature_selection_report_{period}.md` |
| Step 4 | 按周期自动选择模型并训练 | 训练好的模型 |
| Step 5 | 含交易成本和仓位约束的回测评估 | `backtest_report_{period}.md` |
| Step 6 | 保存模型、特征列表、标准化器和预测结果 | `models/`, `predictions_{period}.csv` |

### `scripts/load_real_data.py` — 真实数据加载模块

加载白银(ag)等品种的K线CSV数据，自动完成列名映射（`dt` → `datetime`, `close_oi` → `open_interest`）、数据校验和类型转换。

| 函数名 | 功能 |
|--------|------|
| `load_kline_csv(file_path)` | 读取CSV并返回标准化DataFrame |
| `get_default_data_path(period)` | 获取默认数据路径 |
| `load_or_generate(period, n_rows)` | 加载真实数据，文件不存在时回退到模拟数据 |

### `models/cost_model.py` — 交易成本与止盈止损模块

精细化交易成本计算和止损止盈逻辑（改进点 1.1）。

### `models/var_stress.py` — VaR风险度量与压力测试模块

提供三种VaR计算方法（历史法/参数法/蒙特卡洛）和压力测试（改进点 1.3）。

### `hyperopt_runner.py` — 超参搜索模块

Optuna和贝叶斯优化超参搜索（改进点 3.1/3.3），含L2归一化Pipeline（改进点 5.1）。

### `signal_postprocess.py` — 信号后处理模块

截面TopN排名后处理，按日期分组选取最强信号（改进点 2.5）。

### `factor_loader.py` — 因子加载模块

从多个CSV批量加载因子数据并按日期索引对齐（改进点 5.2）。

### `features/feature_context.py` — 特征缓存上下文

使用 `functools.cached_property` 避免重复计算（改进点 5.3）。

### `features/feature_registry.py` — 特征注册表模块

装饰器式的特征自动发现与注册机制。详见 [特征工程可扩展性](#特征工程可扩展性) 章节。

- `@register_feature(group, level, depends_on, output_names)` — 特征注册装饰器
- `FeatureRegistry` — 全局单例注册表，管理所有自定义特征
- 自动依赖检查、输出列名采集、层级/分组映射

### `features/custom_features.py` — 自定义特征（82个注册特征）

使用 `@register_feature` 装饰器注册的自定义特征文件，包含：

**原有示例特征 (2个)**：
- `rsi_14_slope_custom` — RSI(14)的5周期斜率
- `volume_acceleration` — 成交量变化加速度

**依据《hfml特征工程增强报告-56项目挖掘》集成的9个因子**：

| 特征 | 分组 | 层级 | 综合评分 | 说明 |
|------|------|------|----------|------|
| `divergence` | 微观结构 | level4_micro | 22/25 | 资金-价格方向一致性 (sign(ΔOI)×sign(ΔP)) |
| `vwap_dev` | 量价关系 | level4_micro | 21/25 | 价格相对VWAP偏离度 |
| `vol_state` | 波动率 | level5_cross | 20/25 | ATR(14)/60日均价归一化波动状态 |
| `mom_slope` | 动量指标 | level3_momentum | 19/25 | 价格5周期滚动线性回归斜率归一化 |
| `rsi_slope` | 动量指标 | level3_momentum | 19/25 | RSI(14)的5周期滚动斜率 |
| `vol_zscore` | 成交量 | level4_micro | 20/25 | 成交量20周期Z分数标准化 |
| `gap_decay` | 价格形态 | level4_micro | 18/25 | 夜盘缺口指数衰减因子 |
| `oi_price_alignment` | 跨周期结构 | level5_cross | 20/25 | 持仓变化与价格方向一致性 |
| `oi_price_magnitude` | 跨周期结构 | level5_cross | 20/25 | OI变化/价格变化幅度比 |

**依据《hfml特征工程增强报告-精选10特征》集成的10个精选因子**：

| 特征 | 分组 | 层级 | 优先级 | 说明 | Numba加速 |
|------|------|------|--------|------|-----------|
| `buy_sell_pressure` | 微观结构 | level4_micro | P0 | K线位置买卖净压力 [-1,1] | ✅ |
| `volatility_skew` | 波动率 | level5_cross | P0 | 收益率滚动偏度（window=20） | ❌ |
| `momentum_cross` | 动量指标 | level5_cross | P1 | 快慢周期动量差值（fast=5, slow=20） | ❌ |
| `autocorrelation_1` | 时间序列 | level6_transforms | P1 | 1阶自相关系数（window=20） | ✅ |
| `vwap_std` | 成交量 | level5_cross | P2 | VWAP滚动标准差（window=20） | ❌ |
| `tick_imbalance_proxy` | 微观结构 | level4_micro | P2 | 价格变动方向不平衡代理（window=10） | ❌ |
| `volatility_of_volatility` | 波动率 | level5_cross | P3 | 波动率的波动率 | ❌ |
| `trend_strength_ratio` | 趋势指标 | level5_cross | P3 | 短/长期趋势强度比率（tanh归一化） | 依赖 `_rolling_slope` |
| `volume_profile_skew` | 成交量 | level6_transforms | P4 | 成交量分布偏度（window=20, bins=10） | ✅ |
| `hurst_exponent_approx` | 时间序列 | level6_transforms | P4 | Hurst指数近似（min_periods=100） | ✅ |

**依据《hfml特征工程增强报告-精选20特征-第二辑》集成的21个因子**：

| 特征 | 分组 | 层级 | 优先级 | 说明 | Numba加速 |
|------|------|------|--------|------|-----------|
| `parkinson_volatility` | 波动率 | level3_momentum | P0 | Parkinson极差波动率估计 | ✅ |
| `rogers_satchell_vol` | 波动率 | level3_momentum | P0 | Rogers-Satchell波动率（考虑漂移） | ✅ |
| `yang_zhang_vol` | 波动率 | level3_momentum | P1 | Yang-Zhang最优极差波动率 | ✅ |
| `roll_impact` | 流动性 | level4_micro | P1 | Roll冲击成本估计 | ✅ |
| `amihud_illiquidity` | 流动性 | level4_micro | P1 | Amihud非流动性指标 (×1e6) | ✅ |
| `pastor_stambaugh` | 流动性 | level4_micro | P2 | Pastor-Stambaugh反转流动性代理 | ❌ |
| `roll_spread_estimate` | 微观结构 | level4_micro | P2 | Roll有效价差估计 | ✅ |
| `corwin_schultz_spread` | 微观结构 | level4_micro | P2 | Corwin-Schultz高频价差 | ✅ |
| `volume_synchronized_vol` | 成交量 | level5_cross | P1 | 成交量同步波动率 | ✅ |
| `volume_weighted_atr` | 成交量 | level5_cross | P1 | 成交量加权ATR | ❌ |
| `serial_correlation` | 时间序列 | level6_transforms | P2 | 收益率序列相关性强度 | ✅ |
| `partial_autocorrelation` | 时间序列 | level6_transforms | P2 | 一阶偏自相关系数 | ✅ |
| `variance_ratio` | 时间序列 | level6_transforms | P2 | 方差比率（>1趋势,<1均值回归） | ✅ |
| `bid_ask_spread_proxy` | 微观结构 | level4_micro | P0 | 买卖价差代理（相对价差） | ❌ |
| `effective_spread_proxy` | 微观结构 | level4_micro | P1 | 有效价差代理 | ❌ |
| `price_reversal_metric` | 微观结构 | level4_micro | P1 | 价格反转强度 (-autocorr) | ✅ |
| `volume_price_correlation` | 量价关系 | level4_micro | P1 | 量价滚动相关性 | ✅ |
| `open_interest_momentum` | 持仓分析 | level5_cross | P2 | 持仓量快慢动量差值 | ❌ |
| `long_short_ratio_proxy` | 持仓分析 | level5_cross | P2 | 多空比代理 (OI×price×|OI|) | ❌ |
| `kurtosis_returns` | 高阶统计 | level6_transforms | P3 | 收益率峰度（尾部风险） | ✅ |
| `herfindahl_volume` | 成交量 | level4_micro | P3 | 成交量赫芬达尔集中度 | ✅ |

**依据《hfml特征工程增强报告-精选20特征-第三辑》集成的20个因子**：

| 特征名 | 类别 | 层级 | 优先级 | 说明 | Numba |
|--------|------|------|--------|------|-------|
| `fractal_dimension` | 分形分析 | level6_transforms | P1 | Higuchi分形维数（复杂度/趋势强度） | ❌ |
| `lyapunov_exponent` | 分形分析 | level6_transforms | P2 | 李雅普诺夫指数（混沌/可预测性） | ❌ |
| `hurst_exponent_refined` | 分形分析 | level6_transforms | P2 | 改进Hurst指数（基于DFA） | ❌ |
| `detrended_fluctuation` | 分形分析 | level6_transforms | P2 | 去趋势波动分析指数 | ❌ |
| `approximate_entropy` | 信息熵 | level6_transforms | P1 | 近似熵（序列规律性） | ❌ |
| `sample_entropy` | 信息熵 | level6_transforms | P1 | 样本熵（改进近似熵） | ❌ |
| `permutation_entropy` | 信息熵 | level6_transforms | P2 | 排列熵（模式复杂度） | ❌ |
| `skewness_3rd` | 高阶矩 | level6_transforms | P1 | 加权三阶矩偏度 | ❌ |
| `co_skewness` | 高阶矩 | level6_transforms | P2 | 收益率-成交量协偏度 | ❌ |
| `co_kurtosis` | 高阶矩 | level6_transforms | P3 | 收益率-成交量协峰度 | ❌ |
| `market_microstructure_noise` | 微观结构 | level4_micro | P1 | 微观结构噪声 | ❌ |
| `price_delay` | 微观结构 | level4_micro | P2 | 价格延迟/市场效率 | ❌ |
| `volume_synchronized_returns` | 订单流代理 | level4_micro | P1 | 成交量同步收益 | ❌ |
| `tick_rule_imbalance` | 订单流代理 | level4_micro | P1 | Tick规则不平衡 | ❌ |
| `volume_weighted_price_range` | 订单流代理 | level4_micro | P2 | 成交量加权价格区间 | ❌ |
| `volatility_term_structure` | 波动率曲面 | level5_cross | P1 | 波动率期限结构 | ❌ |
| `volatility_convexity` | 波动率曲面 | level5_cross | P2 | 波动率凸性 | ❌ |
| `cross_asset_correlation` | 相关性网络 | level5_cross | P2 | 品种间相关性（无基准时用自相关） | ❌ |
| `correlation_breakdown` | 相关性网络 | level5_cross | P2 | 相关性断裂指标 | ❌ |
| `regime_switching_probability` | 状态转换 | level5_cross | P2 | 高波动状态概率估计 | ❌ |

> **注**: `divergence`~`gap_decay` 这7个特征同时存在于 `feature_engineering_enhanced.py` 中（Numba加速版本），
> 通过 `compute_all_features()` 的去重逻辑，增强模块版本优先使用。注册表版本提供元数据（分组、层级、描述）。
> `oi_price_alignment`、`oi_price_magnitude` 和全部10+21+20+20个精选特征是纯新增特征，仅通过注册表计算。

**依据《hfml特征工程增强报告-精选20特征-第四辑》集成的20个因子**（含公式验证与Bug修复）：

| 特征名 | 类别 | 层级 | 优先级 | 说明 | Bug修复 |
|--------|------|------|--------|------|---------|
| `spectral_ratio` | 谱分析 | level6_transforms | P2 | 高频/低频谱能量比 | 起始索引修复 |
| `dominant_frequency` | 谱分析 | level6_transforms | P2 | 主导频率（最大自相关滞后倒数） | 起始索引修复 |
| `wavelet_energy_s1`~`s5` | 小波变换 | level6_transforms | P2 | 5尺度Haar小波能量 | — |
| `wavelet_entropy` | 小波变换 | level6_transforms | P2 | 多尺度能量分布熵（归一化） | 依赖解析修复 |
| `extreme_value_index` | 极值理论 | level6_transforms | P1 | Hill estimator尾部指数 | 公式修正（Hill→GPD） |
| `tail_dependence` | 极值理论 | level6_transforms | P2 | 尾部条件概率（需基准） | — |
| `copula_dependence` | Copula依赖 | level6_transforms | P2 | Copula相关性（Kendall→Gauss） | — |
| `rank_correlation` | Copula依赖 | level6_transforms | P1 | Spearman秩相关 | — |
| `ml_derived_volatility` | 机器学习衍生 | level6_transforms | P1 | GARCH(1,1)条件波动率 | — |
| `ml_derived_trend` | 机器学习衍生 | level6_transforms | P1 | 标准化趋势斜率 | **前视偏差修复** |
| `order_book_imbalance_proxy` | 微观结构深度 | level4_micro | P1 | 订单簿不平衡代理 | — |
| `depth_pressure` | 微观结构深度 | level4_micro | P1 | 市场深度压力 | — |
| `herding_behavior` | 行为金融 | level6_transforms | P2 | 羊群行为CSAD指标 | **公式修正** |
| `overreaction_score` | 行为金融 | level6_transforms | P2 | 过度反应得分 | **前视偏差修复** |
| `morning/afternoon/night_session` | 日历效应 | level1_price | P1 | 交易时段标识 | — |
| `session_vol_ratio` | 日历效应 | level1_price | P1 | 时段波动率比率 | — |
| `hour/weekday_seasonality` | 日历效应 | level5_cross | P2 | 小时/星期季节性强度 | **to_period崩溃修复** |
| `cumulant_3`, `cumulant_4` | 高阶统计 | level6_transforms | P2 | 三阶/四阶累积量 | — |
| `z_score_of_z_scores` | 高阶统计 | level6_transforms | P1 | 极端异常值检测 | — |
| `network_centrality` | 复杂网络 | level6_transforms | P3 | 特征向量中心度（需多品种） | — |
| `community_strength` | 复杂网络 | level6_transforms | P3 | 板块群落强度（需多品种） | — |

**依据《hfml特征工程增强报告-精选20特征-第五辑》集成的20个因子**（含公式验证与Bug修复）：

| 特征名 | 类别 | 层级 | 优先级 | 说明 | Bug修复 |
|--------|------|------|--------|------|---------|
| `lyapunov_exponent_refined` | 混沌理论 | level6_transforms | P2 | 改进李雅普诺夫指数（相空间重构） | — |
| `correlation_dimension` | 混沌理论 | level6_transforms | P2 | Grassberger-Procaccia关联维数 | — |
| `martingale_difference` | 鞅测度 | level6_transforms | P1 | 非参数条件期望鞅差检验 | — |
| `variance_ratio_test` | 鞅测度 | level6_transforms | P1 | Lo-MacKinlay方差比检验统计量 | — |
| `mutual_information` | 信息论 | level6_transforms | P1 | 直方图法互信息 | **窗口切片对齐修复** |
| `transfer_entropy` | 信息论 | level6_transforms | P2 | Schreiber传递熵 | **实现补齐（纯NumPy）** |
| `conditional_value_at_risk` | 风险测度 | level5_cross | P1 | CVaR尾部损失均值 | — |
| `expected_shortfall` | 风险测度 | level5_cross | P1 | 预期亏损（负收益CVaR） | — |
| `market_microstructure_efficiency` | 市场微观结构 | level4_micro | P2 | 市场效率系数 | **方差去均值修复** |
| `price_discovery_ratio` | 市场微观结构 | level4_micro | P2 | 开盘缺口/日内波幅比 | — |
| `cointegration_residual` | 统计套利 | level6_transforms | P1 | 简化ADF t统计量（close vs EMA） | **实现补齐** |
| `pairs_trading_signal` | 统计套利 | level6_transforms | P1 | Z-score配对交易信号 | **平仓条件修复** |
| `chart_pattern_strength` | 模式识别 | level6_transforms | P2 | 头肩/双顶底/三角形识别 | **起始索引修复** |
| `candlestick_pattern_score` | 模式识别 | level1_price | P2 | K线形态（十字星、吞没、早晚之星） | — |
| `multifractal_spectrum` | 分形市场 | level6_transforms | P2 | 配分函数多重分形谱宽度 | **起始索引修复** |
| `liquidity_adjusted_var` | 风险测度 | level5_cross | P1 | 流动性调整VaR | **流动性冲击退化修复** |
| `volatility_smile_slope` | 波动率微笑 | level5_cross | P2 | 分位数法波动率偏斜 | — |
| `term_structure_curvature` | 期限结构 | level5_cross | P2 | 波动率蝶式价差曲率 | — |
| `bayesian_volatility` | 贝叶斯推断 | level5_cross | P2 | 共轭先验波动率估计 | — |
| `markov_regime_probability` | 马尔可夫场 | level5_cross | P2 | 高波动状态概率 | **概率退化修复** |

---

## 特征工程可扩展性

### 设计目标

**新增一个特征只需 1 处定义、0 处下游修改**：开发者在 `features/custom_features.py`（或任意文件）中用 `@register_feature` 装饰器定义特征计算函数，系统自动完成特征计算、层级归类、分组归类、筛选参与和模型训练。

### 改进前 vs 改进后

| 操作 | 改进前 | 改进后 |
|------|--------|--------|
| 定义特征计算逻辑 | 修改 `feature_engineering.py` | 在任意文件中定义函数 |
| 更新层级结构 | 手动修改 `config.py` 的 `FEATURE_HIERARCHY` | **自动** |
| 更新特征分组 | 手动修改 `feature_selector.py` 的 `get_recommended_feature_groups()` | **自动** |
| 特征筛选/模型训练 | 无需修改 | 无需修改 |
| **总修改文件数** | **3~4个** | **1个** |

### 使用方法：添加新特征

在 `features/custom_features.py` 中添加（或创建新文件后在 `compute_all_features` 中 `importlib.import_module` 导入）：

```python
from features.feature_registry import register_feature
import pandas as pd

@register_feature(
    group="动量指标",                    # 特征分组（对应 feature_selector 的键）
    level="level3_momentum",            # 层级（对应 FEATURE_HIERARCHY 的键）
    description="RSI 14周期的5周期斜率", # 简短描述
    depends_on=["rsi_14"],              # 依赖的已有特征（自动检查）
    output_names=["rsi_14_slope_custom"], # 输出列名
)
def compute_rsi_14_slope(df, features_df=None, **kwargs):
    """计算RSI(14)的5周期斜率。"""
    slope = features_df["rsi_14"].diff(5) / 5.0
    return pd.DataFrame({"rsi_14_slope_custom": slope}, index=df.index)
```

**系统自动处理的事项**：

1. `compute_all_features()` 自动发现并计算该特征
2. `get_feature_hierarchy()` 自动将 `rsi_14_slope_custom` 归入 `level3_momentum`
3. `get_recommended_feature_groups()` 自动将其加入 `动量指标` 分组
4. 特征筛选（ExtraTrees/RF/XGBoost）自动包含该特征
5. 模型训练自动使用该特征（无需修改 `ml_pipeline.py`）

### 注册表 API

| 方法 | 说明 |
|------|------|
| `FeatureRegistry()` | 获取全局单例注册表 |
| `.register(func, group, level, ...)` | 注册一个特征函数 |
| `.get_all_entries()` | 获取所有已注册条目 |
| `.get_group_mapping()` | 获取 `{分组名: [特征名]}` 映射 |
| `.get_hierarchy_mapping()` | 获取 `{层级名: [特征名]}` 映射 |
| `.compute_registered_features(df, features_df)` | 执行所有注册特征的计算 |
| `.clear()` | 清空注册表（仅用于测试） |

### 函数签名规范

所有注册的特征函数必须遵循以下签名：

```python
def compute_xxx(df: pd.DataFrame, features_df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    """
    参数:
        df: 原始OHLCV数据
        features_df: 已计算的特征矩阵（可用来引用 depends_on 中声明的依赖）
    返回:
        pd.DataFrame，列名即为新增特征名
    """
```

---

## 真实数据流水线

### 概述

`run_real_data_pipeline.py` 实现了从真实K线CSV数据到模型落地的**完整6步自动化流程**。若真实数据文件不存在，自动回退到模拟数据运行。

### 数据格式要求

CSV文件需放置在 `data/klines/KQi@SHFEag/` 目录下：

| CSV列名 | hfml标准列名 | 说明 |
|----------|-------------|------|
| `dt` | `datetime`（索引） | 时间戳 |
| `open` | `open` | 开盘价 |
| `high` | `high` | 最高价 |
| `low` | `low` | 最低价 |
| `close` | `close` | 收盘价 |
| `volume` | `volume` | 成交量 |
| `close_oi` | `open_interest` | 收盘持仓量 |

### 15分钟LSTM特殊处理流程

15分钟周期采用 **XGBoost特征预筛选 → LSTM训练** 的双阶段流程：

```
原始数据 (15min OHLCV+仓差)
    ↓
特征工程 (121+个特征)
    ↓
基础筛选: ExtraTrees + RandomForest (稳定性评估)
    ↓
XGBoost 特征预筛选 (5折TimeSeriesSplit CV)
    ↓
动态搜索最佳特征数量 (达到95%最大准确率的最少特征)
    ↓
筛选后特征子集 (25~30个) → LSTM模型训练
    ↓
模型评估 + 保存 (含XGBoost筛选器)
```

查看筛选报告: `data/mx/feature_selection_report_15min.md`

### 使用示例

```bash
# 5分钟周期（默认）— 完整流程
python run_real_data_pipeline.py --period 5min

# 15分钟周期 — 含XGBoost预筛选（保留30个特征）
python run_real_data_pipeline.py --period 15min --xgb-n-features 30

# 15分钟周期 — 自动搜索最佳特征数
python run_real_data_pipeline.py --period 15min --xgb-n-features 0

# 1分钟周期 — 跳过特征筛选（快速测试）
python run_real_data_pipeline.py --period 1min --skip-feature-selection

# 减少模拟数据量（快速验证）
python run_real_data_pipeline.py --period 5min --n-rows 1000
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--period` | str | `5min` | K线周期：`1min`、`5min`、`15min` |
| `--target` | str | `future_direction` | 预测目标 |
| `--n-rows` | int | `5000` | 模拟数据行数（真实数据存在时忽略） |
| `--skip-feature-selection` | flag | — | 跳过特征筛选 |
| `--n-trials` | int | `30` | Optuna超参搜索次数 |
| `--xgb-n-features` | int | `25` | 15min XGBoost预筛选保留特征数（0=自动搜索） |

### 输出文件

运行完成后在 `data/mx/` 目录生成以下文件：

| 文件 | 说明 |
|------|------|
| `feature_selection_report_{period}.md` | 特征筛选报告（含ExtraTrees/RF/XGBoost分析） |
| `backtest_report_{period}.md` | 回测报告（胜率/盈亏比/Sharpe/最大回撤） |
| `predictions_{period}.csv` | 测试集预测结果（actual vs predicted） |
| `models/{period}_{model}_{date}.pkl` | 已训练模型 |
| `models/{period}_{model}_{date}_features.txt` | 特征名称列表 |
| `models/{period}_{model}_{date}_scaler.pkl` | 标准化器（LSTM） |

### 示例输出

```bash
python run_real_data_pipeline.py --period 5min --n-rows 2000
```

```
======================================================================
  商品期货ML量化模型 — 真实数据全自动流水线
  周期: 5min  目标: future_direction  框架: xgboost
======================================================================

Step 1: 数据加载与预处理
  数据类型: 模拟数据
  数据形状: (2000, 6)

Step 2: 特征工程全量计算
  特征数量: 121, 样本数量: 1901

Step 3: 特征筛选与稳定性评估
  ExtraTrees Top 10: ...
  RandomForest Top 10: ...

Step 4: 模型训练
  使用 LightGBM+XGBoost 集成模型...
  测试集指标: accuracy=0.5524, f1=0.6322

Step 5: 回测与绩效评估
  回测报告已保存: data/mx/backtest_report_5min.md

Step 6: 模型保存与预测输出
  模型已保存: data/mx/models/5min_xgboost_20260212.pkl

======================================================================
  关键绩效摘要
  特征数量: 40 / 原始 121
  accuracy: 0.5524
  win_rate: 0.5588
======================================================================
```

---

## 各周期机器学习模型选择

### 单模型推荐

| 周期 | 推荐模型 | 选择理由 | 数据特点 |
|------|----------|----------|----------|
| **1分钟** | **LightGBM** | 数据量大（每天约240根K线），需要极快的训练速度 | 数据量大、噪声多、信号弱 |
| **5分钟** | **XGBoost** | 数据量适中（每天约48根K线），平衡精度与效率 | 数据量适中、信噪比较好 |
| **15分钟** | **LSTM** | 数据量较少（每天约16根K线），时序依赖更明显 | 数据量少、时序特征强 |

### 集成模型（1分钟/5分钟默认启用）

1分钟和5分钟周期默认使用 **LightGBM + XGBoost 加权集成模型**，15分钟使用单独的LSTM。

---

## 15分钟LSTM特征预筛选

### 为什么需要特征预筛选？

LSTM模型无法直接评估特征重要性（`get_feature_importance()` 返回 None），且LSTM对噪声特征敏感。系统原始特征数为84个，直接输入LSTM会导致：
- **过拟合风险**：15分钟数据量较少（每天约16根K线），过多特征加剧过拟合
- **训练效率低**：特征越多，LSTM训练时间越长
- **可解释性差**：无法知道哪些特征对预测有贡献

### 解决方案：XGBoost作为特征筛选器

使用XGBoost作为代理模型，通过TimeSeriesSplit交叉验证评估每个特征的重要性，筛选出Top N个特征再输入LSTM训练。

**完整流程：**

```
原始数据 (15分钟OHLCV+仓差)
    ↓
特征工程 (生成84个特征)
    ↓
XGBoost特征筛选 (5折TimeSeriesSplit CV)
    ↓
特征重要性排名 + 特征组分析
    ↓
动态搜索最佳特征数量 (10-40个)
    ↓
筛选Top N特征 (默认25个)
    ↓
LSTM模型训练 (使用筛选后的特征)
    ↓
对比实验: 全量特征 vs 筛选特征
```

### 运行特征预筛选

```bash
# 使用默认配置（25个特征）
python ml_pipeline.py --feature-selection

# 指定特征数量
python ml_pipeline.py --feature-selection --n-features 20

# 指定数据量
python ml_pipeline.py --feature-selection --n-rows 3000
```

### 配置参数

在 `config.py` 中的 `LSTM_FEATURE_SELECTION_CONFIG`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `xgb_params.n_estimators` | 200 | XGBoost筛选器的树数量 |
| `xgb_params.max_depth` | 5 | XGBoost筛选器的树深度 |
| `selection.n_features` | 25 | 最终选择的特征数量 |
| `selection.threshold` | `"median"` | 重要性阈值方法 |
| `selection.cv_splits` | 5 | 时间序列交叉验证折数 |

### 输出示例

```
============================================================
15分钟LSTM模型训练 — XGBoost特征预筛选
============================================================

Step 1: 生成15分钟模拟数据...
原始特征数量: 84, 样本数量: 1902

Step 2: XGBoost特征筛选 (目标: 25个特征)...

Step 3: 特征重要性排名 (Top 15):
  oi_ma_20          importance=0.0169  std=0.0019
  ema_40            importance=0.0167  std=0.0034
  vwap              importance=0.0158  std=0.0028
  regime_volatility importance=0.0153  std=0.0075
  atr_pct           importance=0.0151  std=0.0022
  ...

Step 3b: 特征组重要性分析:
  布林带       score=0.0134  入选=3/5
  价格趋势     score=0.0131  入选=5/9
  持仓量(仓差)  score=0.0127  入选=2/9
  动量指标     score=0.0126  入选=4/12
  ...

Step 4: 动态特征数量搜索...
  建议最佳特征数量: 10
    n= 10  accuracy=0.4905 ←
    n= 15  accuracy=0.5004
    n= 20  accuracy=0.5011
    n= 25  accuracy=0.4765
    ...

Step 5: 使用筛选后的 25 个特征训练LSTM...
Step 6: 对比实验 — 全量特征 vs 筛选特征
  全量特征 (84个): 准确率 = 0.48xx
  筛选特征 (25个): 准确率 = 0.50xx
  特征减少: 84→25 (-70%)
```

### 筛选策略说明

| 筛选准则 | 说明 |
|----------|------|
| **重要性排名** | 按5折CV平均XGBoost特征重要性降序排列 |
| **稳定性** | 标准差(std)反映特征在不同折中的稳定性，std越小越稳定 |
| **多样性** | 特征组分析确保每个重要组都有代表入选 |
| **动态数量** | 搜索达到95%最大准确率时所需的最少特征数 |

### 预期效果

- **训练时间减少**：30-50%（特征减少约70%）
- **过拟合降低**：去除噪声特征后，验证集/测试集表现差距缩小
- **模型稳定性提高**：不同时间段表现更一致

---

## 深度特征变换

### 设计理念

> "真正的Alpha不是来自于更多数据，而是来自于更深刻的理解和数据关系挖掘。"

在现有数据边界（OHLCV + 持仓量/仓差）内，通过对已有特征进行深度加工来最大化信息提取，不新增数据源，只提升信息密度。

### 五类深度变换

| 变换类别 | 说明 | 代表特征 | 交易价值 |
|----------|------|----------|----------|
| **非线性变换** | 对关键指标做平方/对数/排名/Z-score | `rsi_14_squared`, `macd_hist_zscore` | 捕捉极端值，消除异方差 |
| **跨周期比率** | 用不同窗口模拟多周期对比 | `rsi_fast_slow_ratio`, `volatility_ratio_fast_slow` | 短期vs长期相对强弱 |
| **特征变化率** | 指标的一阶/二阶导数 | `rsi_14_velocity`, `macd_hist_acceleration` | 趋势反转提前预警 |
| **条件特征** | 基于市场状态的动态特征 | `rsi_high_vol`, `momentum_in_trend` | 同一指标在不同环境下的差异化表达 |
| **特征交互** | 指标与量/仓的乘法交互 | `rsi_volume_interaction`, `oi_price_alignment` | 交叉验证信号可信度 |

### 变换流程

```
原始84个特征 → 非线性变换(16个) + 跨周期比率(3个) + 变化率(9个)
             + 条件特征(4个) + 交互项(4个) = 34个新特征
             → 总计 ~118个特征
```

---

## 智能标签系统

### 从二元分类到五级信号

传统方法使用简单的 `future_direction` (0=跌, 1=涨) 作为预测目标，存在以下问题：
- 忽略了价格变化的**幅度**信息
- 所有信号同等对待，不区分**信号质量**
- 未考虑**持仓量和成交量**的确认作用

智能标签系统通过五级信号和质量评分解决这些问题。

### 运行智能标签

```bash
# 使用默认配置
python ml_pipeline.py --smart-labels

# 指定周期和数据量
python ml_pipeline.py --smart-labels --period 5min --n-rows 5000
```

### 输出示例

```
============================================================
智能标签生成系统
周期: 5min
============================================================

标签统计 (共5000个有效样本):
  strong_sell (强势看空): 352 (7.0%)
  weak_sell (弱看空): 1505 (30.1%)
  neutral (中性/不交易): 1190 (23.8%)
  weak_buy (弱看多): 1611 (32.2%)
  strong_buy (强势看多): 342 (6.8%)

信号质量统计:
  平均质量: 0.666
  高质量信号 (>0.6): 3272
```

---

## 混合智能交易系统

### 设计理念

> "70%胜率不是来自于更复杂的模型，而是来自于更聪明的特征工程、更精细的信号定义、更严格的风险控制。"

不依赖单一ML模型，而是让多个具有不同"交易员思维"的专家各司其职，通过加权投票决定最终信号。

### 运行混合系统

```bash
# 使用默认配置
python ml_pipeline.py --hybrid

# 指定周期和数据量
python ml_pipeline.py --hybrid --period 5min --n-rows 5000
```

### 输出示例

```
============================================================
混合智能交易系统
周期: 5min
============================================================

单点预测 (最新K线):
  最终信号: -1 (1=做多, -1=做空, 0=观望)
  建议仓位: 0.0245
  综合置信度: 0.1519
  融合信号: -0.5225
  专家一致性: 50.00%

各专家信号明细:
  趋势跟踪: 信号=+1.0, 置信度=0.003
  均值回归: 信号=+0.0, 置信度=0.363
  突破交易: 信号=+0.0, 置信度=0.000
  量价关系: 信号=-1.0, 置信度=0.796
  持仓确认: 信号=+0.0, 置信度=0.357

批量预测 (窗口=100):
  总样本: 4900
  做多信号: 1294 (26.4%)
  做空信号: 1256 (25.6%)
  观望: 2350 (48.0%)
  平均置信度: 0.1374
  平均一致性: 75.24%
```

---

## 特征列表

### 分层特征结构 (Feature Hierarchy)

系统采用 **6层分层特征结构**，从基础价格到高阶深度变换逐层构建。这种层级设计的优势是：
- **低层特征**（Level 1-2）计算快、解释性强，可用于基础过滤
- **中层特征**（Level 3）技术指标振荡器，覆盖动量和波动率信号
- **高层特征**（Level 4-5）信息密度大，适合捕捉复杂市场模式
- **深度变换**（Level 6）通过非线性/跨周期/交互变换最大化信息提取
- 模型可以按层级选择性地加载特征，灵活控制复杂度

```
优化前（平面特征列表）：
  features = ['ma_5', 'rsi_14', 'vol_ma_5', 'divergence', ...]

优化后（分层特征结构）：
  FEATURE_HIERARCHY = {
      'level1_price':    基础价格特征 — OHLCV直接衍生 (13个)
      'level2_trend':    趋势特征 — 均线与布林带 (16个)
      'level3_momentum': 动量与波动率特征 — 技术指标振荡器 (30个)
      'level4_micro':    微观结构特征 — 资金流向与持仓分析 (16个)
      'level5_cross':    跨周期与高级组合特征 (9个)
      'level6_transforms': 深度变换特征 — 非线性/跨周期/变化率/交互 (34个)
  }
```

#### Level 1: 基础价格特征（13个）— OHLCV直接衍生

| 特征名 | 说明 |
|--------|------|
| `close_pos` | K线内相对位置 |
| `body_ratio` / `upper_shadow_ratio` / `lower_shadow_ratio` | 实体比和影线比 |
| `candle_direction` / `amplitude` | 涨跌方向和振幅 |
| `gap` / `gap_ratio` | 跳空缺口 |
| `price_position` / `dist_to_high` / `dist_to_low` | 价格在区间中的位置 |
| `log_return` | 对数收益率 |
| `range_pct` | 振幅百分比 |

#### Level 2: 趋势特征（16个）— 均线与布林带

| 特征名 | 说明 |
|--------|------|
| `ma_5`, `ma_10`, `ma_20`, `ma_60` | 简单移动平均线 |
| `ema_5`, `ema_10`, `ema_20`, `ema_60` | 指数移动平均线 |
| `boll_upper` / `boll_mid` / `boll_lower` / `boll_width` / `boll_pct_b` | 布林带系列 |
| `trend_strength` / `trend_direction` / `trend_acceleration` | 趋势强度、方向、加速度 |

#### Level 3: 动量与波动率特征（30个）— 技术指标振荡器

| 特征名 | 说明 |
|--------|------|
| `rsi_14` / `rsi_6` | RSI |
| `macd_dif` / `macd_dea` / `macd_hist` | MACD |
| `kdj_k` / `kdj_d` / `kdj_j` | KDJ |
| `cci` / `williams_r` / `roc_12` / `roc_6` | CCI / 威廉 / 变动率 |
| `obv` / `vwap` | 能量潮 / 加权均价 |
| `tr` / `atr` / `atr_pct` | 真实波幅 |
| `volatility_*` / `return_ma_*` | 滚动波动率和收益率均值 |
| `vol_ma_*` / `vol_ratio_*` / `vol_change` | 成交量指标 |

#### Level 4: 微观结构特征（16个）— 资金流向与持仓分析

| 特征名 | 计算公式 | 说明 |
|--------|----------|------|
| `divergence` | `sign(持仓变化) × sign(价格变化)` | 资金流向与价格背离 |
| `vwap_dev` | `(close - VWAP) / VWAP` | 价格与VWAP偏离 |
| `vol_state` | `ATR14 / 60日均价` | 波动率状态（标准化） |
| `mom_slope` | `5周期价格斜率 / 当前价格` | 动量加速度 |
| `rsi_slope` | RSI的5周期线性回归斜率 | RSI变化速率 |
| `vol_zscore` | `(volume - vol_ma20) / vol_std20` | 成交量Z分数 |
| `gap_decay` | `gap × exp(-秒数/300)` | 夜盘缺口衰减 |
| `oi_change` / `oi_change_pct` | — | 仓差和仓差变化率 |
| `oi_ma_*` / `oi_change_ma_*` | — | 持仓量和仓差移动平均 |
| `vol_oi_ratio` | `volume / open_interest` | 量仓比 |

#### Level 5: 跨周期与高级组合特征（9个）— 市场状态

| 特征名 | 说明 |
|--------|------|
| `regime_volatility` | 滚动波动率 |
| `volatility_rank` / `volatility_change` | 波动率排名和变化率 |
| `market_regime` | 市场状态编码（0~4，含反转状态） |
| `volatility_regime` | 收益率的20周期滚动标准差 |
| `vol_ratio` | ATR14 / ATR14的20周期均值 |
| `reversal_score` | 反转信号强度（0~1，综合RSI极端值+均线偏离+趋势减弱） |
| `regime_duration` | 当前市场状态已持续的K线数量 |
| `regime_change_prob` | 滚动窗口内市场状态转换概率 |

#### Level 6: 深度变换特征（~34个）— 非线性/跨周期/变化率/交互

| 特征名 | 说明 |
|--------|------|
| `rsi_14_squared`, `macd_hist_squared`, ... | 平方项（放大极端值信号） |
| `rsi_14_log`, `macd_hist_log`, ... | 对数变换（降低异方差） |
| `rsi_14_rank`, `macd_hist_rank`, ... | 滚动百分位排名（相对位置） |
| `rsi_14_zscore`, `macd_hist_zscore`, ... | Z-score标准化（偏离度） |
| `rsi_fast_slow_ratio` | 快速RSI(6) / 慢速RSI(24) 比率 |
| `volatility_ratio_fast_slow` | 短期波动率(5) / 长期波动率(20) 比率 |
| `ma_cross_ratio` | 快速MA(5) / 慢速MA(20) 比率 |
| `rsi_14_velocity`, `macd_hist_velocity`, ... | 一阶导数（3周期变化率） |
| `rsi_14_acceleration`, `macd_hist_acceleration`, ... | 二阶导数（加速度） |
| `*_direction_consistency` | 5周期内变化方向一致性 |
| `rsi_high_vol` / `rsi_low_vol` | 高/低波动环境下的RSI |
| `momentum_in_trend` / `momentum_in_range` | 趋势/震荡环境下的动量 |
| `rsi_volume_interaction` | RSI × 量比（超买超卖+放量=强信号） |
| `momentum_vol_interaction` | MACD柱 / ATR（波动率标准化动量） |
| `oi_price_alignment` | 持仓变化×价格变化方向（+1=同向=延续, -1=反向=反转） |
| `oi_price_magnitude` | 持仓变化幅度 / 价格变化幅度 |

### 各周期窗口参数差异

| 参数 | 1分钟 | 5分钟 | 15分钟 |
|------|-------|-------|--------|
| MA/EMA窗口 | 5, 10, 20, 60, 120 | 5, 10, 20, 60 | 5, 10, 20, 40 |
| 成交量窗口 | 5, 10, 20, 60 | 5, 10, 20 | 5, 10, 20 |
| 总特征数 | ~126 | ~118 | ~118 |

---

## 预测目标

| 目标名 | 类型 | 说明 | 预测步长 (1min/5min/15min) |
|--------|------|------|--------------------------|
| `future_direction` | 二分类 (0/1) | 未来价格方向 **推荐** | 5 / 3 / 2 |
| `future_regime` | 五分类 (0~4) | 未来价格五分位状态 | 5 / 3 / 2 |
| `future_return` | 回归 | 未来N根K线的收益率 | 5 / 3 / 2 |
| `future_volatility` | 回归 | 未来N根K线的波动率 | 5 / 3 / 2 |

---

## 特征评估报告

### 评估方法

系统通过以下方法评估特征的预测能力：

1. **时间序列交叉验证 (TimeSeriesSplit)** — 5折，避免前视偏差
2. **特征重要性排序** — 基于模型 `feature_importances_`
3. **互信息排序** — 基于 `mutual_info_classif`，发现非线性关联
4. **前向特征选择** — 逐步添加最优特征
5. **稳定性选择** — LassoCV多次采样，统计特征选中频率
6. **递归特征消除 (RFE)** — 逐步移除最不重要特征
7. **特征组评估** — 将全部特征分为 **11个组**（含深度变换组），分别评估成功率
8. **综合最佳特征选择** — 综合重要性 + 互信息 + CV验证

### 11个特征组

| 特征组 | 特征数 | 说明 |
|--------|--------|------|
| 价格趋势 | 11 | MA/EMA/价格位置 |
| 布林带 | 5 | 布林带系列 |
| 动量指标 | 12 | RSI/MACD/KDJ等 |
| 成交量 | 9 | 量比/OBV/VWAP |
| 波动率 | 9 | ATR/波动率标准差 |
| 持仓量(仓差) | 9 | 仓差/量仓比 |
| K线形态 | 7 | 实体比/影线/缺口 |
| 市场状态 | 10 | 趋势/波动率/反转/持续时间/转换概率 |
| **微观结构** | **8** | **divergence/vwap_dev/vol_state/mom_slope/close_pos/rsi_slope/vol_zscore/gap_decay** |
| **高级波动率** | **4** | **volatility_regime/vol_ratio/atr_pct/range_pct** |
| **深度变换** | **~24** | **非线性/跨周期/变化率/条件/交互变换特征** |

---

## Numba性能加速

### `features/rolling_numba.py` 模块

使用 Numba JIT编译 加速核心数值计算，比pandas原生函数快 **10-50倍**。

**性能对比（1分钟高频数据，100,000根K线）：**

| 计算函数 | pandas原生 | Numba加速 | 加速倍率 |
|----------|-----------|-----------|----------|
| `rolling_mean` | ~50ms | ~1ms | 50x |
| `rolling_std` | ~80ms | ~2ms | 40x |
| `rsi` | ~120ms | ~3ms | 40x |
| `macd_hist` | ~200ms | ~5ms | 40x |

**特点：**
- 首次调用有JIT编译开销（~1-2秒），后续调用接近C语言速度
- 设置 `cache=True`，编译结果缓存到磁盘，程序重启无需重新编译
- numba未安装时自动降级为纯numpy实现，不影响功能正确性
- 增强特征模块 `feature_engineering_enhanced.py` 直接使用numba加速函数

---

## 安装与运行

### 环境要求

- Python >= 3.8

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `lightgbm >= 3.3.0`
- `xgboost >= 1.6.0`
- `tensorflow >= 2.18.0`（仅15分钟LSTM模型需要）
- `numba >= 0.56.0`（Numba加速，可选但强烈推荐）
- `optuna >= 3.0.0`（超参搜索）
- `bayesian-optimization >= 1.4.0`（贝叶斯优化）
- `scipy >= 1.7.0`（VaR计算）
- `joblib >= 1.1.0`（模型持久化）

### 运行真实数据流水线（推荐）

```bash
# 将白银(ag)K线CSV放入 data/klines/KQi@SHFEag/ 目录
# 若无真实数据，自动回退到模拟数据

# 5分钟周期（默认）
python run_real_data_pipeline.py --period 5min

# 15分钟周期（含XGBoost预筛选→LSTM）
python run_real_data_pipeline.py --period 15min --xgb-n-features 30

# 1分钟周期（跳过特征筛选快速测试）
python run_real_data_pipeline.py --period 1min --skip-feature-selection
```

### 运行主流程（模拟数据）

```bash
# 5分钟周期，预测涨跌方向（默认配置，使用集成模型）
python ml_pipeline.py

# 指定1分钟周期
python ml_pipeline.py --period 1min --target future_direction

# 指定5分钟周期
python ml_pipeline.py --period 5min --target future_direction

# 指定15分钟周期（使用LSTM模型）
python ml_pipeline.py --period 15min --target future_direction

# 五分位预测
python ml_pipeline.py --period 5min --target future_regime

# 回归任务
python ml_pipeline.py --period 5min --target future_return

# 指定数据量
python ml_pipeline.py --period 5min --n-rows 10000

# 跳过特征选择（仅训练和评估）
python ml_pipeline.py --period 5min --skip-feature-selection

# 多时间框架协同交易系统（15min→5min→1min层级协同）
python ml_pipeline.py --multi-timeframe

# 多时间框架 + 指定数据量
python ml_pipeline.py --multi-timeframe --n-rows 2000

# 智能标签系统（五级信号+质量评分）
python ml_pipeline.py --smart-labels

# 智能标签 + 指定周期
python ml_pipeline.py --smart-labels --period 5min --n-rows 5000

# 混合智能交易系统（五专家加权投票）
python ml_pipeline.py --hybrid

# 混合系统 + 指定数据量
python ml_pipeline.py --hybrid --period 5min --n-rows 5000

# 15分钟LSTM特征预筛选（XGBoost筛选→LSTM训练对比）
python ml_pipeline.py --feature-selection

# 特征预筛选 + 指定特征数量
python ml_pipeline.py --feature-selection --n-features 20
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--period` | str | `5min` | K线周期：`1min`、`5min`、`15min` |
| `--target` | str | `future_direction` | 预测目标：`future_direction`、`future_return`、`future_regime` |
| `--n-rows` | int | `5000` | 模拟数据行数 |
| `--skip-feature-selection` | flag | — | 跳过特征选择步骤 |
| `--multi-timeframe` | flag | — | 运行多时间框架协同交易系统（自动训练三个周期模型并协同生成信号） |
| `--smart-labels` | flag | — | 运行智能标签生成系统（五级信号+质量评分） |
| `--hybrid` | flag | — | 运行混合智能交易系统（五专家加权投票） |
| `--feature-selection` | flag | — | 运行15分钟LSTM特征预筛选流程（XGBoost筛选→LSTM训练对比） |
| `--n-features` | int | `25` | 特征预筛选：选择的特征数量（配合 `--feature-selection` 使用） |

---

## 运行示例与结果解读

### 示例：5分钟周期 + 集成模型 + 涨跌预测

```bash
python ml_pipeline.py --period 5min --target future_direction --n-rows 2000
```

**输出解读：**

```
============================================================
商品期货机器学习量化模型
周期: 5min, 目标: future_direction, 任务: classification
推荐ML框架: xgboost
============================================================

# 步骤1: 数据准备（分层特征结构）
特征数量: 118, 样本数量: 4854
  level1_price (基础价格特征 — OHLCV直接衍生): 13个特征
  level2_trend (趋势特征 — 均线与布林带): 16个特征
  level3_momentum (动量与波动率特征 — 技术指标振荡器): 30个特征
  level4_micro (微观结构特征 — 资金流向与持仓分析): 16个特征
  level5_cross (跨周期与高级组合特征 — 市场状态与波动率状态): 9个特征
  level6_transforms (深度变换特征 — 非线性/跨周期/变化率/交互): 34个特征

# 步骤2: 模型训练（集成模型）
使用 LightGBM+XGBoost 集成模型...
集成模型权重: {'lightgbm': 0.504, 'xgboost': 0.496}

# 步骤3: 特征重要性 Top 10
vwap_dev             0.020     ← 新增微观结构特征入选Top 10

# 步骤4: 仓位管理（增强版 — 含回撤保护和绩效指标）
仓位管理: 测试集产生 278 个交易信号
  回撤保护乘数: 1.00
  模拟绩效: 胜率=53.96%, 盈亏比=1.17, Sharpe=1.26

# 步骤4.5: 自适应模块（漂移检测 + 动态阈值）
自适应模块: 漂移=none, 阈值=0.554, 需要重训=False

# 步骤5: 特征组评估（10个组，含新增2组）
特征组评估结果:
     group  n_features  accuracy
   高级波动率           5     0.527   ← 新增组排名第1！
      布林带           5     0.516
  持仓量(仓差)          9     0.511
      成交量           9     0.510
      波动率           9     0.509
     微观结构           7     0.508   ← 新增组排名第6
     K线形态           7     0.497
     价格趋势         11     0.492
     市场状态           7     0.485
     动量指标         12     0.480

# 步骤6: 综合特征选择
最佳特征: [..., 'vwap_dev', 'vol_zscore', ...]   ← 新增特征入选
最终成功率: 0.5228
============================================================
流程完成!
```

### 关键结果

1. **高级波动率组排名第1**（0.527）— 新增的 `volatility_regime`, `vol_ratio`, `atr_pct`, `range_pct` 特征预测能力最强
2. **微观结构特征入选最佳特征集** — `vwap_dev` 和 `vol_zscore` 被综合特征选择选入Top 20
3. **特征总数提升** — 从62-70个增至118+个（含10个市场状态特征 + 12个增强特征 + 34个深度变换特征）
4. **Numba加速** — 增强特征模块使用numba加速的rolling计算，适合高频数据场景
5. **回撤保护** — 自动追踪账户回撤，达到预警水平(8%)后线性缩减仓位，达到最大回撤(15%)后停止开仓
6. **自适应阈值** — 基于波动率和模型性能动态调整交易阈值，高波动率/低性能时自动提高阈值
7. **概念漂移检测** — 实时监控模型性能，当准确率下降超过10%时触发重训建议
8. **多时间框架协同** — 通过15min→5min→1min层级决策链，产生A/B/C等级交易信号，A级(三周期共振)信号最强
9. **LSTM特征预筛选** — XGBoost代理筛选器将118个特征筛选至25个，特征减少约80%，降低LSTM过拟合风险
10. **智能标签系统** — 五级信号（-2~+2）替代简单二元分类，综合价格/持仓/成交量生成信号质量评分
11. **混合专家系统** — 五个交易专家（趋势/回归/突破/量价/持仓）加权投票，48%观望过滤低质量信号

### 示例：多时间框架协同交易

```bash
python ml_pipeline.py --multi-timeframe --n-rows 1000
```

**输出解读：**

```
============================================================
多时间框架协同交易系统
============================================================

# 步骤1: 三个周期独立训练
训练15分钟模型（趋势方向）...
  15分钟模型测试集: {'accuracy': 0.468, ...}

训练5分钟模型（入场时机）...
  集成模型权重: {'lightgbm': 0.496, 'xgboost': 0.504}

训练1分钟模型（入场价格）...
  集成模型权重: {'lightgbm': 0.498, 'xgboost': 0.502}

# 步骤2: 创建协同系统并生成信号
创建多时间框架协同系统...
协同信号生成: 139 个样本

# 步骤3: 协同信号统计
多时间框架协同信号统计:
  总样本数: 139
  做多信号: 0
  做空信号: 97
  中性(无信号): 42             ← 42个样本被15分钟中性过滤
  A级信号(三周期共振): 9       ← 最强信号，三个周期完全一致
  B级信号(双周期确认): 5       ← 15分钟+5分钟一致，1分钟部分确认
  C级信号(部分确认): 83        ← 15分钟+5分钟一致，1分钟未确认
  平均信号强度: 0.556
  平均一致性评分: 0.698
============================================================
多时间框架协同交易系统完成!
```

**协同信号解读：**
- **42个中性(无信号)**: 15分钟模型判定方向不明确，自动过滤（核心保护机制）
- **9个A级信号**: 三个周期（15min/5min/1min）方向完全一致的最强信号
- **5个B级信号**: 15分钟和5分钟方向一致，1分钟部分确认（入场评分>0.5）
- **83个C级信号**: 15分钟和5分钟方向一致，但1分钟未确认（强度打7折）
- **信号强度0.556**: 综合三个周期的加权信号强度
- **一致性0.698**: 三个周期之间的方向一致程度

> **注意**：当前使用模拟随机数据，实际商品期货数据的预测效果会因市场行情不同而有差异。接入真实行情数据后需要重新训练和评估。

---

## 7天执行清单（每天命令 + 验收指标）

适用目标：先跑通，再稳步提精度，最后做实盘前风控验证。  
执行原则：每天只改一类变量（特征/阈值/模型其一），避免混改导致无法归因。

### Day 1 — 跑通全链路并建立基线（5min）

**目标**：拿到第一版可复现基线。  
**命令**：

```bash
python run_real_data_pipeline.py --period 5min
```

**验收**：
- 生成 `data/mx/feature_selection_report_5min.md`
- 生成 `data/mx/backtest_report_5min.md`
- 生成 `data/mx/predictions_5min.csv`
- 记录基线指标：`accuracy`、`win_rate`、`profit_factor`、`sharpe_ratio`

---

### Day 2 — 数据质量与目标分布检查

**目标**：确认样本分布可训练，避免“伪效果”。  
**命令**（快速复跑并观察日志中的目标分布）：

```bash
python run_real_data_pipeline.py --period 5min --skip-feature-selection
```

**验收**：
- `future_direction` 正负样本不过度失衡（经验上不建议超过 70:30）
- 缺失值、无穷值处理后样本量无异常骤降
- 将 Day1/Day2 指标写入同一对照表（CSV 或 Markdown）

---

### Day 3 — 特征筛选稳定化（5min）

**目标**：减少噪声特征，提升稳健性。  
**命令**：

```bash
python run_real_data_pipeline.py --period 5min
```

**验收**：
- 对比 `feature_selection_report_5min.md`，确认 Top 特征与特征组排名
- 重点关注稳定特征（非一次性“冲高”特征）
- 若准确率变化不大但回测稳定性提升，也记为正向结果

---

### Day 4 — 15min特征预筛选 + LSTM收敛验证

**目标**：验证中周期模型是否提供有效方向信息。  
**命令**：

```bash
python run_real_data_pipeline.py --period 15min --xgb-n-features 0
```

**验收**：
- 自动得到动态建议特征数（`xgb-n-features 0`）
- 15min 报告中看到筛选后特征列表与组别贡献
- 记录 15min 测试集 `accuracy/f1_score`，作为后续多周期方向过滤基线

---

### Day 5 — 小规模超参搜索（先5min）

**目标**：在不大幅加复杂度前提下提升指标。  
**命令**：

```bash
python run_real_data_pipeline.py --period 5min --n-trials 20
```

**验收**：
- 与 Day1 基线比较：至少一个核心指标稳定提升（如 `accuracy` 或 `profit_factor`）
- 若指标互相冲突（准确率升、收益降），优先看“净收益相关指标”
- 保存本轮参数与结果（避免遗忘最优配置）

---

### Day 6 — 多时间框架协同验证（15m→5m→1m）

**目标**：提高信号质量，减少低质量开仓。  
**命令**：

```bash
python ml_pipeline.py --multi-timeframe --n-rows 1000
```

**验收**：
- 输出 A/B/C 等级信号统计
- A级信号样本虽少但质量应更高（重点看胜率和盈亏比）
- 中性过滤机制有效（减少无把握交易）

---

### Day 7 — 风控联调与实盘前门控检查

**目标**：将“可预测”转为“可交易”。  
**命令**：

```bash
python run_real_data_pipeline.py --period 5min
python run_real_data_pipeline.py --period 15min --xgb-n-features 25
```

**验收**：
- 检查回撤保护、日内限额、阈值自适应是否触发并生效
- 检查成本扣减后策略是否仍为正收益倾向
- 检查 VaR 门控是否在高风险样本自动降仓

---

## 每天都要更新的最小结果表（建议）

建议维护一张 `data/mx/experiment_log.csv`，字段如下：

- `date`
- `period`
- `features_used`
- `key_params`
- `accuracy`
- `f1_score`
- `win_rate`
- `profit_factor`
- `max_drawdown`
- `sharpe_ratio`
- `notes`

只要持续7天有结构化记录，后续优化效率会明显高于“凭感觉调参”。

### 自动写入实验日志（推荐）

跑完回测后可直接执行：

```bash
python scripts/update_experiment_log.py --period 5min --features-used all_features --key-params "n_trials=20; feature_selection=on" --notes "day5 tuning"
```

说明：
- 默认读取 `data/mx/backtest_report_{period}.md`
- 自动提取 `accuracy/f1_score/win_rate/profit_factor/max_drawdown/sharpe_ratio`
- 自动写入（同日期+同周期会更新，不重复插入）`data/mx/experiment_log.csv`
- 可加 `--run-pipeline` 先执行流水线再写入日志

一键（先跑再记）单周期：

```bash
python scripts/update_experiment_log.py --period 5min --run-pipeline --pipeline-extra-args "--n-trials 20" --features-used all_features --key-params "n_trials=20" --notes "daily auto run"
```

批量写入三个周期：

```bash
python scripts/update_experiment_log.py --periods 1min,5min,15min --features-used all_features --key-params "daily_batch" --notes "batch update"
```

一键（先跑再记）批量三周期：

```bash
python scripts/update_experiment_log.py --periods 1min,5min,15min --run-pipeline --features-used all_features --key-params "daily_batch" --notes "batch auto run"
```

严格模式（任一周期失败立即退出，适合CI/计划任务）：

```bash
python scripts/update_experiment_log.py --periods 1min,5min,15min --run-pipeline --strict --features-used all_features --key-params "daily_batch" --notes "batch strict"
```

### Windows 任务计划器（PowerShell日更脚本）

手动执行（先跑流水线、再更新日志、日志落盘到 `logs/`）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\daily_batch_update.ps1 -Workspace . -Periods "1min,5min,15min" -Strict
```

创建每天 21:35 自动任务（示例）：

```powershell
schtasks /Create /TN "HFML_DailyBatch" /SC DAILY /ST 21:35 /TR "powershell -ExecutionPolicy Bypass -File e:\Cursor\hfml\scripts\daily_batch_update.ps1 -Workspace e:\Cursor\hfml -Periods 1min,5min,15min -Strict" /F
```

查看任务执行结果日志：`logs/daily_batch_update_*.log`
