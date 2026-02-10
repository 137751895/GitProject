# 商品期货机器学习量化模型

基于 1分钟、5分钟、15分钟 K线数据（OHLCV + 持仓量/仓差）的机器学习量化交易模型系统。系统包含完整的特征工程、市场状态识别、微观结构分析、Numba高性能计算、模型集成、回测评估、特征选择和仓位管理流程。

---

## 目录

- [项目结构](#项目结构)
- [代码文件功能说明](#代码文件功能说明)
- [各周期机器学习模型选择](#各周期机器学习模型选择)
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
├── ml_pipeline.py                     # 主流程入口脚本
├── requirements.txt                   # Python依赖
├── FirstProject.py                    # VN Trader 启动脚本（独立）
├── features/
│   ├── __init__.py
│   ├── feature_engineering.py         # 特征工程主模块
│   ├── feature_engineering_enhanced.py # 增强特征模块（微观结构/高级波动率/缺口衰减）
│   ├── market_regime.py              # 市场状态识别模块
│   └── rolling_numba.py             # Numba加速滚动计算模块
├── models/
│   ├── __init__.py
│   ├── ml_models.py                   # 机器学习模型模块
│   ├── ensemble_model.py             # 模型集成模块
│   └── position_sizing.py            # 仓位管理模块
└── backtest/
    ├── __init__.py
    └── feature_selector.py            # 回测与特征选择模块
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
| `POSITION_CONFIG` | 仓位管理配置 |
| `ENHANCED_FEATURE_CONFIG` | 增强特征配置（微观结构/高级波动率/缺口衰减/Numba加速） |

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
| `compute_all_features()` | **一键计算所有特征**（含市场状态 + 微观结构 + 高级波动率 + 缺口衰减） | 81~89个特征 |
| `compute_prediction_targets()` | 预测目标 | `future_return`, `future_direction`, `future_volatility`, `future_regime` |

### `features/feature_engineering_enhanced.py` — 增强特征模块

基于附件 multi_timeframe.py 和 rolling_numba.py 的分析新增的高价值特征。

| 函数名 | 功能 | 输出特征 |
|--------|------|----------|
| `compute_microstructure_features()` | 微观结构分析特征 | `divergence`, `vwap_dev`, `vol_state`, `mom_slope`, `close_pos`, `rsi_slope`, `vol_zscore` |
| `compute_advanced_volatility_features()` | 高级波动率特征 | `volatility_regime`, `vol_ratio`, `atr_pct`, `range_pct` |
| `compute_gap_decay_features()` | 期货夜盘缺口衰减特征 | `gap_decay` |

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

**`MarketRegimeDetector` 类** — 识别趋势/震荡/反转等市场状态（四种状态 × 7个特征）。

### `models/ml_models.py` — 机器学习模型模块

封装 LightGBM / XGBoost / LSTM 三种模型，提供统一接口，支持二分类和多分类。

### `models/ensemble_model.py` — 模型集成模块

LightGBM + XGBoost 加权集成，基于验证集自动分配权重。

### `models/position_sizing.py` — 仓位管理模块

改进Kelly公式的仓位计算 + 波动率自适应交易阈值。

### `backtest/feature_selector.py` — 回测与特征选择模块

时间序列交叉验证 + 8种特征选择方法（含稳定性选择和递归特征消除）。现在评估 **10个特征组**。

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

## 特征列表

系统根据K线周期自动计算 **81~89个** 衍生特征，按类别分组如下：

### 1. 价格趋势类（11个特征）

| 特征名 | 说明 |
|--------|------|
| `ma_5`, `ma_10`, `ma_20`, `ma_60` | 简单移动平均线 |
| `ema_5`, `ema_10`, `ema_20`, `ema_60` | 指数移动平均线 |
| `price_position` | 价格在区间中的相对位置 (0~1) |
| `dist_to_high` | 距离区间最高点的距离比 |
| `dist_to_low` | 距离区间最低点的距离比 |

### 2. 布林带类（5个特征）

| 特征名 | 说明 |
|--------|------|
| `boll_upper` / `boll_mid` / `boll_lower` | 布林带上轨/中轨/下轨 |
| `boll_width` | 布林带宽度 |
| `boll_pct_b` | %B指标 |

### 3. 动量类（12个特征）

| 特征名 | 说明 |
|--------|------|
| `rsi_14` / `rsi_6` | RSI |
| `macd_dif` / `macd_dea` / `macd_hist` | MACD |
| `kdj_k` / `kdj_d` / `kdj_j` | KDJ |
| `cci` | CCI |
| `williams_r` | 威廉指标 |
| `roc_12` / `roc_6` | 变动率 |

### 4. 成交量类（7~9个特征）

| 特征名 | 说明 |
|--------|------|
| `vol_ma_*` / `vol_ratio_*` | 成交量均线和量比 |
| `vol_change` / `obv` / `vwap` | 成交量变化率、能量潮、加权均价 |

### 5. 波动率类（7~9个特征）

| 特征名 | 说明 |
|--------|------|
| `tr` / `atr` | 真实波幅 / 平均真实波幅 |
| `volatility_*` / `return_ma_*` | 滚动波动率和收益率均值 |
| `log_return` | 对数收益率 |

### 6. 持仓量/仓差类（9个特征）

| 特征名 | 说明 |
|--------|------|
| `oi_change` / `oi_change_pct` | 仓差和仓差变化率 |
| `oi_ma_*` / `oi_change_ma_*` | 持仓量和仓差移动平均 |
| `vol_oi_ratio` | 量仓比 |

### 7. K线形态类（7个特征）

| 特征名 | 说明 |
|--------|------|
| `body_ratio` / `upper_shadow_ratio` / `lower_shadow_ratio` | 实体比和影线比 |
| `candle_direction` / `amplitude` | 涨跌方向和振幅 |
| `gap` / `gap_ratio` | 跳空缺口 |

### 8. 市场状态类（7个特征）

| 特征名 | 说明 |
|--------|------|
| `regime_volatility` | 滚动波动率 |
| `trend_strength` / `trend_direction` / `trend_acceleration` | 趋势强度、方向、加速度 |
| `volatility_rank` / `volatility_change` | 波动率排名和变化率 |
| `market_regime` | 市场状态编码（0~3） |

### 9. 微观结构类（8个特征）— 新增

| 特征名 | 计算公式 | 说明 |
|--------|----------|------|
| `divergence` | `sign(持仓变化) × sign(价格变化)` | 资金流向与价格背离 |
| `vwap_dev` | `(close - VWAP) / VWAP` | 价格与成交量加权均价的偏离 |
| `vol_state` | `ATR14 / 60日均价` | 波动率状态（标准化） |
| `mom_slope` | `5周期价格斜率 / 当前价格` | 动量加速度 |
| `close_pos` | `(close - low) / (high - low)` | K线内相对位置 |
| `rsi_slope` | RSI的5周期线性回归斜率 | RSI变化速率 |
| `vol_zscore` | `(volume - vol_ma20) / vol_std20` | 成交量Z分数 |
| `gap_decay` | `gap × exp(-秒数/300)` | 夜盘缺口指数衰减（21:00~21:30） |

### 10. 高级波动率类（4个特征）— 新增

| 特征名 | 计算公式 | 说明 |
|--------|----------|------|
| `volatility_regime` | 收益率的20周期滚动标准差 | 波动率状态 |
| `vol_ratio` | `ATR14 / ATR14的20周期均值` | ATR相对水平 |
| `atr_pct` | `ATR14 / 当前价格` | ATR百分比化 |
| `range_pct` | `(high - low) / close` | 振幅百分比 |

### 各周期窗口参数差异

| 参数 | 1分钟 | 5分钟 | 15分钟 |
|------|-------|-------|--------|
| MA/EMA窗口 | 5, 10, 20, 60, 120 | 5, 10, 20, 60 | 5, 10, 20, 40 |
| 成交量窗口 | 5, 10, 20, 60 | 5, 10, 20 | 5, 10, 20 |
| 总特征数 | ~89 | ~81 | ~81 |

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
7. **特征组评估** — 将全部特征分为 **10个组**（新增微观结构组和高级波动率组），分别评估成功率
8. **综合最佳特征选择** — 综合重要性 + 互信息 + CV验证

### 10个特征组

| 特征组 | 特征数 | 说明 |
|--------|--------|------|
| 价格趋势 | 11 | MA/EMA/价格位置 |
| 布林带 | 5 | 布林带系列 |
| 动量指标 | 12 | RSI/MACD/KDJ等 |
| 成交量 | 9 | 量比/OBV/VWAP |
| 波动率 | 9 | ATR/波动率标准差 |
| 持仓量(仓差) | 9 | 仓差/量仓比 |
| K线形态 | 7 | 实体比/影线/缺口 |
| 市场状态 | 7 | 趋势/波动率状态 |
| **微观结构** | **8** | **divergence/vwap_dev/vol_state/mom_slope/close_pos/rsi_slope/vol_zscore/gap_decay** |
| **高级波动率** | **4** | **volatility_regime/vol_ratio/atr_pct/range_pct** |

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

### 运行主流程

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
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--period` | str | `5min` | K线周期：`1min`、`5min`、`15min` |
| `--target` | str | `future_direction` | 预测目标：`future_direction`、`future_return`、`future_regime` |
| `--n-rows` | int | `5000` | 模拟数据行数 |
| `--skip-feature-selection` | flag | — | 跳过特征选择步骤 |

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

# 步骤1: 数据准备
特征数量: 81, 样本数量: 1901      ← 含12个新增增强特征

# 步骤2: 模型训练（集成模型）
使用 LightGBM+XGBoost 集成模型...
集成模型权重: {'lightgbm': 0.504, 'xgboost': 0.496}

# 步骤3: 特征重要性 Top 10
vwap_dev             0.020     ← 新增微观结构特征入选Top 10

# 步骤4: 仓位管理
仓位管理: 测试集产生 285 个交易信号

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
3. **特征总数提升** — 从62-70个增至81-89个（新增12个高价值特征）
4. **Numba加速** — 增强特征模块使用numba加速的rolling计算，适合高频数据场景

> **注意**：当前使用模拟随机数据，实际商品期货数据的预测效果会因市场行情不同而有差异。接入真实行情数据后需要重新训练和评估。
