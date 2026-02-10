# 商品期货机器学习量化模型

基于 1分钟、5分钟、15分钟 K线数据（OHLCV + 持仓量/仓差）的机器学习量化交易模型系统。系统包含完整的特征工程、市场状态识别、模型集成、回测评估、特征选择和仓位管理流程。

---

## 目录

- [项目结构](#项目结构)
- [代码文件功能说明](#代码文件功能说明)
- [各周期机器学习模型选择](#各周期机器学习模型选择)
- [特征列表](#特征列表)
- [预测目标](#预测目标)
- [特征评估报告](#特征评估报告)
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
│   ├── feature_engineering.py         # 特征工程模块
│   └── market_regime.py              # 市场状态识别模块
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
| `PREDICTION_TARGETS` | 预测目标定义及各周期的预测步长（horizon），支持 `future_direction`、`future_return`、`future_regime` |
| `DEFAULT_TARGET` | 默认预测目标：`future_direction`（涨跌方向） |
| `BACKTEST_CONFIG` | 回测参数：训练/验证/测试集比例、交叉验证折数、最低成功率阈值 |
| `LSTM_CONFIG` | LSTM模型超参数：序列长度、隐藏层单元数、dropout率等 |
| `LGBM_CONFIG` | LightGBM模型超参数：树数量、最大深度、学习率等 |
| `XGB_CONFIG` | XGBoost模型超参数：树数量、最大深度、学习率等 |
| `REGIME_CONFIG` | 市场状态识别参数：波动率窗口、趋势阈值等 |
| `ENSEMBLE_CONFIG` | 集成模型配置：是否启用、包含的基础模型 |
| `POSITION_CONFIG` | 仓位管理配置：账户风险比例、最大仓位、信号阈值 |
| `REGIME_LABELS` | 五分位预测目标标签映射 |

### `features/feature_engineering.py` — 特征工程模块

负责从原始K线数据（OHLCV + 持仓量）计算所有衍生技术指标，并集成市场状态特征。

**核心函数：**

| 函数名 | 功能 | 输出特征 |
|--------|------|----------|
| `compute_ma()` | 计算简单移动平均线 | `ma_5`, `ma_10`, `ma_20`, `ma_60` 等 |
| `compute_ema()` | 计算指数移动平均线 | `ema_5`, `ema_10`, `ema_20`, `ema_60` 等 |
| `compute_bollinger_bands()` | 计算布林带 | `boll_upper`, `boll_mid`, `boll_lower`, `boll_width`, `boll_pct_b` |
| `compute_rsi()` | 计算相对强弱指标 | `rsi_14`, `rsi_6` |
| `compute_macd()` | 计算MACD | `macd_dif`, `macd_dea`, `macd_hist` |
| `compute_kdj()` | 计算KDJ指标 | `kdj_k`, `kdj_d`, `kdj_j` |
| `compute_atr()` | 计算平均真实波幅 | `tr`, `atr` |
| `compute_cci()` | 计算CCI指标 | `cci` |
| `compute_williams_r()` | 计算威廉指标 | `williams_r` |
| `compute_roc()` | 计算变动率 | `roc_12`, `roc_6` |
| `compute_obv()` | 计算能量潮 | `obv` |
| `compute_vwap()` | 计算成交量加权平均价 | `vwap` |
| `compute_volume_features()` | 计算成交量相关特征 | `vol_ma_*`, `vol_ratio_*`, `vol_change` |
| `compute_oi_features()` | 计算持仓量/仓差特征 | `oi_change`, `oi_change_pct`, `oi_ma_*`, `vol_oi_ratio` |
| `compute_candle_features()` | 计算K线形态特征 | `body_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio`, `candle_direction`, `amplitude`, `gap`, `gap_ratio` |
| `compute_volatility_features()` | 计算波动率特征 | `volatility_*`, `return_ma_*`, `log_return` |
| `compute_price_position()` | 计算价格位置特征 | `price_position`, `dist_to_high`, `dist_to_low` |
| `compute_all_features()` | **一键计算所有特征**（含市场状态特征，根据周期自动调整窗口参数） | 69~77个特征 |
| `compute_prediction_targets()` | 计算预测目标变量 | `future_return`, `future_direction`, `future_volatility`, `future_regime` |

### `features/market_regime.py` — 市场状态识别模块

**`MarketRegimeDetector` 类** — 识别趋势/震荡/反转等市场状态，为模型提供环境上下文。

**四种市场状态：**
| 状态编码 | 名称 | 描述 |
|----------|------|------|
| 0 | `low_vol_range` | 低波动震荡市 |
| 1 | `low_vol_trend` | 低波动趋势市（平稳趋势） |
| 2 | `high_vol_range` | 高波动震荡市（反转可能） |
| 3 | `high_vol_trend` | 高波动趋势市 |

**输出特征（7个）：**
| 特征名 | 说明 |
|--------|------|
| `regime_volatility` | 滚动波动率 |
| `trend_strength` | 趋势强度（价格与时间的相关系数） |
| `volatility_rank` | 波动率分位数排名 |
| `market_regime` | 市场状态编码（0~3） |
| `trend_direction` | 趋势方向（+1/-1） |
| `volatility_change` | 波动率变化率 |
| `trend_acceleration` | 趋势加速度 |

### `models/ml_models.py` — 机器学习模型模块

封装三种机器学习模型，提供统一接口（train / predict / evaluate / get_feature_importance）。支持二分类和多分类任务。

| 模型类 | 推荐周期 | 底层框架 | 关键特点 |
|--------|----------|----------|----------|
| `LightGBMModel` | 1分钟 | LightGBM | 训练速度快、内存占用低、适合大规模数据 |
| `XGBoostModel` | 5分钟 | XGBoost | 精度高、正则化强、不易过拟合、支持多分类 |
| `LSTMModel` | 15分钟 | TensorFlow/Keras | 捕捉时序依赖关系、学习复杂非线性模式 |

**工厂函数：** `create_model(period, task)` 根据周期自动创建推荐模型。

**统一接口：**
- `train(X_train, y_train, X_val, y_val)` — 训练模型
- `predict(X)` — 预测
- `predict_proba(X)` — 预测概率（仅分类任务）
- `evaluate(X, y)` — 评估模型，返回指标字典（自动适配二分类/多分类）
- `get_feature_importance()` — 获取特征重要性排序

### `models/ensemble_model.py` — 模型集成模块

**`EnsembleModel` 类** — LightGBM + XGBoost 加权集成，提升单一模型稳定性。

| 功能 | 说明 |
|------|------|
| 自动权重分配 | 基于验证集表现动态调整两个模型的权重 |
| 加权投票预测 | 分类任务使用加权概率投票（支持多分类） |
| 加权平均预测 | 回归任务使用加权平均 |
| 综合特征重要性 | 归一化加权平均两个模型的特征重要性 |

### `models/position_sizing.py` — 仓位管理模块

将机器学习预测信号转化为实际仓位建议。

| 函数/类 | 功能 |
|---------|------|
| `calculate_position_size()` | 基于改进Kelly公式计算仓位（信号强度 × 风险预算 / 波动率） |
| `adaptive_threshold()` | 自适应交易阈值（高波动时提高阈值减少交易） |
| `RiskBudgetManager` | 风险预算管理器（综合信号、波动率、风险约束输出交易信号和仓位） |

### `backtest/feature_selector.py` — 回测与特征选择模块

通过回测挑选成功率最高的特征组合，支持稳定性选择和递归特征消除。

**`FeatureSelector` 类方法：**

| 方法 | 功能 | 说明 |
|------|------|------|
| `time_series_cv_evaluate()` | 时间序列交叉验证 | 使用 `TimeSeriesSplit` 避免前视偏差，评估模型成功率 |
| `feature_importance_ranking()` | 特征重要性排序 | 基于模型训练结果的特征重要性排名 |
| `mutual_information_ranking()` | 互信息特征排序 | 基于互信息衡量特征与目标的非线性依赖 |
| `forward_feature_selection()` | 前向特征选择 | 逐步添加使成功率提升最大的特征 |
| `stability_selection()` | 稳定性选择 | 多次LassoCV采样，统计特征被选中频率，筛选稳定特征 |
| `recursive_feature_elimination()` | 递归特征消除 | 逐步移除不重要特征 + 时间序列CV评估，找最佳特征数 |
| `evaluate_feature_subsets()` | 特征子集评估 | 评估8个特征组（含市场状态组）的成功率 |
| `select_best_features()` | 综合最佳特征选择 | 综合特征重要性 + 互信息 + CV验证，输出最终特征集 |

**辅助函数：** `get_recommended_feature_groups()` 返回按类别分组的推荐特征字典（含市场状态组）。

### `ml_pipeline.py` — 主流程入口脚本

端到端的运行流程，整合数据生成、特征工程、模型训练（支持集成模型）、回测评估、仓位管理。

**核心函数：**

| 函数 | 功能 |
|------|------|
| `generate_sample_data()` | 生成模拟K线数据（随机游走 + OHLCV + 持仓量） |
| `prepare_data()` | 数据预处理：计算全部特征（含市场状态） + 预测目标（含多分类），处理缺失值和无穷值 |
| `train_and_evaluate()` | 按 70/15/15 划分数据，训练并评估模型（支持集成模式），输出特征重要性和仓位管理建议 |
| `run_feature_selection()` | 运行完整特征选择流程：特征组评估 + 综合最佳特征选择 |
| `main()` | 命令行入口，支持参数指定周期、目标（含 `future_regime`）、数据量 |

### `FirstProject.py` — VN Trader 启动脚本

基于 vnpy 框架的交易平台启动脚本（独立于ML模型系统），用于连接 CTP 网关进行实盘交易。

### `requirements.txt` — Python依赖

项目所需的全部 Python 包及最低版本要求。

---

## 各周期机器学习模型选择

### 单模型推荐

| 周期 | 推荐模型 | 选择理由 | 数据特点 | 模型优势 |
|------|----------|----------|----------|----------|
| **1分钟** | **LightGBM** | 1分钟数据量最大（每天约240根K线），需要极快的训练速度 | 数据量大、噪声多、信号弱 | 基于直方图算法训练速度快；内存效率高；支持类别特征无需额外编码 |
| **5分钟** | **XGBoost** | 5分钟数据量适中（每天约48根K线），平衡精度与效率 | 数据量适中、信噪比较好 | 精度高且稳定；正则化能力强，不易过拟合；特征重要性评估完善 |
| **15分钟** | **LSTM** | 15分钟数据量较少（每天约16根K线），时序依赖更明显 | 数据量少、时序特征强 | 擅长捕捉长期时序依赖；适合学习复杂非线性模式；序列建模能力强 |

### 集成模型（1分钟/5分钟默认启用）

对于1分钟和5分钟周期，系统默认使用 **LightGBM + XGBoost 加权集成模型**：
- 基于验证集表现自动分配权重
- 通过集合两种不同的梯度提升算法提升预测稳定性
- 15分钟周期使用单独的LSTM模型（LSTM不参与集成）

### 模型超参数配置

**LightGBM (1分钟)**
```python
n_estimators=500, max_depth=6, learning_rate=0.05,
num_leaves=31, min_child_samples=20, subsample=0.8,
colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1
```

**XGBoost (5分钟)**
```python
n_estimators=300, max_depth=5, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1
```

**LSTM (15分钟)**
```python
sequence_length=20, hidden_units=64, dropout_rate=0.2,
epochs=50, batch_size=32, learning_rate=0.001
```

---

## 特征列表

系统根据K线周期自动计算 **69~77个** 衍生特征，按类别分组如下：

### 1. 价格趋势类（11个特征）

| 特征名 | 说明 |
|--------|------|
| `ma_5`, `ma_10`, `ma_20`, `ma_60` | 简单移动平均线（不同窗口） |
| `ema_5`, `ema_10`, `ema_20`, `ema_60` | 指数移动平均线（不同窗口） |
| `price_position` | 价格在区间中的相对位置 (0~1) |
| `dist_to_high` | 距离区间最高点的距离比 |
| `dist_to_low` | 距离区间最低点的距离比 |

### 2. 布林带类（5个特征）

| 特征名 | 说明 |
|--------|------|
| `boll_upper` | 布林带上轨 |
| `boll_mid` | 布林带中轨（20日均线） |
| `boll_lower` | 布林带下轨 |
| `boll_width` | 布林带宽度（标准化） |
| `boll_pct_b` | %B指标（价格在布林带中的相对位置） |

### 3. 动量类（12个特征）

| 特征名 | 说明 |
|--------|------|
| `rsi_14` | 14周期RSI |
| `rsi_6` | 6周期RSI |
| `macd_dif` | MACD DIF线 |
| `macd_dea` | MACD DEA线 |
| `macd_hist` | MACD柱状图 |
| `kdj_k` | KDJ K值 |
| `kdj_d` | KDJ D值 |
| `kdj_j` | KDJ J值 |
| `cci` | 商品通道指标 |
| `williams_r` | 威廉指标 |
| `roc_12` | 12周期变动率 |
| `roc_6` | 6周期变动率 |

### 4. 成交量类（7~9个特征）

| 特征名 | 说明 |
|--------|------|
| `vol_ma_5`, `vol_ma_10`, `vol_ma_20` | 成交量移动平均（不同窗口） |
| `vol_ratio_5`, `vol_ratio_10`, `vol_ratio_20` | 量比（当前成交量 / 均量） |
| `vol_change` | 成交量变化率 |
| `obv` | 能量潮（OBV） |
| `vwap` | 成交量加权平均价 |

### 5. 波动率类（7~9个特征）

| 特征名 | 说明 |
|--------|------|
| `tr` | 真实波幅 |
| `atr` | 平均真实波幅（14周期） |
| `volatility_5`, `volatility_10`, `volatility_20` | 收益率滚动标准差（不同窗口） |
| `return_ma_5`, `return_ma_10`, `return_ma_20` | 收益率滚动均值（不同窗口） |
| `log_return` | 对数收益率 |

### 6. 持仓量/仓差类（9个特征）

| 特征名 | 说明 |
|--------|------|
| `oi_change` | 仓差（持仓量变化量） |
| `oi_change_pct` | 仓差变化率 |
| `oi_ma_5`, `oi_ma_10`, `oi_ma_20` | 持仓量移动平均 |
| `oi_change_ma_5`, `oi_change_ma_10`, `oi_change_ma_20` | 仓差移动平均 |
| `vol_oi_ratio` | 量仓比（成交量 / 持仓量） |

### 7. K线形态类（7个特征）

| 特征名 | 说明 |
|--------|------|
| `body_ratio` | K线实体比（实体 / 总振幅） |
| `upper_shadow_ratio` | 上影线比 |
| `lower_shadow_ratio` | 下影线比 |
| `candle_direction` | 涨跌方向 (+1/-1) |
| `amplitude` | 振幅比 |
| `gap` | 跳空缺口（绝对值） |
| `gap_ratio` | 跳空缺口比 |

### 8. 市场状态类（7个特征）

| 特征名 | 说明 |
|--------|------|
| `regime_volatility` | 滚动波动率（20周期） |
| `trend_strength` | 趋势强度（价格与时间的Pearson相关系数） |
| `volatility_rank` | 波动率历史分位数排名 |
| `market_regime` | 市场状态编码（0=低波震荡, 1=低波趋势, 2=高波震荡, 3=高波趋势） |
| `trend_direction` | 趋势方向 (+1/-1) |
| `volatility_change` | 波动率5周期变化率 |
| `trend_acceleration` | 趋势加速度（趋势强度的5周期变化） |

### 各周期窗口参数差异

| 参数 | 1分钟 | 5分钟 | 15分钟 |
|------|-------|-------|--------|
| MA/EMA窗口 | 5, 10, 20, 60, 120 | 5, 10, 20, 60 | 5, 10, 20, 40 |
| 成交量窗口 | 5, 10, 20, 60 | 5, 10, 20 | 5, 10, 20 |
| 持仓量窗口 | 5, 10, 20, 60 | 5, 10, 20 | 5, 10, 20 |
| 总特征数 | ~77 | ~69 | ~69 |

---

## 预测目标

| 目标名 | 类型 | 说明 | 预测步长 (1min/5min/15min) |
|--------|------|------|--------------------------|
| `future_direction` | 二分类 (0/1) | 未来价格方向（1=涨, 0=跌），**推荐** | 5 / 3 / 2 |
| `future_regime` | 五分类 (0~4) | 未来价格五分位状态（强跌/弱跌/中性/弱涨/强涨） | 5 / 3 / 2 |
| `future_return` | 回归 | 未来N根K线的收益率 | 5 / 3 / 2 |
| `future_volatility` | 回归 | 未来N根K线的波动率 | 5 / 3 / 2 |

### `future_regime` 五分位标签

| 编码 | 标签 | 收益率范围 |
|------|------|-----------|
| 0 | `strong_down` | < -0.5% |
| 1 | `weak_down` | -0.5% ~ -0.1% |
| 2 | `neutral` | -0.1% ~ +0.1% |
| 3 | `weak_up` | +0.1% ~ +0.5% |
| 4 | `strong_up` | > +0.5% |

---

## 特征评估报告

### 评估方法

系统通过以下方法评估特征的预测能力，挑选成功率最高的特征组合：

#### 1. 时间序列交叉验证 (TimeSeriesSplit)

```
Fold 1: [==训练==]  [测试]
Fold 2: [====训练====]  [测试]
Fold 3: [======训练======]  [测试]
Fold 4: [========训练========]  [测试]
Fold 5: [==========训练==========]  [测试]
```

- 使用 `sklearn.model_selection.TimeSeriesSplit`，默认 5 折
- 确保训练数据始终在测试数据**之前**，避免前视偏差（look-ahead bias）
- 输出每折的准确率、精确率、召回率、F1分数

#### 2. 特征重要性排序

- 基于 LightGBM / XGBoost 模型训练后的 `feature_importances_`
- 反映每个特征对模型预测贡献的大小
- LSTM 无直接特征重要性，改用 XGBoost 辅助评估

#### 3. 互信息排序 (Mutual Information)

- 使用 `sklearn.feature_selection.mutual_info_classif`（分类）或 `mutual_info_regression`（回归）
- 衡量特征与目标变量之间的非线性统计依赖性
- 不假设线性关系，能发现复杂的特征-目标关联

#### 4. 前向特征选择 (Forward Selection)

- 每步添加使成功率提升最大的特征
- 直到达到最大特征数或成功率不再提升
- 输出特征选择历史（每步的特征和对应成功率）

#### 5. 稳定性选择 (Stability Selection)

- 使用 LassoCV 进行多次随机采样（默认20次）
- 统计每个特征被选中的频率
- 频率高于阈值（默认60%）的特征被认为是稳定重要的特征
- 适合发现在不同数据子集上一致重要的特征

#### 6. 递归特征消除 (RFE)

- 逐步移除最不重要的特征（每步移除5个）
- 在每一步使用时间序列交叉验证评估模型性能
- 记录每步的特征数量和性能得分
- 输出达到最优性能时的特征子集

#### 7. 特征组评估

将全部特征分为 **8 个组**（价格趋势、布林带、动量指标、成交量、波动率、持仓量/仓差、K线形态、市场状态），分别评估每组在时间序列交叉验证中的成功率，确定哪类特征对预测贡献最大。

#### 8. 综合最佳特征选择

流程：
1. 计算模型特征重要性排名 → 选取 Top N
2. 计算互信息排名 → 选取 Top N
3. 两种排名取并集，按综合分数排序
4. 使用时间序列CV验证最终特征集的成功率

### 评估指标

**分类任务（future_direction / future_regime）：**

| 指标 | 说明 |
|------|------|
| Accuracy（准确率/成功率） | 预测正确的样本比例 |
| Precision（精确率） | 预测为涨中实际为涨的比例（多分类使用加权平均） |
| Recall（召回率） | 实际为涨中被正确预测的比例（多分类使用加权平均） |
| F1 Score | 精确率和召回率的调和平均 |

**回归任务（future_return）：**

| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| R² | 决定系数 |

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

### 运行主流程

```bash
# 5分钟周期，预测涨跌方向（默认配置，使用集成模型）
python ml_pipeline.py

# 指定1分钟周期（使用集成模型）
python ml_pipeline.py --period 1min --target future_direction

# 指定5分钟周期
python ml_pipeline.py --period 5min --target future_direction

# 指定15分钟周期（使用LSTM模型）
python ml_pipeline.py --period 15min --target future_direction

# 五分位预测（强跌/弱跌/中性/弱涨/强涨）
python ml_pipeline.py --period 5min --target future_regime

# 回归任务（预测收益率）
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
python ml_pipeline.py --period 5min --target future_direction --n-rows 5000
```

**输出解读：**

```
============================================================
商品期货机器学习量化模型
周期: 5min, 目标: future_direction, 任务: classification
推荐ML框架: xgboost
============================================================

# 步骤1: 数据准备
生成模拟数据...
数据形状: (5000, 6)
计算 5min 周期衍生指标...
特征数量: 69, 样本数量: 4901      ← 含7个市场状态特征

# 步骤2: 模型训练与评估（集成模型）
创建 5min 周期模型 (推荐框架: xgboost)...
使用 LightGBM+XGBoost 集成模型...   ← 自动启用集成
训练模型...
训练集指标: {'accuracy': 1.0, ...}
测试集指标: {'accuracy': 0.44, ...}

# 步骤3: 特征重要性 Top 10（集成加权）
Top 10 重要特征:
oi_change_ma_20      0.023
vol_ma_5             0.022
volatility_change    0.022     ← 市场状态特征入选
market_regime        0.021     ← 市场状态特征入选
atr                  0.021
...

# 步骤4: 集成模型权重
集成模型权重: {'lightgbm': 0.496, 'xgboost': 0.504}

# 步骤5: 仓位管理
仓位管理: 测试集产生 274 个交易信号

# 步骤6: 特征组评估（8个组，含市场状态组）
特征组评估结果:
     group  n_features  accuracy
      布林带           5     0.516
  持仓量(仓差)          9     0.511
      成交量           9     0.510
      波动率           9     0.509
     K线形态           7     0.497
     价格趋势         11     0.492
     市场状态           7     0.485
     动量指标         12     0.480

# 步骤7: 综合特征选择
最佳特征数量: 20
最佳特征: ['boll_upper', ..., 'market_regime', ...]
最终成功率: 0.5209
============================================================
流程完成!
```

### 示例：五分位多分类预测

```bash
python ml_pipeline.py --period 1min --target future_regime --n-rows 2000
```

### 结果解读

1. **集成模型权重**：LightGBM和XGBoost的权重基于验证集自动分配，接近50/50说明两者表现相当
2. **市场状态特征**：`market_regime` 和 `volatility_change` 等新特征进入Top 10重要特征
3. **特征组成功率**：8个特征组分别评估，可识别哪类特征对预测贡献最大
4. **仓位管理**：基于预测概率和波动率自适应生成交易信号，带仓位大小建议
5. **五分位预测**：`future_regime` 提供更细粒度的预测（强跌/弱跌/中性/弱涨/强涨）

> **注意**：当前使用模拟随机数据，实际商品期货数据的预测效果会因市场行情不同而有差异。接入真实行情数据后需要重新训练和评估。
