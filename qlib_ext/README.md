# qlib_ext：HFML → Qlib 适配层

## 简介

`qlib_ext` 是一个独立的适配层，旨在将 **HFML 商品期货机器学习量化研究框架**的核心模块无缝集成到**微软 Qlib 开源量化平台**的标准工作流中。通过本适配层，研究人员和工程师可以复用 HFML 的特色模块——包括深度特征工程、智能标签系统和混合交易系统——同时享受 Qlib 提供的标准化数据管理、回测框架和实验记录能力。

> **当前版本状态**：基于 Qlib 主分支（2026-03），HFML 提交 `c310016`。适配层已完成骨架搭建，P0/P1/P2 接口问题已修正（见[已知问题](#已知问题与待办事项)）。

---

## 主要特性

- **完整的特征工程流水线**：基础特征（价格/动量/波动率/持仓量）→ 增强特征（微观结构/缺口衰减）→ 深度非线性变换，两阶段分离、无重复计算。
- **智能标签系统**：五维交易信号（`-2/-1/0/+1/+2`）+ 信号质量评分 + 预期收益标签。
- **树模型 & LSTM 模型包装**：LightGBM（1min 推荐）、XGBoost（5min 推荐）、LSTM（15min 推荐），均实现 Qlib `Model` 协议，统一标签二分类映射。
- **混合交易系统信号生成**：`HFMLHybridSignal` 继承 Qlib `Signal` 基类，融合多专家系统输出。
- **多时间框架策略协同**：`HFMLMultiTimeframeStrategy` 基于 `MultiTimeframeCoordinator` 实现三周期（1min/5min/15min）信号融合。
- **期货风控集成**：`HFMLHybridStrategy` 接入 `RiskBudgetManager`/`DrawdownTracker`/`DailyRiskLimit`，实现开多/平多/开空/平空期货执行语义。
- **特征预筛选与漂移监控记录器**：`HFMLFeatureSelectionRecord`（基于 `XGBFeatureSelector.fit_select()`）、`DriftMonitoringRecord`（基于 `ConceptDriftDetector.update()`）、及 `FeatureGroupImportanceRecord`、`FeatureStabilityRecord`、`MultiTimeframeGradeRecord`。
- **在线更新任务框架**：`online_update_job` 支持增量学习与概念漂移响应，可接入 Airflow / cron 等调度系统。

---

## 安装

### 1. 环境准备（推荐使用独立 conda 环境）

```bash
conda create -n hfml_qlib python=3.9
conda activate hfml_qlib
```

### 2. 安装 Qlib

```bash
# 推荐从源码安装最新版
pip install pyqlib
# 或从源码
git clone https://github.com/microsoft/qlib.git
pip install -e ./qlib
```

### 3. 安装 HFML 依赖

```bash
# 在项目根目录安装 HFML 所需依赖
pip install -r requirements.txt
```

`requirements.txt` 主要包含：

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.6.0
tensorflow>=2.18.0
scipy>=1.10.0
pyyaml
```

### 4. 确认项目结构可导入

```bash
# 将项目根目录加入 PYTHONPATH，使 qlib_ext、features、models 等包可被导入
export PYTHONPATH=/path/to/GitProject:$PYTHONPATH
```

或在代码中：

```python
import sys
sys.path.insert(0, "/path/to/GitProject")
```

---

## 快速开始

### 数据格式要求

`HFMLDataLoader` 期望读取 **CSV 格式的 K 线文件**，必须包含以下列：

| 列名            | 说明                |
|----------------|---------------------|
| `open`         | 开盘价              |
| `high`         | 最高价              |
| `low`          | 最低价              |
| `close`        | 收盘价              |
| `volume`       | 成交量              |
| `open_interest`| 持仓量（期货特有）  |

索引应为 `datetime` 类型，示例路径：`data/klines/KQi@SHFEag/KQi@SHFEag_15min.csv`

### 修改配置文件

编辑 `workflow_config_15min.yaml`，至少修改以下路径：

```yaml
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          data_loader:
            kwargs:
              csv_path: data/klines/<合约名>/<合约名>_15min.csv  # ← 改为你的数据路径
          start_time: 2022-01-01   # ← 数据起始时间
          end_time: 2024-12-31     # ← 数据结束时间
```

### 运行工作流

```bash
cd /path/to/GitProject
qrun workflow_config_15min.yaml
```

### 预期输出

- Qlib 实验目录下生成 `pred.pkl`、`label.pkl` 等预测文件
- 自动生成以下 artifacts：
  - `hfml_feature_selection/selected_features.pkl`：筛选后的特征列表
  - `hfml_feature_selection/feature_importance.pkl`：特征重要性
  - `hfml_drift_monitoring/drift_result.pkl`：漂移检测结果
  - 回测报告（IC、Rank IC、累计收益曲线等）

---

## 核心模块说明

### 数据接入 (`qlib_ext/data.py`)

| 类名              | 继承自                     | 说明                                        |
|------------------|---------------------------|---------------------------------------------|
| `HFMLDataLoader` | —（实现 `load()` 接口）   | 从 CSV 文件加载期货 K 线数据，返回 DataFrame |
| `HFMLDataHandler`| `qlib.data.dataset.handler.DataHandlerLP` | Qlib 标准 DataHandler，挂载处理器流水线 |

```yaml
# YAML 配置示例
handler:
  class: HFMLDataHandler
  module_path: qlib_ext.data
  kwargs:
    data_loader:
      class: HFMLDataLoader
      module_path: qlib_ext.data
      kwargs:
        csv_path: data/klines/KQi@SHFEag/KQi@SHFEag_15min.csv
    period: 15min
```

---

### 特征处理器 (`qlib_ext/processors.py`)

| 类名                     | 说明                                                |
|-------------------------|-----------------------------------------------------|
| `HFMLFeatureProcessor`  | 计算基础 + 增强 + 注册表特征（**不含**深度变换）      |
| `HFMLTransformProcessor`| 在基础特征之上应用 `compute_all_transforms` 深度变换 |
| `HFMLQualityFilter`     | 按信号质量分过滤低质量训练样本（仅用于训练阶段）       |

**特征流水线分层设计（避免重复计算）：**

```
原始 OHLCV → HFMLFeatureProcessor
               ├─ 基础价格/动量/波动率/持仓量特征
               ├─ 市场状态特征
               ├─ 增强特征（微观结构、高级波动率、缺口衰减）
               └─ 注册表自定义特征
                        ↓
             HFMLTransformProcessor
               └─ 深度非线性变换（Z-score、变化率、交互项等）
```

```yaml
infer_processors:
  - class: HFMLFeatureProcessor
    module_path: qlib_ext.processors
    kwargs:
      period: 15min          # "1min" / "5min" / "15min"
  - class: HFMLTransformProcessor
    module_path: qlib_ext.processors
  - class: RobustZScoreNorm
    module_path: qlib.data.dataset.processor
    kwargs:
      fields_group: feature
  - class: Fillna
    module_path: qlib.data.dataset.processor
    kwargs:
      fields_group: feature
```

---

### 智能标签处理器 (`qlib_ext/labels.py`)

`HFMLSmartLabelProcessor` 生成以下标签列（均在 `("label", col)` 列组下）：

| 标签列              | 含义                              | 取值范围         |
|--------------------|----------------------------------|-----------------|
| `trading_signal`   | 五维交易信号                      | -2, -1, 0, +1, +2 |
| `signal_quality`   | 信号质量评分                      | 0.0 ~ 1.0       |
| `signal_strength`  | 信号强度                          | 0.0 ~ 1.0       |
| `expected_return`  | 预期收益率                        | float            |
| `oi_confirmation`  | 持仓量确认度                      | 0.0 ~ 1.0       |

> **注意**：模型包装器（`HFMLLSTMQlibModel` 等）默认将 `trading_signal` 映射为**二分类方向标签**（`signal > 0 → 1，否则 → 0`），以与二分类模型头对齐。如需五分类或回归，可通过 `label_col` 参数指定 `expected_return`。

```yaml
learn_processors:
  - class: HFMLSmartLabelProcessor
    module_path: qlib_ext.labels
    kwargs:
      period: 15min
      config:
        horizon: 2
        base_threshold: 0.001
        strong_multiplier: 2.0
        vol_window: 20
  - class: HFMLQualityFilter
    module_path: qlib_ext.processors
    kwargs:
      min_quality: 0.55
```

---

### 模型包装 (`qlib_ext/models.py`)

| 类名                  | 内部模型         | 推荐周期  | 说明                          |
|----------------------|-----------------|---------|-------------------------------|
| `HFMLLSTMQlibModel`  | TF LSTM         | 15min   | 序列模型；预测时自动对齐索引（截去前 `sequence_length` 行）|
| `HFMLLGBMQlibModel`  | LightGBM        | 1min    | 树模型；支持增量学习            |
| `HFMLXGBQlibModel`   | XGBoost         | 5min    | 树模型；可配合 XGBFeatureSelector 预筛选 |

> ⚠️ **LSTM 索引对齐**：`HFMLLSTMQlibModel.predict()` 返回的预测序列比输入少前 `sequence_length`（默认 20）行。本实现已自动截断索引对齐，使用时请确保测试集长度 > `sequence_length`。

```yaml
model:
  class: HFMLLSTMQlibModel
  module_path: qlib_ext.models
  kwargs:
    task: classification
    label_col: trading_signal   # 内部自动映射为二分类
    params:
      sequence_length: 20
      hidden_units: 64
      dropout_rate: 0.2
      epochs: 50
      batch_size: 32
```

---

### 信号与策略 (`qlib_ext/signals.py`, `qlib_ext/strategies.py`)

#### `HFMLHybridSignal`

继承 `qlib.backtest.signal.Signal`，封装 `HybridTradingSystem` 多专家融合信号。

```python
from qlib_ext.signals import HFMLHybridSignal

signal = HFMLHybridSignal(window=100, signal_threshold=0.3)
signal.set_market_data(market_df)
result = signal._get_hybrid_signal()
# result: {"final_signal": 1, "position_size": 0.15, "confidence": 0.72, ...}
```

#### `HFMLHybridStrategy`

单时间框架混合专家策略，集成完整期货风控。

```yaml
strategy:
  class: HFMLHybridStrategy
  module_path: qlib_ext.strategies
  kwargs:
    instrument: KQi@SHFEag
    signal:
      class: HFMLHybridSignal
      module_path: qlib_ext.signals
      kwargs:
        window: 100
        signal_threshold: 0.3
    risk_kwargs:
      account_risk: 0.02        # 每笔最大风险比例
      max_position: 0.2         # 最大仓位
      max_drawdown: 0.15        # 最大允许回撤
      max_daily_loss: 0.03      # 单日最大亏损
      max_daily_trades: 20      # 单日最大交易次数
```

**执行语义**（期货四方向）：
- 开多：无仓位 → 买入
- 平多：持多头 → 卖出平仓
- 开空：无仓位 → 卖出开空
- 平空（回补）：持空头 → 买入平仓

#### `HFMLMultiTimeframeStrategy`

三周期（15min/5min/1min）协同策略。从三个独立 Recorder 加载各周期预测分数，通过 `MultiTimeframeCoordinator` 融合为最终信号。

```yaml
strategy:
  class: HFMLMultiTimeframeStrategy
  module_path: qlib_ext.strategies
  kwargs:
    instrument: KQi@SHFEag
    recorder_id_15m: <15min_recorder_id>
    recorder_id_5m:  <5min_recorder_id>
    recorder_id_1m:  <1min_recorder_id>
    coordinator_kwargs:
      neutral_zone: 0.10
      entry_threshold: 0.60
      min_grade: "C"
```

---

### 记录器 (`qlib_ext/records.py`)

| 类名                          | 依赖               | 产出 Artifact                            |
|------------------------------|-------------------|------------------------------------------|
| `HFMLFeatureSelectionRecord` | `SignalRecord`    | `selected_features.pkl`, `feature_importance.pkl`, `feature_group_importance.pkl` |
| `FeatureGroupImportanceRecord`| `HFMLFeatureSelectionRecord` | `feature_group_importance.pkl`  |
| `FeatureStabilityRecord`     | `SignalRecord`    | `feature_stability.pkl`                  |
| `DriftMonitoringRecord`      | `SignalRecord`    | `drift_result.pkl`                       |
| `HybridSignalBreakdownRecord`| `SignalRecord`    | `signal_breakdown.pkl`                   |
| `MultiTimeframeGradeRecord`  | `SignalRecord`    | `multiframe_grade.pkl`                   |

> **前提条件**：训练脚本需预先保存 `dataset` artifact，记录器才能加载训练集：
> ```python
> with R.start(experiment_name="hfml"):
>     R.save_objects(dataset=dataset)
> ```

```yaml
record:
  - class: SignalRecord
    module_path: qlib.workflow.record_temp
    kwargs:
      model: <MODEL>
      dataset: <DATASET>

  - class: HFMLFeatureSelectionRecord
    module_path: qlib_ext.records
    kwargs:
      selector_type: xgb
      selector_kwargs:
        n_features: 25
        threshold: median
        cv_splits: 5
        xgb_params:
          n_estimators: 200
          max_depth: 5
          learning_rate: 0.05

  - class: DriftMonitoringRecord
    module_path: qlib_ext.records
```

---

### 在线更新 (`qlib_ext/online.py`)

`online_update_job` 将 `AdaptiveModelManager.step()` 封装为可调度的任务函数，支持：
- 概念漂移检测（`ConceptDriftDetector`）
- 在线增量学习（`OnlineLearner`，LightGBM/XGBoost warm-start）
- 自适应阈值调整（`AdaptiveThresholdManager`）

```python
from qlib_ext.online import online_update_job

result = online_update_job(
    y_true=actual_labels,       # 真实标签（array-like）
    y_pred=model_predictions,   # 模型预测值（array-like）
    X_new=new_feature_df,       # 增量特征数据（可选）
    y_new=new_label_series,     # 增量标签（可选）
    current_volatility=0.018,
    mean_volatility=0.015,
    experiment_name="hfml_experiment",
)

print(result["drift_level"])    # "none" / "warning" / "drift"
print(result["threshold"])      # 当前自适应交易阈值
print(result["needs_retrain"])  # 是否需要完全重训
```

**调度示例（cron）：**

```bash
# 每个交易日收盘后执行在线更新
0 16 * * 1-5 python -c "from qlib_ext.online import online_update_job; ..."
```

---

## 配置文件示例

以下是 `workflow_config_15min.yaml` 的简化结构，展示各模块的挂载方式：

```yaml
sys:
  rel_path:
    - .                         # 将项目根目录加入 Python 路径

qlib_init:
  provider_uri: ~/.qlib/qlib_data/hfml_data
  region: cn

task:
  model:
    class: HFMLLSTMQlibModel
    module_path: qlib_ext.models
    kwargs:
      task: classification
      label_col: trading_signal
      params:
        sequence_length: 20
        epochs: 50

  dataset:
    class: TSDatasetH
    module_path: qlib.data.dataset
    kwargs:
      step_len: 20
      segments:
        train: [2022-01-01, 2023-06-30]
        valid: [2023-07-01, 2023-12-31]
        test:  [2024-01-01, 2024-12-31]
      handler:
        class: HFMLDataHandler
        module_path: qlib_ext.data
        kwargs:
          data_loader:
            class: HFMLDataLoader
            module_path: qlib_ext.data
            kwargs:
              csv_path: data/klines/KQi@SHFEag/KQi@SHFEag_15min.csv
          period: 15min
          start_time: 2022-01-01
          end_time: 2024-12-31
          infer_processors:
            - class: HFMLFeatureProcessor
              module_path: qlib_ext.processors
              kwargs:
                period: 15min
            - class: HFMLTransformProcessor
              module_path: qlib_ext.processors
          learn_processors:
            - class: HFMLSmartLabelProcessor
              module_path: qlib_ext.labels
              kwargs:
                period: 15min
                config:
                  horizon: 2
            - class: HFMLQualityFilter
              module_path: qlib_ext.processors
              kwargs:
                min_quality: 0.55

  record:
    - class: SignalRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        model: <MODEL>
        dataset: <DATASET>
    - class: HFMLFeatureSelectionRecord
      module_path: qlib_ext.records
      kwargs:
        selector_type: xgb
        selector_kwargs:
          n_features: 25
          xgb_params:
            n_estimators: 200
    - class: DriftMonitoringRecord
      module_path: qlib_ext.records
    - class: PortAnaRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        config:
          strategy:
            class: HFMLHybridStrategy
            module_path: qlib_ext.strategies
            kwargs:
              instrument: KQi@SHFEag
              signal:
                class: HFMLHybridSignal
                module_path: qlib_ext.signals
                kwargs:
                  window: 100
          executor:
            class: SimulatorExecutor
            module_path: qlib.backtest.executor
            kwargs:
              time_per_step: 15min
          backtest:
            start_time: 2024-01-01
            end_time: 2024-12-31
            account: 10000000
            exchange_kwargs:
              freq: 15min
              deal_price: close
              open_cost: 0.0003
              close_cost: 0.0003
```

---

## 测试

运行集成测试（覆盖接口存在性、数据流、标签映射等关键链路）：

```bash
cd /path/to/GitProject
python test_integration.py
```

测试覆盖范围（17 项）：

| 编号 | 测试内容                                    |
|-----|---------------------------------------------|
| 1   | `qlib_ext` 所有模块导入                      |
| 2   | `HFMLDataLoader` 错误处理                    |
| 3   | `HFMLFeatureProcessor` 特征计算与 MultiIndex 输出 |
| 4   | `HFMLTransformProcessor` 深度变换列数增加    |
| 5   | `HFMLSmartLabelProcessor` 标签生成           |
| 6   | `HFMLQualityFilter` 样本过滤                 |
| 7   | 三种模型类 `fit`/`predict` 接口存在          |
| 8   | `HFMLHybridSignal` 信号生成                  |
| 9   | `utils` 工具函数（`normalize_pred` 等）      |
| 10  | YAML 配置文件格式与 `selector_kwargs` 结构   |
| 11  | `trading_signal` → 二分类方向映射正确性      |
| 12  | `XGBFeatureSelector.fit_select()` 接口存在   |
| 13  | `ConceptDriftDetector.update()` 接口存在     |
| 14  | `online_update_job` 参数签名正确             |
| 15  | 特征流水线不含重复 `compute_all_transforms`  |
| 16  | `_ScorePredictorWrapper.predict_proba()` 输出 |
| 17  | 三个新增记录器类已定义                       |

---

## 已知问题与待办事项

以下问题基于 `integration_summary.md` 第 7 节复核报告，当前版本（`c310016`）中已修正 P0/P1/P2 核心问题，但仍有若干端到端验证需在真实 Qlib 环境中完成：

| 优先级 | 问题描述                                            | 当前状态          |
|-------|---------------------------------------------------|-----------------|
| P0    | 标签五分类 → 二分类映射                              | ✅ 已修正         |
| P0    | `HFMLFeatureSelectionRecord` 调用不存在的 `select_features()` | ✅ 已修正 |
| P0    | `DriftMonitoringRecord` 调用不存在的 `update_batch()` | ✅ 已修正       |
| P0    | `online_update_job` 参数顺序错误                    | ✅ 已修正         |
| P1    | 特征流水线 `compute_all_transforms` 重复执行         | ✅ 已修正         |
| P1    | `HFMLMultiTimeframeStrategy` 输入标量分数而非 DataFrame | ✅ 已修正     |
| P1    | `HFMLHybridSignal` 未继承 Qlib `Signal` 基类        | ✅ 已修正         |
| P2    | 策略缺乏期货执行语义与风控集成                       | ✅ 已修正         |
| P2    | `workflow_config_15min.yaml` `selector_kwargs` 结构不匹配 | ✅ 已修正  |
| P2    | 缺少 `FeatureGroupImportanceRecord` 等记录器         | ✅ 已修正         |
| —     | 端到端真实 Qlib 环境验证（numpy 依赖缺失环境受限）   | ⏳ 待验证         |
| —     | `integration_summary.md` 修正总结章节               | ⏳ 待补充         |
| —     | `HFMLDataLoader` 未正式继承 Qlib `DataLoader` 抽象类 | ⏳ 低优先级      |
| —     | 多时间框架策略完整多周期 YAML 配置示例              | ⏳ 待补充         |

---

## 贡献指南

### 报告问题

在 GitHub Issues 中提交问题时，请提供：
- 操作系统和 Python 版本
- Qlib 版本（`pip show pyqlib`）
- 最小可复现代码片段
- 完整错误堆栈信息

### 提交 PR

1. Fork 本仓库并创建特性分支：`git checkout -b feature/my-fix`
2. 在 `qlib_ext/` 内完成修改，**不得修改** HFML 原始模块（`features/`、`models/` 等）
3. 在 `test_integration.py` 中添加或更新对应测试
4. 确保 `python test_integration.py` 全部通过
5. 提交 PR，在描述中说明修改的问题编号和测试结果

### 代码风格

- 遵循 [PEP 8](https://pep8.org/) 规范
- 公共方法使用 NumPy 风格 docstring
- 新增自定义 Processor 须实现 `fit()`、`__call__()`、`is_for_infer()` 三个方法
- 新增自定义 Record 须继承 `qlib.workflow.record_temp.ACRecordTemp`，实现 `_generate()` 和 `list()`

### 添加新自定义特征

通过 HFML 的 `FeatureRegistry` 注册，无需修改 `HFMLFeatureProcessor`：

```python
# features/custom_features.py
from features.feature_registry import register_feature

@register_feature(name="my_feature", level="custom", hierarchy="custom")
def compute_my_feature(df, features):
    return (features["close"] / features["vwap"]).rename("my_feature")
```

---

## 许可证

本项目采用 **MIT License**，与 Qlib 保持一致。详见根目录 `LICENSE` 文件。

---

## 致谢

- [Microsoft Qlib](https://github.com/microsoft/qlib) — 开源量化投资平台，提供数据管理、模型训练、回测和记录等标准化工作流。
- HFML 研究团队 — 提供商品期货特征工程、智能标签、混合交易系统等核心研究模块。
- 感谢所有在 `integration_summary.md` 中提出复核意见的贡献者，推动了本适配层的完善。
