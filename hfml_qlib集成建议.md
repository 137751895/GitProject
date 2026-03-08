# HFML 项目接入 Qlib 集成建议

> 当前项目 命名叫 HFML  与 开源项目Qlib （已将源码下载到当前项目目录下的qlib文件夹内）扩展机制整理
> 
> 分析对象：hfml/config.py、hfml/features、hfml/models、hfml/backtest、hfml/scripts、hfml/ml_pipeline.py、hfml/run_real_data_pipeline.py

## 1. 概述

HFML 项目是一个面向商品期货的机器学习交易系统，核心输入是 1 分钟、5 分钟、15 分钟 K 线与持仓量数据，核心输出是预测、标签、交易信号、仓位建议与风险控制结果。项目当前已经具备完整的研究链路：

1. 数据加载：scripts/load_real_data.py 中的 load_kline_csv、load_or_generate。
2. 特征工程：features/feature_engineering.py 中的 compute_all_features。
3. 增强特征：features/feature_engineering_enhanced.py 中的微观结构、高级波动率、缺口衰减特征。
4. 深度变换：features/feature_transforms.py 中的非线性、跨周期、变化率、条件、交互特征。
5. 特征注册表：features/feature_registry.py 中的 FeatureRegistry 与 register_feature。
6. 标签系统：models/smart_labels.py 中的 SmartLabelGenerator。
7. 模型层：models/ml_models.py 中的 LightGBMModel、XGBoostModel、LSTMModel。
8. 模型集成：models/ensemble_model.py 中的 EnsembleModel。
9. 15 分钟预筛选：models/xgb_feature_selector.py 中的 XGBFeatureSelector。
10. 交易系统：models/hybrid_trading.py 中的 HybridTradingSystem。
11. 多时间框架：models/multi_timeframe.py 中的 MultiTimeframeCoordinator。
12. 自适应与风控：models/adaptive_learning.py、models/position_sizing.py、models/var_stress.py。

将 HFML 接入 Qlib 的价值，不是把原有逻辑推倒重写，而是把 HFML 的已有模块挂到 Qlib 的标准骨架上：

1. 数据接入改为 Qlib 的 DataLoader / DataHandlerLP / DatasetH / TSDatasetH 链路。
2. 训练执行改为 qrun 或 task_train 驱动，统一实验记录与复现。
3. 回测改为 Signal + Strategy + Executor + Exchange + PortAnaRecord 链路。
4. 评估改为 SignalRecord、SigAnaRecord、自定义 RecordTemp 统一产出。
5. 在线更新改为 workflow/online 或自定义增量工作流接管。

建议的集成原则只有两句话：

1、HFML 保留算法与业务逻辑，Qlib 接管装配、运行、记录、回测、分析。

2、不能直接在当前项目和 Qlib源码基础上直接修改，qlib会以pip第三方库的方式安装到venv虚拟环境中，当前目录下面的 HFML会与集成重构后的代码分离开成为独立项目。

## 2. 数据接入与特征工程

### 2.1 HFML 数据源与特征总装配现状

HFML 的原始数据入口在 scripts/load_real_data.py：

1. load_kline_csv 读取 CSV。
2. 将 dt 映射为 datetime 索引。
3. 将 close_oi 映射为 open_interest。
4. 输出标准列：open、high、low、close、volume、open_interest。

HFML 的全量特征入口在 features/feature_engineering.py 的 compute_all_features，装配顺序非常清晰：

1. 基础价格特征：MA、EMA、布林带、价格位置。
2. 动量特征：RSI、MACD、KDJ、CCI、Williams %R、ROC。
3. 成交量特征：vol_ma、vol_ratio、obv、vwap。
4. 波动率特征：ATR、volatility、return_ma、log_return。
5. 持仓量特征：oi_change、oi_ma、vol_oi_ratio。
6. K 线形态特征：body_ratio、shadow_ratio、gap。
7. 市场状态特征：MarketRegimeDetector.detect_regime。
8. 增强特征：compute_microstructure_features、compute_advanced_volatility_features、compute_gap_decay_features。
9. 深度特征变换：compute_all_transforms。
10. 注册表特征：FeatureRegistry.compute_registered_features。

这一结构和 Qlib 的最佳实践高度兼容，因为它天然可以分解为：

1. 原始数据加载。
2. 原始特征计算。
3. 训练可学习变换。
4. 推理可复用变换。
5. 数据集切片与时序样本构造。

### 2.2 推荐的 Qlib 集成路径

HFML 的特征链路不建议直接塞进一个超大的 Model.fit。更合理的映射如下：

| HFML 组件   | 当前入口                          | Qlib 扩展点                   | 建议角色        |
| --------- | ----------------------------- | -------------------------- | ----------- |
| CSV K 线加载 | scripts/load_real_data.py     | DataLoader                 | 原始数据读取      |
| 因子批量对齐    | factor_loader.py              | DataLoader 或 Processor     | 外部因子拼接      |
| 基础与增强特征   | compute_all_features          | DataHandlerLP 或 Processor  | 特征主流水线      |
| 注册表机制     | FeatureRegistry               | Handler 内部调度或自定义 Processor | 动态特征发现      |
| 深度变换      | compute_all_transforms        | Processor                  | 训练/推理共享特征变换 |
| 滚动上下文特征   | rolling_numba、feature_context | Processor + TSDatasetH     | 时序依赖特征      |

建议分两层接入：

1. DataLoader 负责把原始期货 K 线与外部因子统一成一个 DataFrame。
2. DataHandlerLP 负责调用 HFML 特征管线，将结果拆成 feature 与 label 两个 group。

### 2.3 推荐实现一：使用 StaticDataLoader 先打通最小链路

如果当前目标是先把 HFML 研究结果挂入 Qlib，而不是一步到位重构数据底座，最快路径是：

1. 离线跑 scripts/load_real_data.py 与 compute_all_features。
2. 把结果保存为 parquet 或 pickle。
3. 用 Qlib 的 StaticDataLoader 读取。
4. 用 DataHandlerLP 组织 infer_processors 与 learn_processors。

这种方式最适合第一阶段验证：

1. 调试成本最低。
2. 最容易检查未来信息泄漏。
3. 最容易和现有 HFML 输出逐行对比。

### 2.4 推荐实现二：自定义 HFMLDataLoader 直接复用现有源码

当你希望把 HFML 特征计算纳入 Qlib 训练链路时，建议直接做一个自定义 DataLoader：

```python
import pandas as pd

from qlib.data.dataset.loader import DataLoader
from scripts.load_real_data import load_kline_csv
from factor_loader import load_factors


class HFMLDataLoader(DataLoader):
    def __init__(
        self,
        csv_path,
        factor_index_pkl=None,
        factors_dir=None,
        factor_list=None,
        start_time=None,
        end_time=None,
    ):
        self.csv_path = csv_path
        self.factor_index_pkl = factor_index_pkl
        self.factors_dir = factors_dir
        self.factor_list = factor_list or []
        self.start_time = start_time
        self.end_time = end_time

    def load(self, instruments=None, start_time=None, end_time=None):
        df = load_kline_csv(self.csv_path)
        st = start_time or self.start_time
        et = end_time or self.end_time
        if st is not None or et is not None:
            df = df.loc[st:et]

        if self.factor_index_pkl and self.factors_dir and self.factor_list:
            factor_df = load_factors(
                index_pkl_path=self.factor_index_pkl,
                factors_dir=self.factors_dir,
                factor_list=self.factor_list,
                start_date=str(df.index.min().date()),
                end_date=str(df.index.max().date()),
            )
            factor_df = factor_df.reset_index().rename(columns={"trade_date": "datetime"})
            factor_df["datetime"] = pd.to_datetime(factor_df["datetime"])
            factor_df = factor_df.set_index("datetime")
            factor_df = factor_df.drop(columns=[c for c in ["ts_code"] if c in factor_df.columns])
            df = df.join(factor_df, how="left")

        return df.sort_index()
```

这里的关键点是：

1. scripts/load_real_data.py 的列标准化逻辑不用重写。
2. factor_loader.py 的外部因子拼接逻辑可以保留。
3. DataLoader 只负责原始数据，不负责模型特征。

### 2.5 推荐实现三：把 HFML 特征链封装为 Processor

HFML 的 compute_all_features 目前是一个总装配函数。接入 Qlib 时，建议把它拆成两个 Processor 层次：

1. HFMLFeatureProcessor：基础特征 + 增强特征 + 注册表特征。
2. HFMLTransformProcessor：深度变换特征。

这样做的原因是：

1. compute_all_features 中的前半部分更像原始特征工程。
2. compute_all_transforms 更像特征后处理。
3. 两层拆开后，方便分别复用、缓存与灰度启停。

示例：

```python
import pandas as pd

from qlib.data.dataset.processor import Processor
from features.feature_engineering import compute_all_features
from features.feature_transforms import compute_all_transforms


class HFMLFeatureProcessor(Processor):
    def __init__(self, period="5min"):
        self.period = period

    def __call__(self, df: pd.DataFrame):
        base_features = compute_all_features(df, period=self.period)
        out = df.copy()
        for col in base_features.columns:
            out[("feature", col)] = base_features[col]
        return out.sort_index(axis=1)


class HFMLTransformProcessor(Processor):
    def __init__(self, raw_feature_group="feature"):
        self.raw_feature_group = raw_feature_group

    def __call__(self, df: pd.DataFrame):
        feat = df[self.raw_feature_group].copy()
        transformed = compute_all_transforms(feat, raw_df=feat)
        out = df.copy()
        for col in transformed.columns:
            out[("feature", col)] = transformed[col]
        return out.sort_index(axis=1)
```

### 2.6 FeatureRegistry 如何映射到 Qlib

HFML 的 FeatureRegistry 是一个装饰器驱动的自动发现系统，注册信息包括：

1. group
2. level
3. description
4. depends_on
5. output_names

这和 Qlib 的关系不是一一替代，而是分层映射：

1. 注册本身保留在 HFML 内部，不需要迁移为 Qlib registry。
2. 注册表产物在 Handler 或 Processor 中被统一调用。
3. group 可映射到 Qlib 的评估分组或自定义 RecordTemp。
4. level 可映射到特征层级分析报表。

最推荐的方案是写一个 HFMLDataHandler，内部直接调用 FeatureRegistry，而不是试图把所有注册特征逐个改写为 Qlib 的 custom_ops。

### 2.7 什么时候使用 custom_ops，什么时候不要用

HFML 的很多特征并不适合硬改成 Qlib custom_ops。

适合改成 custom_ops 的只有这类特征：

1. 单列或少量列滚动表达式。
2. 不需要外部上下文缓存。
3. 不依赖复杂 DataFrame 级联组合。

例如：

1. ma、ema、rsi、roc 这类可表达滚动特征。
2. 少量简单交互特征。

不适合改成 custom_ops 的是：

1. compute_microstructure_features 这类一次依赖多列的批量计算。
2. compute_gap_decay_features 这类依赖 session 时间上下文的逻辑。
3. FeatureRegistry 这种动态发现与依赖检查机制。
4. compute_all_transforms 这种整批后处理逻辑。

因此，HFML 的正确映射不是“全量 custom_ops 化”，而是“少量表达式特征可选 custom_ops，其余继续走 Processor”。

### 2.8 TSDatasetH 在 HFML 特征侧的作用

HFML 中有两类上下文依赖：

1. 特征计算阶段的滚动窗口依赖，例如 rolling_numba、MarketRegimeDetector。
2. 模型输入阶段的时序窗口依赖，例如 15 分钟 LSTM 的 sequence_length。

前者属于 Processor 或 Handler 内的特征构造责任。
后者属于 TSDatasetH 的样本构造责任。

也就是说：

1. 不要把所有滚动逻辑都丢给 TSDatasetH。
2. TSDatasetH 只负责把已经算好的截面表切成 step_len 序列样本。

### 2.9 推荐的 HFMLDataHandler 结构

```python
from qlib.data.dataset.handler import DataHandlerLP


class HFMLDataHandler(DataHandlerLP):
    def __init__(self, data_loader, period="5min", **kwargs):
        super().__init__(data_loader=data_loader, **kwargs)
        self.period = period
```

推荐在 kwargs 中配置：

1. shared_processors：基础列清洗、索引对齐、类型修正。
2. infer_processors：HFMLFeatureProcessor、HFMLTransformProcessor、标准化处理。
3. learn_processors：标签生成、标签过滤、仅训练期可见的样本筛选。

## 3. 智能标签系统

### 3.1 HFML 标签机制分析

HFML 的标签逻辑不只是一个 future_return。models/smart_labels.py 中的 SmartLabelGenerator.create_labels 会同时生成：

1. trading_signal：五级信号，取值为 -2、-1、0、1、2。
2. signal_strength：信号强度。
3. signal_quality：质量评分。
4. expected_return：未来收益率。
5. oi_confirmation：持仓确认方向。

内部逻辑包含四层：

1. 未来收益率 raw_return。
2. 基于 rolling volatility 的 dynamic_threshold。
3. 基于 open_interest 的 oi_signal。
4. 基于成交量均值的 volume_confirmed。

这个设计比 Qlib 常见的单一 label 更丰富，因此接入 Qlib 时建议不要只保留一个标签列，而是保留一个标签组。

### 3.2 建议映射到 Qlib 的方式

最合理的映射方式是：

1. label/trading_signal 作为主监督标签。
2. label/signal_quality 作为样本权重或过滤依据。
3. label/expected_return 作为回归评估辅助列。
4. label/oi_confirmation 作为诊断列。

如果任务是分类：

1. 可以先将 trading_signal 转成三分类或五分类。
2. 也可以单独把 strong 与 weak 折叠为方向二分类。

如果任务是回归：

1. 使用 expected_return 作为主标签。
2. trading_signal 和 signal_quality 留作分析与策略过滤。

### 3.3 为什么标签应该进入 learn_processors

SmartLabelGenerator 明确依赖未来价格：

1. future_price = close.shift(-horizon)
2. raw_return = future_price / close - 1

这决定了它绝不能进入 infer_processors。

在 Qlib 中建议这样处理：

1. 把 SmartLabelGenerator 包装成 Processor。
2. 放进 learn_processors。
3. 显式令 is_for_infer 返回 False。

示例：

```python
import pandas as pd

from qlib.data.dataset.processor import Processor
from models.smart_labels import SmartLabelGenerator


class HFMLSmartLabelProcessor(Processor):
    def __init__(self, period="5min", config=None):
        config = config or {}
        self.period = period
        self.generator = SmartLabelGenerator(**config)

    def __call__(self, df: pd.DataFrame):
        labels = self.generator.create_labels(df)
        out = df.copy()
        for col in labels.columns:
            out[("label", col)] = labels[col]
        return out.sort_index(axis=1)

    def is_for_infer(self):
        return False
```

### 3.4 标签过滤与样本加权建议

HFML 的 signal_quality 非常适合变成训练样本过滤器或权重列。建议两种方式二选一：

1. 硬过滤：只保留 signal_quality 大于阈值的样本。
2. 软加权：把 signal_quality 作为 sample_weight 传给模型。

在 Qlib 中，第一种更容易先落地。示例：

```python
class HFMLQualityFilter(Processor):
    def __init__(self, min_quality=0.55):
        self.min_quality = min_quality

    def __call__(self, df):
        quality_key = ("label", "signal_quality")
        return df[df[quality_key] >= self.min_quality]

    def is_for_infer(self):
        return False
```

### 3.5 配置示例

```yaml
learn_processors:
  - class: HFMLSmartLabelProcessor
    module_path: hfml.qlib_ext.processors
    kwargs:
      period: 5min
      config:
        horizon: 3
        base_threshold: 0.001
        strong_multiplier: 2.0
        vol_window: 20
        volume_window: 60
        quality_weights: [0.4, 0.3, 0.2, 0.1]
  - class: HFMLQualityFilter
    module_path: hfml.qlib_ext.processors
    kwargs:
      min_quality: 0.55
```

## 4. 深度特征变换模型

### 4.1 HFML 模型分工现状

HFML 在 config.py 与 models/ml_models.py 中明确规定：

1. 1 分钟使用 LightGBMModel。
2. 5 分钟使用 XGBoostModel。
3. 15 分钟使用 LSTMModel。

同时，15 分钟链路前面还增加了 models/xgb_feature_selector.py 中的 XGBFeatureSelector，用于 Top N 预筛选后再送入 LSTM。

这和 Qlib 的最佳接入方式也很清晰：

1. 1 分钟、5 分钟尽量使用 Qlib 现成树模型封装。
2. 15 分钟单独使用 TSDatasetH + 自定义 Model 或 GeneralPTNN。

### 4.2 1 分钟与 5 分钟模型的接入建议

HFML 的 LightGBMModel 和 XGBoostModel 是轻量包装，核心逻辑并不复杂。因此接入 Qlib 时，优先级建议如下：

1. 若只需要训练与预测，直接改用 Qlib 的 LightGBM / XGBoost contrib 模型。
2. 若必须保留 HFML 的 evaluate、get_feature_importance 接口风格，可以写一个极薄包装器适配 Qlib Model 协议。

推荐做法是：

1. 数据与标签保留 HFML 逻辑。
2. 模型训练改走 Qlib 现有封装。
3. 把 HFML 的 EnsembleModel 作为第二阶段可选扩展，而不是第一阶段阻塞项。

### 4.3 15 分钟 LSTM 的接入建议

HFML 的 LSTMModel 有两个关键特征：

1. 通过 _create_sequences 按 sequence_length 构造序列。
2. 用 StandardScaler 做数值标准化。

这和 Qlib 的 TSDatasetH 非常匹配。推荐改造方向：

1. 不再在模型内部自己切序列。
2. 由 TSDatasetH 统一构造 step_len 窗口。
3. 模型只接收形如 batch_size × step_len × d_feat 的张量。

### 4.4 两种可选实现

#### 方案 A：GeneralPTNN + 自定义 nn.Module

如果希望最大化复用 Qlib 训练循环，建议把 HFML 的 TensorFlow LSTMModel 改成 PyTorch 网络，挂到 GeneralPTNN。

优点：

1. 更贴近 Qlib 现有深度模型生态。
2. 训练、保存、预测、记录更统一。
3. 与 TSDatasetH 的契合度最好。

#### 方案 B：保留现有 LSTMModel，外包一层 Qlib Model

如果希望最少改动，可直接包装现有 HFML LSTMModel：

```python
import pandas as pd

from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandlerLP
from models.ml_models import LSTMModel


class HFMLLSTMQlibModel(Model):
    def __init__(self, params=None):
        self.inner = LSTMModel(task="classification", params=params)

    def fit(self, dataset, reweighter=None):
        train_df, valid_df = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        X_train = train_df["feature"]
        y_train = train_df["label"]["trading_signal"]
        X_valid = valid_df["feature"]
        y_valid = valid_df["label"]["trading_signal"]
        self.inner.train(X_train, y_train, X_valid, y_valid)

    def predict(self, dataset, segment="test"):
        test_df = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        X_test = test_df["feature"]
        proba = self.inner.predict_proba(X_test)
        index = X_test.index[self.inner.params["sequence_length"] :]
        return pd.Series(proba[:, 1], index=index, name="score")
```

这个方案能快速接入，但有一个明显问题：

模型内部还在重复做序列切片，不够“Qlib 原生”。因此它适合作为迁移过渡版本，不建议作为最终形态。

### 4.5 15 分钟预筛选如何接入 Qlib

HFML 的 XGBFeatureSelector 做了三件事：

1. fit_select：时间序列交叉验证计算平均特征重要性。
2. get_feature_groups_importance：按特征组聚合重要性。
3. dynamic_feature_count：动态搜索最佳特征数。

在 Qlib 中推荐两种落地方式：

1. 训练前离线预处理：最稳，最容易排查。
2. 自定义 RecordTemp：训练前或训练后输出特征筛选报告。

我的建议是：

1. 第一阶段把 XGBFeatureSelector 放到训练前离线流水线中。
2. 选中的特征列表写入一个 artifact，例如 selected_features.json。
3. DataHandler 或 Dataset 初始化时读取该列表。

如果一定要把它放进 Qlib 工作流内，最合适的位置不是 Processor，而是一个自定义任务前置步骤或自定义 RecordTemp。原因是：

1. 它本质上是模型辅助选择，不是纯数据清洗。
2. 它依赖 y。
3. 它还会产生报告和动态搜索结果。

### 4.6 推荐配置

```yaml
task:
  model:
    class: HFMLLSTMQlibModel
    module_path: hfml.qlib_ext.models
    kwargs:
      params:
        sequence_length: 20
        hidden_units: 64
        dropout_rate: 0.2
        epochs: 50
        batch_size: 32
        learning_rate: 0.001
  dataset:
    class: TSDatasetH
    module_path: qlib.data.dataset
    kwargs:
      step_len: 20
      handler:
        class: HFMLDataHandler
        module_path: hfml.qlib_ext.data
```

## 5. 混合智能交易系统

### 5.1 HFML 交易系统分析

models/hybrid_trading.py 的 HybridTradingSystem 是一个典型的三层结构：

1. 专家层：TrendFollowingExpert、MeanReversionExpert、BreakoutExpert、VolumePriceExpert、OIConfirmationExpert。
2. 融合层：_ensemble_fusion，按 confidence 加权投票。
3. 风控层：_risk_adjustment，根据 volatility、Kelly fraction、agreement、signal_threshold 决定 final_signal 与 position_size。

HybridTradingSystem.predict 的输出已经接近 Qlib Strategy 所需信息：

1. final_signal
2. position_size
3. confidence
4. fused_signal
5. agreement
6. experts_breakdown

### 5.2 推荐映射到 Qlib 的方式

最推荐的映射方式不是把五个专家分别做成五个 Qlib Strategy，而是：

1. 若专家只依赖行情特征，将它们整合到一个自定义 Signal 中。
2. 若专家还依赖模型预测或账户状态，将它们整合到一个自定义 Strategy 中。

建议按下列角色拆分：

| HFML 模块                     | Qlib 角色                           |
| --------------------------- | --------------------------------- |
| 五个专家 generate_signal        | Signal 内部子逻辑                      |
| HybridTradingSystem.predict | Strategy 内部决策器                    |
| _ensemble_fusion            | Signal 融合或 Strategy 融合            |
| _risk_adjustment            | generate_trade_decision 内的仓位与方向控制 |

### 5.3 推荐实现：HFMLHybridSignal + HFMLHybridStrategy

```python
from qlib.backtest.signal import Signal
from qlib.strategy.base import BaseStrategy

from models.hybrid_trading import HybridTradingSystem


class HFMLHybridSignal(Signal):
    def __init__(self, window=100, **kwargs):
        self.window = window
        self.system = HybridTradingSystem(**kwargs)
        self.market_data = None

    def set_market_data(self, market_data):
        self.market_data = market_data

    def get_signal(self, start_time, end_time):
        df = self.market_data.loc[:end_time].tail(self.window)
        result = self.system.predict(df)
        return result


class HFMLHybridStrategy(BaseStrategy):
    def __init__(self, signal, instrument, trade_exchange=None, **kwargs):
        super().__init__(trade_exchange=trade_exchange, **kwargs)
        self.signal = signal
        self.instrument = instrument

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        start_time, end_time = self.trade_calendar.get_step_time(trade_step, shift=0)
        sig = self.signal.get_signal(start_time, end_time)
        # 依据 sig["final_signal"] 与 sig["position_size"] 生成订单
        return self._build_trade_decision(sig)
```

### 5.4 如果专家依赖模型输出，如何组织

用户提出的场景很重要：某些专家可能不只看 K 线，还要看模型输出。此时建议做法是：

1. 先训练多个模型，分别产出各自的 pred.pkl。
2. 在自定义 Signal 或 Strategy 中读取多个 recorder 的预测结果。
3. 让五个专家把预测值作为一个附加输入，而不是直接耦合到训练脚本里。

推荐结构：

1. trend 专家用 15 分钟模型输出。
2. entry 专家用 5 分钟模型输出。
3. execution 专家用 1 分钟模型输出。
4. 量价、持仓专家继续直接读行情特征。

### 5.5 最终信号如何变成 Qlib 交易决策

HFML 的 final_signal 是做多、做空、观望三态，position_size 是建议仓位比例。映射到 Qlib 时需要进一步落成：

1. 目标头寸。
2. 订单方向。
3. 订单数量。
4. 手续费、滑点、最小变动价位、交易单位规则。

因此，真正的交易决策生成应该在 Strategy 里，而不是只停留在 Signal 层。

下面给一个更接近可落地实现的 `_build_trade_decision` 模板。这里假设：

1. `sig` 是一个字典或 Series，至少包含 `instrument`、`final_signal`、`position_size`。
2. `final_signal` 取值为 `1/-1/0`，分别表示做多、做空、观望。
3. `position_size` 是目标仓位比例，例如 `0.3` 表示使用 30% 可用资金。
4. 当前策略以单合约单时刻决策为例，多标的时只需在外层循环即可。

```python
import copy
from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.strategy.base import BaseStrategy


class HFMLHybridStrategy(BaseStrategy):
  def _build_trade_decision(self, sig):
    if sig is None:
      return TradeDecisionWO([], self)

    trade_step = self.trade_calendar.get_trade_step()
    trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

    instrument = sig["instrument"]
    final_signal = int(sig.get("final_signal", 0))
    target_ratio = float(max(0.0, min(1.0, sig.get("position_size", 0.0))))

    current_pos = copy.deepcopy(self.trade_position)
    current_amount = current_pos.get_stock_amount(instrument)
    cash = current_pos.get_cash()

    deal_price = self.trade_exchange.get_deal_price(
      stock_id=instrument,
      start_time=trade_start_time,
      end_time=trade_end_time,
      direction=Order.BUY if final_signal >= 0 else Order.SELL,
    )
    if deal_price is None or deal_price <= 0:
      return TradeDecisionWO([], self)

    contract_factor = self.trade_exchange.get_factor(
      stock_id=instrument,
      start_time=trade_start_time,
      end_time=trade_end_time,
    )

    target_value = cash * target_ratio
    target_amount = target_value / deal_price
    target_amount = self.trade_exchange.round_amount_by_trade_unit(target_amount, contract_factor)

    order_list = []

    if final_signal > 0:
      delta_amount = max(0.0, target_amount - current_amount)
      if delta_amount > 0:
        order_list.append(
          Order(
            stock_id=instrument,
            amount=delta_amount,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=Order.BUY,
          )
        )
    elif final_signal < 0:
      sell_amount = current_amount if current_amount > 0 else target_amount
      sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, contract_factor)
      if sell_amount > 0:
        order_list.append(
          Order(
            stock_id=instrument,
            amount=sell_amount,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=Order.SELL,
          )
        )
    else:
      if current_amount > 0:
        order_list.append(
          Order(
            stock_id=instrument,
            amount=current_amount,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=Order.SELL,
          )
        )

    executable_orders = [order for order in order_list if self.trade_exchange.check_order(order)]
    return TradeDecisionWO(executable_orders, self)
```

这段代码的核心意思是：

1. Signal 只表达方向和建议仓位。
2. Strategy 读取当前持仓、可用现金、交易价格、合约乘数、最小交易单位。
3. 最后再把目标头寸差额变成 `Order` 列表并封装成 `TradeDecisionWO`。

如果你的期货场景允许显式做空，那么通常不应直接复用股票式的 `current_amount -> 全卖出` 逻辑，而应继续扩展为：

1. 区分多头平仓、空头开仓。
2. 区分主力合约切换时的移仓单。
3. 在 `Exchange` 或更细粒度执行层处理中金所、商品所不同的手续费和保证金规则。

## 6. 多时间框架协同

### 6.1 HFML 的现有机制

models/multi_timeframe.py 中的 MultiTimeframeCoordinator 采用标准的三级协同：

1. TrendDirectionModel：15 分钟定方向。
2. EntryTimingModel：5 分钟找时机。
3. EntryPriceOptimizer：1 分钟做精确入场。

generate_signals 的输出包括：

1. final_signal
2. signal_grade
3. signal_strength
4. trend_direction
5. trend_confidence
6. entry_signal
7. entry_strength
8. entry_score
9. optimal_entry
10. agreement_score

这套设计本质上不是回测执行器问题，而是“多模型预测结果按时间对齐后在策略层做层级过滤”的问题。

### 6.2 在 Qlib 中的推荐实现

推荐方案是：

1. 分别训练 15 分钟、5 分钟、1 分钟三个任务。
2. 每个任务单独生成 pred.pkl。
3. 在自定义 Strategy 中按时间戳对齐这三个预测序列。
4. 复现 MultiTimeframeCoordinator.generate_signals 的过滤与评级逻辑。

这是最直接也最稳的实现，因为：

1. Qlib 的训练任务天然是单任务、单数据频率更清晰。
2. 多频协调属于交易策略层，而不是模型训练层。
3. 各周期可以独立回测与诊断。

如果三路预测来自不同实验或不同 recorder，推荐在 Strategy 初始化时显式加载，而不是在每个 bar 内反复查 recorder。典型写法如下：

```python
import pandas as pd
from qlib.workflow import R


def load_pred_from_recorder(recorder_id, artifact_name="pred.pkl"):
  rec = R.get_recorder(recorder_id=recorder_id)
  pred = rec.load_object(artifact_name)
  if isinstance(pred, pd.Series):
    pred = pred.to_frame("score")
  if not isinstance(pred.index, pd.MultiIndex):
    raise ValueError("HFML 多周期预测建议统一为 datetime/instrument 的 MultiIndex")
  pred = pred.sort_index()
  return pred
```

若某个实验把预测保存到子目录，例如 `prediction/pred.pkl`，则把 `artifact_name` 改成对应相对路径即可。Qlib 的 `Recorder.load_object` 支持这种形式。

### 6.3 是否使用 NestedExecutor

NestedExecutor 可以帮助你做多层执行粒度控制，但它不负责多周期预测信号融合。对于 HFML 这种 15 分钟到 5 分钟到 1 分钟的层级过滤场景：

1. 多周期预测协调应放在 Strategy。
2. NestedExecutor 更适合用在“信号已定，执行细节再分层”的场景。

因此建议是：

1. 第一阶段不用 NestedExecutor 实现多周期信号逻辑。
2. 若后续要把 5 分钟信号拆成 1 分钟执行调度，再考虑 NestedExecutor。

### 6.4 推荐实现结构

```python
import pandas as pd
from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.strategy.base import BaseStrategy
from qlib.workflow import R


def _normalize_pred_frame(pred, score_col="score"):
  if isinstance(pred, pd.Series):
    pred = pred.to_frame(score_col)
  pred = pred.sort_index()
  if not isinstance(pred.index, pd.MultiIndex):
    raise ValueError("prediction index must be MultiIndex(datetime, instrument)")
  return pred


class HFMLMultiTimeframeStrategy(BaseStrategy):
  def __init__(
    self,
    pred_15m=None,
    pred_5m=None,
    pred_1m=None,
    recorder_id_15m=None,
    recorder_id_5m=None,
    recorder_id_1m=None,
    coordinator_kwargs=None,
    **kwargs,
  ):
        super().__init__(**kwargs)
        self.coordinator = MultiTimeframeCoordinator(**(coordinator_kwargs or {}))
    self.pred_15m = self._init_pred(pred_15m, recorder_id_15m)
    self.pred_5m = self._init_pred(pred_5m, recorder_id_5m)
    self.pred_1m = self._init_pred(pred_1m, recorder_id_1m)

  def _init_pred(self, pred, recorder_id):
    if pred is not None:
      return _normalize_pred_frame(pred)
    if recorder_id is None:
      raise ValueError("pred and recorder_id cannot both be None")
    rec = R.get_recorder(recorder_id=recorder_id)
    return _normalize_pred_frame(rec.load_object("pred.pkl"))

  def _slice_one(self, pred_df, dt, instrument):
    try:
      return pred_df.loc[(dt, instrument)]
    except KeyError:
      return None

    def generate_trade_decision(self, execute_result=None):
    trade_step = self.trade_calendar.get_trade_step()
    trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
    instrument = self.trade_position.get_stock_list()[0]

    pred_15m = self._slice_one(self.pred_15m, trade_start_time, instrument)
    pred_5m = self._slice_one(self.pred_5m, trade_start_time, instrument)
    pred_1m = self._slice_one(self.pred_1m, trade_start_time, instrument)

    if pred_15m is None or pred_5m is None or pred_1m is None:
      return TradeDecisionWO([], self)

    fused = self.coordinator.generate_signals(
      trend_pred=pred_15m,
      entry_pred=pred_5m,
      execution_pred=pred_1m,
    )

    final_signal = fused.get("final_signal", 0)
    if final_signal == 0:
      return TradeDecisionWO([], self)

    amount = max(0.0, float(fused.get("position_size", 0.0)))
    order = Order(
      stock_id=instrument,
      amount=amount,
      start_time=trade_start_time,
      end_time=trade_end_time,
      direction=Order.BUY if final_signal > 0 else Order.SELL,
    )
    return TradeDecisionWO([order], self)
```

这个模板强调两个要点：

1. 预测结果在 `__init__` 时一次性从 recorder 读取到内存，避免回测时重复 IO。
2. Strategy 内部统一约束预测索引格式为 `(datetime, instrument)`，否则多周期对齐会非常脆弱。

如果你的主时钟不是 15 分钟整点，而是“最近一个可用 15 分钟预测 + 最近一个可用 5 分钟确认 + 当前 1 分钟执行信号”，则 `_slice_one` 应改成 `asof` 风格查找，而不是严格等时刻索引。

### 6.5 工程建议

建议把 MultiTimeframeCoordinator 保留为一个纯业务组件，不直接改写其内部逻辑。Qlib 侧只负责：

1. 给它喂对齐后的三路输入。
2. 接收它的输出。
3. 把输出映射为交易订单。

这样 HFML 与 Qlib 的边界最清晰。

## 7. 自适应学习与风险管理

### 7.1 HFML 现有模块分析

HFML 的自适应与风控模块非常完整：

1. models/adaptive_learning.py
   - ConceptDriftDetector
   - OnlineLearner
   - AdaptiveThresholdManager
   - AdaptiveModelManager
2. models/position_sizing.py
   - calculate_position_size
   - adaptive_threshold
   - DrawdownTracker
   - DailyRiskLimit
   - RiskBudgetManager
3. models/var_stress.py
   - RiskModels
   - calculate_comprehensive_var
   - perform_stress_testing

这些模块的共同特点是：

1. 强依赖实时预测结果。
2. 强依赖账户权益、回撤、日内状态。
3. 一部分属于模型更新管理，一部分属于交易执行前风控。

### 7.2 在 Qlib 中应该放在哪一层

建议明确分层：

1. ConceptDriftDetector、OnlineLearner、AdaptiveModelManager 放到训练后与在线更新工作流。
2. DrawdownTracker、DailyRiskLimit、RiskBudgetManager、RiskModels 放到 Strategy 或 Executor 的交易决策阶段。

### 7.3 概念漂移与在线学习的接入建议

这部分不建议塞进 Model.fit。推荐路径：

1. 训练阶段：正常 qrun 训练，生成 recorder。
2. 评估阶段：自定义 RecordTemp，记录最近窗口准确率衰减、漂移级别、阈值变化建议。
3. 在线阶段：使用 workflow/online 或单独批处理脚本调用 AdaptiveModelManager.step。

更具体一点：

1. 概念漂移检测结果可以作为 recorder metrics。
2. 增量更新历史可以作为 artifact 保存。
3. needs_retrain 可以作为在线调度器的触发标志。

更具体地说，有两种接入方式。

第一种是接入 Qlib online 工作流，把 HFML 的自适应更新视为在线 routine 的一部分：

```python
from qlib.model.trainer import TrainerR
from qlib.workflow.online.manager import OnlineManager


online_manager = OnlineManager(
  strategies=[online_strategy],
  trainer=TrainerR(),
  begin_time="2024-01-02",
  freq="day",
)

online_manager.first_train()
online_manager.routine(cur_time="2024-01-03")
online_manager.routine(cur_time="2024-01-04")
```

这种模式适合以下场景：

1. 你的 HFML 更新周期与 Qlib online 的任务生产节奏一致。
2. 模型更新后仍希望走标准 recorder、signal、collector 体系。
3. 需要回放历史 routine，验证“哪些日子触发了漂移更新”。

第二种通常更实用，即把 `AdaptiveModelManager.step` 单独做成定时任务，和 qrun 训练解耦：

```python
import qlib
from qlib.workflow import R
from hfml.models.adaptive_learning import AdaptiveModelManager


def online_update_job(provider_uri, region, experiment_name, recorder_id, latest_batch):
  qlib.init(provider_uri=provider_uri, region=region)

  rec = R.get_recorder(experiment_name=experiment_name, recorder_id=recorder_id)
  model = rec.load_object("trained_model")
  adaptive_state = rec.load_object("adaptive_state.pkl")

  manager = AdaptiveModelManager(**adaptive_state["manager_kwargs"])
  manager.model = model
  manager.threshold_manager = adaptive_state["threshold_manager"]
  manager.drift_detector = adaptive_state["drift_detector"]

  result = manager.step(latest_batch["X"], latest_batch["y"])

  rec.log_metrics(
    drift_level=float(result.get("drift_level", 0.0)),
    adaptive_threshold=float(result.get("threshold", 0.0)),
    needs_retrain=int(bool(result.get("needs_retrain", False))),
  )
  rec.save_objects(
    **{
      "trained_model": manager.model,
      "adaptive_state.pkl": {
        "manager_kwargs": adaptive_state["manager_kwargs"],
        "threshold_manager": manager.threshold_manager,
        "drift_detector": manager.drift_detector,
        "last_result": result,
      },
      "adaptive_step_result.pkl": result,
    }
  )
  return result
```

这个定时任务可以由 Airflow、cron、Windows Task Scheduler 或你现有的行情采集主进程触发。更推荐这种方式的原因是：

1. HFML 的在线学习往往依赖更细粒度的最新样本，不一定和 qrun 周期一致。
2. 漂移检测失败、更新失败、回滚重试都更容易单独治理。
3. `needs_retrain` 为真时，可以再触发完整 qrun 或新的 online routine，而不是强行在一次 routine 里做所有事情。

### 7.4 风控建议放在 generate_trade_decision 中

用户要求里这一点是正确的。HFML 风控逻辑最适合放在 Strategy.generate_trade_decision 中，因为它依赖：

1. 当前信号强度。
2. 当前波动率。
3. 当前权益与回撤。
4. 当日已交易次数。
5. VaR 是否超限。

这些信息都更接近回测时态，而不是训练时态。

推荐伪代码：

```python
def generate_trade_decision(self, execute_result=None):
    raw_signal = self.signal.get_signal(...)
    risk_view = self.risk_budget.compute_signals(
        probabilities=raw_signal["probabilities"],
        volatility=raw_signal["volatility"],
    )
    var_result = self.risk_models.calculate_comprehensive_var(...)
    if var_result and var_result[0].var_value > self.var_gate:
        # 缩仓或禁开新仓
        pass
    # 再根据 drawdown、daily limit、position usage 决定最终订单
```

### 7.5 与 Qlib Account / Position / Exchange 的关系

HFML 风控并不需要替换 Qlib 的 Account 和 Position，而是应建立在它们之上：

1. Account 维护资金与权益。
2. Position 维护持仓。
3. Strategy 根据这些状态调用 HFML 风控模块。
4. Exchange 负责期货交易规则与成交成本。

换句话说：

1. Qlib 的 Account/Position 是账本。
2. HFML 的 RiskBudgetManager / RiskModels 是决策约束器。

## 8. 回测与评估

### 8.1 HFML 当前评估现状

HFML 当前已经在 ml_pipeline.py、run_real_data_pipeline.py、models/position_sizing.py 中实现了：

1. accuracy、precision、recall、f1。
2. profit_factor。
3. calmar_ratio。
4. win_rate。
5. avg_win、avg_loss。
6. max_drawdown。
7. sharpe_ratio。

此外，XGBFeatureSelector 与 FeatureSelector 还提供：

1. 特征重要性。
2. 稳定性。
3. 特征组重要性。
4. 交叉验证成功率。

### 8.2 如何复用 Qlib 的标准记录器

HFML 接入 Qlib 后，标准记录器建议至少保留三类：

1. SignalRecord：统一产出 pred.pkl 与 label.pkl。
2. SigAnaRecord：统一输出 IC、Rank IC、long-short 等标准信号分析。
3. PortAnaRecord：统一跑组合回测与风险分析。

这里的关键要求只有一个：

模型预测输出必须是 Qlib 认可的索引格式，通常为 datetime 与 instrument 组成的 MultiIndex。

### 8.3 HFML 需要额外补充的自定义 RecordTemp

HFML 有几类分析是 Qlib 默认没有的，应写成自定义 RecordTemp：

1. FeatureGroupImportanceRecord
   - 调用 XGBFeatureSelector.get_feature_groups_importance。
2. FeatureStabilityRecord
   - 调用 FeatureSelector.stability_selection 或 ExtraTrees 稳定性分析。
3. DriftMonitoringRecord
   - 记录 AdaptiveModelManager 的 drift_level、threshold、needs_retrain。
4. HybridSignalBreakdownRecord
   - 记录五个专家的信号分解、agreement、fused_signal。
5. MultiTimeframeGradeRecord
   - 记录 A、B、C 等级信号数量、收益贡献、过滤率。

### 8.4 示例

```python
import logging
import pandas as pd
from qlib.log import get_module_logger
from qlib.workflow.record_temp import ACRecordTemp, SignalRecord

from hfml.backtest.feature_selector import FeatureSelector
from hfml.models.xgb_feature_selector import XGBFeatureSelector


class HFMLFeatureSelectionRecord(ACRecordTemp):
    artifact_path = "hfml_feature_selection"
    depend_cls = SignalRecord

  def __init__(
    self,
    recorder,
    selector_type="xgb",
    feature_col="feature",
    label_col="label",
    selector_kwargs=None,
    skip_existing=False,
  ):
    super().__init__(recorder=recorder, skip_existing=skip_existing)
    self.selector_type = selector_type
    self.feature_col = feature_col
    self.label_col = label_col
    self.selector_kwargs = selector_kwargs or {}
    self.logger = get_module_logger(self.__class__.__name__, level=logging.INFO)

  def _load_train_frame(self):
    dataset = self.load("dataset")
    if dataset is None:
      raise ValueError("dataset artifact is required for HFMLFeatureSelectionRecord")

    train_frame = dataset.prepare("train", col_set=[self.feature_col, self.label_col])
    if isinstance(train_frame, tuple):
      features, labels = train_frame
    else:
      features = train_frame[self.feature_col]
      labels = train_frame[self.label_col]
    return features, labels

  def _create_selector(self):
    if self.selector_type == "xgb":
      return XGBFeatureSelector(**self.selector_kwargs)
    if self.selector_type == "classic":
      return FeatureSelector(**self.selector_kwargs)
    raise ValueError(f"unsupported selector_type: {self.selector_type}")

    def _generate(self, **kwargs):
    params = self.recorder.load_object("params.pkl")
    features, labels = self._load_train_frame()
    selector = self._create_selector()

    if isinstance(labels, pd.DataFrame):
      labels = labels.iloc[:, 0]

    self.logger.info("start HFML feature selection record generation")

    selected_features = selector.select_features(features, labels)

    artifacts = {
      "selected_features.pkl": selected_features,
      "feature_columns.pkl": list(features.columns),
      "selector_params.pkl": params,
    }
    metrics = {
      "feature_count_before": int(features.shape[1]),
      "feature_count_after": int(len(selected_features)),
    }

    if hasattr(selector, "get_feature_importance"):
      feature_importance = selector.get_feature_importance()
      artifacts["feature_importance.pkl"] = feature_importance

    if hasattr(selector, "get_feature_groups_importance"):
      group_importance = selector.get_feature_groups_importance()
      artifacts["feature_group_importance.pkl"] = group_importance

    if hasattr(selector, "stability_selection"):
      stability_report = selector.stability_selection(features, labels)
      artifacts["feature_stability.pkl"] = stability_report

    self.recorder.log_metrics(**metrics)
    return artifacts

  def list(self):
    return [
      "selected_features.pkl",
      "feature_columns.pkl",
      "selector_params.pkl",
      "feature_importance.pkl",
      "feature_group_importance.pkl",
      "feature_stability.pkl",
    ]
```

这个模板有几个关键点：

1. 它依赖训练实验已经把 `dataset` 和 `params.pkl` 保存到 recorder。
2. 特征筛选只对训练段执行，避免把验证集、测试集信息泄漏进来。
3. 结果同时保存为 artifact 和 metrics，便于后续 YAML 工作流或分析脚本复用。

如果你的训练脚本当前还没有保存 `dataset`，建议在训练 recorder 中补上：

```python
R.save_objects(dataset=dataset, **{"params.pkl": params})
```

### 8.5 期货场景的 PortAnaRecord 注意事项

Qlib 默认很多示例偏股票，因此接入 HFML 时需要明确修改：

1. benchmark 不一定可用。
2. trade_unit 不能沿用股票默认值。
3. limit_threshold 不适合很多期货场景。
4. 多空交易与换月逻辑应在 Strategy 或 Exchange 中重写。

## 9. 配置驱动集成示例

下面给出一个面向 HFML 的完整 Qlib YAML 示例。这个示例目标是：

1. 使用 HFML 原始数据加载。
2. 使用 HFML 特征与智能标签。
3. 15 分钟采用时序数据集。
4. 使用自定义 Strategy 融合混合专家与多时间框架逻辑。
5. 复用 Qlib 的标准记录器，并追加 HFML 自定义记录器。

```yaml
sys:
  rel_path:
    - ../

qlib_init:
  provider_uri: ~/.qlib/qlib_data/cn_data
  region: cn

task:
  model:
    class: HFMLLSTMQlibModel
    module_path: hfml.qlib_ext.models
    kwargs:
      params:
        sequence_length: 20
        hidden_units: 64
        dropout_rate: 0.2
        epochs: 50
        batch_size: 32
        learning_rate: 0.001
        patience: 10

  dataset:
    class: TSDatasetH
    module_path: qlib.data.dataset
    kwargs:
      step_len: 20
      segments:
        train: [2022-01-01, 2023-06-30]
        valid: [2023-07-01, 2023-12-31]
        test: [2024-01-01, 2024-12-31]
      handler:
        class: HFMLDataHandler
        module_path: hfml.qlib_ext.data
        kwargs:
          start_time: 2022-01-01
          end_time: 2024-12-31
          fit_start_time: 2022-01-01
          fit_end_time: 2023-12-31
          data_loader:
            class: HFMLDataLoader
            module_path: hfml.qlib_ext.data
            kwargs:
              csv_path: data/klines/KQi@SHFEag/KQi@SHFEag_15min.csv
              factor_index_pkl: data/factors/index.pkl
              factors_dir: data/factors
              factor_list: ["$alpha1", "$alpha2"]
          infer_processors:
            - class: HFMLFeatureProcessor
              module_path: hfml.qlib_ext.processors
              kwargs:
                period: 15min
            - class: HFMLTransformProcessor
              module_path: hfml.qlib_ext.processors
            - class: RobustZScoreNorm
              module_path: qlib.data.dataset.processor
              kwargs:
                fields_group: feature
            - class: Fillna
              module_path: qlib.data.dataset.processor
              kwargs:
                fields_group: feature
          learn_processors:
            - class: HFMLSmartLabelProcessor
              module_path: hfml.qlib_ext.processors
              kwargs:
                period: 15min
                config:
                  horizon: 2
                  base_threshold: 0.001
                  strong_multiplier: 2.0
                  vol_window: 20
                  volume_window: 60
                  quality_weights: [0.4, 0.3, 0.2, 0.1]
            - class: HFMLQualityFilter
              module_path: hfml.qlib_ext.processors
              kwargs:
                min_quality: 0.55

  record:
    - class: SignalRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        model: <MODEL>
        dataset: <DATASET>

    - class: SigAnaRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        model: <MODEL>
        dataset: <DATASET>

    - class: HFMLFeatureSelectionRecord
      module_path: hfml.qlib_ext.records

    - class: HFMLDriftMonitoringRecord
      module_path: hfml.qlib_ext.records

    - class: PortAnaRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        config:
          strategy:
            class: HFMLMultiTimeframeStrategy
            module_path: hfml.qlib_ext.strategy
            kwargs:
              instrument: KQi@SHFEag
              pred_15m: <PRED>
              pred_5m_recorder: hfml_5m_latest
              pred_1m_recorder: hfml_1m_latest
              coordinator_kwargs:
                neutral_zone: 0.10
                entry_threshold: 0.60
                optimization_threshold: 0.55
                min_grade: C
              hybrid_kwargs:
                signal_threshold: 0.3
                max_position: 0.2
                target_win_rate: 0.7
                target_win_loss_ratio: 2.0
              risk_kwargs:
                account_risk: 0.02
                max_position: 1.0
                base_threshold: 0.55
                max_drawdown: 0.15
                drawdown_warning: 0.08
                max_daily_loss: 0.03
                max_daily_trades: 20
              var_kwargs:
                var99_gate: 0.02
                position_scale_on_breach: 0.5

          executor:
            class: SimulatorExecutor
            module_path: qlib.backtest.executor
            kwargs:
              time_per_step: 15min
              generate_portfolio_metrics: true

          backtest:
            start_time: 2024-01-01
            end_time: 2024-12-31
            account: 10000000
            benchmark: null
            exchange_kwargs:
              freq: 15min
              deal_price: close
              open_cost: 0.0003
              close_cost: 0.0003
              min_cost: 0
              trade_unit: 1
              limit_threshold: null
```

## 10. 潜在问题与注意事项

### 10.1 期货与股票差异

HFML 是商品期货项目，接入 Qlib 时必须显式处理以下差异：

1. 多空都是一等公民，不能默认只做多。
2. 交易单位、最小价格变动、手续费、保证金与股票不同。
3. benchmark 常常没有直接可比对象。
4. 换月与连续合约处理不在 Qlib 默认股票范式里。
5. 涨跌停与可成交性规则需按期货市场重写。

因此，真正需要期货化改造的地方主要是：

1. DataLoader
2. Strategy
3. Exchange 参数，必要时是自定义 Exchange

### 10.2 未来信息泄漏

HFML 中以下逻辑天然带未来信息：

1. SmartLabelGenerator 的 future_price。
2. XGBFeatureSelector 的有监督筛选。
3. 任意基于 y 的样本过滤。

在 Qlib 里要遵守三个规则：

1. 所有标签生成与标签过滤都放 learn_processors。
2. Processor.fit 只允许使用训练期数据。
3. 特征筛选列表若由全样本生成，会导致严重泄漏，必须按训练窗口单独生成。

### 10.3 自定义类路径与 sys.path

Qlib 通过 class 与 module_path 动态实例化对象。HFML 接入时经常踩坑的点是：

1. module_path 写成文件路径而不是包路径。
2. rel_path 没把 hfml 根目录加入搜索路径。
3. 自定义类模块 import 时依赖相对路径错误。

建议做法：

1. 给 hfml 增加一层专门的 qlib_ext 包，集中放适配器代码。
2. 所有 Qlib 配置只引用 qlib_ext 下的类。
3. 不直接在 YAML 里引用过多原始研究脚本。

推荐目录结构如下：

```text
hfml/
  qlib_ext/
   __init__.py
   data.py
   processors.py
   labels.py
   models.py
   signals.py
   strategies.py
   records.py
   online.py
   utils.py
```

可以按下面的职责拆分：

1. `data.py`
   - `HFMLDataLoader`
   - `HFMLDataHandler`
2. `processors.py`
   - `HFMLFeatureProcessor`
   - `HFMLTransformProcessor`
   - `HFMLQualityFilter`
3. `labels.py`
   - `HFMLSmartLabelProcessor`
4. `models.py`
   - `HFMLLSTMQlibModel`
   - 其他对 HFML 原始模型的 Qlib 包装器
5. `signals.py`
   - `HFMLHybridSignal`
   - 多模型融合辅助函数
6. `strategies.py`
   - `HFMLHybridStrategy`
   - `HFMLMultiTimeframeStrategy`
7. `records.py`
   - `HFMLFeatureSelectionRecord`
   - `DriftMonitoringRecord`
   - `HybridSignalBreakdownRecord`
8. `online.py`
   - `online_update_job`
   - `AdaptiveModelManager` 的 Qlib workflow 封装

对应 YAML 中就尽量只出现这一类包路径，例如：

```yaml
sys:
  rel_path:
   - ../

task:
  model:
   class: HFMLLSTMQlibModel
   module_path: hfml.qlib_ext.models

  record:
   - class: HFMLFeatureSelectionRecord
    module_path: hfml.qlib_ext.records
```

这样做的直接收益是：

1. HFML 原始研究代码与 Qlib 适配层解耦。
2. 后续替换底层模型实现时，YAML 不需要大幅改动。
3. `sys.rel_path` 只需要稳定指向项目根目录，而不需要为每个脚本单独补 import hack。

### 10.4 模型输出对齐问题

HFML 的 LSTMModel 因为内部序列切片，会导致预测长度比输入短。这在 Qlib 中必须显式处理，否则 PortAnaRecord 会出现 pred 与 label 不对齐。推荐方案：

1. 用 TSDatasetH 解决大部分窗口问题。
2. predict 返回时显式裁掉前 step_len 个索引。
3. 在自定义 RecordTemp 中加一层索引校验。

### 10.5 多时间框架信号对齐问题

15 分钟、5 分钟、1 分钟预测天然不在同一频率。不要简单 inner join 后直接用，需要先定义交易主时钟：

1. 若以 15 分钟为主交易时钟，则 5 分钟和 1 分钟用于窗口内确认。
2. 若以 5 分钟为主交易时钟，则 15 分钟只作为上层过滤器。

这一点应在 Strategy 中固定下来，不能让不同脚本各自定义。

### 10.6 建议的落地顺序

如果要最稳地把 HFML 接到 Qlib，建议按下面顺序推进：

1. 先实现 HFMLDataLoader，把真实 K 线与外部因子接入。
2. 再实现 HFMLFeatureProcessor 与 HFMLSmartLabelProcessor。
3. 用 Qlib 的树模型先跑通 1 分钟和 5 分钟任务。
4. 再接 15 分钟 TSDatasetH + LSTM。
5. 再把 HybridTradingSystem 与 MultiTimeframeCoordinator 包到 Strategy。
6. 最后接 DriftMonitoringRecord、FeatureSelectionRecord、在线更新流程。

这个顺序的好处是：

1. 每一步都能独立验证。
2. 数据、标签、模型、策略、风控不会混在同一个调试回合里。
3. 便于逐步替换 HFML 原脚本，而不是一次性大迁移。

## 11. 结论

HFML 与 Qlib 的结构兼容度很高，原因在于 HFML 本身已经是“模块化量化链路”，而 Qlib 恰好提供的是“标准化装配骨架”。从源码角度看，最合理的集成边界是：

1. HFML 负责算法与业务逻辑。
2. Qlib 负责对象装配、配置驱动、实验管理、回测、分析、在线工作流。

对本项目来说，最关键的五个映射关系是：

1. scripts/load_real_data.py、factor_loader.py 映射到 DataLoader。
2. compute_all_features、FeatureRegistry、compute_all_transforms 映射到 DataHandlerLP + Processor。
3. SmartLabelGenerator 映射到 learn_processors 中的标签处理器。
4. LSTMModel、XGBFeatureSelector、多周期模型链路映射到 TSDatasetH + Model + 多任务训练记录。
5. HybridTradingSystem、MultiTimeframeCoordinator、RiskBudgetManager、RiskModels 映射到 Signal + Strategy + 自定义 RecordTemp。

如果按本文结构实施，开发者不需要把 HFML 改造成另一个框架，只需要为它补一层 Qlib 适配层，就可以把已有商品期货逻辑无缝纳入 Qlib 的标准工作流。


