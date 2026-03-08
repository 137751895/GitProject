"""
HFML-Qlib 适配层
================
将 HFML 商品期货机器学习项目接入 Qlib 标准工作流的适配器包。

目录结构:
    data.py       - HFMLDataLoader, HFMLDataHandler
    processors.py - HFMLFeatureProcessor, HFMLTransformProcessor, HFMLQualityFilter
    labels.py     - HFMLSmartLabelProcessor
    models.py     - HFMLLSTMQlibModel
    signals.py    - HFMLHybridSignal
    strategies.py - HFMLHybridStrategy, HFMLMultiTimeframeStrategy
    records.py    - HFMLFeatureSelectionRecord, DriftMonitoringRecord
    online.py     - online_update_job
    utils.py      - 辅助函数
"""
