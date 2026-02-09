"""
商品期货机器学习量化模型 - 配置文件
Configuration for commodity futures ML quantitative model.
"""

# K线周期配置
PERIODS = ["1min", "5min", "15min"]

# 各周期推荐的机器学习框架
# 1分钟: LightGBM - 数据量大，需要高速训练，LightGBM速度快且内存占用低
# 5分钟: XGBoost - 数据量适中，XGBoost精度高且稳定
# 15分钟: LSTM (Keras) - 数据量较少但时序特征更重要，LSTM擅长捕捉长期依赖关系
ML_FRAMEWORKS = {
    "1min": "lightgbm",
    "5min": "xgboost",
    "15min": "lstm",
}

# 预测目标配置
# 推荐的预测目标特征：
#   - future_return: 未来N根K线的收益率（最常用）
#   - future_direction: 未来价格方向（涨/跌，分类任务）
#   - future_volatility: 未来波动率
PREDICTION_TARGETS = {
    "future_return": {
        "description": "未来N根K线的收益率",
        "horizon": {"1min": 5, "5min": 3, "15min": 2},
    },
    "future_direction": {
        "description": "未来价格方向 (1=涨, 0=跌)",
        "horizon": {"1min": 5, "5min": 3, "15min": 2},
    },
}

# 默认预测目标
DEFAULT_TARGET = "future_direction"

# 回测配置
BACKTEST_CONFIG = {
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "test_ratio": 0.15,
    "n_splits": 5,  # 时间序列交叉验证折数
    "min_success_rate": 0.55,  # 最低成功率阈值
}

# LSTM模型配置
LSTM_CONFIG = {
    "sequence_length": 20,
    "hidden_units": 64,
    "dropout_rate": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
}

# LightGBM配置
LGBM_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# XGBoost配置
XGB_CONFIG = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}
