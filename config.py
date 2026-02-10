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
    "future_regime": {
        "description": "未来价格五分位状态 (0=强跌, 1=弱跌, 2=中性, 3=弱涨, 4=强涨)",
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

# 市场状态识别配置
REGIME_CONFIG = {
    "vol_window": 20,
    "trend_window": 20,
    "vol_quantile_high": 0.7,
    "vol_quantile_low": 0.3,
    "trend_threshold": 0.3,
    # 反转检测参数
    "rsi_window": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "ma_deviation_threshold": 0.02,
    "reversal_lookback": 5,
    "trend_accel_threshold": 0.1,
    "reversal_weights": (0.4, 0.3, 0.3),
    # 五分位预测目标的收益率分箱阈值: [强跌 | -0.5% | 弱跌 | -0.1% | 中性 | +0.1% | 弱涨 | +0.5% | 强涨]
    "bins": [float("-inf"), -0.005, -0.001, 0.001, 0.005, float("inf")],
}

# 集成模型配置
ENSEMBLE_CONFIG = {
    "enabled": True,
    "models": ["lightgbm", "xgboost"],
}

# 仓位管理配置
POSITION_CONFIG = {
    "account_risk": 0.02,
    "max_position": 1.0,
    "base_threshold": 0.55,
}

# 预测目标: future_regime 的分类标签
REGIME_LABELS = {
    0: "strong_down",
    1: "weak_down",
    2: "neutral",
    3: "weak_up",
    4: "strong_up",
}

# 分层特征结构 (Feature Hierarchy)
# 将所有特征按层级组织，从基础价格到高阶跨周期特征
# Level 1: 基础价格特征 — 原始OHLCV衍生
# Level 2: 趋势特征 — 移动平均、趋势方向
# Level 3: 动量与波动率特征 — 技术指标振荡器
# Level 4: 微观结构特征 — 资金流向、订单不平衡
# Level 5: 跨周期/高级特征 — 市场状态、跨层级组合
FEATURE_HIERARCHY = {
    "level1_price": {
        "description": "基础价格特征 — OHLCV直接衍生",
        "features": [
            "close_pos", "body_ratio", "upper_shadow_ratio",
            "lower_shadow_ratio", "candle_direction", "amplitude",
            "gap", "gap_ratio", "price_position", "dist_to_high",
            "dist_to_low", "log_return", "range_pct",
        ],
    },
    "level2_trend": {
        "description": "趋势特征 — 均线与布林带",
        "features": [
            "ma_5", "ma_10", "ma_20", "ma_60",
            "ema_5", "ema_10", "ema_20", "ema_60",
            "boll_upper", "boll_mid", "boll_lower",
            "boll_width", "boll_pct_b",
            "trend_strength", "trend_direction", "trend_acceleration",
        ],
    },
    "level3_momentum": {
        "description": "动量与波动率特征 — 技术指标振荡器",
        "features": [
            "rsi_14", "rsi_6", "macd_dif", "macd_dea", "macd_hist",
            "kdj_k", "kdj_d", "kdj_j", "cci", "williams_r",
            "roc_12", "roc_6", "obv", "vwap",
            "tr", "atr", "atr_pct",
            "volatility_5", "volatility_10", "volatility_20",
            "return_ma_5", "return_ma_10", "return_ma_20",
            "vol_ma_5", "vol_ma_10", "vol_ma_20",
            "vol_ratio_5", "vol_ratio_10", "vol_ratio_20",
            "vol_change",
        ],
    },
    "level4_micro": {
        "description": "微观结构特征 — 资金流向与持仓分析",
        "features": [
            "divergence", "vwap_dev", "vol_state", "mom_slope",
            "rsi_slope", "vol_zscore", "gap_decay",
            "oi_change", "oi_change_pct",
            "oi_ma_5", "oi_ma_10", "oi_ma_20",
            "oi_change_ma_5", "oi_change_ma_10", "oi_change_ma_20",
            "vol_oi_ratio",
        ],
    },
    "level5_cross": {
        "description": "跨周期与高级组合特征 — 市场状态与波动率状态",
        "features": [
            "regime_volatility", "volatility_rank", "volatility_change",
            "market_regime", "volatility_regime", "vol_ratio",
            "reversal_score", "regime_duration", "regime_change_prob",
        ],
    },
}

# 增强特征配置
ENHANCED_FEATURE_CONFIG = {
    # 微观结构特征
    "microstructure": {
        "enabled": True,
        "features": [
            "divergence", "vwap_dev", "vol_state", "mom_slope",
            "close_pos", "rsi_slope", "vol_zscore",
        ],
    },
    # 高级波动率特征
    "advanced_volatility": {
        "enabled": True,
        "features": [
            "volatility_regime", "vol_ratio", "atr_pct", "range_pct",
        ],
    },
    # 缺口衰减特征
    "gap_decay": {
        "enabled": True,
        "features": ["gap_decay"],
        "night_session_start_hour": 21,
        "night_session_start_max_minute": 30,
        "decay_constant": 300,
    },
    # Numba加速
    "numba_acceleration": {
        "enabled": True,
        "description": "使用numba加速核心指标计算（rolling_mean, rsi, macd等），比pandas快10-50倍",
    },
}
