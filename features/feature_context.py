# ===== 修改说明 =====
# 依据《当前项目性能提升指南-最终版v2.md》改进点 5.3
# 新增内容已用 # [新增] 标记
# ===================
"""
商品期货机器学习量化模型 - 特征上下文缓存模块
Feature context with cached_property to deduplicate intermediate computations.
"""

from cached_property import cached_property  # [新增]

import numpy as np  # [新增]
import pandas as pd  # [新增]

from features.rolling_numba import rolling_mean  # [新增]


class FeatureContext:  # [新增]
    def __init__(self, df: pd.DataFrame):  # [新增]
        self.df = df  # [新增]

    @cached_property  # [新增]
    def close(self) -> np.ndarray:  # [新增]
        return self.df["close"].values.astype(np.float64)  # [新增]

    @cached_property  # [新增]
    def high(self) -> np.ndarray:  # [新增]
        return self.df["high"].values.astype(np.float64)  # [新增]

    @cached_property  # [新增]
    def low(self) -> np.ndarray:  # [新增]
        return self.df["low"].values.astype(np.float64)  # [新增]

    @cached_property  # [新增]
    def prev_close(self) -> np.ndarray:  # [新增]
        pc = np.roll(self.close, 1)  # [新增]
        pc[0] = np.nan  # [新增]
        return pc  # [新增]

    @cached_property  # [新增]
    def tr(self) -> np.ndarray:  # [新增]
        tr = np.maximum(  # [新增]
            self.high - self.low,  # [新增]
            np.maximum(np.abs(self.high - self.prev_close), np.abs(self.low - self.prev_close)),  # [新增]
        )  # [新增]
        return np.nan_to_num(tr, nan=0.0)  # [新增]

    @cached_property  # [新增]
    def atr_14(self) -> np.ndarray:  # [新增]
        return rolling_mean(self.tr, 14)  # [新增]
