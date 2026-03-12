"""
HFML-Qlib 标签处理器模块
========================
HFMLSmartLabelProcessor : 将 SmartLabelGenerator 包装为 Qlib learn_processor。

标签列（均放在 "label" 列组）：
    trading_signal  - 五级信号 (-2/-1/0/+1/+2)
    signal_strength - 信号强度 (0~2)
    signal_quality  - 质量评分 (0~1)
    expected_return - 预期收益率
    oi_confirmation - 持仓确认方向
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


try:
    from qlib.data.dataset.processor import Processor

    class HFMLSmartLabelProcessor(Processor):
        """Qlib Processor 实现，将 HFML SmartLabelGenerator 结果写入 label 列组。

        此处理器仅用于 learn_processors（is_for_infer 返回 False），
        因为标签生成依赖未来价格数据，不能在推理阶段使用。

        Parameters
        ----------
        period : str
            K 线周期，用于确定默认预测 horizon。
        config : dict, optional
            传递给 SmartLabelGenerator 的参数。支持字段：
            horizon, base_threshold, strong_multiplier, vol_window,
            oi_window, volume_window, quality_weights。
        """

        # 各周期默认 horizon（K 线数）
        DEFAULT_HORIZON = {"1min": 5, "5min": 3, "15min": 2}

        def __init__(self, period: str = "5min", config: dict = None):
            self.period = period
            config = config or {}
            if "horizon" not in config:
                config["horizon"] = self.DEFAULT_HORIZON.get(period, 3)
            from models.smart_labels import SmartLabelGenerator
            self.generator = SmartLabelGenerator(**config)

        def fit(self, df: pd.DataFrame):
            return self

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            """生成多维度标签并附加到 df 的 "label" 列组中。

            Parameters
            ----------
            df : pd.DataFrame
                至少包含 close 列（及可选的 open_interest、volume 列）的 DataFrame。

            Returns
            -------
            pd.DataFrame
                带 MultiIndex 列的 DataFrame，新增列组 "label"。
            """
            # 提取原始 OHLCV 数据供标签生成使用
            raw_df = _extract_raw(df)

            try:
                labels = self.generator.create_labels(raw_df)
            except Exception as exc:
                logger.warning(f"HFMLSmartLabelProcessor: create_labels 失败 ({exc})，跳过标签生成")
                return df

            out = df.copy()
            for col in labels.columns:
                out[("label", col)] = labels[col].values
            return out.sort_index(axis=1)

        def is_for_infer(self) -> bool:
            """标签处理器不参与推理流水线。"""
            return False

except ImportError:
    class HFMLSmartLabelProcessor:  # type: ignore[no-redef]
        """占位实现（Qlib 未安装）。"""

        DEFAULT_HORIZON = {"1min": 5, "5min": 3, "15min": 2}

        def __init__(self, period="5min", config=None):
            self.period = period
            config = config or {}
            if "horizon" not in config:
                config["horizon"] = self.DEFAULT_HORIZON.get(period, 3)
            from models.smart_labels import SmartLabelGenerator
            self.generator = SmartLabelGenerator(**config)

        def fit(self, df):
            return self

        def __call__(self, df):
            raw_df = _extract_raw(df)
            labels = self.generator.create_labels(raw_df)
            out = df.copy()
            for col in labels.columns:
                out[("label", col)] = labels[col].values
            return out

        def is_for_infer(self):
            return False


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _extract_raw(df: pd.DataFrame) -> pd.DataFrame:
    """从可能带 MultiIndex 列的 DataFrame 提取原始 OHLCV 层。"""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    ohlcv_cols = ["open", "high", "low", "close", "volume", "open_interest"]
    top_levels = df.columns.get_level_values(0).unique().tolist()
    flat_cols = {}
    for col in ohlcv_cols:
        for top in top_levels:
            if (top, col) in df.columns:
                flat_cols[col] = df[(top, col)]
                break
        if col not in flat_cols and col in df.columns:
            flat_cols[col] = df[col]
    if flat_cols:
        return pd.DataFrame(flat_cols, index=df.index)
    flat = df.copy()
    flat.columns = [f"{a}_{b}" if a else str(b) for a, b in flat.columns]
    return flat
