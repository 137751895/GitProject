"""
HFML-Qlib 特征处理器模块
========================
HFMLFeatureProcessor   : 调用 compute_all_features 生成基础与增强特征。
HFMLTransformProcessor : 调用 compute_all_transforms 生成深度变换特征。
HFMLQualityFilter      : 基于 signal_quality 过滤低质量训练样本。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具：安全地向 MultiIndex 列 DataFrame 写入列
# ---------------------------------------------------------------------------

def _assign_columns(out: pd.DataFrame, group: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """将 new_df 的所有列以 (group, col_name) 的形式写入 out。"""
    for col in new_df.columns:
        out[(group, col)] = new_df[col].values
    return out


# ---------------------------------------------------------------------------
# HFMLFeatureProcessor
# ---------------------------------------------------------------------------

try:
    from qlib.data.dataset.processor import Processor

    class HFMLFeatureProcessor(Processor):
        """Qlib Processor 实现，封装 HFML 的全量特征工程入口。

        调用顺序与 compute_all_features 一致：
        1. 基础价格/动量/成交量/波动率/持仓量/K 线形态特征
        2. 市场状态特征
        3. 增强特征（微观结构、高级波动率、缺口衰减）
        4. 注册表特征

        Parameters
        ----------
        period : str
            K 线周期，如 "1min"、"5min"、"15min"。
        """

        def __init__(self, period: str = "5min"):
            self.period = period

        def fit(self, df: pd.DataFrame):
            """无需拟合，特征计算为无状态变换。"""
            return self

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            """计算 HFML 全量特征并附加到 df 中。

            Parameters
            ----------
            df : pd.DataFrame
                来自 DataLoader 的原始 K 线 DataFrame。
                索引为 datetime，列为 open/high/low/close/volume/open_interest 等。

            Returns
            -------
            pd.DataFrame
                带 MultiIndex 列的 DataFrame，特征列组名为 "feature"。
            """
            from features.feature_engineering import compute_all_features

            # 兼容 MultiIndex 列：若 df 已带列组，提取原始 OHLCV 层
            raw_df = _extract_raw(df)

            try:
                feature_df = compute_all_features(raw_df, period=self.period)
            except Exception as exc:
                logger.warning(f"HFMLFeatureProcessor: compute_all_features 失败 ({exc})，返回原始 df")
                return df

            out = df.copy()
            out = _assign_columns(out, "feature", feature_df)
            return out.sort_index(axis=1)

        def is_for_infer(self) -> bool:
            return True

    class HFMLTransformProcessor(Processor):
        """Qlib Processor 实现，封装 HFML 的深度特征变换入口。

        在 HFMLFeatureProcessor 之后运行，对已计算的基础特征做：
        1. 非线性变换（平方、对数、百分位排名、Z-score）
        2. 跨周期比率
        3. 变化率与加速度
        4. 条件特征
        5. 特征交互项

        Parameters
        ----------
        raw_feature_group : str
            基础特征列组名，默认 "feature"。
        """

        def __init__(self, raw_feature_group: str = "feature"):
            self.raw_feature_group = raw_feature_group

        def fit(self, df: pd.DataFrame):
            return self

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            from features.feature_transforms import compute_all_transforms

            # 提取基础特征层
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    feat_df = df[self.raw_feature_group].copy()
                except KeyError:
                    feat_df = _extract_raw(df)
            else:
                feat_df = df.copy()

            try:
                transformed = compute_all_transforms(feat_df, raw_df=feat_df)
            except Exception as exc:
                logger.warning(f"HFMLTransformProcessor: compute_all_transforms 失败 ({exc})，跳过变换")
                return df

            out = df.copy()
            out = _assign_columns(out, "feature", transformed)
            return out.sort_index(axis=1)

        def is_for_infer(self) -> bool:
            return True

    class HFMLQualityFilter(Processor):
        """Qlib Processor 实现，过滤信号质量低于阈值的训练样本。

        仅在 learn_processors 中使用（is_for_infer 返回 False）。

        Parameters
        ----------
        min_quality : float
            最低信号质量评分阈值（0~1）。默认 0.55。
        quality_key : tuple
            signal_quality 列的 MultiIndex 键，默认 ("label", "signal_quality")。
        """

        def __init__(self, min_quality: float = 0.55,
                     quality_key: tuple = ("label", "signal_quality")):
            self.min_quality = min_quality
            self.quality_key = quality_key

        def fit(self, df: pd.DataFrame):
            return self

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            key = self.quality_key
            if key in df.columns:
                mask = df[key] >= self.min_quality
                filtered = df[mask]
                logger.info(
                    f"HFMLQualityFilter: 过滤前 {len(df)} 行，过滤后 {len(filtered)} 行"
                    f"（min_quality={self.min_quality}）"
                )
                return filtered
            else:
                logger.warning(
                    f"HFMLQualityFilter: 未找到列 {key}，跳过过滤"
                )
                return df

        def is_for_infer(self) -> bool:
            return False

except ImportError:
    # Qlib 未安装时的占位实现
    class HFMLFeatureProcessor:  # type: ignore[no-redef]
        def __init__(self, period="5min"):
            self.period = period

        def fit(self, df):
            return self

        def __call__(self, df):
            from features.feature_engineering import compute_all_features
            raw_df = _extract_raw(df)
            feature_df = compute_all_features(raw_df, period=self.period)
            return feature_df

        def is_for_infer(self):
            return True

    class HFMLTransformProcessor:  # type: ignore[no-redef]
        def __init__(self, raw_feature_group="feature"):
            self.raw_feature_group = raw_feature_group

        def fit(self, df):
            return self

        def __call__(self, df):
            from features.feature_transforms import compute_all_transforms
            return compute_all_transforms(df, raw_df=df)

        def is_for_infer(self):
            return True

    class HFMLQualityFilter:  # type: ignore[no-redef]
        def __init__(self, min_quality=0.55, quality_key=("label", "signal_quality")):
            self.min_quality = min_quality
            self.quality_key = quality_key

        def fit(self, df):
            return self

        def __call__(self, df):
            key = self.quality_key
            if key in df.columns:
                return df[df[key] >= self.min_quality]
            return df

        def is_for_infer(self):
            return False


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _extract_raw(df: pd.DataFrame) -> pd.DataFrame:
    """从可能带 MultiIndex 列的 DataFrame 中提取原始 OHLCV 层。"""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    # 尝试提取顶层为空字符串或 "raw" 的列
    top_levels = df.columns.get_level_values(0).unique().tolist()
    # 优先取 OHLCV 基础列
    ohlcv_cols = ["open", "high", "low", "close", "volume", "open_interest"]
    flat_cols = {}
    for col in ohlcv_cols:
        for top in top_levels:
            if (top, col) in df.columns:
                flat_cols[col] = df[(top, col)]
                break
            elif col in df.columns:
                flat_cols[col] = df[col]
                break
    if flat_cols:
        return pd.DataFrame(flat_cols, index=df.index)
    # 回退：拍平所有列
    flat = df.copy()
    flat.columns = [
        f"{a}_{b}" if a else str(b) for a, b in flat.columns
    ]
    return flat
