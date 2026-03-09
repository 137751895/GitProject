"""
HFML-Qlib 特征处理器模块
========================
HFMLFeatureProcessor   : 计算基础/增强/注册表特征（不含深度变换）。
HFMLTransformProcessor : 在基础特征之上应用 compute_all_transforms 深度变换。
HFMLQualityFilter      : 基于 signal_quality 过滤低质量训练样本。

特征链设计（修正后）
------------------
1. HFMLFeatureProcessor  → 调用各底层特征函数（price/momentum/volume/volatility/
                            OI/candle/regime/microstructure/enhanced/registry），
                            结果存入 ("feature", col) 列；
                            同时保留原始 OHLCV 于 ("raw", col) 列供后续处理器使用。
2. HFMLTransformProcessor → 从 ("feature", ...) 提取基础特征、从 ("raw", ...) 提取
                            OHLCV，调用 compute_all_transforms(base_features, raw_df)
                            生成深度变换特征，追加到 ("feature", col) 列。

这样避免了 compute_all_features() 内部已调用一次 compute_all_transforms
后又在 HFMLTransformProcessor 重复调用的问题。
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具：安全地向 MultiIndex 列 DataFrame 写入列
# ---------------------------------------------------------------------------

def _assign_columns(out: pd.DataFrame, group: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """将 new_df 的所有列以 (group, col_name) 的形式合并到 out 中。

    使用 pd.concat 确保 MultiIndex 列正确保留。
    """
    if new_df is None or new_df.empty:
        return out

    # 用相同的长度截断，避免 data/index 不匹配
    n = min(len(out), len(new_df))
    new_mi = pd.MultiIndex.from_tuples([(group, str(col)) for col in new_df.columns])
    new_df_mi = pd.DataFrame(
        new_df.values[:n],
        index=out.index[:n],
        columns=new_mi,
    )

    # 去掉 out 中已存在的同名列（允许覆盖）
    out_truncated = out.iloc[:n]
    if isinstance(out_truncated.columns, pd.MultiIndex):
        existing = [c for c in out_truncated.columns if c in new_mi]
        if existing:
            out_truncated = out_truncated.drop(columns=existing)

    return pd.concat([out_truncated, new_df_mi], axis=1)


def _build_multiindex_df(groups_data: dict, index) -> pd.DataFrame:
    """从 {group_name: df_or_dict} 构造带 MultiIndex 列的 DataFrame。

    Parameters
    ----------
    groups_data : dict
        {group_name: pd.DataFrame or dict_of_arrays}
    index :
        DataFrame 的行索引。
    """
    col_tuples = []
    arrays = []
    for group, data in groups_data.items():
        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                col_tuples.append((group, str(col)))
                arrays.append(data[col].values)
        elif isinstance(data, dict):
            for col, arr in data.items():
                col_tuples.append((group, str(col)))
                arrays.append(np.asarray(arr))

    if not col_tuples:
        return pd.DataFrame(index=index)

    mi = pd.MultiIndex.from_tuples(col_tuples)
    # 用 np.column_stack + pd.DataFrame(columns=mi) 确保产生真正的 MultiIndex
    arr2d = np.column_stack(arrays)
    return pd.DataFrame(arr2d, index=index, columns=mi)


# ---------------------------------------------------------------------------
# 内部：不含 compute_all_transforms 的基础+增强特征计算
# ---------------------------------------------------------------------------

def _compute_base_features(df: pd.DataFrame, period: str = "5min") -> pd.DataFrame:
    """计算 HFML 基础 + 增强 + 注册表特征，不包含深度变换 (compute_all_transforms)。

    这是对 compute_all_features() 的拆分：只保留非变换部分，让
    HFMLTransformProcessor 在下一步统一调用 compute_all_transforms。

    Parameters
    ----------
    df : pd.DataFrame
        原始 OHLCV K 线，列名含 open/high/low/close/volume/open_interest。
    period : str
        K 线周期。

    Returns
    -------
    pd.DataFrame
        基础 + 增强 + 注册表特征 DataFrame（平铺列，无 MultiIndex）。
    """
    from features.feature_engineering import (
        compute_ma, compute_ema, compute_bollinger_bands,
        compute_price_position, compute_rsi, compute_macd,
        compute_kdj, compute_cci, compute_williams_r, compute_roc,
        compute_volume_features, compute_obv, compute_vwap,
        compute_atr, compute_volatility_features,
        compute_oi_features, compute_candle_features,
    )

    required_cols = {"open", "high", "low", "close", "volume", "open_interest"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"_compute_base_features: 缺少必需列 {missing}")

    o = df["open"]
    h = df["high"]
    low = df["low"]
    c = df["close"]
    v = df["volume"]
    oi = df["open_interest"]

    if period == "1min":
        ma_windows = [5, 10, 20, 60, 120]
        vol_windows = [5, 10, 20, 60]
        oi_windows  = [5, 10, 20, 60]
    elif period == "5min":
        ma_windows = [5, 10, 20, 60]
        vol_windows = [5, 10, 20]
        oi_windows  = [5, 10, 20]
    else:  # 15min
        ma_windows = [5, 10, 20, 40]
        vol_windows = [5, 10, 20]
        oi_windows  = [5, 10, 20]

    features = pd.DataFrame(index=df.index)

    # 价格类
    features = pd.concat([features, compute_ma(c, ma_windows)], axis=1)
    features = pd.concat([features, compute_ema(c, ma_windows)], axis=1)
    features = pd.concat([features, compute_bollinger_bands(c)], axis=1)
    features = pd.concat([features, compute_price_position(c, h, low)], axis=1)

    # 动量类
    features["rsi_14"]    = compute_rsi(c, 14)
    features["rsi_6"]     = compute_rsi(c, 6)
    features = pd.concat([features, compute_macd(c)], axis=1)
    features = pd.concat([features, compute_kdj(h, low, c)], axis=1)
    features["cci"]       = compute_cci(h, low, c)
    features["williams_r"]= compute_williams_r(h, low, c)
    features["roc_12"]    = compute_roc(c, 12)
    features["roc_6"]     = compute_roc(c, 6)

    # 成交量类
    features = pd.concat([features, compute_volume_features(v, vol_windows)], axis=1)
    features["obv"]  = compute_obv(c, v)
    features["vwap"] = compute_vwap(h, low, c, v)

    # 波动率类
    features = pd.concat([features, compute_atr(h, low, c)], axis=1)
    features = pd.concat([features, compute_volatility_features(c, vol_windows)], axis=1)

    # 持仓量类
    features = pd.concat([features, compute_oi_features(oi, v, oi_windows)], axis=1)

    # K 线形态
    features = pd.concat([features, compute_candle_features(o, h, low, c)], axis=1)

    # 市场状态
    try:
        from features.market_regime import MarketRegimeDetector
        regime_features = MarketRegimeDetector().detect_regime(df)
        features = pd.concat([features, regime_features], axis=1)
    except Exception as exc:
        logger.warning(f"market_regime 计算失败，跳过: {exc}")

    # 增强特征（微观结构、高级波动率、缺口衰减）
    try:
        from features.feature_engineering_enhanced import (
            compute_microstructure_features,
            compute_advanced_volatility_features,
            compute_gap_decay_features,
        )
        features = pd.concat([features, compute_microstructure_features(df)], axis=1)
        features = pd.concat([features, compute_advanced_volatility_features(df)], axis=1)
        features = pd.concat([features, compute_gap_decay_features(df)], axis=1)
    except Exception as exc:
        logger.warning(f"feature_engineering_enhanced 计算失败，跳过: {exc}")

    # 注册表自定义特征
    try:
        from features.feature_registry import FeatureRegistry
        import importlib
        try:
            importlib.import_module("features.custom_features")
        except ImportError:
            pass
        registry = FeatureRegistry()
        if registry.get_all_entries():
            custom = registry.compute_registered_features(df, features)
            if len(custom.columns) > 0:
                new_cols = [col for col in custom.columns if col not in features.columns]
                if new_cols:
                    features = pd.concat([features, custom[new_cols]], axis=1)
    except Exception as exc:
        logger.warning(f"feature_registry 计算失败，跳过: {exc}")

    return features


# ---------------------------------------------------------------------------
# HFMLFeatureProcessor
# ---------------------------------------------------------------------------

try:
    from qlib.data.dataset.processor import Processor

    class HFMLFeatureProcessor(Processor):
        """Qlib Processor 实现，计算 HFML 基础 + 增强 + 注册表特征（不含深度变换）。

        特征结果存入 ("feature", col) 列组；
        原始 OHLCV 额外保存到 ("raw", col) 列组，供后续
        HFMLTransformProcessor 正确传递 raw_df 参数。

        Parameters
        ----------
        period : str
            K 线周期，如 "1min"、"5min"、"15min"。
        """

        def __init__(self, period: str = "5min"):
            self.period = period

        def fit(self, df: pd.DataFrame):
            return self

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            raw_df = _extract_raw(df)

            try:
                feature_df = _compute_base_features(raw_df, period=self.period)
            except Exception as exc:
                logger.warning(f"HFMLFeatureProcessor: 基础特征计算失败 ({exc})，返回原始 df")
                return df

            # 从头构建带 MultiIndex 列的输出 DataFrame，避免 tuple-key 赋值的 pandas 歧义
            # ("raw", col)     → 原始 OHLCV，供 HFMLTransformProcessor 使用
            # ("feature", col) → 基础 + 增强 + 注册表特征
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume", "open_interest"]
                          if c in raw_df.columns]
            out = _build_multiindex_df(
                {
                    "raw":     raw_df[ohlcv_cols],
                    "feature": feature_df,
                },
                index=raw_df.index,
            )
            return out.sort_index(axis=1)

        def is_for_infer(self) -> bool:
            return True

    class HFMLTransformProcessor(Processor):
        """Qlib Processor 实现，在基础特征之上应用 compute_all_transforms 深度变换。

        从前序 HFMLFeatureProcessor 的输出中提取：
        - ("feature", ...) 列组 → 基础特征矩阵（作为 compute_all_transforms 的主输入）
        - ("raw", ...) 列组 → 原始 OHLCV（作为 raw_df 参数，确保变换依赖正确来源）

        生成的深度变换特征追加到 ("feature", col) 列组中。

        Parameters
        ----------
        feature_group : str
            基础特征列组名，默认 "feature"。
        raw_group : str
            原始 OHLCV 列组名，默认 "raw"。
        """

        def __init__(self, feature_group: str = "feature", raw_group: str = "raw"):
            self.feature_group = feature_group
            self.raw_group = raw_group

        def fit(self, df: pd.DataFrame):
            return self

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            from features.feature_transforms import compute_all_transforms

            # 提取基础特征（"feature" 组）和原始 OHLCV（"raw" 组）
            if isinstance(df.columns, pd.MultiIndex):
                top_levels = df.columns.get_level_values(0).unique().tolist()
                if self.feature_group in top_levels:
                    feat_df = df.xs(self.feature_group, level=0, axis=1).copy()
                else:
                    feat_df = _extract_raw(df)
                if self.raw_group in top_levels:
                    raw_ohlcv_df = df.xs(self.raw_group, level=0, axis=1).copy()
                else:
                    raw_ohlcv_df = feat_df
            else:
                feat_df = df.copy()
                raw_ohlcv_df = df.copy()

            try:
                transformed = compute_all_transforms(feat_df, raw_df=raw_ohlcv_df)
            except Exception as exc:
                logger.warning(f"HFMLTransformProcessor: compute_all_transforms 失败 ({exc})，跳过变换")
                return df

            if transformed is None or transformed.empty:
                return df

            # 将变换特征追加为 ("feature", col) 列组，使用 pd.concat 避免 tuple-key 赋值歧义
            transform_mi = _build_multiindex_df(
                {"feature": transformed},
                index=df.index,
            )
            # 去除与已有列重名的部分
            if isinstance(df.columns, pd.MultiIndex):
                existing = set(df.columns.tolist())
                new_cols = [c for c in transform_mi.columns if c not in existing]
                transform_mi = transform_mi[new_cols]

            return pd.concat([df, transform_mi], axis=1).sort_index(axis=1)

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
                logger.warning(f"HFMLQualityFilter: 未找到列 {key}，跳过过滤")
                return df

        def is_for_infer(self) -> bool:
            return False

except ImportError:
    # Qlib 未安装时的占位实现（与 Qlib 版本保持相同的 MultiIndex 输出格式）
    class HFMLFeatureProcessor:  # type: ignore[no-redef]
        def __init__(self, period="5min"):
            self.period = period

        def fit(self, df):
            return self

        def __call__(self, df):
            raw_df = _extract_raw(df)
            feature_df = _compute_base_features(raw_df, period=self.period)
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume", "open_interest"]
                          if c in raw_df.columns]
            return _build_multiindex_df(
                {"raw": raw_df[ohlcv_cols], "feature": feature_df},
                index=raw_df.index,
            ).sort_index(axis=1)

        def is_for_infer(self):
            return True

    class HFMLTransformProcessor:  # type: ignore[no-redef]
        def __init__(self, feature_group="feature", raw_group="raw"):
            self.feature_group = feature_group
            self.raw_group = raw_group

        def fit(self, df):
            return self

        def __call__(self, df):
            from features.feature_transforms import compute_all_transforms

            if isinstance(df.columns, pd.MultiIndex):
                top_levels = df.columns.get_level_values(0).unique().tolist()
                feat_df = df.xs(self.feature_group, level=0, axis=1).copy() if self.feature_group in top_levels else _extract_raw(df)
                raw_ohlcv_df = df.xs(self.raw_group, level=0, axis=1).copy() if self.raw_group in top_levels else feat_df
            else:
                feat_df = df.copy()
                raw_ohlcv_df = df.copy()

            try:
                transformed = compute_all_transforms(feat_df, raw_df=raw_ohlcv_df)
            except Exception as exc:
                logger.warning(f"HFMLTransformProcessor(fallback): compute_all_transforms 失败 ({exc})")
                return df

            if transformed is None or transformed.empty:
                return df

            transform_mi = _build_multiindex_df({"feature": transformed}, index=df.index)
            if isinstance(df.columns, pd.MultiIndex):
                existing = set(df.columns.tolist())
                new_cols = [c for c in transform_mi.columns if c not in existing]
                transform_mi = transform_mi[new_cols]
            return pd.concat([df, transform_mi], axis=1).sort_index(axis=1)

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
