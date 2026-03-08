"""
HFML-Qlib 辅助函数模块
======================
提供跨模块共用的辅助函数，包括：
- 预测结果加载与对齐
- MultiIndex 规范化
- 日期范围工具
"""

import logging
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 预测结果处理
# ---------------------------------------------------------------------------

def load_pred_from_recorder(
    recorder_id: str,
    artifact_name: str = "pred.pkl",
    score_col: str = "score",
) -> pd.DataFrame:
    """从 Qlib Recorder 加载预测结果并规范化为 MultiIndex DataFrame。

    Parameters
    ----------
    recorder_id : str
        实验记录 ID。
    artifact_name : str
        artifact 文件名，默认 "pred.pkl"。
    score_col : str
        预测分数列名，默认 "score"。

    Returns
    -------
    pd.DataFrame
        带 MultiIndex(datetime, instrument) 的 DataFrame。
    """
    from qlib.workflow import R

    rec = R.get_recorder(recorder_id=recorder_id)
    pred = rec.load_object(artifact_name)
    return normalize_pred(pred, score_col=score_col)


def normalize_pred(
    pred: Union[pd.Series, pd.DataFrame],
    score_col: str = "score",
) -> pd.DataFrame:
    """将预测对象规范化为带 MultiIndex 的 DataFrame。

    Parameters
    ----------
    pred : pd.Series or pd.DataFrame
        预测结果。
    score_col : str
        若 pred 为 Series，使用此名称作为列名。

    Returns
    -------
    pd.DataFrame
        排序后的 MultiIndex DataFrame。
    """
    if isinstance(pred, pd.Series):
        pred = pred.to_frame(score_col)
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"不支持的预测类型: {type(pred)}")
    if not isinstance(pred.index, pd.MultiIndex):
        raise ValueError(
            "预测结果索引必须为 MultiIndex(datetime, instrument)，"
            "请检查模型 predict 方法的输出格式"
        )
    return pred.sort_index()


def align_predictions(
    pred_primary: pd.DataFrame,
    pred_secondary: pd.DataFrame,
    method: str = "ffill",
) -> pd.DataFrame:
    """将辅助预测对齐到主预测的时间戳。

    Parameters
    ----------
    pred_primary : pd.DataFrame
        主预测 DataFrame（以其时间戳为准）。
    pred_secondary : pd.DataFrame
        辅助预测 DataFrame（将被对齐）。
    method : str
        填充方法，"ffill"（向前填充）或 "bfill"（向后填充）。

    Returns
    -------
    pd.DataFrame
        对齐后的辅助预测 DataFrame。
    """
    if pred_primary.empty or pred_secondary.empty:
        return pred_secondary

    primary_idx = pred_primary.index.get_level_values(0).unique()
    secondary_idx = pred_secondary.index.get_level_values(0).unique()

    # 找出辅助预测中的标的列表
    instruments = pred_secondary.index.get_level_values(1).unique()

    aligned_parts = []
    for inst in instruments:
        try:
            sub = pred_secondary.xs(inst, level=1)
        except KeyError:
            continue
        sub_aligned = sub.reindex(primary_idx, method=method)
        sub_aligned.index = pd.MultiIndex.from_product(
            [[ts for ts in primary_idx], [inst]],
            names=pred_secondary.index.names,
        )
        aligned_parts.append(sub_aligned)

    if not aligned_parts:
        return pred_secondary

    return pd.concat(aligned_parts).sort_index()


# ---------------------------------------------------------------------------
# MultiIndex 辅助
# ---------------------------------------------------------------------------

def slice_pred_at(
    pred_df: pd.DataFrame,
    dt,
    instrument: str,
    fallback_ffill: bool = True,
) -> Optional[float]:
    """从 MultiIndex 预测 DataFrame 取特定时刻与标的的预测分数。

    Parameters
    ----------
    pred_df : pd.DataFrame
        带 MultiIndex(datetime, instrument) 的预测 DataFrame。
    dt : datetime-like
        目标时刻。
    instrument : str
        目标标的代码。
    fallback_ffill : bool
        若精确匹配失败，使用向前填充查找最近可用值。

    Returns
    -------
    float or None
        预测分数，若找不到则返回 None。
    """
    try:
        row = pred_df.loc[(dt, instrument)]
        val = row.iloc[0] if isinstance(row, pd.Series) else float(row)
        return float(val)
    except KeyError:
        pass

    if not fallback_ffill:
        return None

    try:
        sub = pred_df.xs(instrument, level=1)
        loc = sub.index.get_indexer([dt], method="ffill")[0]
        if loc >= 0:
            return float(sub.iloc[loc, 0])
    except Exception:
        pass

    return None


def build_multiindex_pred(
    scores: "ArrayLike",
    datetimes,
    instrument: str,
    name: str = "score",
) -> pd.Series:
    """将分数数组与时间序列构造为带 MultiIndex 的 pd.Series。

    Parameters
    ----------
    scores : array-like
        预测分数数组。
    datetimes : array-like
        对应时间戳序列。
    instrument : str
        标的代码。
    name : str
        Series 名称。

    Returns
    -------
    pd.Series
        带 MultiIndex(datetime, instrument) 的预测序列。
    """
    import numpy as np

    scores = np.asarray(scores)
    datetimes = pd.DatetimeIndex(datetimes)
    index = pd.MultiIndex.from_arrays(
        [datetimes, [instrument] * len(datetimes)],
        names=["datetime", "instrument"],
    )
    return pd.Series(scores, index=index, name=name)


# ---------------------------------------------------------------------------
# 日期范围工具
# ---------------------------------------------------------------------------

def infer_segments(df: pd.DataFrame, train_ratio: float = 0.7,
                   valid_ratio: float = 0.15) -> dict:
    """根据 DataFrame 的时间范围自动推断 train/valid/test 分段。

    Parameters
    ----------
    df : pd.DataFrame
        时间索引 DataFrame。
    train_ratio : float
        训练集比例，默认 0.7。
    valid_ratio : float
        验证集比例，默认 0.15。

    Returns
    -------
    dict
        含 "train", "valid", "test" 键的时间段字典，每项为 (start, end) 元组。
    """
    times = df.index if not isinstance(df.index, pd.MultiIndex) else df.index.get_level_values(0)
    times = pd.DatetimeIndex(sorted(times.unique()))

    n = len(times)
    if n < 3:
        return {
            "train": (str(times[0].date()), str(times[-1].date())),
            "valid": (str(times[0].date()), str(times[-1].date())),
            "test":  (str(times[0].date()), str(times[-1].date())),
        }

    train_end_idx = max(0, int(n * train_ratio) - 1)
    valid_end_idx = max(train_end_idx + 1, int(n * (train_ratio + valid_ratio)) - 1)
    valid_end_idx = min(valid_end_idx, n - 2)  # 留至少一个 test 样本

    return {
        "train": (str(times[0].date()), str(times[train_end_idx].date())),
        "valid": (str(times[train_end_idx + 1].date()),
                  str(times[valid_end_idx].date())),
        "test": (str(times[valid_end_idx + 1].date()),
                 str(times[-1].date())),
    }
