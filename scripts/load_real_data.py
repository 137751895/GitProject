"""
商品期货机器学习量化模型 - 真实数据加载模块
Load real commodity futures kline CSV data for the hfml pipeline.

支持加载白银(ag)等品种的 1min/5min/15min K线数据，
将CSV列名映射为 hfml 标准列名，并进行基本校验。

使用方法:
    from scripts.load_real_data import load_kline_csv
    df = load_kline_csv("data/klines/KQi@SHFEag/KQi@SHFEag_5min.csv")
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认数据路径模板
DEFAULT_DATA_DIR = os.path.join("data", "klines", "KQi@SHFEag")
FILENAME_TEMPLATE = "KQi@SHFEag_{period}.csv"


def load_kline_csv(file_path: str) -> pd.DataFrame:
    """读取K线CSV文件，返回符合hfml标准的DataFrame。

    CSV列映射:
        dt -> datetime (索引)
        open -> open
        high -> high
        low -> low
        close -> close
        volume -> volume
        close_oi -> open_interest
        open_oi -> 丢弃

    Parameters
    ----------
    file_path : str
        CSV文件路径

    Returns
    -------
    pd.DataFrame
        标准化后的K线数据，列: open, high, low, close, volume, open_interest
        索引: datetime

    Raises
    ------
    FileNotFoundError
        文件不存在
    ValueError
        缺少必要列
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    logger.info(f"加载数据: {file_path}")
    df = pd.read_csv(file_path)

    # 检查必要列
    required_cols = {"dt", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")

    # 时间列处理
    df["datetime"] = pd.to_datetime(df["dt"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # 持仓量映射: close_oi -> open_interest
    if "close_oi" in df.columns:
        df["open_interest"] = df["close_oi"].astype(float)
    elif "open_interest" not in df.columns:
        logger.warning("CSV中没有 close_oi 或 open_interest 列，使用默认值50000")
        df["open_interest"] = 50000.0

    # 保留标准列
    standard_cols = ["open", "high", "low", "close", "volume", "open_interest"]
    df = df[standard_cols].copy()

    # 数据类型
    for col in standard_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除全空行
    df.dropna(subset=["close"], inplace=True)

    # 确保持仓量为正
    df["open_interest"] = df["open_interest"].clip(lower=1.0)

    logger.info(f"  数据形状: {df.shape}, 时间范围: {df.index[0]} ~ {df.index[-1]}")
    return df


def get_default_data_path(period: str) -> str:
    """获取默认数据文件路径。

    Parameters
    ----------
    period : str
        K线周期: 1min, 5min, 15min

    Returns
    -------
    str
        文件路径
    """
    filename = FILENAME_TEMPLATE.format(period=period)
    return os.path.join(DEFAULT_DATA_DIR, filename)


def load_or_generate(period: str, n_rows: int = 5000) -> pd.DataFrame:
    """加载真实数据，若不存在则回退到生成模拟数据。

    Parameters
    ----------
    period : str
        K线周期
    n_rows : int
        模拟数据行数（仅在回退时使用）

    Returns
    -------
    tuple
        (df, is_real) — DataFrame 和 是否为真实数据的标志
    """
    data_path = get_default_data_path(period)
    if os.path.exists(data_path):
        logger.info(f"检测到真实数据文件: {data_path}")
        df = load_kline_csv(data_path)
        return df, True
    else:
        logger.warning(f"未找到真实数据: {data_path}，使用模拟数据 (n_rows={n_rows})")
        from ml_pipeline import generate_sample_data
        df = generate_sample_data(n_rows=n_rows, period=period)
        return df, False
