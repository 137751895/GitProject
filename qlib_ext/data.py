"""
HFML-Qlib 数据接入模块
======================
HFMLDataLoader: 封装 load_kline_csv 与 load_factors，作为 Qlib DataLoader。
HFMLDataHandler: 继承 DataHandlerLP，组织 shared/infer/learn processors。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HFMLDataLoader
# ---------------------------------------------------------------------------

class HFMLDataLoader:
    """Qlib DataLoader 实现，封装 HFML 的 K 线加载与外部因子拼接逻辑。

    Parameters
    ----------
    csv_path : str
        K 线 CSV 文件路径（由 scripts/load_real_data.load_kline_csv 读取）。
    factor_index_pkl : str, optional
        因子索引 pickle 文件路径（factor_loader 使用）。
    factors_dir : str, optional
        因子数据目录。
    factor_list : list of str, optional
        需要加载的因子名列表。
    start_time : str or None
        数据起始时间（ISO 格式字符串，如 "2022-01-01"）。
    end_time : str or None
        数据终止时间。
    """

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
        """加载 K 线数据并拼接外部因子。

        Parameters
        ----------
        instruments : ignored
            为与 Qlib DataLoader 接口保持兼容而保留（HFML 使用单品种 CSV）。
        start_time : str or None
            优先使用此参数，其次 self.start_time。
        end_time : str or None
            优先使用此参数，其次 self.end_time。

        Returns
        -------
        pd.DataFrame
            标准化 K 线 DataFrame，索引为 datetime。
        """
        from scripts.load_real_data import load_kline_csv

        df = load_kline_csv(self.csv_path)

        st = start_time or self.start_time
        et = end_time or self.end_time
        if st is not None:
            df = df[df.index >= pd.Timestamp(st)]
        if et is not None:
            df = df[df.index <= pd.Timestamp(et)]

        if self.factor_index_pkl and self.factors_dir and self.factor_list:
            try:
                from factor_loader import load_factors

                factor_df = load_factors(
                    index_pkl_path=self.factor_index_pkl,
                    factors_dir=self.factors_dir,
                    factor_list=self.factor_list,
                    start_date=str(df.index.min().date()) if len(df) > 0 else "",
                    end_date=str(df.index.max().date()) if len(df) > 0 else "",
                )
                if not factor_df.empty:
                    factor_df = factor_df.reset_index()
                    if "trade_date" in factor_df.columns:
                        factor_df = factor_df.rename(columns={"trade_date": "datetime"})
                    factor_df["datetime"] = pd.to_datetime(factor_df["datetime"])
                    factor_df = factor_df.set_index("datetime")
                    drop_cols = [c for c in ["ts_code"] if c in factor_df.columns]
                    factor_df = factor_df.drop(columns=drop_cols)
                    df = df.join(factor_df, how="left")
            except Exception as exc:
                logger.warning(f"外部因子加载失败，跳过: {exc}")

        return df.sort_index()


# ---------------------------------------------------------------------------
# HFMLDataHandler
# ---------------------------------------------------------------------------

try:
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.processor import Processor

    class HFMLDataHandler(DataHandlerLP):
        """Qlib DataHandlerLP 实现，整合 HFML 特征工程与标签处理链路。

        Parameters
        ----------
        data_loader : HFMLDataLoader or dict
            数据加载器实例或 Qlib 配置字典。
        period : str
            K 线周期，如 "1min"、"5min"、"15min"。
        **kwargs :
            传递给 DataHandlerLP 的额外参数（start_time, end_time,
            fit_start_time, fit_end_time, infer_processors, learn_processors 等）。
        """

        def __init__(self, data_loader, period="5min", **kwargs):
            self.period = period

            # 若未显式传入 infer_processors，使用默认特征工程流水线
            if "infer_processors" not in kwargs:
                from qlib_ext.processors import (
                    HFMLFeatureProcessor,
                    HFMLTransformProcessor,
                )
                from qlib.data.dataset.processor import RobustZScoreNorm, Fillna

                kwargs["infer_processors"] = [
                    HFMLFeatureProcessor(period=period),
                    HFMLTransformProcessor(),
                    RobustZScoreNorm(fields_group="feature"),
                    Fillna(fields_group="feature"),
                ]

            # 若未显式传入 learn_processors，使用默认标签生成流水线
            if "learn_processors" not in kwargs:
                from qlib_ext.labels import HFMLSmartLabelProcessor
                from qlib_ext.processors import HFMLQualityFilter

                kwargs["learn_processors"] = [
                    HFMLSmartLabelProcessor(period=period),
                    HFMLQualityFilter(),
                ]

            super().__init__(data_loader=data_loader, **kwargs)

except ImportError:
    # Qlib 未安装时提供占位实现，便于单元测试
    class HFMLDataHandler:  # type: ignore[no-redef]
        """占位实现（Qlib 未安装）。"""

        def __init__(self, data_loader, period="5min", **kwargs):
            self.data_loader = data_loader
            self.period = period
            self.kwargs = kwargs

        def setup_data(self, *args, **kwargs):
            pass
