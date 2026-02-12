"""
商品期货机器学习量化模型 - 特征注册表模块
Feature registry for automatic feature discovery and extensibility.

本模块实现了装饰器式的特征注册机制，使新增特征只需在一处定义、
一处注册（通过装饰器），即可自动参与特征计算、筛选和模型训练。

使用方式::

    from features.feature_registry import register_feature, FeatureRegistry

    @register_feature(
        group="动量指标",
        level="level3_momentum",
        description="RSI 14周期的5周期斜率",
        depends_on=["rsi_14"],
    )
    def compute_rsi_14_slope(df, features_df=None, **kwargs):
        slope = features_df["rsi_14"].diff(5) / 5.0
        return pd.DataFrame({"rsi_14_slope": slope}, index=df.index)
"""

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class _FeatureEntry:                                          # [新增]
    """单个已注册特征函数的元数据。"""                        # [新增]

    __slots__ = (                                             # [新增]
        "func", "group", "level", "description",              # [新增]
        "depends_on", "output_names",                         # [新增]
    )                                                         # [新增]

    def __init__(                                             # [新增]
        self,                                                 # [新增]
        func: Callable,                                       # [新增]
        group: str,                                           # [新增]
        level: str,                                           # [新增]
        description: str,                                     # [新增]
        depends_on: Optional[List[str]],                      # [新增]
        output_names: Optional[List[str]],                    # [新增]
    ):                                                        # [新增]
        self.func = func                                      # [新增]
        self.group = group                                    # [新增]
        self.level = level                                    # [新增]
        self.description = description                        # [新增]
        self.depends_on = depends_on or []                    # [新增]
        self.output_names = output_names or []                # [新增]


class FeatureRegistry:                                        # [新增]
    """全局特征注册表，收集所有通过装饰器注册的自定义特征。

    该类采用单例模式，确保全局只有一份注册表。开发者通过
    ``@register_feature(...)`` 装饰器注册特征函数后，下游模块
    （``compute_all_features``、``get_feature_hierarchy``、
    ``get_recommended_feature_groups``）即可自动发现新特征。

    Attributes
    ----------
    _entries : dict
        键为函数名，值为 ``_FeatureEntry``。
    """                                                       # [新增]

    _instance: Optional["FeatureRegistry"] = None             # [新增]
    _entries: Dict[str, _FeatureEntry]                        # [新增]

    def __new__(cls) -> "FeatureRegistry":                    # [新增]
        if cls._instance is None:                             # [新增]
            cls._instance = super().__new__(cls)              # [新增]
            cls._instance._entries = {}                       # [新增]
        return cls._instance                                  # [新增]

    # ---- registration API ------------------------------------------------

    def register(                                             # [新增]
        self,                                                 # [新增]
        func: Callable,                                       # [新增]
        group: str = "自定义",                                # [新增]
        level: str = "level6_transforms",                     # [新增]
        description: str = "",                                # [新增]
        depends_on: Optional[List[str]] = None,               # [新增]
        output_names: Optional[List[str]] = None,             # [新增]
    ) -> None:                                                # [新增]
        """将一个特征计算函数注册到全局注册表中。

        Parameters
        ----------
        func : Callable
            签名为 ``func(df, features_df=None, **kwargs) -> pd.DataFrame``。
            ``df`` 是原始 OHLCV DataFrame；``features_df`` 是已计算完毕
            的当前特征矩阵（可用来引用依赖特征）。返回 DataFrame 的列即
            为新增特征。
        group : str
            特征分组名称（对应 ``get_recommended_feature_groups`` 的键）。
        level : str
            层级名称（对应 ``FEATURE_HIERARCHY`` 的键）。
        description : str
            简短描述。
        depends_on : list of str, optional
            本特征依赖的其他特征名列表。
        output_names : list of str, optional
            显式指定输出列名（不指定则在首次计算后自动采集）。
        """                                                   # [新增]
        key = func.__name__                                   # [新增]
        self._entries[key] = _FeatureEntry(                   # [新增]
            func=func,                                        # [新增]
            group=group,                                      # [新增]
            level=level,                                      # [新增]
            description=description,                          # [新增]
            depends_on=depends_on,                            # [新增]
            output_names=output_names,                        # [新增]
        )                                                     # [新增]

    # ---- query API -------------------------------------------------------

    def get_all_entries(self) -> Dict[str, _FeatureEntry]:    # [新增]
        """返回所有已注册特征条目。"""                        # [新增]
        return dict(self._entries)                            # [新增]

    def get_group_mapping(self) -> Dict[str, List[str]]:      # [新增]
        """返回 {分组名: [特征名, ...]} 映射。

        若某个条目的 ``output_names`` 为空，该条目暂不包含在映射中
        （需在首次计算后通过 ``update_output_names`` 补充）。
        """                                                   # [新增]
        mapping: Dict[str, List[str]] = {}                    # [新增]
        for entry in self._entries.values():                  # [新增]
            if not entry.output_names:                        # [新增]
                continue                                      # [新增]
            mapping.setdefault(entry.group, []).extend(        # [新增]
                entry.output_names                            # [新增]
            )                                                 # [新增]
        return mapping                                        # [新增]

    def get_hierarchy_mapping(self) -> Dict[str, List[str]]:  # [新增]
        """返回 {层级名: [特征名, ...]} 映射。"""            # [新增]
        mapping: Dict[str, List[str]] = {}                    # [新增]
        for entry in self._entries.values():                  # [新增]
            if not entry.output_names:                        # [新增]
                continue                                      # [新增]
            mapping.setdefault(entry.level, []).extend(        # [新增]
                entry.output_names                            # [新增]
            )                                                 # [新增]
        return mapping                                        # [新增]

    def update_output_names(                                  # [新增]
        self, func_name: str, names: List[str]                # [新增]
    ) -> None:                                                # [新增]
        """首次计算后回填输出列名。"""                        # [新增]
        if func_name in self._entries:                        # [新增]
            self._entries[func_name].output_names = names     # [新增]

    def compute_registered_features(                          # [新增]
        self, df, features_df, **kwargs                       # [新增]
    ):                                                        # [新增]
        """执行所有已注册的自定义特征计算。

        Parameters
        ----------
        df : pd.DataFrame
            原始 OHLCV 数据。
        features_df : pd.DataFrame
            当前已计算的特征矩阵（供依赖引用）。
        **kwargs
            透传给各特征函数的额外参数。

        Returns
        -------
        pd.DataFrame
            所有自定义特征合并后的 DataFrame（可能为空）。
        """                                                   # [新增]
        import pandas as pd                                   # [新增]

        frames = []                                           # [新增]
        for key, entry in self._entries.items():              # [新增]
            # 检查依赖是否满足                                # [新增]
            missing = [                                       # [新增]
                d for d in entry.depends_on                   # [新增]
                if d not in features_df.columns               # [新增]
            ]                                                 # [新增]
            if missing:                                       # [新增]
                logger.warning(                               # [新增]
                    "跳过 %s: 依赖特征缺失 %s", key, missing  # [新增]
                )                                             # [新增]
                continue                                      # [新增]

            try:                                              # [新增]
                result = entry.func(                          # [新增]
                    df, features_df=features_df, **kwargs     # [新增]
                )                                             # [新增]
                if result is not None and len(result.columns) > 0:  # [新增]
                    # 自动采集输出列名                        # [新增]
                    if not entry.output_names:                # [新增]
                        self.update_output_names(             # [新增]
                            key, list(result.columns)         # [新增]
                        )                                     # [新增]
                    frames.append(result)                     # [新增]
            except Exception:                                 # [新增]
                logger.exception("注册特征 %s 计算失败", key)  # [新增]

        if frames:                                            # [新增]
            return pd.concat(frames, axis=1)                  # [新增]
        return pd.DataFrame(index=df.index)                   # [新增]

    def clear(self) -> None:                                  # [新增]
        """清空注册表（仅用于测试）。"""                      # [新增]
        self._entries.clear()                                 # [新增]


# ---- 公共装饰器 ----------------------------------------------------------

def register_feature(                                         # [新增]
    group: str = "自定义",                                    # [新增]
    level: str = "level6_transforms",                         # [新增]
    description: str = "",                                    # [新增]
    depends_on: Optional[List[str]] = None,                   # [新增]
    output_names: Optional[List[str]] = None,                 # [新增]
) -> Callable:                                                # [新增]
    """特征注册装饰器。

    用法::

        @register_feature(
            group="动量指标",
            level="level3_momentum",
            depends_on=["rsi_14"],
            output_names=["rsi_14_slope"],
        )
        def compute_rsi_14_slope(df, features_df=None, **kwargs):
            slope = features_df["rsi_14"].diff(5) / 5.0
            return pd.DataFrame({"rsi_14_slope": slope}, index=df.index)

    被装饰的函数会自动注册到全局 ``FeatureRegistry``，
    无需修改任何下游代码即可参与特征计算、筛选和模型训练。

    Parameters
    ----------
    group : str
        特征分组名称。
    level : str
        层级名称。
    description : str
        简短描述。
    depends_on : list of str, optional
        依赖的特征名列表。
    output_names : list of str, optional
        输出列名列表。
    """                                                       # [新增]

    def decorator(func: Callable) -> Callable:                # [新增]
        registry = FeatureRegistry()                          # [新增]
        registry.register(                                    # [新增]
            func=func,                                        # [新增]
            group=group,                                      # [新增]
            level=level,                                      # [新增]
            description=description,                          # [新增]
            depends_on=depends_on,                            # [新增]
            output_names=output_names,                        # [新增]
        )                                                     # [新增]
        return func                                           # [新增]

    return decorator                                          # [新增]
