# ===== 修改说明 =====
# 依据《当前项目性能提升指南-最终版v2.md》改进点 2.5
# 新增内容已用 # [新增] 标记
# ===================
"""
商品期货机器学习量化模型 - 信号后处理模块
Signal post-processing: cross-sectional TopN ranking.
"""

import pandas as pd  # [新增]


def apply_topn_by_date_rank(preds: pd.DataFrame, topn: int = 10) -> pd.DataFrame:  # [新增]
    out = preds.copy()  # [新增]
    out["rank"] = out.groupby("date")["pred"].rank(ascending=False, method="first")  # [新增]  # [BUGFIX] P0-5: rank descending to select top predictions; method="first" for deterministic tie-breaking
    out["signal"] = (out["rank"] <= float(topn)).astype(int)  # [新增]
    return out  # [新增]
