# ===== 修改说明 =====
# 依据《当前项目性能提升指南-最终版v2.md》改进点 5.2
# 新增内容已用 # [新增] 标记
# ===================
"""
商品期货机器学习量化模型 - 因子批量加载与索引对齐模块
Factor batch loading with index alignment.
"""

import os  # [新增]

import pandas as pd  # [新增]


def load_factors(index_pkl_path: str, factors_dir: str, factor_list: list, start_date: str = "", end_date: str = "") -> pd.DataFrame:  # [新增]
    index_df = pd.read_pickle(index_pkl_path)  # [新增]
    index_df["trade_date"] = index_df["trade_date"].astype(str)  # [新增]

    if start_date != "":  # [新增]
        index_df = index_df[index_df["trade_date"] >= start_date]  # [新增]

    if end_date != "":  # [新增]
        index_df = index_df[index_df["trade_date"] <= end_date]  # [新增]

    if index_df.empty:  # [新增]
        return index_df  # [新增]

    start_index = index_df.index[0]  # [新增]
    end_index = index_df.index[-1]  # [新增]

    factor_dfs = []  # [新增]

    for factor in factor_list:  # [新增]
        factor = factor.replace("$", "")  # [新增]
        factor_file = os.path.join(factors_dir, f"{factor}.pkl")  # [新增]
        if os.path.isfile(factor_file):  # [新增]
            factor_data = pd.read_pickle(factor_file)  # [新增]
            factor_data[factor] = pd.to_numeric(factor_data[factor], errors="coerce")  # [新增]
            factor_data = factor_data.iloc[start_index:end_index + 1]  # [新增]
            factor_dfs.append(factor_data)  # [新增]

    combined_factors_df = pd.concat(factor_dfs, axis=1)  # [新增]
    df_factor = index_df.join(combined_factors_df, how="left")  # [新增]
    df_factor = df_factor.set_index(["ts_code", "trade_date"]).sort_index()  # [新增]
    return df_factor  # [新增]
