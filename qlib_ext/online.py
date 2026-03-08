"""
HFML-Qlib 在线更新模块
======================
online_update_job : 将 AdaptiveModelManager.step 封装为可被调度的定时任务函数。

该函数可由 Airflow、cron、Windows Task Scheduler 等外部调度器触发，
也可直接在 Python 脚本中调用。
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def online_update_job(
    latest_X: pd.DataFrame,
    latest_y: pd.Series,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    region: str = "cn",
    experiment_name: str = "hfml_experiment",
    recorder_id: Optional[str] = None,
    manager_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行一次在线自适应更新步骤。

    流程：
    1. 初始化 qlib（若尚未初始化）。
    2. 从 recorder 加载模型与自适应状态。
    3. 调用 AdaptiveModelManager.step 进行增量更新与漂移检测。
    4. 将更新后的状态与指标保存回 recorder。

    Parameters
    ----------
    latest_X : pd.DataFrame
        最新一批特征数据（来自实盘或最新回测数据）。
    latest_y : pd.Series
        对应的真实标签。
    provider_uri : str
        Qlib 数据源 URI。
    region : str
        Qlib 地区设置（"cn" 或 "us"）。
    experiment_name : str
        Qlib 实验名称，用于查找 recorder。
    recorder_id : str, optional
        指定 recorder ID。若为 None，使用最新记录。
    manager_kwargs : dict, optional
        传递给 AdaptiveModelManager 的参数。

    Returns
    -------
    dict
        包含 drift_level, needs_retrain, threshold 等键的更新结果字典。
    """
    try:
        import qlib
        qlib.init(provider_uri=provider_uri, region=region)
    except Exception as exc:
        logger.warning(f"qlib.init 已完成或失败: {exc}")

    try:
        from qlib.workflow import R
        if recorder_id is not None:
            rec = R.get_recorder(recorder_id=recorder_id)
        else:
            exp = R.get_exp(experiment_name=experiment_name)
            recorders = list(exp.list_recorders().values())
            if not recorders:
                raise ValueError(f"实验 '{experiment_name}' 中没有可用的 recorder")

            def _get_start_time(r):
                raw = r.info.get("start_time", None)
                if raw is None:
                    return pd.Timestamp.min
                try:
                    return pd.Timestamp(raw)
                except Exception:
                    return pd.Timestamp.min

            rec = sorted(recorders, key=_get_start_time)[-1]
    except Exception as exc:
        logger.error(f"加载 recorder 失败: {exc}")
        return {"error": str(exc)}

    # 加载模型与自适应状态
    model = None
    adaptive_state: Dict[str, Any] = {}
    try:
        model = rec.load_object("trained_model")
    except Exception:
        logger.warning("未找到 trained_model artifact，将使用空模型")

    try:
        adaptive_state = rec.load_object("adaptive_state.pkl")
    except Exception:
        logger.info("未找到 adaptive_state.pkl，使用默认配置初始化")

    # 执行自适应更新
    from models.adaptive_learning import AdaptiveModelManager

    mkwargs = {**(manager_kwargs or {})}
    manager = AdaptiveModelManager(**mkwargs)
    if model is not None:
        manager.model = model

    if "threshold_manager" in adaptive_state:
        manager.threshold_manager = adaptive_state["threshold_manager"]
    if "drift_detector" in adaptive_state:
        manager.drift_detector = adaptive_state["drift_detector"]

    try:
        result = manager.step(latest_X, latest_y)
    except Exception as exc:
        logger.error(f"AdaptiveModelManager.step 执行失败: {exc}")
        return {"error": str(exc)}

    # 记录指标与保存状态
    metrics_to_log = {
        k: float(v)
        for k, v in result.items()
        if isinstance(v, (int, float)) and k not in ("needs_retrain",)
    }
    if "needs_retrain" in result:
        metrics_to_log["needs_retrain"] = int(bool(result["needs_retrain"]))

    try:
        rec.log_metrics(**metrics_to_log)
        rec.save_objects(
            **{
                "trained_model": manager.model,
                "adaptive_state.pkl": {
                    "manager_kwargs": mkwargs,
                    "threshold_manager": manager.threshold_manager,
                    "drift_detector": manager.drift_detector,
                    "last_result": result,
                },
                "adaptive_step_result.pkl": result,
            }
        )
        logger.info(f"在线更新完成，结果: {metrics_to_log}")
    except Exception as exc:
        logger.warning(f"保存在线更新结果失败: {exc}")

    return result
