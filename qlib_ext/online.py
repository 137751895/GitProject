"""
HFML-Qlib 在线更新模块（修正版）
================================
online_update_job : 将 AdaptiveModelManager.step 封装为可被调度的定时任务函数。

修正点（对应 7.4 问题）：
- AdaptiveModelManager 初始化时传入 base_model（必须参数）
- manager.step() 正确传参：step(y_true, y_pred, X_new, y_new, ...)
  其中 y_true/y_pred 是标签与预测，X_new/y_new 是增量学习数据

该函数可由 Airflow、cron、Windows Task Scheduler 等外部调度器触发，
也可直接在 Python 脚本中调用。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def online_update_job(
    y_true: "pd.Series | np.ndarray",
    y_pred: "pd.Series | np.ndarray",
    X_new: Optional[pd.DataFrame] = None,
    y_new: Optional[pd.Series] = None,
    current_volatility: float = 0.0,
    mean_volatility: float = 0.0,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    region: str = "cn",
    experiment_name: str = "hfml_experiment",
    recorder_id: Optional[str] = None,
    manager_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行一次在线自适应更新步骤。

    流程：
    1. 初始化 qlib（若尚未初始化）。
    2. 从 recorder 加载 base_model 与自适应状态。
    3. 构造 AdaptiveModelManager(base_model=...) 并恢复状态。
    4. 调用 manager.step(y_true, y_pred, X_new, y_new, ...) 执行自适应更新。
    5. 将更新后的状态与指标保存回 recorder。

    Parameters
    ----------
    y_true : array-like
        真实标签（与最新预测对应）。
    y_pred : array-like
        模型对应的预测值（用于漂移检测）。
    X_new : pd.DataFrame, optional
        最新增量特征数据（用于在线学习）。
    y_new : pd.Series, optional
        增量数据对应的标签（用于在线学习）。
    current_volatility : float
        当前波动率（用于自适应阈值调整）。
    mean_volatility : float
        历史平均波动率。
    provider_uri : str
        Qlib 数据源 URI。
    region : str
        Qlib 地区设置（"cn" 或 "us"）。
    experiment_name : str
        Qlib 实验名称，用于查找 recorder。
    recorder_id : str, optional
        指定 recorder ID。若为 None，使用最新记录。
    manager_kwargs : dict, optional
        传递给 AdaptiveModelManager 的额外参数
        （drift_window, drift_warning, drift_threshold,
          update_interval, n_incremental_trees, base_threshold）。

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

    # 加载 recorder
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

    # 加载 base_model（AdaptiveModelManager 必须参数）
    base_model = None
    try:
        base_model = rec.load_object("trained_model")
        logger.info("已从 recorder 加载 trained_model")
    except Exception:
        logger.warning("未找到 trained_model artifact，漂移检测将跳过在线学习部分")

    # 加载自适应状态（可选）
    adaptive_state: Dict[str, Any] = {}
    try:
        adaptive_state = rec.load_object("adaptive_state.pkl") or {}
    except Exception:
        logger.info("未找到 adaptive_state.pkl，使用默认初始化")

    # 如果没有 base_model，仍可使用独立的漂移检测
    if base_model is None:
        from models.adaptive_learning import ConceptDriftDetector, AdaptiveThresholdManager
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()

        detector = ConceptDriftDetector()
        if "drift_detector" in adaptive_state:
            detector = adaptive_state["drift_detector"]
        result = detector.update(y_true_arr, y_pred_arr)

        threshold_mgr = AdaptiveThresholdManager()
        if "threshold_manager" in adaptive_state:
            threshold_mgr = adaptive_state["threshold_manager"]
        thr = threshold_mgr.compute_threshold(
            current_volatility=current_volatility,
            mean_volatility=mean_volatility,
            current_accuracy=result.get("current_accuracy"),
            baseline_accuracy=result.get("baseline_accuracy"),
        )
        result["threshold"]    = thr["threshold"]
        result["needs_retrain"] = result.get("drift_level") == "drift"
        return result

    # 构造 AdaptiveModelManager 并恢复状态
    from models.adaptive_learning import AdaptiveModelManager

    mkwargs = {**(manager_kwargs or {})}
    manager = AdaptiveModelManager(base_model=base_model, **mkwargs)

    if "threshold_manager" in adaptive_state:
        manager.threshold_manager = adaptive_state["threshold_manager"]
    if "drift_detector" in adaptive_state:
        manager.drift_detector = adaptive_state["drift_detector"]
    if "online_learner" in adaptive_state:
        manager.online_learner = adaptive_state["online_learner"]

    # 正确调用 step(y_true, y_pred, X_new, y_new, ...)
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    try:
        result = manager.step(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            X_new=X_new,
            y_new=y_new,
            current_volatility=current_volatility,
            mean_volatility=mean_volatility,
        )
    except Exception as exc:
        logger.error(f"AdaptiveModelManager.step 执行失败: {exc}")
        return {"error": str(exc)}

    # 记录指标与保存状态
    metrics_to_log = {}
    for k, v in result.items():
        if k == "needs_retrain":
            metrics_to_log[k] = int(bool(v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            metrics_to_log[k] = float(v)

    try:
        rec.log_metrics(**metrics_to_log)
        saved_model = getattr(getattr(manager, "online_learner", None), "base_model", base_model)
        rec.save_objects(**{
            "trained_model": saved_model or base_model,
            "adaptive_state.pkl": {
                "manager_kwargs":    mkwargs,
                "threshold_manager": manager.threshold_manager,
                "drift_detector":    manager.drift_detector,
                "online_learner":    manager.online_learner,
                "last_result":       result,
            },
            "adaptive_step_result.pkl": result,
        })
        logger.info(f"在线更新完成，结果: {metrics_to_log}")
    except Exception as exc:
        logger.warning(f"保存在线更新结果失败: {exc}")

    return result
