"""
HFML-Qlib 自定义记录器模块
==========================
HFMLFeatureSelectionRecord : 特征选择报告记录器（依赖 SignalRecord）。
DriftMonitoringRecord      : 概念漂移监控记录器。
HybridSignalBreakdownRecord: 混合专家信号分解记录器。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HFMLFeatureSelectionRecord
# ---------------------------------------------------------------------------

try:
    from qlib.workflow.record_temp import ACRecordTemp, SignalRecord
    from qlib.log import get_module_logger

    class HFMLFeatureSelectionRecord(ACRecordTemp):
        """特征选择报告记录器。

        在 SignalRecord 完成后，对训练集特征执行 XGBoost 或经典特征选择，
        并将结果保存为 artifact。

        Parameters
        ----------
        recorder : qlib.workflow.recorder.Recorder
            当前实验记录对象。
        selector_type : str
            "xgb" 使用 XGBFeatureSelector，"classic" 使用 FeatureSelector。
        feature_col : str
            特征列组名，默认 "feature"。
        label_col : str
            标签列组名，默认 "label"。
        selector_kwargs : dict, optional
            传递给特征选择器的参数。
        skip_existing : bool
            若 artifact 已存在则跳过，默认 False。
        """

        artifact_path = "hfml_feature_selection"
        depend_cls = SignalRecord

        def __init__(
            self,
            recorder,
            selector_type: str = "xgb",
            feature_col: str = "feature",
            label_col: str = "label",
            selector_kwargs: dict = None,
            skip_existing: bool = False,
        ):
            super().__init__(recorder=recorder, skip_existing=skip_existing)
            self.selector_type = selector_type
            self.feature_col = feature_col
            self.label_col = label_col
            self.selector_kwargs = selector_kwargs or {}
            self.logger = get_module_logger(self.__class__.__name__, level=logging.INFO)

        def _load_train_frame(self):
            """从 recorder 的 dataset artifact 加载训练集。"""
            dataset = self.load("dataset")
            if dataset is None:
                raise ValueError(
                    "dataset artifact 不存在，请确保训练脚本已保存 dataset"
                )
            train_frame = dataset.prepare(
                "train", col_set=[self.feature_col, self.label_col]
            )
            if isinstance(train_frame, tuple):
                features, labels = train_frame
            else:
                features = train_frame.get(self.feature_col, train_frame)
                labels = train_frame.get(self.label_col, None)
            return features, labels

        def _create_selector(self):
            if self.selector_type == "xgb":
                from models.xgb_feature_selector import XGBFeatureSelector
                return XGBFeatureSelector(**self.selector_kwargs)
            if self.selector_type == "classic":
                from backtest.feature_selector import FeatureSelector
                return FeatureSelector(**self.selector_kwargs)
            raise ValueError(f"不支持的 selector_type: {self.selector_type}")

        def _generate(self, **kwargs):
            features, labels = self._load_train_frame()
            selector = self._create_selector()

            if isinstance(labels, pd.DataFrame):
                labels = labels.iloc[:, 0]

            self.logger.info("开始 HFML 特征选择记录生成")
            selected_features = selector.select_features(features, labels)

            artifacts = {
                "selected_features.pkl": selected_features,
                "feature_columns.pkl": list(features.columns),
            }
            metrics = {
                "feature_count_before": int(features.shape[1]),
                "feature_count_after": int(len(selected_features)),
            }

            if hasattr(selector, "get_feature_importance"):
                artifacts["feature_importance.pkl"] = selector.get_feature_importance()

            if hasattr(selector, "get_feature_groups_importance"):
                artifacts["feature_group_importance.pkl"] = selector.get_feature_groups_importance()

            if hasattr(selector, "stability_selection"):
                try:
                    artifacts["feature_stability.pkl"] = selector.stability_selection(
                        features, labels
                    )
                except Exception as exc:
                    self.logger.warning(f"stability_selection 失败: {exc}")

            self.recorder.log_metrics(**metrics)
            return artifacts

        def list(self):
            return [
                "selected_features.pkl",
                "feature_columns.pkl",
                "feature_importance.pkl",
                "feature_group_importance.pkl",
                "feature_stability.pkl",
            ]

    class DriftMonitoringRecord(ACRecordTemp):
        """概念漂移监控记录器。

        在 SignalRecord 完成后，使用 AdaptiveModelManager 检测最新数据
        上的模型漂移程度，并将结果记录为 metrics 与 artifact。

        Parameters
        ----------
        recorder : qlib.workflow.recorder.Recorder
            当前实验记录对象。
        manager_kwargs : dict, optional
            传递给 AdaptiveModelManager 的参数。
        skip_existing : bool
            若 artifact 已存在则跳过，默认 False。
        """

        artifact_path = "hfml_drift_monitoring"
        depend_cls = SignalRecord

        def __init__(
            self,
            recorder,
            manager_kwargs: dict = None,
            skip_existing: bool = False,
        ):
            super().__init__(recorder=recorder, skip_existing=skip_existing)
            self.manager_kwargs = manager_kwargs or {}
            self.logger = get_module_logger(self.__class__.__name__, level=logging.INFO)

        def _generate(self, **kwargs):
            from models.adaptive_learning import AdaptiveModelManager

            # 尝试从 recorder 加载模型与历史自适应状态
            try:
                model = self.recorder.load_object("trained_model")
                adaptive_state = self.recorder.load_object("adaptive_state.pkl")
            except Exception:
                model = None
                adaptive_state = {}

            manager = AdaptiveModelManager(**self.manager_kwargs)
            if model is not None:
                manager.model = model

            # 加载测试集预测与标签用于漂移检测
            try:
                pred = self.recorder.load_object("pred.pkl")
                label = self.recorder.load_object("label.pkl")
                if isinstance(pred, pd.DataFrame):
                    pred = pred.iloc[:, 0]
                if isinstance(label, pd.DataFrame):
                    label = label.iloc[:, 0]
                # 按共同索引对齐预测与标签，避免长度不一致
                pred_aligned, label_aligned = pred.align(label, join="inner")
                result = manager.drift_detector.update_batch(
                    predictions=pred_aligned.values,
                    true_labels=label_aligned.values,
                )
            except Exception as exc:
                self.logger.warning(f"漂移检测数据加载失败: {exc}")
                result = {"drift_level": "none", "needs_retrain": False}

            metrics = {
                "drift_level_str": str(result.get("drift_level", "none")),
                "needs_retrain": int(bool(result.get("needs_retrain", False))),
            }
            if "threshold" in result:
                metrics["adaptive_threshold"] = float(result["threshold"])

            self.recorder.log_metrics(**metrics)
            self.logger.info(f"漂移监控结果: {metrics}")

            return {"drift_result.pkl": result}

        def list(self):
            return ["drift_result.pkl"]

    class HybridSignalBreakdownRecord(ACRecordTemp):
        """混合专家信号分解记录器。

        记录五个专家信号的分解、agreement 和 fused_signal，
        便于调试与归因分析。

        Parameters
        ----------
        recorder : qlib.workflow.recorder.Recorder
            当前实验记录对象。
        market_data : pd.DataFrame, optional
            行情数据，用于重新计算专家信号。
        hybrid_kwargs : dict, optional
            传递给 HybridTradingSystem 的参数。
        skip_existing : bool
            若 artifact 已存在则跳过，默认 False。
        """

        artifact_path = "hfml_hybrid_signal_breakdown"
        depend_cls = SignalRecord

        def __init__(
            self,
            recorder,
            market_data: pd.DataFrame = None,
            hybrid_kwargs: dict = None,
            skip_existing: bool = False,
        ):
            super().__init__(recorder=recorder, skip_existing=skip_existing)
            self.market_data = market_data
            self.hybrid_kwargs = hybrid_kwargs or {}
            self.logger = get_module_logger(self.__class__.__name__, level=logging.INFO)

        def _generate(self, **kwargs):
            if self.market_data is None or self.market_data.empty:
                self.logger.warning("HybridSignalBreakdownRecord: 行情数据为空，跳过")
                return {}

            from models.hybrid_trading import HybridTradingSystem
            system = HybridTradingSystem(**self.hybrid_kwargs)

            try:
                result = system.predict(self.market_data)
                breakdown_df = pd.DataFrame(
                    [result.get("experts_breakdown", {})]
                )
                metrics = {
                    "fused_signal": float(result.get("fused_signal", 0.0)),
                    "agreement": float(result.get("agreement", 0.0)),
                    "final_signal": float(result.get("final_signal", 0.0)),
                    "position_size": float(result.get("position_size", 0.0)),
                }
                self.recorder.log_metrics(**metrics)
                return {"signal_breakdown.pkl": breakdown_df}
            except Exception as exc:
                self.logger.warning(f"信号分解计算失败: {exc}")
                return {}

        def list(self):
            return ["signal_breakdown.pkl"]

except ImportError:
    # Qlib 未安装时的占位实现
    class HFMLFeatureSelectionRecord:  # type: ignore[no-redef]
        def __init__(self, recorder, **kwargs):
            self.recorder = recorder

        def generate(self):
            pass

    class DriftMonitoringRecord:  # type: ignore[no-redef]
        def __init__(self, recorder, **kwargs):
            self.recorder = recorder

        def generate(self):
            pass

    class HybridSignalBreakdownRecord:  # type: ignore[no-redef]
        def __init__(self, recorder, **kwargs):
            self.recorder = recorder

        def generate(self):
            pass
