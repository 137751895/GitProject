"""
HFML-Qlib 自定义记录器模块
==========================
HFMLFeatureSelectionRecord  : 特征选择报告记录器（依赖 SignalRecord）。
FeatureGroupImportanceRecord: 特征组重要性记录器。
FeatureStabilityRecord      : 特征稳定性记录器。
DriftMonitoringRecord       : 概念漂移监控记录器。
HybridSignalBreakdownRecord : 混合专家信号分解记录器。
MultiTimeframeGradeRecord   : 多时间框架信号等级记录器。
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
        """特征选择报告记录器（修正版）。

        修正点：
        - XGBFeatureSelector 使用 fit_select() 而非不存在的 select_features()
        - get_feature_groups_importance() 传入 feature_groups 参数
        - selector_kwargs 仅传 XGBFeatureSelector 构造函数支持的参数：
          n_features, threshold, cv_splits, xgb_params

        Parameters
        ----------
        recorder :
            当前实验记录对象。
        selector_type : str
            "xgb" 使用 XGBFeatureSelector，"classic" 使用 FeatureSelector。
        feature_col : str
            特征列组名，默认 "feature"。
        label_col : str
            标签列组名，默认 "label"。
        selector_kwargs : dict, optional
            传递给特征选择器构造函数的参数。
            XGBFeatureSelector 接受: n_features, threshold, cv_splits, xgb_params
            FeatureSelector 接受: n_splits, min_success_rate
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
            """从 recorder 的 dataset artifact 加载训练集。

            注意：训练脚本须提前保存 dataset artifact：
              R.save_objects(dataset=dataset)
            """
            dataset = self.load("dataset")
            if dataset is None:
                raise ValueError(
                    "dataset artifact 不存在。请在训练脚本中调用 "
                    "R.save_objects(dataset=dataset) 保存 dataset。"
                )
            train_frame = dataset.prepare(
                "train", col_set=[self.feature_col, self.label_col]
            )
            if isinstance(train_frame, tuple):
                features, labels = train_frame
            elif isinstance(train_frame, dict):
                features = train_frame.get(self.feature_col, pd.DataFrame())
                labels   = train_frame.get(self.label_col, pd.Series())
            else:
                # 若返回单个 DataFrame，按列组拆分
                if isinstance(train_frame.columns, pd.MultiIndex):
                    features = train_frame[self.feature_col] if self.feature_col in train_frame.columns.get_level_values(0) else train_frame
                    labels   = train_frame[self.label_col]   if self.label_col   in train_frame.columns.get_level_values(0) else pd.Series()
                else:
                    features = train_frame
                    labels   = pd.Series()
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

            if isinstance(labels, pd.DataFrame):
                labels = labels.iloc[:, 0]

            # 二分类映射：与模型包装器保持一致（NaN 安全版本）
            if labels.name == "trading_signal" or (
                labels.dtype in ["int64", "int32", "float64"]
                and set(labels.dropna().unique().astype(int)).issubset({-2, -1, 0, 1, 2})
            ):
                labels = (labels.fillna(0) > 0).astype(int)

            selector = self._create_selector()
            self.logger.info("开始 HFML 特征选择记录生成")

            artifacts = {}
            metrics = {
                "feature_count_before": int(features.shape[1]),
            }

            if self.selector_type == "xgb":
                # XGBFeatureSelector 使用 fit_select() 返回 (selected_features, importance_df)
                selected_features, importance_df = selector.fit_select(features, labels)
                artifacts["selected_features.pkl"] = selected_features
                artifacts["feature_columns.pkl"]   = list(features.columns)
                artifacts["feature_importance.pkl"] = importance_df
                metrics["feature_count_after"] = int(len(selected_features))

                # get_feature_groups_importance() 需要 feature_groups 参数
                try:
                    from config import FEATURE_HIERARCHY
                    feature_groups = {
                        level: info["features"]
                        for level, info in FEATURE_HIERARCHY.items()
                    }
                    group_importance = selector.get_feature_groups_importance(feature_groups)
                    artifacts["feature_group_importance.pkl"] = group_importance
                except Exception as exc:
                    self.logger.warning(f"特征组重要性计算失败（跳过）: {exc}")

            elif self.selector_type == "classic":
                # FeatureSelector 使用 select_best_features()
                from models.ml_models import LightGBMModel
                selected_features = selector.select_best_features(
                    LightGBMModel, features, labels,
                    task="classification",
                    top_n=self.selector_kwargs.get("top_n", 20),
                )
                artifacts["selected_features.pkl"] = selected_features
                artifacts["feature_columns.pkl"]   = list(features.columns)
                metrics["feature_count_after"] = int(len(selected_features))

                try:
                    stability = selector.extra_trees_importance_stability(
                        features, labels, task="classification"
                    )
                    artifacts["feature_stability.pkl"] = stability
                except Exception as exc:
                    self.logger.warning(f"特征稳定性计算失败（跳过）: {exc}")

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

    # ---------------------------------------------------------------------------
    # FeatureGroupImportanceRecord
    # ---------------------------------------------------------------------------

    class FeatureGroupImportanceRecord(ACRecordTemp):
        """特征组重要性记录器。

        调用 XGBFeatureSelector.get_feature_groups_importance() 按 FEATURE_HIERARCHY
        汇总各特征组的重要性评分，并保存为 artifact。

        Parameters
        ----------
        recorder :
            当前实验记录对象。
        skip_existing : bool
            若 artifact 已存在则跳过，默认 False。
        """

        artifact_path = "hfml_feature_group_importance"
        depend_cls = HFMLFeatureSelectionRecord

        def __init__(self, recorder, skip_existing: bool = False):
            super().__init__(recorder=recorder, skip_existing=skip_existing)
            self.logger = get_module_logger(self.__class__.__name__, level=logging.INFO)

        def _generate(self, **kwargs):
            # 加载 HFMLFeatureSelectionRecord 已保存的重要性 artifact
            importance_df = self.load("feature_importance.pkl")
            if importance_df is None:
                raise ValueError("feature_importance.pkl 不存在，请先运行 HFMLFeatureSelectionRecord")

            from config import FEATURE_HIERARCHY
            from models.xgb_feature_selector import XGBFeatureSelector

            selector = XGBFeatureSelector()
            selector.feature_importance_df = importance_df
            selected = self.load("selected_features.pkl") or []
            selector.selected_features = selected

            feature_groups = {
                level: info["features"]
                for level, info in FEATURE_HIERARCHY.items()
            }
            group_importance = selector.get_feature_groups_importance(feature_groups)

            self.recorder.log_metrics(
                n_feature_groups=int(len(group_importance)),
                top_group_score=float(group_importance["score"].max()) if "score" in group_importance.columns else 0.0,
            )
            return {"feature_group_importance.pkl": group_importance}

        def list(self):
            return ["feature_group_importance.pkl"]

    # ---------------------------------------------------------------------------
    # FeatureStabilityRecord
    # ---------------------------------------------------------------------------

    class FeatureStabilityRecord(ACRecordTemp):
        """特征稳定性记录器。

        调用 FeatureSelector.extra_trees_importance_stability() 评估特征重要性稳定性，
        输出 stability 报告并保存为 artifact。

        Parameters
        ----------
        recorder :
            当前实验记录对象。
        feature_col : str
            特征列组名，默认 "feature"。
        label_col : str
            标签列组名，默认 "label"。
        top_n : int
            保留前 N 个稳定特征，默认 50。
        skip_existing : bool
            若 artifact 已存在则跳过，默认 False。
        """

        artifact_path = "hfml_feature_stability"
        depend_cls = SignalRecord

        def __init__(
            self,
            recorder,
            feature_col: str = "feature",
            label_col: str = "label",
            top_n: int = 50,
            skip_existing: bool = False,
        ):
            super().__init__(recorder=recorder, skip_existing=skip_existing)
            self.feature_col = feature_col
            self.label_col   = label_col
            self.top_n       = top_n
            self.logger = get_module_logger(self.__class__.__name__, level=logging.INFO)

        def _generate(self, **kwargs):
            dataset = self.load("dataset")
            if dataset is None:
                raise ValueError("dataset artifact 不存在，请先保存 dataset")

            train_frame = dataset.prepare("train", col_set=[self.feature_col, self.label_col])
            if isinstance(train_frame, tuple):
                features, labels = train_frame
            elif isinstance(train_frame, dict):
                features = train_frame.get(self.feature_col, pd.DataFrame())
                labels   = train_frame.get(self.label_col, pd.Series())
            else:
                features = train_frame
                labels   = pd.Series()

            if isinstance(labels, pd.DataFrame):
                labels = labels.iloc[:, 0]
            labels = (labels > 0).astype(int)

            from backtest.feature_selector import FeatureSelector
            selector = FeatureSelector()
            stability_report = selector.extra_trees_importance_stability(
                features, labels, top_n=self.top_n, task="classification"
            )
            self.recorder.log_metrics(n_stable_features=int(self.top_n))
            return {"feature_stability.pkl": stability_report}

        def list(self):
            return ["feature_stability.pkl"]

    # ---------------------------------------------------------------------------
    # DriftMonitoringRecord
    # ---------------------------------------------------------------------------

    class DriftMonitoringRecord(ACRecordTemp):
        """概念漂移监控记录器（修正版）。

        修正点：
        - AdaptiveModelManager 初始化时从 recorder 加载 base_model 并传入
        - 使用 drift_detector.update(y_true, y_pred) 而非不存在的 update_batch()
        - pred 与 label 按共同索引对齐后再传入

        Parameters
        ----------
        recorder :
            当前实验记录对象。
        manager_kwargs : dict, optional
            传递给 AdaptiveModelManager 的额外参数（drift_window 等）。
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
            from models.adaptive_learning import AdaptiveModelManager, ConceptDriftDetector

            # 加载 base_model（AdaptiveModelManager 必须传入）
            base_model = None
            try:
                base_model = self.recorder.load_object("trained_model")
            except Exception:
                self.logger.warning("未找到 trained_model，将使用占位模型进行漂移检测")

            # 加载预测结果与真实标签
            try:
                pred  = self.recorder.load_object("pred.pkl")
                label = self.recorder.load_object("label.pkl")
                if isinstance(pred, pd.DataFrame):
                    pred = pred.iloc[:, 0]
                if isinstance(label, pd.DataFrame):
                    label = label.iloc[:, 0]
                # 按共同索引对齐
                pred_aligned, label_aligned = pred.align(label, join="inner")
                # 转为二分类方向预测
                y_pred = (pred_aligned >= 0.5).astype(int).values
                y_true = (label_aligned > 0).astype(int).values
            except Exception as exc:
                self.logger.warning(f"漂移检测数据加载失败: {exc}")
                return {"drift_result.pkl": {"drift_level": "none", "needs_retrain": False}}

            # 使用独立的 ConceptDriftDetector（不依赖 base_model）进行漂移检测
            drift_cfg = {
                k: v for k, v in self.manager_kwargs.items()
                if k in ("drift_window", "drift_warning", "drift_threshold")
            }
            detector = ConceptDriftDetector(
                window_size=drift_cfg.get("drift_window", 100),
                warning_threshold=drift_cfg.get("drift_warning", 0.05),
                drift_threshold=drift_cfg.get("drift_threshold", 0.10),
            )
            result = detector.update(y_true, y_pred)

            # 若 base_model 存在，也通过 AdaptiveModelManager.step 更新完整状态
            if base_model is not None:
                try:
                    manager = AdaptiveModelManager(base_model=base_model, **self.manager_kwargs)
                    result = manager.step(
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                except Exception as exc:
                    self.logger.warning(f"AdaptiveModelManager.step 失败，使用独立检测器结果: {exc}")

            metrics = {
                "drift_level_str": str(result.get("drift_level", "none")),
                "needs_retrain":   int(bool(result.get("needs_retrain", False))),
                "current_accuracy": float(result.get("current_accuracy", 0.0) or 0.0),
                "accuracy_drop":    float(result.get("accuracy_drop", 0.0) or 0.0),
            }
            if "threshold" in result:
                metrics["adaptive_threshold"] = float(result["threshold"])

            self.recorder.log_metrics(**metrics)
            self.logger.info(f"漂移监控结果: {metrics}")
            return {"drift_result.pkl": result}

        def list(self):
            return ["drift_result.pkl"]

    # ---------------------------------------------------------------------------
    # HybridSignalBreakdownRecord
    # ---------------------------------------------------------------------------

    class HybridSignalBreakdownRecord(ACRecordTemp):
        """混合专家信号分解记录器。"""

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
            self.market_data  = market_data
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
                breakdown_df = pd.DataFrame([result.get("experts_breakdown", {})])
                metrics = {
                    "fused_signal":  float(result.get("fused_signal",  0.0)),
                    "agreement":     float(result.get("agreement",     0.0)),
                    "final_signal":  float(result.get("final_signal",  0.0)),
                    "position_size": float(result.get("position_size", 0.0)),
                }
                self.recorder.log_metrics(**metrics)
                return {"signal_breakdown.pkl": breakdown_df}
            except Exception as exc:
                self.logger.warning(f"信号分解计算失败: {exc}")
                return {}

        def list(self):
            return ["signal_breakdown.pkl"]

    # ---------------------------------------------------------------------------
    # MultiTimeframeGradeRecord
    # ---------------------------------------------------------------------------

    class MultiTimeframeGradeRecord(ACRecordTemp):
        """多时间框架信号等级记录器。

        记录 A/B/C 等级信号的数量、信号强度分布、过滤率等统计信息，
        便于分析多周期协同效果。

        Parameters
        ----------
        recorder :
            当前实验记录对象。
        pred_col : str
            预测分数列名，默认 "score"。
        min_grade : str
            最低等级阈值，默认 "C"。
        coordinator_kwargs : dict, optional
            传递给 MultiTimeframeCoordinator 的参数。
        skip_existing : bool
            若 artifact 已存在则跳过，默认 False。
        """

        artifact_path = "hfml_multiframe_grade"
        depend_cls = SignalRecord

        def __init__(
            self,
            recorder,
            pred_col: str = "score",
            min_grade: str = "C",
            coordinator_kwargs: dict = None,
            skip_existing: bool = False,
        ):
            super().__init__(recorder=recorder, skip_existing=skip_existing)
            self.pred_col = pred_col
            self.min_grade = min_grade
            self.coordinator_kwargs = coordinator_kwargs or {}
            self.logger = get_module_logger(self.__class__.__name__, level=logging.INFO)

        def _generate(self, **kwargs):
            # 加载预测结果
            try:
                pred = self.recorder.load_object("pred.pkl")
                if isinstance(pred, pd.DataFrame):
                    pred = pred.iloc[:, 0]
            except Exception as exc:
                self.logger.warning(f"MultiTimeframeGradeRecord: 预测加载失败 ({exc})，跳过")
                return {}

            # 使用自定义阈值映射信号到 A/B/C 等级
            # A: |score - 0.5| > 0.3 且高信心
            # B: 0.15 < |score - 0.5| <= 0.3
            # C: 0.05 < |score - 0.5| <= 0.15
            # -: |score - 0.5| <= 0.05 (中性，过滤)
            strength = (pred - 0.5).abs()
            grade = pd.Series("-", index=pred.index)
            grade[strength > 0.30] = "A"
            grade[(strength > 0.15) & (strength <= 0.30)] = "B"
            grade[(strength > 0.05) & (strength <= 0.15)] = "C"

            grade_order = {"A": 3, "B": 2, "C": 1, "-": 0}
            min_level  = grade_order.get(self.min_grade, 1)
            filtered   = grade[grade.map(lambda g: grade_order.get(g, 0)) < min_level]

            n_total   = len(grade)
            n_A       = (grade == "A").sum()
            n_B       = (grade == "B").sum()
            n_C       = (grade == "C").sum()
            n_neutral = (grade == "-").sum()
            n_filtered = len(filtered)
            filter_rate = n_filtered / n_total if n_total > 0 else 0.0

            metrics = {
                "n_total":     int(n_total),
                "n_grade_A":   int(n_A),
                "n_grade_B":   int(n_B),
                "n_grade_C":   int(n_C),
                "n_neutral":   int(n_neutral),
                "n_filtered":  int(n_filtered),
                "filter_rate": float(filter_rate),
            }
            self.recorder.log_metrics(**metrics)
            self.logger.info(f"多周期等级统计: {metrics}")

            grade_df = pd.DataFrame({
                "score": pred,
                "strength": strength,
                "grade": grade,
            })
            return {"multiframe_grade.pkl": grade_df}

        def list(self):
            return ["multiframe_grade.pkl"]

except ImportError:
    # Qlib 未安装时的占位实现
    class HFMLFeatureSelectionRecord:  # type: ignore[no-redef]
        def __init__(self, recorder, **kwargs):
            self.recorder = recorder
        def generate(self):
            pass

    class FeatureGroupImportanceRecord:  # type: ignore[no-redef]
        def __init__(self, recorder, **kwargs):
            self.recorder = recorder
        def generate(self):
            pass

    class FeatureStabilityRecord:  # type: ignore[no-redef]
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

    class MultiTimeframeGradeRecord:  # type: ignore[no-redef]
        def __init__(self, recorder, **kwargs):
            self.recorder = recorder
        def generate(self):
            pass
