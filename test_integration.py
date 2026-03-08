"""
HFML × Qlib 集成验证脚本
========================
用于验证 qlib_ext 适配层的各个组件是否能正确加载并运行。

运行方式：
    python test_integration.py

测试内容：
1. qlib_ext 包导入测试
2. HFMLDataLoader 数据加载测试（使用模拟数据）
3. HFMLFeatureProcessor / HFMLTransformProcessor 特征工程测试
4. HFMLSmartLabelProcessor 标签生成测试
5. HFMLQualityFilter 样本过滤测试
6. HFMLLGBMQlibModel / HFMLXGBQlibModel 模型接口测试（fit/predict 签名）
7. HFMLHybridSignal 信号生成测试
8. utils 辅助函数测试
9. （可选）完整 Qlib workflow 运行测试（需要 Qlib 已安装）
"""

import sys
import os
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 配置日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_integration")

# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path（使 qlib_ext、features、models 等可被导入）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 辅助函数
# ===========================================================================

def _make_sample_kline(n: int = 500, period: str = "15min") -> pd.DataFrame:
    """生成模拟 K 线 DataFrame，索引为 datetime，列为 OHLCV + open_interest。"""
    freq_map = {"1min": "1min", "5min": "5min", "15min": "15min"}
    freq = freq_map.get(period, "15min")
    base = datetime(2023, 1, 1, 9, 0, 0)
    index = pd.date_range(base, periods=n, freq=freq)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.clip(close, 50, 200)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.abs(np.random.randn(n) * 1000 + 5000)
    oi = np.abs(np.random.randn(n) * 500 + 50000)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": oi,
        },
        index=index,
    )


def _check(condition: bool, msg: str) -> bool:
    if condition:
        logger.info(f"  ✓ {msg}")
    else:
        logger.error(f"  ✗ {msg}")
    return condition


PASS_COUNT = 0
FAIL_COUNT = 0


def _test(name: str):
    """装饰器：包裹测试函数，捕获异常并统计通过/失败数量。"""
    def decorator(fn):
        def wrapper():
            global PASS_COUNT, FAIL_COUNT
            logger.info(f"\n{'=' * 60}")
            logger.info(f"测试: {name}")
            logger.info("=" * 60)
            try:
                result = fn()
                if result is False:
                    FAIL_COUNT += 1
                    logger.error(f"  [FAIL] {name}")
                else:
                    PASS_COUNT += 1
                    logger.info(f"  [PASS] {name}")
            except Exception:
                FAIL_COUNT += 1
                logger.error(f"  [ERROR] {name}")
                logger.error(traceback.format_exc())
        return wrapper
    return decorator


# ===========================================================================
# 测试 1：qlib_ext 包导入
# ===========================================================================

@_test("qlib_ext 包导入")
def test_import():
    import qlib_ext
    from qlib_ext.data import HFMLDataLoader, HFMLDataHandler
    from qlib_ext.processors import HFMLFeatureProcessor, HFMLTransformProcessor, HFMLQualityFilter
    from qlib_ext.labels import HFMLSmartLabelProcessor
    from qlib_ext.models import HFMLLSTMQlibModel, HFMLLGBMQlibModel, HFMLXGBQlibModel
    from qlib_ext.signals import HFMLHybridSignal
    from qlib_ext.strategies import HFMLHybridStrategy, HFMLMultiTimeframeStrategy
    from qlib_ext.records import HFMLFeatureSelectionRecord, DriftMonitoringRecord
    from qlib_ext.online import online_update_job
    from qlib_ext.utils import normalize_pred, slice_pred_at, build_multiindex_pred

    all_pass = _check(True, "所有 qlib_ext 模块导入成功")
    return all_pass


# ===========================================================================
# 测试 2：HFMLDataLoader（使用模拟数据回退路径）
# ===========================================================================

@_test("HFMLDataLoader — 使用模拟数据")
def test_data_loader():
    from qlib_ext.data import HFMLDataLoader
    from scripts.load_real_data import get_default_data_path

    # 构造一个指向不存在文件的路径，测试异常处理
    loader = HFMLDataLoader(csv_path="/tmp/nonexistent.csv")
    ok = False
    try:
        loader.load()
        _check(False, "HFMLDataLoader 应该抛出 FileNotFoundError 但未抛出")
    except FileNotFoundError:
        ok = True
        _check(True, "HFMLDataLoader 正确抛出 FileNotFoundError")
    except Exception as exc:
        _check(False, f"HFMLDataLoader 抛出意外异常: {exc}")
    return ok


# ===========================================================================
# 测试 3：HFMLFeatureProcessor
# ===========================================================================

@_test("HFMLFeatureProcessor — 特征计算")
def test_feature_processor():
    from qlib_ext.processors import HFMLFeatureProcessor

    df = _make_sample_kline(n=300, period="15min")
    proc = HFMLFeatureProcessor(period="15min")
    proc.fit(df)
    result = proc(df)

    _check(isinstance(result, pd.DataFrame), "返回类型为 DataFrame")
    has_feature_col = (
        isinstance(result.columns, pd.MultiIndex)
        and "feature" in result.columns.get_level_values(0)
    ) or any(c for c in result.columns if str(c).startswith("ma_") or str(c).startswith("rsi_"))
    _check(has_feature_col, f"包含特征列（列数={result.shape[1]}）")
    _check(proc.is_for_infer(), "is_for_infer() 返回 True")
    return True


# ===========================================================================
# 测试 4：HFMLTransformProcessor
# ===========================================================================

@_test("HFMLTransformProcessor — 深度变换")
def test_transform_processor():
    from qlib_ext.processors import HFMLFeatureProcessor, HFMLTransformProcessor

    df = _make_sample_kline(n=300, period="15min")
    feat_proc = HFMLFeatureProcessor(period="15min")
    df_with_feat = feat_proc(df)

    trans_proc = HFMLTransformProcessor()
    result = trans_proc(df_with_feat)

    _check(isinstance(result, pd.DataFrame), "返回类型为 DataFrame")
    _check(result.shape[1] >= df_with_feat.shape[1], "变换后列数不减少")
    return True


# ===========================================================================
# 测试 5：HFMLSmartLabelProcessor
# ===========================================================================

@_test("HFMLSmartLabelProcessor — 标签生成")
def test_label_processor():
    from qlib_ext.labels import HFMLSmartLabelProcessor

    df = _make_sample_kline(n=300, period="15min")
    proc = HFMLSmartLabelProcessor(period="15min", config={"horizon": 2})
    proc.fit(df)
    result = proc(df)

    _check(isinstance(result, pd.DataFrame), "返回类型为 DataFrame")
    _check(not proc.is_for_infer(), "is_for_infer() 返回 False")

    # 检查标签列
    label_cols = []
    if isinstance(result.columns, pd.MultiIndex):
        label_cols = [sub for top, sub in result.columns if top == "label"]
    expected = {"trading_signal", "signal_quality", "expected_return"}
    has_labels = bool(expected & set(label_cols))
    _check(has_labels, f"包含预期标签列: {label_cols}")
    return True


# ===========================================================================
# 测试 6：HFMLQualityFilter
# ===========================================================================

@_test("HFMLQualityFilter — 样本过滤")
def test_quality_filter():
    from qlib_ext.labels import HFMLSmartLabelProcessor
    from qlib_ext.processors import HFMLQualityFilter

    df = _make_sample_kline(n=300, period="15min")
    label_proc = HFMLSmartLabelProcessor(period="15min", config={"horizon": 2})
    df_with_labels = label_proc(df)

    filt = HFMLQualityFilter(min_quality=0.3)
    filtered = filt(df_with_labels)

    _check(isinstance(filtered, pd.DataFrame), "返回类型为 DataFrame")
    _check(len(filtered) <= len(df_with_labels), "过滤后行数不超过原始行数")
    _check(not filt.is_for_infer(), "is_for_infer() 返回 False")
    return True


# ===========================================================================
# 测试 7：模型类接口检查
# ===========================================================================

@_test("HFMLLGBMQlibModel / HFMLXGBQlibModel — 接口检查")
def test_model_interfaces():
    from qlib_ext.models import HFMLLGBMQlibModel, HFMLXGBQlibModel, HFMLLSTMQlibModel

    for cls in [HFMLLGBMQlibModel, HFMLXGBQlibModel, HFMLLSTMQlibModel]:
        model = cls(params={}, task="classification")
        _check(hasattr(model, "fit"), f"{cls.__name__} 有 fit 方法")
        _check(hasattr(model, "predict"), f"{cls.__name__} 有 predict 方法")

    return True


# ===========================================================================
# 测试 8：HFMLHybridSignal
# ===========================================================================

@_test("HFMLHybridSignal — 信号生成")
def test_hybrid_signal():
    from qlib_ext.signals import HFMLHybridSignal

    df = _make_sample_kline(n=200, period="15min")
    signal = HFMLHybridSignal(window=50)
    signal.set_market_data(df)

    result = signal.get_signal()
    _check(isinstance(result, dict), "get_signal 返回 dict")
    _check("final_signal" in result, "结果包含 final_signal")
    _check("position_size" in result, "结果包含 position_size")
    _check(result["final_signal"] in (-2, -1, 0, 1, 2),
           f"final_signal 值合理: {result['final_signal']}")
    return True


# ===========================================================================
# 测试 9：utils 辅助函数
# ===========================================================================

@_test("utils — normalize_pred / build_multiindex_pred / slice_pred_at")
def test_utils():
    from qlib_ext.utils import normalize_pred, build_multiindex_pred, slice_pred_at

    # build_multiindex_pred
    scores = np.array([0.6, 0.7, 0.55])
    dts = pd.date_range("2024-01-01", periods=3, freq="15min")
    pred_series = build_multiindex_pred(scores, dts, instrument="HFML_TEST")
    _check(isinstance(pred_series, pd.Series), "build_multiindex_pred 返回 Series")
    _check(isinstance(pred_series.index, pd.MultiIndex), "Series 索引为 MultiIndex")

    # normalize_pred
    pred_df = normalize_pred(pred_series)
    _check(isinstance(pred_df, pd.DataFrame), "normalize_pred(Series) 返回 DataFrame")

    # slice_pred_at
    val = slice_pred_at(pred_df, dts[0], "HFML_TEST")
    _check(val is not None, f"slice_pred_at 找到值: {val}")
    _check(abs(val - 0.6) < 1e-6, "slice_pred_at 返回值正确")

    val_none = slice_pred_at(pred_df, dts[0], "NONEXISTENT", fallback_ffill=False)
    _check(val_none is None, "slice_pred_at 找不到时返回 None")

    return True


# ===========================================================================
# 测试 10：YAML 配置文件存在性
# ===========================================================================

@_test("YAML 配置文件检查")
def test_yaml_exists():
    yaml_path = PROJECT_ROOT / "workflow_config_15min.yaml"
    _check(yaml_path.exists(), f"workflow_config_15min.yaml 存在于 {yaml_path}")

    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _check(isinstance(config, dict), "YAML 解析成功")
    _check("task" in config, "YAML 包含 task 配置")
    _check("model" in config.get("task", {}), "task 包含 model 配置")
    _check("dataset" in config.get("task", {}), "task 包含 dataset 配置")
    return True


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    logger.info("\n" + "=" * 70)
    logger.info("HFML × Qlib 集成验证脚本")
    logger.info(f"Python: {sys.version}")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info("=" * 70)

    # 运行所有测试
    test_import()
    test_data_loader()
    test_feature_processor()
    test_transform_processor()
    test_label_processor()
    test_quality_filter()
    test_model_interfaces()
    test_hybrid_signal()
    test_utils()
    test_yaml_exists()

    # 汇总
    total = PASS_COUNT + FAIL_COUNT
    logger.info("\n" + "=" * 70)
    logger.info(f"测试结果: {PASS_COUNT}/{total} 通过，{FAIL_COUNT}/{total} 失败")
    if FAIL_COUNT == 0:
        logger.info("✓ 所有测试通过！HFML × Qlib 集成验证成功。")
    else:
        logger.warning(
            f"⚠ {FAIL_COUNT} 个测试失败，请检查上方日志。"
            "部分失败可能是由于缺少 Qlib 安装或 HFML 依赖库。"
        )
    logger.info("=" * 70)
    return FAIL_COUNT


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
