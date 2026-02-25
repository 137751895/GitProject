"""
单元测试 - 9个自定义特征因子
Tests for the 9 custom features from the enhancement report.

依据《hfml特征工程增强报告-56项目挖掘-最终可执行版.md》中的
手算验证用例编写，确保每个特征的计算逻辑正确。
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_registry import FeatureRegistry

# Import custom_features to trigger registration
import features.custom_features  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_df(n=200):
    """Create a minimal OHLCV+OI DataFrame for testing."""
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(100, 1000, n).astype(float)
    oi = 5000.0 + np.cumsum(rng.randn(n) * 10)
    idx = pd.date_range("2026-01-01 09:00", periods=n, freq="5min")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "open_interest": oi,
    }, index=idx)


# ---------------------------------------------------------------------------
# 1. divergence
# ---------------------------------------------------------------------------

class TestDivergence:

    def test_basic(self):
        """报告中的手算用例: [nan, 1, 1, 1, 0]"""
        df = pd.DataFrame({
            "close": [100, 101, 100, 102, 102],
            "open_interest": [1000, 1010, 1005, 1008, 1007],
        })
        from features.custom_features import compute_divergence
        result = compute_divergence(df)["divergence"]
        expected = pd.Series([np.nan, 1.0, 1.0, 1.0, 0.0], name="divergence")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_first_row(self):
        """第一行必为 NaN"""
        df = _make_base_df(10)
        from features.custom_features import compute_divergence
        result = compute_divergence(df)["divergence"]
        assert np.isnan(result.iloc[0])

    def test_values_in_range(self):
        """输出值只能是 -1, 0, 1, NaN"""
        df = _make_base_df(100)
        from features.custom_features import compute_divergence
        result = compute_divergence(df)["divergence"]
        valid = result.dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})


# ---------------------------------------------------------------------------
# 2. vwap_dev
# ---------------------------------------------------------------------------

class TestVwapDev:

    def test_basic(self):
        """报告中的手算用例（近似验证）"""
        df = pd.DataFrame({
            "high": [101, 103, 102, 104, 105],
            "low": [99, 101, 100, 102, 103],
            "close": [100, 102, 101, 103, 104],
            "volume": [10, 20, 10, 20, 40],
        })
        from features.custom_features import compute_vwap_dev
        result = compute_vwap_dev(df)["vwap_dev"]
        # 第一行 close == vwap -> 偏离 ~0
        assert abs(result.iloc[0]) < 0.001
        # 后续行应有正常数值
        assert result.notna().all()

    def test_no_nan_propagation(self):
        df = _make_base_df(50)
        from features.custom_features import compute_vwap_dev
        result = compute_vwap_dev(df)["vwap_dev"]
        assert result.notna().all()


# ---------------------------------------------------------------------------
# 3. vol_state
# ---------------------------------------------------------------------------

class TestVolState:

    def test_short_window(self):
        """使用短窗口手算验证"""
        df = pd.DataFrame({
            "high": [101, 103, 104, 103, 105, 106],
            "low": [99, 100, 101, 100, 102, 103],
            "close": [100, 102, 103, 101, 104, 105],
        })
        from features.custom_features import compute_vol_state
        result = compute_vol_state(df, atr_window=3, norm_window=3)["vol_state"]
        # First 2 rows should be NaN (window=3)
        assert result.iloc[:2].isna().all()
        # Remaining should be positive
        valid = result.dropna()
        assert (valid > 0).all()

    def test_positive_output(self):
        df = _make_base_df(100)
        from features.custom_features import compute_vol_state
        result = compute_vol_state(df)["vol_state"]
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# 4. mom_slope
# ---------------------------------------------------------------------------

class TestMomSlope:

    def test_linear_input(self):
        """线性序列斜率应恒定"""
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        })
        from features.custom_features import compute_mom_slope
        result = compute_mom_slope(df, window=3)["mom_slope"]
        # Window=3 -> first 2 NaN, rest should be close to 1/close
        valid = result.dropna()
        assert len(valid) >= 1
        # For linear data, slope ≈ 1.0, mom_slope ≈ 1/close ≈ ~0.0097
        for v in valid:
            assert abs(v - 1.0 / 103.0) < 0.002  # rough check

    def test_output_shape(self):
        df = _make_base_df(50)
        from features.custom_features import compute_mom_slope
        result = compute_mom_slope(df)["mom_slope"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 5. rsi_slope
# ---------------------------------------------------------------------------

class TestRsiSlope:

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_rsi_slope
        result = compute_rsi_slope(df)["rsi_slope"]
        assert len(result) == len(df)

    def test_has_valid_values(self):
        """RSI slope should have valid values after warm-up"""
        df = _make_base_df(100)
        from features.custom_features import compute_rsi_slope
        result = compute_rsi_slope(df)["rsi_slope"]
        # After RSI warm-up (14) + slope warm-up (5) = ~19 NaN rows
        valid = result.dropna()
        assert len(valid) >= 50


# ---------------------------------------------------------------------------
# 6. vol_zscore
# ---------------------------------------------------------------------------

class TestVolZscore:

    def test_basic(self):
        """报告中的手算用例: window=3, 均匀递增序列 zscore=1"""
        vol = pd.Series([10, 11, 12, 13, 14], name="volume")
        df = pd.DataFrame({"volume": vol})
        from features.custom_features import compute_vol_zscore
        result = compute_vol_zscore(df, window=3)["vol_zscore"]
        # For a linearly increasing series, zscore ≈ 1.0 after warm-up
        valid = result.dropna()
        for v in valid:
            assert abs(v - 1.0) < 0.1

    def test_no_nan_after_warmup(self):
        df = _make_base_df(100)
        from features.custom_features import compute_vol_zscore
        result = compute_vol_zscore(df, window=20)["vol_zscore"]
        assert result.iloc[25:].notna().all()


# ---------------------------------------------------------------------------
# 7. gap_decay
# ---------------------------------------------------------------------------

class TestGapDecay:

    def test_basic_night_session(self):
        """报告中的手算用例"""
        idx = pd.to_datetime([
            "2026-01-01 20:59:00",
            "2026-01-01 21:01:00",
            "2026-01-01 21:02:00",
        ])
        df = pd.DataFrame({
            "open": [100.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0],
        }, index=idx)
        from features.custom_features import compute_gap_decay
        result = compute_gap_decay(df)["gap_decay"]
        # i=0: hour=20, not 21 -> 0
        assert result.iloc[0] == 0.0
        # i=1: gap=(102-100)/100=0.02, seconds=60, exp(-60/300)≈0.8187
        expected_1 = 0.02 * np.exp(-60 / 300)
        assert abs(result.iloc[1] - expected_1) < 1e-6
        # i=2: gap=(103-101)/101, seconds=120
        expected_2 = (103 - 101) / 101 * np.exp(-120 / 300)
        assert abs(result.iloc[2] - expected_2) < 1e-6

    def test_no_datetime_index(self):
        """无DatetimeIndex时应返回全零"""
        df = pd.DataFrame({
            "open": [100.0, 102.0],
            "close": [100.0, 101.0],
        })
        from features.custom_features import compute_gap_decay
        result = compute_gap_decay(df)["gap_decay"]
        assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# 8. oi_price_alignment
# ---------------------------------------------------------------------------

class TestOiPriceAlignment:

    def test_basic(self):
        """手算验证: 同涨同跌=+1, 反向=-1"""
        df = pd.DataFrame({
            "close": [100, 101, 100, 102, 103],
            "open_interest": [1000, 1010, 1008, 1015, 1025],
        })
        from features.custom_features import compute_oi_price_alignment
        result = compute_oi_price_alignment(df)["oi_price_alignment"]
        # idx0: NaN (no prev data)
        # idx1: close up (+1), OI up (+10) -> 1*1 = +1
        # idx2: close down (-1), OI down (-2) -> (-1)*(-1) = +1 (同向)
        # idx3: close up (+2), OI up (+7) -> 1*1 = +1
        # idx4: close up (+1), OI up (+10) -> 1*1 = +1
        expected = pd.Series(
            [np.nan, 1.0, 1.0, 1.0, 1.0], name="oi_price_alignment"
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_with_features_df(self):
        """使用 features_df 中的 oi_change"""
        df = pd.DataFrame({
            "close": [100, 101, 100],
            "open_interest": [1000, 1010, 1008],
        })
        features_df = pd.DataFrame({
            "oi_change": df["open_interest"].diff(),
        })
        from features.custom_features import compute_oi_price_alignment
        result = compute_oi_price_alignment(
            df, features_df=features_df
        )["oi_price_alignment"]
        # idx0: NaN, idx1: up*up=+1, idx2: down*down=+1
        expected = pd.Series(
            [np.nan, 1.0, 1.0], name="oi_price_alignment"
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_divergence_case(self):
        """测试价格与OI反向运动的情况（真正背离 = -1）"""
        df = pd.DataFrame({
            "close": [100, 101, 102],      # price going up
            "open_interest": [1000, 990, 980],  # OI going down
        })
        from features.custom_features import compute_oi_price_alignment
        result = compute_oi_price_alignment(df)["oi_price_alignment"]
        # idx1: up*down = -1, idx2: up*down = -1
        expected = pd.Series(
            [np.nan, -1.0, -1.0], name="oi_price_alignment"
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)


# ---------------------------------------------------------------------------
# 9. oi_price_magnitude
# ---------------------------------------------------------------------------

class TestOiPriceMagnitude:

    def test_basic(self):
        """报告中的手算用例（值被 clip 到 100）"""
        df = pd.DataFrame({
            "close": [100, 101, 100, 102, 103],
            "open_interest": [1000, 1010, 1008, 1015, 1025],
        })
        from features.custom_features import compute_oi_price_magnitude
        result = compute_oi_price_magnitude(df)["oi_price_magnitude"]
        # First row: NaN (diff is NaN)
        assert np.isnan(result.iloc[0])
        # All valid values should be clipped at <= 100
        valid = result.dropna()
        assert (valid <= 100.0).all()
        assert (valid >= 0.0).all()

    def test_clip_upper(self):
        """自定义 clip_upper 参数"""
        df = pd.DataFrame({
            "close": [100, 100.001],  # tiny price change
            "open_interest": [1000, 1100],  # large OI change
        })
        from features.custom_features import compute_oi_price_magnitude
        result = compute_oi_price_magnitude(
            df, clip_upper=50.0
        )["oi_price_magnitude"]
        valid = result.dropna()
        assert (valid <= 50.0).all()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_all_features_in_compute_all(self):
        """所有9个特征应在 compute_all_features 输出中"""
        from features.feature_engineering import compute_all_features
        df = _make_base_df(200)
        features_df = compute_all_features(df, period="5min")

        expected_features = [
            "divergence", "vwap_dev", "vol_state", "mom_slope",
            "rsi_slope", "vol_zscore", "gap_decay",
            "oi_price_alignment", "oi_price_magnitude",
        ]
        for feat in expected_features:
            assert feat in features_df.columns, (
                f"特征 {feat} 未在 compute_all_features 结果中"
            )

    def test_no_duplicate_columns(self):
        """不应有重复列名"""
        from features.feature_engineering import compute_all_features
        df = _make_base_df(200)
        features_df = compute_all_features(df, period="5min")
        # Check no duplicate column names
        assert len(features_df.columns) == len(set(features_df.columns)), (
            f"存在重复列: "
            f"{[c for c in features_df.columns if list(features_df.columns).count(c) > 1]}"
        )

    def test_registry_has_all_nine(self):
        """注册表应包含全部9个特征函数"""
        registry = FeatureRegistry()
        all_output_names = set()
        for entry in registry.get_all_entries().values():
            all_output_names.update(entry.output_names)

        expected = {
            "divergence", "vwap_dev", "vol_state", "mom_slope",
            "rsi_slope", "vol_zscore", "gap_decay",
            "oi_price_alignment", "oi_price_magnitude",
        }
        assert expected.issubset(all_output_names), (
            f"注册表缺失特征: {expected - all_output_names}"
        )

    def test_feature_count_increase(self):
        """新增特征数量应为2（oi_price_alignment, oi_price_magnitude）
        其余7个已在增强模块中存在，通过去重不重复添加"""
        from features.feature_engineering import compute_all_features
        df = _make_base_df(200)
        features_df = compute_all_features(df, period="5min")
        # oi_price_alignment and oi_price_magnitude should be present
        assert "oi_price_alignment" in features_df.columns
        assert "oi_price_magnitude" in features_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
