"""
单元测试 - 自定义特征因子
Tests for the custom features from both enhancement reports.

依据《hfml特征工程增强报告-56项目挖掘-最终可执行版.md》中的9个特征
以及《hfml特征工程增强报告-精选10特征-可执行版.md》中的10个精选特征。
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
# 10 精选特征1: buy_sell_pressure
# ---------------------------------------------------------------------------

class TestBuySellPressure:

    def test_basic(self):
        """报告中的手算验证"""
        input_df = pd.DataFrame({
            "high": [102, 103, 104, 103, 105],
            "low": [98, 100, 99, 100, 102],
            "close": [100, 102, 101, 102, 104],
            "volume": [1000, 1200, 1100, 1300, 1500]
        })
        from features.custom_features import compute_buy_sell_pressure
        result = compute_buy_sell_pressure(input_df)["buy_sell_pressure"]
        # Row 0: hl=4, buy=1000*2/4=500, sell=1000*2/4=500, net=0
        assert abs(result.iloc[0] - 0.0) < 0.01
        # Row 1: hl=3, buy=1200*2/3=800, sell=1200*1/3=400, net=400/1200≈0.333
        assert abs(result.iloc[1] - 0.333333) < 0.01

    def test_output_range(self):
        """输出应在 [-1, 1] 范围内"""
        df = _make_base_df(100)
        from features.custom_features import compute_buy_sell_pressure
        result = compute_buy_sell_pressure(df)["buy_sell_pressure"]
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()


# ---------------------------------------------------------------------------
# 10 精选特征2: volatility_skew
# ---------------------------------------------------------------------------

class TestVolatilitySkew:

    def test_right_vs_left(self):
        """右偏序列的偏度应大于左偏序列"""
        # Use longer sequences with clear asymmetry
        input_df_right = pd.DataFrame({
            "close": [100, 101, 100.5, 102, 101.5, 103, 102.5, 104,
                      103.5, 105, 104.5, 106, 105.5, 110]
        })
        input_df_left = pd.DataFrame({
            "close": [110, 109, 109.5, 108, 108.5, 107, 107.5, 106,
                      106.5, 105, 105.5, 104, 104.5, 100]
        })
        from features.custom_features import compute_volatility_skew
        result_right = compute_volatility_skew(
            input_df_right, window=10
        )["volatility_skew"]
        result_left = compute_volatility_skew(
            input_df_left, window=10
        )["volatility_skew"]
        # Right skew should be > left skew at end
        r_val = result_right.dropna().iloc[-1] if not result_right.dropna().empty else 0
        l_val = result_left.dropna().iloc[-1] if not result_left.dropna().empty else 0
        assert r_val > l_val

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_volatility_skew
        result = compute_volatility_skew(df)["volatility_skew"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 10 精选特征3: momentum_cross
# ---------------------------------------------------------------------------

class TestMomentumCross:

    def test_accel_vs_decel(self):
        """正值表示短期动量强于长期（如反弹），负值表示短期弱于长期（如回调）"""
        # Long-term down then short-term up → positive cross
        input_df_pos = pd.DataFrame({
            "close": [120, 118, 115, 112, 110, 108, 106, 104, 102, 100,
                      99, 100, 102, 105, 110, 116]
        })
        # Long-term up then short-term down → negative cross
        input_df_neg = pd.DataFrame({
            "close": [100, 102, 105, 108, 112, 116, 120, 124, 128, 130,
                      131, 130, 128, 125, 121, 116]
        })
        from features.custom_features import compute_momentum_cross
        result_pos = compute_momentum_cross(
            input_df_pos, fast_window=5, slow_window=15
        )["momentum_cross"]
        result_neg = compute_momentum_cross(
            input_df_neg, fast_window=5, slow_window=15
        )["momentum_cross"]
        assert result_pos.iloc[-1] > 0
        assert result_neg.iloc[-1] < 0

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_momentum_cross
        result = compute_momentum_cross(df)["momentum_cross"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 10 精选特征4: autocorrelation_1
# ---------------------------------------------------------------------------

class TestAutocorrelation1:

    def test_trend_positive(self):
        """强趋势序列的自相关应为正"""
        input_df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107]
        })
        from features.custom_features import compute_autocorrelation_1
        result = compute_autocorrelation_1(
            input_df, window=8
        )["autocorrelation_1"]
        # 趋势序列自相关应 > 0
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.iloc[-1] > 0

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_autocorrelation_1
        result = compute_autocorrelation_1(df)["autocorrelation_1"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 10 精选特征5: vwap_std
# ---------------------------------------------------------------------------

class TestVwapStd:

    def test_dispersed_gt_concentrated(self):
        """分散成交的标准差应大于集中成交"""
        np.random.seed(42)
        n = 50
        # Concentrated: tight price range
        df_conc = pd.DataFrame({
            "high": 100 + np.random.uniform(0, 1, n),
            "low": 100 - np.random.uniform(0, 1, n),
            "close": 100 + np.random.randn(n) * 0.3,
            "volume": np.full(n, 1000.0),
        })
        # Dispersed: wide price range
        df_disp = pd.DataFrame({
            "high": 100 + np.random.uniform(0, 10, n),
            "low": 100 - np.random.uniform(0, 10, n),
            "close": 100 + np.random.randn(n) * 5,
            "volume": np.full(n, 1000.0),
        })
        from features.custom_features import compute_vwap_std
        result_conc = compute_vwap_std(df_conc, window=20)["vwap_std"]
        result_disp = compute_vwap_std(df_disp, window=20)["vwap_std"]
        c = result_conc.dropna().iloc[-1]
        d = result_disp.dropna().iloc[-1]
        assert d > c

    def test_positive_output(self):
        df = _make_base_df(100)
        from features.custom_features import compute_vwap_std
        result = compute_vwap_std(df)["vwap_std"]
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# 10 精选特征6: tick_imbalance_proxy
# ---------------------------------------------------------------------------

class TestTickImbalanceProxy:

    def test_up_vs_down(self):
        """上涨序列应正偏，下跌序列应负偏"""
        input_df_up = pd.DataFrame({
            "close": [100, 101, 102, 101, 103, 104, 103, 105]
        })
        input_df_down = pd.DataFrame({
            "close": [100, 99, 98, 99, 97, 96, 97, 95]
        })
        from features.custom_features import compute_tick_imbalance_proxy
        result_up = compute_tick_imbalance_proxy(
            input_df_up, window=8
        )["tick_imbalance_proxy"]
        result_down = compute_tick_imbalance_proxy(
            input_df_down, window=8
        )["tick_imbalance_proxy"]
        assert result_up.iloc[-1] > 0
        assert result_down.iloc[-1] < 0

    def test_output_range(self):
        df = _make_base_df(100)
        from features.custom_features import compute_tick_imbalance_proxy
        result = compute_tick_imbalance_proxy(df)["tick_imbalance_proxy"]
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()


# ---------------------------------------------------------------------------
# 10 精选特征7: volatility_of_volatility
# ---------------------------------------------------------------------------

class TestVolatilityOfVolatility:

    def test_switch_gt_stable(self):
        """波动切换期的vov应大于稳定期"""
        # Stable: consistent small oscillations
        input_df_stable = pd.DataFrame({
            "close": [100, 101, 99, 100, 101, 99, 100, 101, 99, 100,
                      101, 99, 100, 101, 99]
        })
        # Switching: calm then wild
        input_df_switch = pd.DataFrame({
            "close": [100, 101, 100, 101, 100, 101, 100, 110, 90, 120,
                      80, 130, 70, 140, 60]
        })
        from features.custom_features import compute_volatility_of_volatility
        result_stable = compute_volatility_of_volatility(
            input_df_stable, vol_window=5, vov_window=5
        )["volatility_of_volatility"]
        result_switch = compute_volatility_of_volatility(
            input_df_switch, vol_window=5, vov_window=5
        )["volatility_of_volatility"]
        # Get last valid values
        s_valid = result_stable.dropna()
        w_valid = result_switch.dropna()
        if len(s_valid) > 0 and len(w_valid) > 0:
            assert w_valid.iloc[-1] > s_valid.iloc[-1]

    def test_positive_output(self):
        df = _make_base_df(100)
        from features.custom_features import compute_volatility_of_volatility
        result = compute_volatility_of_volatility(df)["volatility_of_volatility"]
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# 10 精选特征8: trend_strength_ratio
# ---------------------------------------------------------------------------

class TestTrendStrengthRatio:

    def test_output_range(self):
        """tanh归一化后应在 [-1, 1]"""
        df = _make_base_df(100)
        from features.custom_features import compute_trend_strength_ratio
        result = compute_trend_strength_ratio(df)["trend_strength_ratio"]
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_trend_strength_ratio
        result = compute_trend_strength_ratio(df)["trend_strength_ratio"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 10 精选特征9: volume_profile_skew
# ---------------------------------------------------------------------------

class TestVolumeProfileSkew:

    def test_high_vs_low(self):
        """高价区成交多应正偏，低价区成交多应负偏"""
        # Need enough data points (window + extra)
        # High-price volume concentration
        input_df_high = pd.DataFrame({
            "high": [102, 103, 104, 105, 104, 105, 106],
            "low": [98, 99, 100, 101, 100, 101, 102],
            "close": [101, 102, 103, 104, 103, 104, 105],
            "volume": [100, 200, 500, 500, 200, 500, 500]
        })
        # Low-price volume concentration
        input_df_low = pd.DataFrame({
            "high": [102, 103, 104, 105, 104, 105, 106],
            "low": [98, 99, 100, 101, 100, 101, 102],
            "close": [99, 100, 101, 102, 101, 102, 103],
            "volume": [500, 500, 200, 100, 200, 100, 100]
        })
        from features.custom_features import compute_volume_profile_skew
        result_high = compute_volume_profile_skew(
            input_df_high, window=5, bins=5
        )["volume_profile_skew"]
        result_low = compute_volume_profile_skew(
            input_df_low, window=5, bins=5
        )["volume_profile_skew"]
        h_valid = result_high.dropna()
        l_valid = result_low.dropna()
        assert len(h_valid) > 0, "No valid values for high-vol data"
        assert h_valid.iloc[-1] > l_valid.iloc[-1]

    def test_output_range(self):
        df = _make_base_df(100)
        from features.custom_features import compute_volume_profile_skew
        result = compute_volume_profile_skew(df)["volume_profile_skew"]
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()


# ---------------------------------------------------------------------------
# 10 精选特征10: hurst_exponent_approx
# ---------------------------------------------------------------------------

class TestHurstExponentApprox:

    def test_trend_series(self):
        """趋势序列的Hurst指数"""
        np.random.seed(42)
        trend = np.cumsum(np.random.randn(200) * 0.01 + 0.001) + 100
        df = pd.DataFrame({"close": trend})
        from features.custom_features import compute_hurst_exponent_approx
        result = compute_hurst_exponent_approx(
            df, min_window=10, max_window=50, min_periods=100
        )["hurst_exponent_approx"]
        valid = result.dropna()
        assert len(valid) > 0
        # Hurst should be a finite number
        assert np.isfinite(valid.iloc[-1])

    def test_output_shape(self):
        df = _make_base_df(200)
        from features.custom_features import compute_hurst_exponent_approx
        result = compute_hurst_exponent_approx(df)["hurst_exponent_approx"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_all_features_in_compute_all(self):
        """所有19个特征应在 compute_all_features 输出中"""
        from features.feature_engineering import compute_all_features
        df = _make_base_df(200)
        features_df = compute_all_features(df, period="5min")

        # 9个原有增强特征
        enhanced_features = [
            "divergence", "vwap_dev", "vol_state", "mom_slope",
            "rsi_slope", "vol_zscore", "gap_decay",
            "oi_price_alignment", "oi_price_magnitude",
        ]
        # 10个精选新增特征
        selected_features = [
            "buy_sell_pressure", "volatility_skew", "momentum_cross",
            "autocorrelation_1", "vwap_std", "tick_imbalance_proxy",
            "volatility_of_volatility", "trend_strength_ratio",
            "volume_profile_skew", "hurst_exponent_approx",
        ]
        for feat in enhanced_features + selected_features:
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

    def test_registry_has_all(self):
        """注册表应包含全部19个特征函数"""
        registry = FeatureRegistry()
        all_output_names = set()
        for entry in registry.get_all_entries().values():
            all_output_names.update(entry.output_names)

        expected = {
            # 9个增强特征
            "divergence", "vwap_dev", "vol_state", "mom_slope",
            "rsi_slope", "vol_zscore", "gap_decay",
            "oi_price_alignment", "oi_price_magnitude",
            # 10个精选特征
            "buy_sell_pressure", "volatility_skew", "momentum_cross",
            "autocorrelation_1", "vwap_std", "tick_imbalance_proxy",
            "volatility_of_volatility", "trend_strength_ratio",
            "volume_profile_skew", "hurst_exponent_approx",
        }
        assert expected.issubset(all_output_names), (
            f"注册表缺失特征: {expected - all_output_names}"
        )

    def test_feature_count_increase(self):
        """新增特征应在 compute_all_features 输出中"""
        from features.feature_engineering import compute_all_features
        df = _make_base_df(200)
        features_df = compute_all_features(df, period="5min")
        # All 10 new features should be present
        new_features = [
            "buy_sell_pressure", "volatility_skew", "momentum_cross",
            "autocorrelation_1", "vwap_std", "tick_imbalance_proxy",
            "volatility_of_volatility", "trend_strength_ratio",
            "volume_profile_skew", "hurst_exponent_approx",
        ]
        for feat in new_features:
            assert feat in features_df.columns, f"Missing: {feat}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
