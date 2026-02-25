"""
单元测试 - 自定义特征因子
Tests for the custom features from both enhancement reports.

依据《hfml特征工程增强报告-56项目挖掘-最终可执行版.md》中的9个特征
以及《hfml特征工程增强报告-精选10特征-可执行版.md》中的10个精选特征。
依据《hfml特征工程增强报告-精选20特征-第二辑.md》集成的20+1个精选特征。
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
# 第二辑特征1: parkinson_volatility
# ---------------------------------------------------------------------------

class TestParkinsonVolatility:

    def test_basic(self):
        input_df = pd.DataFrame({
            "high": [101, 102, 103, 104, 103],
            "low": [99, 100, 101, 102, 101]
        })
        from features.custom_features import compute_parkinson_volatility
        result = compute_parkinson_volatility(input_df, window=3)["parkinson_volatility"]
        assert not result.isna().all()
        assert result.iloc[-1] > 0

    def test_positive_output(self):
        df = _make_base_df(100)
        from features.custom_features import compute_parkinson_volatility
        result = compute_parkinson_volatility(df)["parkinson_volatility"]
        valid = result.dropna()
        assert (valid > 0).all()


# ---------------------------------------------------------------------------
# 第二辑特征2: rogers_satchell_vol
# ---------------------------------------------------------------------------

class TestRogersSatchellVol:

    def test_basic(self):
        input_df = pd.DataFrame({
            "open": [100, 101, 102, 103, 102],
            "high": [102, 103, 104, 105, 104],
            "low": [98, 99, 100, 101, 100],
            "close": [101, 102, 103, 104, 103]
        })
        from features.custom_features import compute_rogers_satchell_vol
        result = compute_rogers_satchell_vol(input_df, window=3)["rogers_satchell_vol"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_rogers_satchell_vol
        result = compute_rogers_satchell_vol(df)["rogers_satchell_vol"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征3: yang_zhang_vol
# ---------------------------------------------------------------------------

class TestYangZhangVol:

    def test_basic(self):
        input_df = pd.DataFrame({
            "open": [100, 101, 102, 103, 102, 101],
            "high": [102, 103, 104, 105, 104, 103],
            "low": [98, 99, 100, 101, 100, 99],
            "close": [101, 102, 103, 104, 103, 102]
        })
        from features.custom_features import compute_yang_zhang_vol
        result = compute_yang_zhang_vol(input_df, window=3)["yang_zhang_vol"]
        valid = result.dropna()
        assert len(valid) > 0

    def test_positive_output(self):
        df = _make_base_df(100)
        from features.custom_features import compute_yang_zhang_vol
        result = compute_yang_zhang_vol(df)["yang_zhang_vol"]
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# 第二辑特征4: roll_impact
# ---------------------------------------------------------------------------

class TestRollImpact:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_roll_impact
        result = compute_roll_impact(df)["roll_impact"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_shape(self):
        df = _make_base_df(50)
        from features.custom_features import compute_roll_impact
        result = compute_roll_impact(df)["roll_impact"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征5: amihud_illiquidity
# ---------------------------------------------------------------------------

class TestAmihudIlliquidity:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_amihud_illiquidity
        result = compute_amihud_illiquidity(df)["amihud_illiquidity"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_shape(self):
        df = _make_base_df(50)
        from features.custom_features import compute_amihud_illiquidity
        result = compute_amihud_illiquidity(df)["amihud_illiquidity"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征6: pastor_stambaugh
# ---------------------------------------------------------------------------

class TestPastorStambaugh:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_pastor_stambaugh
        result = compute_pastor_stambaugh(df)["pastor_stambaugh"]
        valid = result.dropna()
        assert len(valid) > 0

    def test_output_shape(self):
        df = _make_base_df(50)
        from features.custom_features import compute_pastor_stambaugh
        result = compute_pastor_stambaugh(df)["pastor_stambaugh"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征7: roll_spread_estimate
# ---------------------------------------------------------------------------

class TestRollSpreadEstimate:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_roll_spread_estimate
        result = compute_roll_spread_estimate(df)["roll_spread_estimate"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_shape(self):
        df = _make_base_df(50)
        from features.custom_features import compute_roll_spread_estimate
        result = compute_roll_spread_estimate(df)["roll_spread_estimate"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征8: corwin_schultz_spread
# ---------------------------------------------------------------------------

class TestCorwinSchultzSpread:

    def test_basic(self):
        input_df = pd.DataFrame({
            "high": [102, 103, 104, 105, 104],
            "low": [98, 99, 100, 101, 100]
        })
        from features.custom_features import compute_corwin_schultz_spread
        result = compute_corwin_schultz_spread(input_df)["corwin_schultz_spread"]
        # First row NaN, rest may or may not have values
        assert len(result) == len(input_df)

    def test_non_negative(self):
        df = _make_base_df(100)
        from features.custom_features import compute_corwin_schultz_spread
        result = compute_corwin_schultz_spread(df)["corwin_schultz_spread"]
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# 第二辑特征9: volume_synchronized_vol
# ---------------------------------------------------------------------------

class TestVolumeSynchronizedVol:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_volume_synchronized_vol
        result = compute_volume_synchronized_vol(df)["volume_synchronized_vol"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_shape(self):
        df = _make_base_df(50)
        from features.custom_features import compute_volume_synchronized_vol
        result = compute_volume_synchronized_vol(df)["volume_synchronized_vol"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征10: volume_weighted_atr
# ---------------------------------------------------------------------------

class TestVolumeWeightedAtr:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_volume_weighted_atr
        result = compute_volume_weighted_atr(df)["volume_weighted_atr"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_with_tr(self):
        """Test with precomputed tr in features_df"""
        df = _make_base_df(100)
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        features_df = pd.DataFrame({"tr": tr})
        from features.custom_features import compute_volume_weighted_atr
        result = compute_volume_weighted_atr(df, features_df=features_df)["volume_weighted_atr"]
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# 第二辑特征11: serial_correlation
# ---------------------------------------------------------------------------

class TestSerialCorrelation:

    def test_trend_positive(self):
        """强趋势序列自相关应为正"""
        input_df = pd.DataFrame({
            "close": [100.0 + i * 0.5 for i in range(50)]
        })
        from features.custom_features import compute_serial_correlation
        result = compute_serial_correlation(input_df, window=20)["serial_correlation"]
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.iloc[-1] > 0

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_serial_correlation
        result = compute_serial_correlation(df)["serial_correlation"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征12: partial_autocorrelation
# ---------------------------------------------------------------------------

class TestPartialAutocorrelation:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_partial_autocorrelation
        result = compute_partial_autocorrelation(df)["partial_autocorrelation"]
        valid = result.dropna()
        assert len(valid) > 0

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_partial_autocorrelation
        result = compute_partial_autocorrelation(df)["partial_autocorrelation"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征13: variance_ratio
# ---------------------------------------------------------------------------

class TestVarianceRatio:

    def test_basic(self):
        df = _make_base_df(200)
        from features.custom_features import compute_variance_ratio
        result = compute_variance_ratio(df, q_periods=5, window=50)["variance_ratio"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_output_shape(self):
        df = _make_base_df(200)
        from features.custom_features import compute_variance_ratio
        result = compute_variance_ratio(df)["variance_ratio"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征14: bid_ask_spread_proxy
# ---------------------------------------------------------------------------

class TestBidAskSpreadProxy:

    def test_basic(self):
        input_df = pd.DataFrame({
            "high": [102, 103, 104],
            "low": [98, 99, 100]
        })
        from features.custom_features import compute_bid_ask_spread_proxy
        result = compute_bid_ask_spread_proxy(input_df)["bid_ask_spread_proxy"]
        # (102-98)/100 = 0.04
        assert abs(result.iloc[0] - 0.04) < 0.001
        assert result.notna().all()

    def test_positive_output(self):
        df = _make_base_df(100)
        from features.custom_features import compute_bid_ask_spread_proxy
        result = compute_bid_ask_spread_proxy(df)["bid_ask_spread_proxy"]
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# 第二辑特征15: effective_spread_proxy
# ---------------------------------------------------------------------------

class TestEffectiveSpreadProxy:

    def test_basic(self):
        input_df = pd.DataFrame({
            "high": [102, 103, 104],
            "low": [98, 99, 100],
            "close": [100, 102, 101]
        })
        from features.custom_features import compute_effective_spread_proxy
        result = compute_effective_spread_proxy(input_df)["effective_spread_proxy"]
        # mid=(102+98)/2=100, close=100, spread=0
        assert abs(result.iloc[0]) < 0.001
        assert result.notna().all()

    def test_non_negative(self):
        df = _make_base_df(100)
        from features.custom_features import compute_effective_spread_proxy
        result = compute_effective_spread_proxy(df)["effective_spread_proxy"]
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# 第二辑特征16: price_reversal_metric
# ---------------------------------------------------------------------------

class TestPriceReversalMetric:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_price_reversal_metric
        result = compute_price_reversal_metric(df)["price_reversal_metric"]
        valid = result.dropna()
        assert len(valid) > 0

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_price_reversal_metric
        result = compute_price_reversal_metric(df)["price_reversal_metric"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征17: volume_price_correlation
# ---------------------------------------------------------------------------

class TestVolumePriceCorrelation:

    def test_positive_corr(self):
        """放量大涨、缩量小涨 → returns与volume正相关"""
        # Returns vary: large returns at high volume, small at low volume
        input_df = pd.DataFrame({
            "close": [100, 100.1, 101, 101.1, 103, 103.1, 106, 106.1, 110, 110.1],
            "volume": [100, 100, 500, 100, 500, 100, 500, 100, 500, 100],
        })
        from features.custom_features import compute_volume_price_correlation
        result = compute_volume_price_correlation(input_df, window=8)["volume_price_correlation"]
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.iloc[-1] > 0

    def test_output_range(self):
        df = _make_base_df(100)
        from features.custom_features import compute_volume_price_correlation
        result = compute_volume_price_correlation(df)["volume_price_correlation"]
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()


# ---------------------------------------------------------------------------
# 第二辑特征18: open_interest_momentum
# ---------------------------------------------------------------------------

class TestOpenInterestMomentum:

    def test_basic(self):
        input_df = pd.DataFrame({
            "open_interest": [1000, 1010, 1030, 1060, 1100, 1150, 1210]
        })
        from features.custom_features import compute_open_interest_momentum
        result = compute_open_interest_momentum(input_df, fast_window=2, slow_window=5)["open_interest_momentum"]
        assert len(result) == len(input_df)

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_open_interest_momentum
        result = compute_open_interest_momentum(df)["open_interest_momentum"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征19: long_short_ratio_proxy
# ---------------------------------------------------------------------------

class TestLongShortRatioProxy:

    def test_long_dominant(self):
        """价涨+OI增应为正"""
        input_df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
            "open_interest": [1000, 1010, 1020, 1030, 1040]
        })
        from features.custom_features import compute_long_short_ratio_proxy
        result = compute_long_short_ratio_proxy(input_df, smooth_window=2)["long_short_ratio_proxy"]
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.iloc[-1] > 0

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_long_short_ratio_proxy
        result = compute_long_short_ratio_proxy(df)["long_short_ratio_proxy"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑特征20: kurtosis_returns (herfindahl_volume was replaced by kurtosis)
# ---------------------------------------------------------------------------

class TestKurtosisReturns:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_kurtosis_returns
        result = compute_kurtosis_returns(df)["kurtosis_returns"]
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()  # kurtosis is always positive

    def test_output_shape(self):
        df = _make_base_df(100)
        from features.custom_features import compute_kurtosis_returns
        result = compute_kurtosis_returns(df)["kurtosis_returns"]
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# 第二辑bonus: herfindahl_volume
# ---------------------------------------------------------------------------

class TestHerfindahlVolume:

    def test_basic(self):
        df = _make_base_df(100)
        from features.custom_features import compute_herfindahl_volume
        result = compute_herfindahl_volume(df)["herfindahl_volume"]
        valid = result.dropna()
        assert len(valid) > 0
        # Herfindahl is in (0, 1] for window > 1
        assert (valid > 0).all()
        assert (valid <= 1).all()

    def test_uniform_volume(self):
        """Uniform volume should give HHI = 1/window"""
        df = pd.DataFrame({"volume": np.full(50, 100.0)})
        from features.custom_features import compute_herfindahl_volume
        result = compute_herfindahl_volume(df, window=20)["herfindahl_volume"]
        valid = result.dropna()
        # HHI = 20 * (1/20)^2 = 1/20 = 0.05
        for v in valid:
            assert abs(v - 0.05) < 0.001


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_all_features_in_compute_all(self):
        """所有19+21个特征应在 compute_all_features 输出中"""
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
        # 20+1个第二辑精选特征
        second_batch_features = [
            "parkinson_volatility", "rogers_satchell_vol", "yang_zhang_vol",
            "roll_impact", "amihud_illiquidity", "pastor_stambaugh",
            "roll_spread_estimate", "corwin_schultz_spread",
            "volume_synchronized_vol", "volume_weighted_atr",
            "serial_correlation", "partial_autocorrelation", "variance_ratio",
            "bid_ask_spread_proxy", "effective_spread_proxy", "price_reversal_metric",
            "volume_price_correlation", "open_interest_momentum", "long_short_ratio_proxy",
            "kurtosis_returns", "herfindahl_volume",
        ]
        for feat in enhanced_features + selected_features + second_batch_features:
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
        """注册表应包含全部19+21+20个特征函数"""
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
            # 20+1个第二辑精选特征
            "parkinson_volatility", "rogers_satchell_vol", "yang_zhang_vol",
            "roll_impact", "amihud_illiquidity", "pastor_stambaugh",
            "roll_spread_estimate", "corwin_schultz_spread",
            "volume_synchronized_vol", "volume_weighted_atr",
            "serial_correlation", "partial_autocorrelation", "variance_ratio",
            "bid_ask_spread_proxy", "effective_spread_proxy", "price_reversal_metric",
            "volume_price_correlation", "open_interest_momentum", "long_short_ratio_proxy",
            "kurtosis_returns", "herfindahl_volume",
            # 20个第三辑精选特征
            "fractal_dimension", "lyapunov_exponent", "approximate_entropy",
            "sample_entropy", "permutation_entropy", "skewness_3rd",
            "co_skewness", "co_kurtosis", "market_microstructure_noise",
            "price_delay", "volume_synchronized_returns", "tick_rule_imbalance",
            "volume_weighted_price_range", "volatility_term_structure",
            "volatility_convexity", "cross_asset_correlation",
            "correlation_breakdown", "regime_switching_probability",
            "hurst_exponent_refined", "detrended_fluctuation",
        }
        assert expected.issubset(all_output_names), (
            f"注册表缺失特征: {expected - all_output_names}"
        )

    def test_feature_count_increase(self):
        """新增特征应在 compute_all_features 输出中"""
        from features.feature_engineering import compute_all_features
        df = _make_base_df(300)
        features_df = compute_all_features(df, period="5min")
        # All batches 1-3 features should be present
        new_features = [
            "buy_sell_pressure", "volatility_skew", "momentum_cross",
            "autocorrelation_1", "vwap_std", "tick_imbalance_proxy",
            "volatility_of_volatility", "trend_strength_ratio",
            "volume_profile_skew", "hurst_exponent_approx",
            "parkinson_volatility", "rogers_satchell_vol", "yang_zhang_vol",
            "roll_impact", "amihud_illiquidity", "pastor_stambaugh",
            "roll_spread_estimate", "corwin_schultz_spread",
            "volume_synchronized_vol", "volume_weighted_atr",
            "serial_correlation", "partial_autocorrelation", "variance_ratio",
            "bid_ask_spread_proxy", "effective_spread_proxy", "price_reversal_metric",
            "volume_price_correlation", "open_interest_momentum", "long_short_ratio_proxy",
            "kurtosis_returns", "herfindahl_volume",
            # Batch 3
            "fractal_dimension", "lyapunov_exponent", "approximate_entropy",
            "sample_entropy", "permutation_entropy", "skewness_3rd",
            "co_skewness", "co_kurtosis", "market_microstructure_noise",
            "price_delay", "volume_synchronized_returns", "tick_rule_imbalance",
            "volume_weighted_price_range", "volatility_term_structure",
            "volatility_convexity", "cross_asset_correlation",
            "correlation_breakdown", "regime_switching_probability",
            "hurst_exponent_refined", "detrended_fluctuation",
        ]
        for feat in new_features:
            assert feat in features_df.columns, f"Missing: {feat}"


# ============================================================
# 第三辑特征测试（20个）
# ============================================================


class TestFractalDimension:
    def test_basic(self):
        from features.custom_features import compute_fractal_dimension
        df = _make_base_df(200)
        result = compute_fractal_dimension(df, window=50)
        assert "fractal_dimension" in result.columns
        assert result["fractal_dimension"].notna().sum() > 0

    def test_trend_vs_random(self):
        from features.custom_features import compute_fractal_dimension
        trend = np.cumsum(np.ones(200) * 0.1) + 100
        df_trend = pd.DataFrame({"close": trend, "open": trend, "high": trend + 1, "low": trend - 1, "volume": np.ones(200) * 1000, "open_interest": np.ones(200) * 5000})
        result_trend = compute_fractal_dimension(df_trend, window=50)["fractal_dimension"]
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(200) * 0.1) + 100
        df_random = pd.DataFrame({"close": random_walk, "open": random_walk, "high": random_walk + 1, "low": random_walk - 1, "volume": np.ones(200) * 1000, "open_interest": np.ones(200) * 5000})
        result_random = compute_fractal_dimension(df_random, window=50)["fractal_dimension"]
        fd_trend = result_trend.dropna().iloc[-1] if not result_trend.dropna().empty else 1.0
        fd_random = result_random.dropna().iloc[-1] if not result_random.dropna().empty else 1.5
        assert fd_trend < fd_random


class TestLyapunovExponent:
    def test_basic(self):
        from features.custom_features import compute_lyapunov_exponent
        df = _make_base_df(300)
        result = compute_lyapunov_exponent(df, window=100, tau=5)
        assert "lyapunov_exponent" in result.columns
        assert result["lyapunov_exponent"].notna().sum() > 0


class TestApproximateEntropy:
    def test_basic(self):
        from features.custom_features import compute_approximate_entropy
        df = _make_base_df(200)
        result = compute_approximate_entropy(df, window=50)
        assert "approximate_entropy" in result.columns
        assert result["approximate_entropy"].notna().sum() > 0


class TestSampleEntropy:
    def test_basic(self):
        from features.custom_features import compute_sample_entropy
        df = _make_base_df(200)
        result = compute_sample_entropy(df, window=50)
        assert "sample_entropy" in result.columns
        assert result["sample_entropy"].notna().sum() > 0


class TestPermutationEntropy:
    def test_basic(self):
        from features.custom_features import compute_permutation_entropy
        df = _make_base_df(200)
        result = compute_permutation_entropy(df, window=50)
        assert "permutation_entropy" in result.columns
        assert result["permutation_entropy"].notna().sum() > 0


class TestSkewness3rd:
    def test_basic(self):
        from features.custom_features import compute_skewness_3rd
        df = _make_base_df(100)
        result = compute_skewness_3rd(df, window=20)
        assert "skewness_3rd" in result.columns
        assert result["skewness_3rd"].notna().sum() > 0


class TestCoSkewness:
    def test_basic(self):
        from features.custom_features import compute_co_skewness
        df = _make_base_df(100)
        result = compute_co_skewness(df, window=20)
        assert "co_skewness" in result.columns
        assert result["co_skewness"].notna().sum() > 0


class TestCoKurtosis:
    def test_basic(self):
        from features.custom_features import compute_co_kurtosis
        df = _make_base_df(100)
        result = compute_co_kurtosis(df, window=20)
        assert "co_kurtosis" in result.columns
        assert result["co_kurtosis"].notna().sum() > 0


class TestMarketMicrostructureNoise:
    def test_basic(self):
        from features.custom_features import compute_market_microstructure_noise
        df = _make_base_df(100)
        result = compute_market_microstructure_noise(df, window=20)
        assert "market_microstructure_noise" in result.columns
        assert result["market_microstructure_noise"].notna().sum() > 0


class TestPriceDelay:
    def test_basic(self):
        from features.custom_features import compute_price_delay
        df = _make_base_df(200)
        result = compute_price_delay(df, window=50, max_lag=5)
        assert "price_delay" in result.columns
        assert result["price_delay"].notna().sum() > 0


class TestVolumeSynchronizedReturns:
    def test_basic(self):
        from features.custom_features import compute_volume_synchronized_returns
        df = _make_base_df(100)
        result = compute_volume_synchronized_returns(df, window=20)
        assert "volume_synchronized_returns" in result.columns
        assert result["volume_synchronized_returns"].notna().sum() > 0


class TestTickRuleImbalance:
    def test_basic(self):
        from features.custom_features import compute_tick_rule_imbalance
        df = _make_base_df(100)
        result = compute_tick_rule_imbalance(df, window=20)
        assert "tick_rule_imbalance" in result.columns
        vals = result["tick_rule_imbalance"].dropna()
        assert len(vals) > 0
        assert (vals >= -1.0).all() and (vals <= 1.0).all()


class TestVolumeWeightedPriceRange:
    def test_basic(self):
        from features.custom_features import compute_volume_weighted_price_range
        df = _make_base_df(100)
        result = compute_volume_weighted_price_range(df, window=20)
        assert "volume_weighted_price_range" in result.columns
        assert result["volume_weighted_price_range"].notna().sum() > 0


class TestVolatilityTermStructure:
    def test_basic(self):
        from features.custom_features import compute_volatility_term_structure
        df = _make_base_df(200)
        result = compute_volatility_term_structure(df)
        assert "volatility_term_structure" in result.columns
        assert result["volatility_term_structure"].notna().sum() > 0


class TestVolatilityConvexity:
    def test_basic(self):
        from features.custom_features import compute_volatility_convexity
        df = _make_base_df(200)
        result = compute_volatility_convexity(df)
        assert "volatility_convexity" in result.columns
        assert result["volatility_convexity"].notna().sum() > 0


class TestCrossAssetCorrelation:
    def test_basic(self):
        from features.custom_features import compute_cross_asset_correlation
        df = _make_base_df(200)
        result = compute_cross_asset_correlation(df, window=60)
        assert "cross_asset_correlation" in result.columns
        assert result["cross_asset_correlation"].notna().sum() > 0


class TestCorrelationBreakdown:
    def test_basic(self):
        from features.custom_features import compute_correlation_breakdown
        df = _make_base_df(200)
        result = compute_correlation_breakdown(df, window=60)
        assert "correlation_breakdown" in result.columns
        assert result["correlation_breakdown"].notna().sum() > 0


class TestRegimeSwitchingProbability:
    def test_basic(self):
        from features.custom_features import compute_regime_switching_probability
        df = _make_base_df(200)
        result = compute_regime_switching_probability(df, window=100)
        assert "regime_switching_probability" in result.columns
        vals = result["regime_switching_probability"].dropna()
        assert len(vals) > 0
        assert (vals >= 0).all() and (vals <= 1).all()


class TestHurstExponentRefined:
    def test_basic(self):
        from features.custom_features import compute_hurst_exponent_refined
        df = _make_base_df(300)
        result = compute_hurst_exponent_refined(df, window=100)
        assert "hurst_exponent_refined" in result.columns
        assert result["hurst_exponent_refined"].notna().sum() > 0


class TestDetrendedFluctuation:
    def test_basic(self):
        from features.custom_features import compute_detrended_fluctuation
        df = _make_base_df(300)
        result = compute_detrended_fluctuation(df, window=100)
        assert "detrended_fluctuation" in result.columns
        assert result["detrended_fluctuation"].notna().sum() > 0


class TestBatch3Integration:
    def test_all_20_registered(self):
        from features.feature_registry import FeatureRegistry
        import features.custom_features
        registry = FeatureRegistry()
        all_output_names = set()
        for name, entry in registry.get_all_entries().items():
            all_output_names.update(entry.output_names)
        batch3 = {
            "fractal_dimension", "lyapunov_exponent", "approximate_entropy",
            "sample_entropy", "permutation_entropy", "skewness_3rd",
            "co_skewness", "co_kurtosis", "market_microstructure_noise",
            "price_delay", "volume_synchronized_returns", "tick_rule_imbalance",
            "volume_weighted_price_range", "volatility_term_structure",
            "volatility_convexity", "cross_asset_correlation",
            "correlation_breakdown", "regime_switching_probability",
            "hurst_exponent_refined", "detrended_fluctuation",
        }
        assert batch3.issubset(all_output_names), f"Missing: {batch3 - all_output_names}"

    def test_compute_all_features_includes_batch3(self):
        from features.feature_engineering import compute_all_features
        df = _make_base_df(300)
        features_df = compute_all_features(df, period="5min")
        batch3 = [
            "fractal_dimension", "lyapunov_exponent", "approximate_entropy",
            "sample_entropy", "permutation_entropy", "skewness_3rd",
            "co_skewness", "co_kurtosis", "market_microstructure_noise",
            "price_delay", "volume_synchronized_returns", "tick_rule_imbalance",
            "volume_weighted_price_range", "volatility_term_structure",
            "volatility_convexity", "cross_asset_correlation",
            "correlation_breakdown", "regime_switching_probability",
            "hurst_exponent_refined", "detrended_fluctuation",
        ]
        for feat in batch3:
            assert feat in features_df.columns, f"Missing from compute_all_features: {feat}"




# ===========================================================================
# Batch-4 Tests (20 new features)
# ===========================================================================

def _make_batch4_df(n=300):
    """Create a DataFrame for batch-4 testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:00", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.5) + 0.1
    low = close - np.abs(np.random.randn(n) * 0.5) - 0.1
    open_ = close + np.random.randn(n) * 0.3
    volume = np.abs(np.random.randn(n) * 200) + 100
    oi = 5000.0 + np.cumsum(np.random.randn(n) * 10)
    return pd.DataFrame({
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "open_interest": oi,
    }, index=dates)


class TestSpectralRatio:
    def test_values_in_0_1(self):
        from features.custom_features import compute_spectral_ratio
        df = _make_batch4_df(200)
        result = compute_spectral_ratio(df)
        valid = result["spectral_ratio"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_output_length(self):
        from features.custom_features import compute_spectral_ratio
        df = _make_batch4_df(200)
        result = compute_spectral_ratio(df)
        assert len(result) == 200

    def test_column_name(self):
        from features.custom_features import compute_spectral_ratio
        df = _make_batch4_df(200)
        result = compute_spectral_ratio(df)
        assert "spectral_ratio" in result.columns


class TestDominantFrequency:
    def test_positive_values(self):
        from features.custom_features import compute_dominant_frequency
        df = _make_batch4_df(300)
        result = compute_dominant_frequency(df)
        valid = result["dominant_frequency"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_length(self):
        from features.custom_features import compute_dominant_frequency
        df = _make_batch4_df(300)
        result = compute_dominant_frequency(df)
        assert len(result) == 300


class TestWaveletEnergy:
    def test_five_columns(self):
        from features.custom_features import compute_wavelet_energy
        df = _make_batch4_df(300)
        result = compute_wavelet_energy(df)
        expected_cols = [f"wavelet_energy_s{i}" for i in range(1, 6)]
        for col in expected_cols:
            assert col in result.columns

    def test_non_negative(self):
        from features.custom_features import compute_wavelet_energy
        df = _make_batch4_df(300)
        result = compute_wavelet_energy(df)
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert (valid >= 0).all(), f"{col} has negative values"

    def test_output_length(self):
        from features.custom_features import compute_wavelet_energy
        df = _make_batch4_df(300)
        result = compute_wavelet_energy(df)
        assert len(result) == 300


class TestWaveletEntropy:
    def test_with_precomputed_energy(self):
        from features.custom_features import compute_wavelet_entropy, compute_wavelet_energy
        df = _make_batch4_df(300)
        energy_df = compute_wavelet_energy(df)
        result = compute_wavelet_entropy(df, features_df=energy_df)
        valid = result["wavelet_entropy"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_nan_without_features(self):
        from features.custom_features import compute_wavelet_entropy
        df = _make_batch4_df(300)
        result = compute_wavelet_entropy(df, features_df=None)
        assert result["wavelet_entropy"].isna().all()

    def test_nan_with_incomplete_features(self):
        from features.custom_features import compute_wavelet_entropy
        df = _make_batch4_df(300)
        partial = pd.DataFrame({"wavelet_energy_s1": np.ones(300)}, index=df.index)
        result = compute_wavelet_entropy(df, features_df=partial)
        assert result["wavelet_entropy"].isna().all()


class TestExtremeValueIndex:
    def test_positive_values(self):
        from features.custom_features import compute_extreme_value_index
        df = _make_batch4_df(300)
        result = compute_extreme_value_index(df)
        valid = result["extreme_value_index"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_length(self):
        from features.custom_features import compute_extreme_value_index
        df = _make_batch4_df(300)
        result = compute_extreme_value_index(df)
        assert len(result) == 300


class TestTailDependence:
    def test_nan_without_benchmark(self):
        from features.custom_features import compute_tail_dependence
        df = _make_batch4_df(300)
        result = compute_tail_dependence(df, benchmark_col=None)
        assert result["tail_dependence"].isna().all()

    def test_nan_with_missing_benchmark_col(self):
        from features.custom_features import compute_tail_dependence
        df = _make_batch4_df(300)
        result = compute_tail_dependence(df, benchmark_col="nonexistent")
        assert result["tail_dependence"].isna().all()


class TestCopulaDependence:
    def test_nan_without_benchmark(self):
        from features.custom_features import compute_copula_dependence
        df = _make_batch4_df(300)
        result = compute_copula_dependence(df, benchmark_col=None)
        assert result["copula_dependence"].isna().all()

    def test_nan_with_missing_benchmark_col(self):
        from features.custom_features import compute_copula_dependence
        df = _make_batch4_df(300)
        result = compute_copula_dependence(df, benchmark_col="nonexistent")
        assert result["copula_dependence"].isna().all()


class TestRankCorrelation:
    def test_values_in_minus1_1(self):
        from features.custom_features import compute_rank_correlation
        df = _make_batch4_df(300)
        result = compute_rank_correlation(df, col1="close", col2="volume")
        valid = result["rank_correlation"].dropna()
        assert len(valid) > 0
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_output_length(self):
        from features.custom_features import compute_rank_correlation
        df = _make_batch4_df(300)
        result = compute_rank_correlation(df)
        assert len(result) == 300

    def test_nan_with_missing_col(self):
        from features.custom_features import compute_rank_correlation
        df = _make_batch4_df(300)
        result = compute_rank_correlation(df, col1="close", col2="nonexistent")
        assert result["rank_correlation"].isna().all()


class TestMlDerivedVolatility:
    def test_positive_values(self):
        from features.custom_features import compute_ml_derived_volatility
        df = _make_batch4_df(300)
        result = compute_ml_derived_volatility(df)
        valid = result["ml_derived_volatility"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_length(self):
        from features.custom_features import compute_ml_derived_volatility
        df = _make_batch4_df(300)
        result = compute_ml_derived_volatility(df)
        assert len(result) == 300


class TestMlDerivedTrend:
    def test_outputs_slope_not_price(self):
        from features.custom_features import compute_ml_derived_trend
        df = _make_batch4_df(300)
        result = compute_ml_derived_trend(df)
        valid = result["ml_derived_trend"].dropna()
        assert len(valid) > 0
        # Slope values should be much smaller than price levels
        assert valid.abs().max() < df["close"].max()

    def test_output_length(self):
        from features.custom_features import compute_ml_derived_trend
        df = _make_batch4_df(300)
        result = compute_ml_derived_trend(df)
        assert len(result) == 300


class TestOrderBookImbalanceProxy:
    def test_valid_values(self):
        from features.custom_features import compute_order_book_imbalance_proxy
        df = _make_batch4_df(300)
        result = compute_order_book_imbalance_proxy(df)
        valid = result["order_book_imbalance_proxy"].dropna()
        assert len(valid) > 0
        assert np.isfinite(valid).all()

    def test_output_length(self):
        from features.custom_features import compute_order_book_imbalance_proxy
        df = _make_batch4_df(300)
        result = compute_order_book_imbalance_proxy(df)
        assert len(result) == 300


class TestDepthPressure:
    def test_positive_values(self):
        from features.custom_features import compute_depth_pressure
        df = _make_batch4_df(300)
        result = compute_depth_pressure(df)
        valid = result["depth_pressure"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_output_length(self):
        from features.custom_features import compute_depth_pressure
        df = _make_batch4_df(300)
        result = compute_depth_pressure(df)
        assert len(result) == 300


class TestHerdingBehavior:
    def test_clipped_values(self):
        from features.custom_features import compute_herding_behavior
        df = _make_batch4_df(300)
        result = compute_herding_behavior(df)
        valid = result["herding_behavior"].dropna()
        assert len(valid) > 0
        assert (valid >= -5).all() and (valid <= 5).all()

    def test_output_length(self):
        from features.custom_features import compute_herding_behavior
        df = _make_batch4_df(300)
        result = compute_herding_behavior(df)
        assert len(result) == 300


class TestOverreactionScore:
    def test_non_negative(self):
        from features.custom_features import compute_overreaction_score
        df = _make_batch4_df(300)
        result = compute_overreaction_score(df)
        valid = result["overreaction_score"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_no_look_ahead_bias(self):
        """Verify overreaction_score at index i only depends on data[:i+1]."""
        from features.custom_features import compute_overreaction_score
        df = _make_batch4_df(300)
        full_result = compute_overreaction_score(df)
        # Compute on truncated data; last valid value should match
        trunc = df.iloc[:200]
        trunc_result = compute_overreaction_score(trunc)
        # The value at index 199 should be the same in both
        assert np.isclose(
            full_result["overreaction_score"].iloc[199],
            trunc_result["overreaction_score"].iloc[199],
            equal_nan=True,
        )

    def test_output_length(self):
        from features.custom_features import compute_overreaction_score
        df = _make_batch4_df(300)
        result = compute_overreaction_score(df)
        assert len(result) == 300


class TestCalendarEffect:
    def test_produces_4_columns(self):
        from features.custom_features import compute_calendar_effect
        df = _make_batch4_df(300)
        result = compute_calendar_effect(df)
        expected = {"morning_session", "afternoon_session", "night_session", "session_vol_ratio"}
        assert expected.issubset(set(result.columns))

    def test_session_binary(self):
        from features.custom_features import compute_calendar_effect
        df = _make_batch4_df(300)
        result = compute_calendar_effect(df)
        for col in ["morning_session", "afternoon_session", "night_session"]:
            valid = result[col].dropna()
            assert set(valid.unique()).issubset({0.0, 1.0})

    def test_output_length(self):
        from features.custom_features import compute_calendar_effect
        df = _make_batch4_df(300)
        result = compute_calendar_effect(df)
        assert len(result) == 300


class TestSeasonalityStrength:
    def test_produces_2_columns(self):
        from features.custom_features import compute_seasonality_strength
        df = _make_batch4_df(300)
        result = compute_seasonality_strength(df)
        assert "hour_seasonality" in result.columns
        assert "weekday_seasonality" in result.columns

    def test_output_length(self):
        from features.custom_features import compute_seasonality_strength
        df = _make_batch4_df(300)
        result = compute_seasonality_strength(df)
        assert len(result) == 300


class TestHigherOrderCumulant:
    def test_produces_2_columns(self):
        from features.custom_features import compute_higher_order_cumulant
        df = _make_batch4_df(300)
        result = compute_higher_order_cumulant(df)
        assert "cumulant_3" in result.columns
        assert "cumulant_4" in result.columns

    def test_output_length(self):
        from features.custom_features import compute_higher_order_cumulant
        df = _make_batch4_df(300)
        result = compute_higher_order_cumulant(df)
        assert len(result) == 300

    def test_valid_values(self):
        from features.custom_features import compute_higher_order_cumulant
        df = _make_batch4_df(300)
        result = compute_higher_order_cumulant(df)
        for col in ["cumulant_3", "cumulant_4"]:
            valid = result[col].dropna()
            assert len(valid) > 0
            assert np.isfinite(valid).all()


class TestZScoreOfZScores:
    def test_produces_values(self):
        from features.custom_features import compute_z_score_of_z_scores
        df = _make_batch4_df(300)
        result = compute_z_score_of_z_scores(df)
        valid = result["z_score_of_z_scores"].dropna()
        assert len(valid) > 0
        assert np.isfinite(valid).all()

    def test_output_length(self):
        from features.custom_features import compute_z_score_of_z_scores
        df = _make_batch4_df(300)
        result = compute_z_score_of_z_scores(df)
        assert len(result) == 300


class TestNetworkCentrality:
    def test_nan_without_symbol_cols(self):
        from features.custom_features import compute_network_centrality
        df = _make_batch4_df(300)
        result = compute_network_centrality(df, symbol_cols=None)
        assert result["network_centrality"].isna().all()

    def test_nan_with_insufficient_symbols(self):
        from features.custom_features import compute_network_centrality
        df = _make_batch4_df(300)
        result = compute_network_centrality(df, symbol_cols=["close"])
        assert result["network_centrality"].isna().all()


class TestCommunityStrength:
    def test_nan_without_symbol_cols(self):
        from features.custom_features import compute_community_strength
        df = _make_batch4_df(300)
        result = compute_community_strength(df, symbol_cols=None)
        assert result["community_strength"].isna().all()

    def test_nan_without_sector_map(self):
        from features.custom_features import compute_community_strength
        df = _make_batch4_df(300)
        result = compute_community_strength(df, symbol_cols=["a", "b", "c"], sector_map=None)
        assert result["community_strength"].isna().all()


class TestBatch4Integration:
    def test_all_20_registered(self):
        from features.feature_registry import FeatureRegistry
        import features.custom_features  # noqa: F401
        registry = FeatureRegistry()
        all_output_names = set()
        for name, entry in registry.get_all_entries().items():
            all_output_names.update(entry.output_names)
        batch4 = {
            "spectral_ratio", "dominant_frequency",
            "wavelet_energy_s1", "wavelet_energy_s2", "wavelet_energy_s3",
            "wavelet_energy_s4", "wavelet_energy_s5", "wavelet_entropy",
            "extreme_value_index", "tail_dependence", "copula_dependence",
            "rank_correlation", "ml_derived_volatility", "ml_derived_trend",
            "order_book_imbalance_proxy", "depth_pressure",
            "herding_behavior", "overreaction_score",
            "morning_session", "afternoon_session", "night_session",
            "session_vol_ratio", "hour_seasonality", "weekday_seasonality",
            "cumulant_3", "cumulant_4", "z_score_of_z_scores",
            "network_centrality", "community_strength",
        }
        assert batch4.issubset(all_output_names), f"Missing: {batch4 - all_output_names}"

    def test_compute_all_features_includes_batch4(self):
        from features.feature_engineering import compute_all_features
        df = _make_batch4_df(300)
        features_df = compute_all_features(df, period="5min")
        batch4_cols = [
            "spectral_ratio", "dominant_frequency",
            "wavelet_energy_s1", "wavelet_energy_s2", "wavelet_energy_s3",
            "wavelet_energy_s4", "wavelet_energy_s5", "wavelet_entropy",
            "extreme_value_index", "tail_dependence", "copula_dependence",
            "rank_correlation", "ml_derived_volatility", "ml_derived_trend",
            "order_book_imbalance_proxy", "depth_pressure",
            "herding_behavior", "overreaction_score",
            "morning_session", "afternoon_session", "night_session",
            "session_vol_ratio", "hour_seasonality", "weekday_seasonality",
            "cumulant_3", "cumulant_4", "z_score_of_z_scores",
            "network_centrality", "community_strength",
        ]
        for feat in batch4_cols:
            assert feat in features_df.columns, f"Missing from compute_all_features: {feat}"

    def test_no_look_ahead_overreaction(self):
        """Verify overreaction_score at each index i depends only on data[:i+1]."""
        from features.custom_features import compute_overreaction_score
        df = _make_batch4_df(300)
        full = compute_overreaction_score(df)["overreaction_score"]
        # Check several indices: value should match when computed on truncated data
        for i in [150, 200, 250]:
            trunc = df.iloc[:i + 1]
            trunc_val = compute_overreaction_score(trunc)["overreaction_score"].iloc[i]
            assert np.isclose(full.iloc[i], trunc_val, equal_nan=True), (
                f"Look-ahead detected at index {i}: full={full.iloc[i]}, trunc={trunc_val}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
