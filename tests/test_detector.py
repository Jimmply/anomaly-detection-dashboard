"""
Unit tests for AnomalyDetector
================================
Tests cover:
  - Each detection method returns the correct shape and columns
  - Anomaly flag column is boolean
  - Score column contains no NaN values
  - Known synthetic anomalies are detected at reasonable recall
  - Edge cases: constant series, very short series, series with NaNs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
import os

# Ensure src/ is importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import AnomalyDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_series() -> pd.Series:
    """500-point Gaussian series – mostly anomaly-free."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=50.0, scale=2.0, size=500)
    timestamps = pd.date_range("2024-01-01", periods=500, freq="1min")
    return pd.Series(values, index=timestamps, name="temperature")


@pytest.fixture
def series_with_spikes(normal_series: pd.Series) -> pd.Series:
    """Normal series with 5 injected extreme spikes."""
    s = normal_series.copy()
    spike_idx = [50, 150, 250, 350, 450]
    for idx in spike_idx:
        s.iloc[idx] += 25.0  # ~12.5-sigma spike
    return s


@pytest.fixture
def short_series() -> pd.Series:
    """Minimum viable series for edge-case tests."""
    return pd.Series([1.0, 2.0, 1.5, 100.0, 1.8], name="sensor")


@pytest.fixture
def constant_series() -> pd.Series:
    """Constant-valued series (zero variance)."""
    return pd.Series([5.0] * 100, name="sensor")


@pytest.fixture
def series_with_nans(normal_series: pd.Series) -> pd.Series:
    """Series with scattered NaN values."""
    s = normal_series.copy()
    s.iloc[10] = np.nan
    s.iloc[100] = np.nan
    s.iloc[200] = np.nan
    return s


@pytest.fixture
def detector() -> AnomalyDetector:
    return AnomalyDetector(
        window_size=20,
        zscore_threshold=3.0,
        contamination=0.05,
        iqr_multiplier=1.5,
        dbscan_eps=0.5,
        dbscan_min_samples=5,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


EXPECTED_COLUMNS = {"value", "anomaly", "score"}


def assert_result_schema(result: pd.DataFrame, expected_len: int) -> None:
    """Assert the result has the correct shape, columns, and dtypes."""
    assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
    assert set(result.columns) == EXPECTED_COLUMNS, f"Expected columns {EXPECTED_COLUMNS}, got {set(result.columns)}"
    assert len(result) == expected_len, f"Length mismatch: {len(result)} != {expected_len}"
    assert result["anomaly"].dtype == bool, "anomaly column must be bool"
    assert result["score"].dtype in (float, np.float64, np.float32), "score column must be float"
    assert result["score"].isna().sum() == 0, "score column must have no NaN values"
    assert result["anomaly"].isna().sum() == 0, "anomaly column must have no NaN values"


# ---------------------------------------------------------------------------
# Z-Score tests
# ---------------------------------------------------------------------------


class TestZScore:
    def test_returns_correct_schema(self, detector, normal_series):
        result = detector.detect(normal_series, method="zscore")
        assert_result_schema(result, len(normal_series))

    def test_detects_known_spikes(self, detector, series_with_spikes):
        spike_idx = [50, 150, 250, 350, 450]
        result = detector.detect(series_with_spikes, method="zscore")
        for idx in spike_idx:
            assert result["anomaly"].iloc[idx], (
                f"Expected spike at index {idx} to be flagged as anomaly"
            )

    def test_low_false_positive_rate_on_normal(self, detector, normal_series):
        result = detector.detect(normal_series, method="zscore")
        # With threshold=3 sigma on a Gaussian, expect <<1% false positives
        fp_rate = result["anomaly"].mean()
        assert fp_rate < 0.03, f"False positive rate {fp_rate:.3f} too high for clean Gaussian signal"

    def test_handles_nan_input(self, detector, series_with_nans):
        result = detector.detect(series_with_nans, method="zscore")
        assert_result_schema(result, len(series_with_nans))

    def test_handles_constant_series(self, detector, constant_series):
        result = detector.detect(constant_series, method="zscore")
        assert_result_schema(result, len(constant_series))

    def test_score_is_non_negative(self, detector, normal_series):
        result = detector.detect(normal_series, method="zscore")
        assert (result["score"] >= 0).all(), "Z-score scores must be non-negative"

    def test_index_preserved(self, detector, normal_series):
        result = detector.detect(normal_series, method="zscore")
        pd.testing.assert_index_equal(result.index, normal_series.index)


# ---------------------------------------------------------------------------
# IQR tests
# ---------------------------------------------------------------------------


class TestIQR:
    def test_returns_correct_schema(self, detector, normal_series):
        result = detector.detect(normal_series, method="iqr")
        assert_result_schema(result, len(normal_series))

    def test_detects_known_spikes(self, detector, series_with_spikes):
        spike_idx = [50, 150, 250, 350, 450]
        result = detector.detect(series_with_spikes, method="iqr")
        detected = [i for i in spike_idx if result["anomaly"].iloc[i]]
        recall = len(detected) / len(spike_idx)
        assert recall >= 0.8, f"IQR recall on injected spikes too low: {recall:.2f}"

    def test_non_negative_scores(self, detector, normal_series):
        result = detector.detect(normal_series, method="iqr")
        assert (result["score"] >= 0).all(), "IQR scores must be non-negative"

    def test_inliers_have_zero_score(self, detector, normal_series):
        """Points inside the fence should have score == 0."""
        result = detector.detect(normal_series, method="iqr")
        inlier_scores = result.loc[~result["anomaly"], "score"]
        # Allow a tiny floating-point tolerance
        assert (inlier_scores < 1e-9).all(), "Non-anomalous IQR scores should be 0"

    def test_handles_constant_series(self, detector, constant_series):
        result = detector.detect(constant_series, method="iqr")
        assert_result_schema(result, len(constant_series))

    def test_index_preserved(self, detector, normal_series):
        result = detector.detect(normal_series, method="iqr")
        pd.testing.assert_index_equal(result.index, normal_series.index)


# ---------------------------------------------------------------------------
# Isolation Forest tests
# ---------------------------------------------------------------------------


class TestIsolationForest:
    def test_returns_correct_schema(self, detector, normal_series):
        result = detector.detect(normal_series, method="isolation_forest")
        assert_result_schema(result, len(normal_series))

    def test_scores_in_unit_interval(self, detector, normal_series):
        result = detector.detect(normal_series, method="isolation_forest")
        assert (result["score"] >= 0).all(), "IF scores must be >= 0"
        assert (result["score"] <= 1).all(), "IF scores must be <= 1 (normalised)"

    def test_contamination_respected(self, detector, normal_series):
        """Anomaly fraction should be close to the configured contamination."""
        result = detector.detect(normal_series, method="isolation_forest")
        observed_rate = result["anomaly"].mean()
        # Allow ±3% tolerance around the configured contamination
        assert abs(observed_rate - detector.contamination) < 0.03, (
            f"Observed anomaly rate {observed_rate:.3f} too far from "
            f"contamination {detector.contamination}"
        )

    def test_detects_spikes(self, detector, series_with_spikes):
        spike_idx = [50, 150, 250, 350, 450]
        result = detector.detect(series_with_spikes, method="isolation_forest")
        detected = [i for i in spike_idx if result["anomaly"].iloc[i]]
        recall = len(detected) / len(spike_idx)
        assert recall >= 0.6, f"Isolation Forest spike recall too low: {recall:.2f}"

    def test_index_preserved(self, detector, normal_series):
        result = detector.detect(normal_series, method="isolation_forest")
        pd.testing.assert_index_equal(result.index, normal_series.index)

    def test_handles_nan_input(self, detector, series_with_nans):
        result = detector.detect(series_with_nans, method="isolation_forest")
        assert_result_schema(result, len(series_with_nans))


# ---------------------------------------------------------------------------
# DBSCAN tests
# ---------------------------------------------------------------------------


class TestDBSCAN:
    def test_returns_correct_schema(self, detector, normal_series):
        result = detector.detect(normal_series, method="dbscan")
        assert_result_schema(result, len(normal_series))

    def test_scores_non_negative(self, detector, normal_series):
        result = detector.detect(normal_series, method="dbscan")
        assert (result["score"] >= 0).all(), "DBSCAN scores must be non-negative"

    def test_detects_isolated_points(self):
        """DBSCAN should flag a clearly isolated outlier."""
        det = AnomalyDetector(dbscan_eps=0.3, dbscan_min_samples=3, random_state=42)
        # Tight cluster at 0, one extreme outlier at 100
        values = list(np.random.default_rng(0).normal(0, 0.1, 100)) + [100.0]
        s = pd.Series(values, name="sensor")
        result = det.detect(s, method="dbscan")
        assert result["anomaly"].iloc[-1], "DBSCAN should flag the extreme outlier"

    def test_index_preserved(self, detector, normal_series):
        result = detector.detect(normal_series, method="dbscan")
        pd.testing.assert_index_equal(result.index, normal_series.index)

    def test_handles_nan_input(self, detector, series_with_nans):
        result = detector.detect(series_with_nans, method="dbscan")
        assert_result_schema(result, len(series_with_nans))


# ---------------------------------------------------------------------------
# AnomalyDetector.detect() dispatch tests
# ---------------------------------------------------------------------------


class TestDetectDispatch:
    def test_invalid_method_raises(self, detector, normal_series):
        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect(normal_series, method="magic_method")

    def test_method_case_insensitive(self, detector, normal_series):
        result = detector.detect(normal_series, method="ZScore")
        assert_result_schema(result, len(normal_series))

    def test_too_short_series_raises(self, detector):
        with pytest.raises(ValueError, match="at least 2"):
            detector.detect(pd.Series([42.0]), method="zscore")

    def test_all_methods_return_same_index(self, detector, normal_series):
        for method in AnomalyDetector.SUPPORTED_METHODS:
            result = detector.detect(normal_series, method=method)
            pd.testing.assert_index_equal(result.index, normal_series.index, check_names=False)


# ---------------------------------------------------------------------------
# Summary helper test
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_keys(self, detector, normal_series):
        result = detector.detect(normal_series, method="zscore")
        summary = detector.summary(result)
        expected_keys = {"total_points", "anomaly_count", "anomaly_rate_pct", "mean_score", "max_score"}
        assert set(summary.keys()) == expected_keys

    def test_summary_values_consistent(self, detector, normal_series):
        result = detector.detect(normal_series, method="zscore")
        summary = detector.summary(result)
        assert summary["total_points"] == len(normal_series)
        assert summary["anomaly_count"] == int(result["anomaly"].sum())
        expected_rate = round(100 * summary["anomaly_count"] / summary["total_points"], 2)
        assert summary["anomaly_rate_pct"] == expected_rate
