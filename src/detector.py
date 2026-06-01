"""
Anomaly Detection Engine
========================
Supports multiple detection strategies for univariate time-series sensor data.

Detection methods:
- Z-Score      : rolling-window z-score with configurable sigma threshold
- IQR          : interquartile range fence method
- Isolation Forest : sklearn (or pyod if available)
- DBSCAN       : density-based spatial clustering

Each public method accepts a pandas Series and returns a DataFrame with columns:
    value        : original sensor readings
    anomaly      : bool flag  (True = anomaly)
    score        : continuous anomaly score (higher = more anomalous)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Attempt to import pyod; fall back gracefully
try:
    from pyod.models.iforest import IForest as PyodIForest

    _PYOD_AVAILABLE = True
except ImportError:
    _PYOD_AVAILABLE = False
    logger.info("pyod not found – using sklearn IsolationForest as fallback.")


class AnomalyDetector:
    """
    Unified interface for time-series anomaly detection.

    Parameters
    ----------
    window_size : int
        Rolling window length used by statistical methods (Z-score).
    zscore_threshold : float
        Number of standard deviations beyond which a point is flagged.
    contamination : float
        Expected proportion of anomalies in the dataset (0 < contamination < 0.5).
        Used by Isolation Forest.
    iqr_multiplier : float
        Fence multiplier applied to the IQR (default 1.5 for Tukey fences).
    dbscan_eps : float
        Maximum distance between two samples for DBSCAN neighbourhood.
    dbscan_min_samples : int
        Minimum number of samples in a DBSCAN neighbourhood.
    random_state : int
        Random seed for reproducible results.
    """

    SUPPORTED_METHODS = ["zscore", "iqr", "isolation_forest", "dbscan"]

    def __init__(
        self,
        window_size: int = 20,
        zscore_threshold: float = 3.0,
        contamination: float = 0.05,
        iqr_multiplier: float = 1.5,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        self.window_size = window_size
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.iqr_multiplier = iqr_multiplier
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, series: pd.Series, method: str) -> pd.DataFrame:
        """
        Run anomaly detection using the specified method.

        Parameters
        ----------
        series : pd.Series
            Univariate time-series of sensor readings.
        method : str
            One of 'zscore', 'iqr', 'isolation_forest', 'dbscan'.

        Returns
        -------
        pd.DataFrame
            Columns: value, anomaly (bool), score (float).
        """
        method = method.lower().strip()
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {self.SUPPORTED_METHODS}"
            )

        series = self._validate_series(series)

        dispatch = {
            "zscore": self._zscore,
            "iqr": self._iqr,
            "isolation_forest": self._isolation_forest,
            "dbscan": self._dbscan,
        }
        return dispatch[method](series)

    # ------------------------------------------------------------------
    # Detection implementations
    # ------------------------------------------------------------------

    def _zscore(self, series: pd.Series) -> pd.DataFrame:
        """
        Rolling Z-score anomaly detection.

        Uses an expanding window until enough observations are collected,
        then switches to a fixed rolling window. This avoids NaN flags at
        the start of the series.
        """
        min_periods = min(self.window_size, max(2, len(series) // 10))

        rolling_mean = series.rolling(
            window=self.window_size, min_periods=min_periods
        ).mean()
        rolling_std = series.rolling(
            window=self.window_size, min_periods=min_periods
        ).std()

        # Avoid division by zero on constant segments
        rolling_std = rolling_std.replace(0, np.nan).fillna(series.std() or 1.0)

        z_scores = np.abs((series - rolling_mean) / rolling_std)
        # Fill any remaining NaN z-scores (very start of series) with 0
        z_scores = z_scores.fillna(0.0)

        anomaly_flags = z_scores > self.zscore_threshold

        return self._build_result(series, anomaly_flags, z_scores)

    def _iqr(self, series: pd.Series) -> pd.DataFrame:
        """
        Interquartile range (Tukey fence) anomaly detection.

        Global IQR is computed once and applied uniformly. The anomaly score
        is the normalised distance beyond the fence (0 for points inside the fence).
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_fence = q1 - self.iqr_multiplier * iqr
        upper_fence = q3 + self.iqr_multiplier * iqr

        anomaly_flags = (series < lower_fence) | (series > upper_fence)

        # Score: normalised distance beyond the nearest fence
        distance_above = np.maximum(0, series - upper_fence)
        distance_below = np.maximum(0, lower_fence - series)
        raw_score = distance_above + distance_below
        score = raw_score / (iqr + 1e-9)  # normalise by IQR width

        return self._build_result(series, anomaly_flags, score)

    def _isolation_forest(self, series: pd.Series) -> pd.DataFrame:
        """
        Isolation Forest anomaly detection.

        Prefers pyod.IForest when available for richer diagnostics;
        falls back to sklearn IsolationForest otherwise.
        """
        X = series.values.reshape(-1, 1)

        if _PYOD_AVAILABLE:
            model = PyodIForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
            # pyod decision_scores_: higher = more anomalous
            raw_scores = model.decision_scores_
            labels = model.labels_  # 1 = anomaly, 0 = normal
            anomaly_flags = pd.Series(labels.astype(bool), index=series.index)
        else:
            model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
            # sklearn decision_function: negative = anomaly; flip sign so higher = worse
            raw_scores = -model.decision_function(X)
            predictions = model.predict(X)
            anomaly_flags = pd.Series(predictions == -1, index=series.index)

        # Normalise scores to [0, 1] for consistent reporting
        score_min, score_max = raw_scores.min(), raw_scores.max()
        if score_max > score_min:
            normalised = (raw_scores - score_min) / (score_max - score_min)
        else:
            normalised = np.zeros_like(raw_scores, dtype=float)

        scores = pd.Series(normalised, index=series.index)
        return self._build_result(series, anomaly_flags, scores)

    def _dbscan(self, series: pd.Series) -> pd.DataFrame:
        """
        DBSCAN clustering-based anomaly detection.

        Points labelled as noise (cluster label == -1) are treated as anomalies.
        The anomaly score is the normalised distance to the nearest core point
        (or 1.0 for noise points in low-density regions).
        """
        scaler = StandardScaler()

        # Feature matrix: value + time index (normalised) for temporal context
        time_index = np.arange(len(series)).reshape(-1, 1)
        values = series.values.reshape(-1, 1)
        X_raw = np.hstack([time_index, values])
        X = scaler.fit_transform(X_raw)

        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, n_jobs=-1)
        db.fit(X)
        labels = db.labels_  # -1 means noise / anomaly

        anomaly_flags = pd.Series(labels == -1, index=series.index)

        # Score: distance to nearest core point (proxy for outlierness)
        core_mask = np.zeros(len(X), dtype=bool)
        core_mask[db.core_sample_indices_] = True

        if core_mask.any():
            core_points = X[core_mask]
            from sklearn.metrics import pairwise_distances_argmin_min

            _, distances = pairwise_distances_argmin_min(X, core_points)
            score_norm = distances / (distances.max() + 1e-9)
        else:
            # Edge case: no core points at all
            score_norm = np.ones(len(X))

        scores = pd.Series(score_norm, index=series.index)
        return self._build_result(series, anomaly_flags, scores)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_series(series: pd.Series) -> pd.Series:
        """Ensure the input is a clean, numeric pandas Series."""
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        series = pd.to_numeric(series, errors="coerce")
        n_nan = series.isna().sum()
        if n_nan > 0:
            logger.warning(
                "Series contains %d NaN values – forward-filling before detection.", n_nan
            )
            series = series.ffill().bfill()
        if len(series) < 2:
            raise ValueError("Series must contain at least 2 data points.")
        return series

    @staticmethod
    def _build_result(
        series: pd.Series,
        anomaly_flags: pd.Series,
        scores: pd.Series,
    ) -> pd.DataFrame:
        """Assemble the standardised output DataFrame."""
        return pd.DataFrame(
            {
                "value": series.values,
                "anomaly": anomaly_flags.values.astype(bool),
                "score": scores.values.astype(float),
            },
            index=series.index,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self, result: pd.DataFrame) -> dict:
        """Return a summary dict from a detection result DataFrame."""
        total = len(result)
        n_anomalies = int(result["anomaly"].sum())
        return {
            "total_points": total,
            "anomaly_count": n_anomalies,
            "anomaly_rate_pct": round(100 * n_anomalies / total, 2) if total else 0.0,
            "mean_score": round(float(result["score"].mean()), 4),
            "max_score": round(float(result["score"].max()), 4),
        }

    def multi_channel_detect(
        self,
        df: pd.DataFrame,
        method: str,
        channels: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Run detection on multiple sensor channels in one call.

        Parameters
        ----------
        df : pd.DataFrame
            Each column is a sensor channel time-series.
        method : str
            Detection method to apply to every channel.
        channels : list[str] | None
            Subset of column names to process. Defaults to all numeric columns.

        Returns
        -------
        pd.DataFrame
            Index matches df. Per-channel ``<name>_anomaly`` and ``<name>_score``
            columns, plus ``any_anomaly`` (True if any channel flagged) and
            ``max_score`` (worst score across all channels at each timestamp).
        """
        cols = channels or list(df.select_dtypes(include="number").columns)
        if not cols:
            raise ValueError("No numeric columns found in df.")

        results: dict[str, pd.Series] = {}
        for col in cols:
            res = self.detect(df[col], method)
            results[f"{col}_anomaly"] = pd.Series(res["anomaly"].values, index=df.index)
            results[f"{col}_score"] = pd.Series(res["score"].values, index=df.index)

        out = pd.DataFrame(results, index=df.index)
        anomaly_cols = [c for c in out.columns if c.endswith("_anomaly")]
        score_cols = [c for c in out.columns if c.endswith("_score")]
        out["any_anomaly"] = out[anomaly_cols].any(axis=1)
        out["max_score"] = out[score_cols].max(axis=1)
        return out
