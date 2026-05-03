"""
Real-Time Anomaly Detection Dashboard
======================================
Streamlit application for interactive time-series anomaly detection
on industrial sensor data.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Allow src/ imports when running from project root or src/ directory
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_generator import generate_sample_data
from detector import AnomalyDetector

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

APP_TITLE = os.getenv("APP_TITLE", "Real-Time Anomaly Detection Dashboard")
DEFAULT_CONTAMINATION = float(os.getenv("DEFAULT_CONTAMINATION", "0.05"))
DEFAULT_ZSCORE_THRESHOLD = float(os.getenv("DEFAULT_ZSCORE_THRESHOLD", "3.0"))
DEFAULT_WINDOW_SIZE = int(os.getenv("DEFAULT_WINDOW_SIZE", "20"))

METHOD_LABELS = {
    "zscore": "Z-Score (Rolling Window)",
    "iqr": "IQR (Tukey Fence)",
    "isolation_forest": "Isolation Forest",
    "dbscan": "DBSCAN Clustering",
}

SENSOR_UNITS = {
    "temperature": "°C",
    "pressure": "bar",
    "vibration": "mm/s",
    "power": "kW",
}

ANOMALY_COLOR = "rgba(255, 59, 59, 0.85)"
NORMAL_COLOR = "rgba(59, 130, 246, 0.7)"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/anomaly-detection-dashboard",
        "Report a bug": "https://github.com/yourusername/anomaly-detection-dashboard/issues",
    },
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise all session state keys with sensible defaults."""
    defaults: dict = {
        "df_raw": None,
        "detection_result": None,
        "selected_column": None,
        "last_method": None,
        "last_params_hash": None,
        "data_source": "synthetic",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_synthetic_data(n_points: int, seed: int) -> pd.DataFrame:
    """Generate synthetic sensor data and cache it in session state."""
    return generate_sample_data(n_points=n_points, random_seed=seed)


def _load_uploaded_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse an uploaded CSV file into a DataFrame with a datetime index."""
    try:
        df = pd.read_csv(uploaded_file)
        # Try to detect a timestamp / datetime column
        timestamp_candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ["time", "date", "ts", "index"])
        ]
        if timestamp_candidates:
            ts_col = timestamp_candidates[0]
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=[ts_col]).set_index(ts_col)
            df.index.name = "timestamp"
        else:
            # Create a synthetic integer index
            df.index = pd.RangeIndex(len(df))
            df.index.name = "row"
        return df
    except Exception as exc:
        st.error(f"Could not parse uploaded file: {exc}")
        return None


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------

def _run_detection(
    df: pd.DataFrame,
    column: str,
    method: str,
    window_size: int,
    zscore_threshold: float,
    contamination: float,
    iqr_multiplier: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> pd.DataFrame:
    """
    Run the selected detection method and return the result DataFrame.
    Results are cached in session state keyed by a parameter hash.
    """
    import hashlib, json

    params = {
        "column": column,
        "method": method,
        "window_size": window_size,
        "zscore_threshold": zscore_threshold,
        "contamination": contamination,
        "iqr_multiplier": iqr_multiplier,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "n_rows": len(df),
    }
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    if (
        st.session_state.get("last_params_hash") == params_hash
        and st.session_state.get("detection_result") is not None
    ):
        return st.session_state["detection_result"]

    detector = AnomalyDetector(
        window_size=window_size,
        zscore_threshold=zscore_threshold,
        contamination=contamination,
        iqr_multiplier=iqr_multiplier,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    series = df[column].copy()
    with st.spinner(f"Running {METHOD_LABELS[method]}..."):
        result = detector.detect(series, method=method)

    st.session_state["detection_result"] = result
    st.session_state["last_params_hash"] = params_hash
    return result


# ---------------------------------------------------------------------------
# Plotly chart
# ---------------------------------------------------------------------------

def _build_timeseries_chart(
    df_raw: pd.DataFrame,
    result: pd.DataFrame,
    column: str,
    method_label: str,
) -> go.Figure:
    """Build an interactive Plotly figure showing the sensor signal and anomalies."""
    unit = SENSOR_UNITS.get(column, "")
    y_label = f"{column.title()} ({unit})" if unit else column.title()

    normal_mask = ~result["anomaly"]
    anomaly_mask = result["anomaly"]

    index_vals = result.index

    fig = go.Figure()

    # Normal points – thin line + faint markers
    fig.add_trace(
        go.Scatter(
            x=index_vals[normal_mask],
            y=result.loc[normal_mask, "value"],
            mode="lines+markers",
            name="Normal",
            line=dict(color="#3b82f6", width=1.5),
            marker=dict(size=3, color="#3b82f6", opacity=0.5),
            hovertemplate="%{x}<br>Value: %{y:.3f}<extra>Normal</extra>",
        )
    )

    # Anomaly points – larger, red markers
    if anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=index_vals[anomaly_mask],
                y=result.loc[anomaly_mask, "value"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=10,
                    color=ANOMALY_COLOR,
                    symbol="circle-open",
                    line=dict(width=2.5, color="rgb(220,38,38)"),
                ),
                hovertemplate=(
                    "%{x}<br>"
                    "Value: %{y:.3f}<br>"
                    "Score: %{customdata:.4f}"
                    "<extra>Anomaly</extra>"
                ),
                customdata=result.loc[anomaly_mask, "score"].values,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{column.title()} – Anomaly Detection ({method_label})",
            font=dict(size=18, color="#1e293b"),
        ),
        xaxis=dict(
            title="Timestamp",
            showgrid=True,
            gridcolor="#e2e8f0",
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis=dict(
            title=y_label,
            showgrid=True,
            gridcolor="#e2e8f0",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=80, b=60),
        hovermode="x unified",
    )
    return fig


def _build_score_distribution_chart(result: pd.DataFrame) -> go.Figure:
    """Build a histogram of anomaly scores with a threshold line."""
    anomaly_scores = result.loc[result["anomaly"], "score"]
    normal_scores = result.loc[~result["anomaly"], "score"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=normal_scores,
            name="Normal",
            marker_color="#3b82f6",
            opacity=0.7,
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=anomaly_scores,
            name="Anomaly",
            marker_color="#ef4444",
            opacity=0.8,
            nbinsx=40,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _build_anomaly_report(df_raw: pd.DataFrame, result: pd.DataFrame, column: str, method: str) -> bytes:
    """Construct an anomaly report CSV ready for download."""
    anomalies = result[result["anomaly"]].copy()
    anomalies = anomalies.rename(columns={"value": column, "score": "anomaly_score"})
    anomalies["detection_method"] = METHOD_LABELS.get(method, method)
    anomalies["sensor_channel"] = column

    # Attach surrounding context values if available in the raw DataFrame
    other_cols = [c for c in df_raw.columns if c != column and c != "is_anomaly"]
    if other_cols:
        anomalies = anomalies.join(df_raw[other_cols], how="left")

    return anomalies.to_csv().encode("utf-8")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    # ---- Sidebar ----
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square",
            use_column_width=False,
        )
        st.title("⚙️ Configuration")
        st.divider()

        # -- Data source --
        st.subheader("Data Source")
        data_source = st.radio(
            "Select data source",
            options=["synthetic", "upload"],
            format_func=lambda x: "Generate Synthetic Data" if x == "synthetic" else "Upload CSV File",
            key="data_source",
        )

        df_raw: Optional[pd.DataFrame] = None

        if data_source == "synthetic":
            n_points = st.slider(
                "Number of data points", min_value=200, max_value=5000, value=1000, step=100
            )
            rand_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
            if st.button("Generate Data", type="primary", use_container_width=True):
                with st.spinner("Generating synthetic sensor data..."):
                    st.session_state["df_raw"] = _load_synthetic_data(n_points, rand_seed)
                    st.session_state["detection_result"] = None
                    st.session_state["last_params_hash"] = None
                st.success("Data generated!")

            if st.session_state["df_raw"] is None:
                # Auto-generate on first load
                st.session_state["df_raw"] = _load_synthetic_data(1000, 42)

        else:
            uploaded_file = st.file_uploader(
                "Upload sensor CSV",
                type=["csv"],
                help="CSV must have a timestamp column and at least one numeric sensor column.",
            )
            if uploaded_file is not None:
                df_parsed = _load_uploaded_data(uploaded_file)
                if df_parsed is not None:
                    st.session_state["df_raw"] = df_parsed
                    st.session_state["detection_result"] = None
                    st.session_state["last_params_hash"] = None

            if st.session_state["df_raw"] is None:
                st.info("Upload a CSV file to get started, or switch to synthetic data.")

        df_raw = st.session_state.get("df_raw")

        st.divider()

        # -- Sensor column selector --
        st.subheader("Sensor Channel")
        numeric_cols = []
        if df_raw is not None:
            numeric_cols = [
                c for c in df_raw.select_dtypes(include="number").columns
                if c not in ("is_anomaly",)
            ]

        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            selected_column = None
        else:
            selected_column = st.selectbox(
                "Select sensor column",
                options=numeric_cols,
                index=0,
                key="selected_column",
            )

        st.divider()

        # -- Detection method --
        st.subheader("Detection Method")
        method = st.selectbox(
            "Algorithm",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda k: METHOD_LABELS[k],
        )

        st.divider()

        # -- Hyperparameters (shown contextually) --
        st.subheader("Hyperparameters")

        window_size = DEFAULT_WINDOW_SIZE
        zscore_threshold = DEFAULT_ZSCORE_THRESHOLD
        contamination = DEFAULT_CONTAMINATION
        iqr_multiplier = 1.5
        dbscan_eps = 0.5
        dbscan_min_samples = 5

        if method == "zscore":
            window_size = st.slider(
                "Rolling window size",
                min_value=5,
                max_value=200,
                value=DEFAULT_WINDOW_SIZE,
                step=5,
                help="Number of historical points used to compute the rolling mean and std.",
            )
            zscore_threshold = st.slider(
                "Z-score threshold (σ)",
                min_value=1.0,
                max_value=6.0,
                value=DEFAULT_ZSCORE_THRESHOLD,
                step=0.1,
                help="Points beyond this many standard deviations are flagged as anomalies.",
            )

        elif method == "iqr":
            iqr_multiplier = st.slider(
                "IQR multiplier (k)",
                min_value=0.5,
                max_value=4.0,
                value=1.5,
                step=0.1,
                help="Tukey fence: Q1 − k·IQR  and  Q3 + k·IQR. Lower = more sensitive.",
            )

        elif method == "isolation_forest":
            contamination = st.slider(
                "Expected contamination",
                min_value=0.01,
                max_value=0.40,
                value=DEFAULT_CONTAMINATION,
                step=0.01,
                format="%.2f",
                help="Approximate proportion of anomalies expected in the dataset.",
            )

        elif method == "dbscan":
            dbscan_eps = st.slider(
                "Epsilon (neighbourhood radius)",
                min_value=0.05,
                max_value=3.0,
                value=0.5,
                step=0.05,
                help="Maximum distance between two points to be considered neighbours.",
            )
            dbscan_min_samples = st.slider(
                "Min samples (core point)",
                min_value=2,
                max_value=30,
                value=5,
                step=1,
                help="Minimum neighbourhood size to form a core point.",
            )

        st.divider()
        run_btn = st.button("Run Detection", type="primary", use_container_width=True)

    # ---- Main area ----
    st.title("📡 Real-Time Anomaly Detection Dashboard")
    st.caption(
        "Industrial sensor monitoring · Supports Z-Score, IQR, Isolation Forest, and DBSCAN"
    )

    if df_raw is None or selected_column is None:
        st.info("👈 Configure your data source and sensor channel in the sidebar, then click **Run Detection**.")
        _render_landing_info()
        return

    # Auto-run detection if the button was pressed or no result exists yet
    if run_btn or st.session_state.get("detection_result") is None:
        result = _run_detection(
            df_raw,
            column=selected_column,
            method=method,
            window_size=window_size,
            zscore_threshold=zscore_threshold,
            contamination=contamination,
            iqr_multiplier=iqr_multiplier,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
    else:
        result = st.session_state["detection_result"]

    if result is None:
        st.error("Detection failed. Check logs for details.")
        return

    # ---- Metrics row ----
    total_points = len(result)
    n_anomalies = int(result["anomaly"].sum())
    anomaly_rate = 100 * n_anomalies / total_points if total_points else 0.0
    max_score = float(result["score"].max())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Points", f"{total_points:,}")
    col2.metric("Anomalies Detected", f"{n_anomalies:,}", delta=f"{anomaly_rate:.1f}%", delta_color="inverse")
    col3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    col4.metric("Max Anomaly Score", f"{max_score:.4f}")

    st.divider()

    # ---- Time-series chart ----
    fig_ts = _build_timeseries_chart(df_raw, result, selected_column, METHOD_LABELS[method])
    st.plotly_chart(fig_ts, use_container_width=True)

    # ---- Score distribution + data overview side by side ----
    chart_col, stats_col = st.columns([3, 2])

    with chart_col:
        fig_dist = _build_score_distribution_chart(result)
        st.plotly_chart(fig_dist, use_container_width=True)

    with stats_col:
        st.subheader("Dataset Overview")
        if df_raw is not None:
            numeric_df = df_raw.select_dtypes(include="number").drop(columns=["is_anomaly"], errors="ignore")
            st.dataframe(
                numeric_df.describe().round(3),
                use_container_width=True,
            )

    st.divider()

    # ---- Anomaly detail table ----
    st.subheader(f"Anomaly Detail — {n_anomalies} event(s) detected")

    if n_anomalies == 0:
        st.success("No anomalies detected with the current configuration. Try adjusting the sensitivity parameters.")
    else:
        anomaly_rows = result[result["anomaly"]].copy()
        anomaly_rows = anomaly_rows.rename(
            columns={"value": selected_column, "score": "anomaly_score", "anomaly": "is_anomaly"}
        )
        anomaly_rows["anomaly_score"] = anomaly_rows["anomaly_score"].round(6)

        # Add context columns from raw data
        context_cols = [c for c in df_raw.columns if c not in (selected_column, "is_anomaly")]
        if context_cols:
            anomaly_rows = anomaly_rows.join(df_raw[context_cols], how="left")

        st.dataframe(
            anomaly_rows,
            use_container_width=True,
            height=300,
            column_config={
                "is_anomaly": st.column_config.CheckboxColumn("Anomaly", disabled=True),
                "anomaly_score": st.column_config.NumberColumn("Score", format="%.6f"),
            },
        )

        # ---- Download button ----
        report_csv = _build_anomaly_report(df_raw, result, selected_column, method)
        st.download_button(
            label="⬇️ Download Anomaly Report (CSV)",
            data=report_csv,
            file_name=f"anomaly_report_{selected_column}_{method}.csv",
            mime="text/csv",
            help="Downloads a CSV containing all flagged anomaly rows with scores and context.",
        )

    # ---- Ground-truth comparison (synthetic data only) ----
    if "is_anomaly" in df_raw.columns and data_source == "synthetic":
        st.divider()
        st.subheader("Ground-Truth Comparison (Synthetic Data)")
        st.caption(
            "Because synthetic data has known injected anomalies, we can evaluate detection quality."
        )
        _render_ground_truth_metrics(df_raw, result)


def _render_ground_truth_metrics(df_raw: pd.DataFrame, result: pd.DataFrame) -> None:
    """Compute and display precision/recall against synthetic ground truth."""
    gt = df_raw["is_anomaly"].values.astype(bool)
    pred = result["anomaly"].values.astype(bool)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(gt) if len(gt) > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision:.3f}", help="TP / (TP + FP)")
    m2.metric("Recall", f"{recall:.3f}", help="TP / (TP + FN)")
    m3.metric("F1-Score", f"{f1:.3f}", help="Harmonic mean of Precision and Recall")
    m4.metric("Accuracy", f"{accuracy:.3f}")

    with st.expander("Confusion matrix details"):
        conf_df = pd.DataFrame(
            [[tp, fp], [fn, tn]],
            index=["Predicted Anomaly", "Predicted Normal"],
            columns=["Actual Anomaly", "Actual Normal"],
        )
        st.dataframe(conf_df)


def _render_landing_info() -> None:
    """Render informational cards shown before any data is loaded."""
    st.markdown("---")
    st.subheader("Supported Detection Methods")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            "**Z-Score**\n\nRolling mean ± k·σ window. "
            "Sensitive to local deviations. Best for stationary signals with clear noise profile."
        )
    with c2:
        st.markdown(
            "**IQR**\n\nTukey fence: Q1 − k·IQR / Q3 + k·IQR. "
            "Non-parametric and robust to skewed distributions."
        )
    with c3:
        st.markdown(
            "**Isolation Forest**\n\nTree-based method that isolates anomalies by randomly "
            "partitioning the feature space. Excellent for complex distributions."
        )
    with c4:
        st.markdown(
            "**DBSCAN**\n\nDensity-based clustering. Points in sparse regions are flagged "
            "as noise/anomalies. Handles arbitrary cluster shapes."
        )

    st.markdown("---")
    st.subheader("Expected CSV Format")
    st.code(
        "timestamp,temperature,pressure,vibration,power\n"
        "2024-01-01 00:00:00,74.85,4.52,2.73,18.41\n"
        "2024-01-01 00:01:00,75.12,4.48,2.91,18.67\n"
        "...",
        language="csv",
    )
    st.caption(
        "Timestamp column is auto-detected. Any numeric columns can be used as sensor channels."
    )


if __name__ == "__main__":
    main()
