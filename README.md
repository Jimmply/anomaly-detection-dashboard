# Real-Time Anomaly Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psf/black)

An interactive Streamlit dashboard for real-time anomaly detection on industrial sensor time-series data. Upload your own CSV or generate realistic synthetic sensor data, configure four different detection algorithms through an intuitive UI, and export flagged events as a report — all without writing a single line of code.

Built from 4+ years of hands-on manufacturing analytics experience, this project reflects the kinds of monitoring challenges found in CNC machining, HVAC, pump stations, and process equipment fleets.

---

## Screenshot

> *Screenshot placeholder — run the app locally and capture your own with `streamlit run src/app.py`*

![Dashboard Screenshot](docs/screenshot_placeholder.png)

---

## Key Features

- **Four detection algorithms** with individually tunable hyperparameters, all runnable from the UI
- **Upload any CSV** or use the built-in synthetic sensor data generator (temperature, pressure, vibration, power)
- **Interactive Plotly charts** — zoom, pan, hover for exact values; range-slider for time navigation
- **Anomaly score distribution** histogram for quick threshold intuition
- **Ground-truth evaluation** metrics (precision, recall, F1, confusion matrix) when using synthetic data
- **One-click CSV export** of flagged anomaly rows with scores and surrounding sensor context
- **Configurable via `.env`** — suitable for containerised / cloud deployments
- **Full unit test suite** (`pytest`) covering all four algorithms and edge cases

---

## Detection Methods

### Z-Score (Rolling Window)
Computes a rolling mean and standard deviation over a configurable window. Points that deviate by more than *k* standard deviations from the local mean are flagged. Effective for stationary or slowly trending signals where the noise profile is well understood (e.g., temperature in a stable furnace).

**Key parameters:** `window_size`, `zscore_threshold (σ)`

### IQR — Interquartile Range (Tukey Fence)
Non-parametric method using global quartiles. Fences are set at Q1 − k·IQR and Q3 + k·IQR. Robust against skewed distributions and outliers in the training window, making it suitable for pressure and flow signals. Lower *k* values (e.g., 1.0) increase sensitivity.

**Key parameters:** `iqr_multiplier (k)`

### Isolation Forest
Ensemble tree-based method that isolates anomalies by randomly partitioning the feature space — anomalies require fewer cuts to isolate. Uses [pyod](https://pyod.readthedocs.io/) when available, falling back to scikit-learn. Excellent for multimodal distributions and signals with complex periodic patterns. Particularly strong on vibration and power draw data.

**Key parameters:** `contamination` (expected anomaly fraction)

### DBSCAN Clustering
Density-Based Spatial Clustering of Applications with Noise. Points in sparse, low-density regions are labelled as noise and treated as anomalies. Includes the normalised temporal index as a feature alongside the sensor value, giving the algorithm awareness of time proximity. Handles arbitrary cluster shapes and does not require a pre-specified number of clusters.

**Key parameters:** `epsilon` (neighbourhood radius), `min_samples`

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| App framework | [Streamlit](https://streamlit.io/) | Interactive UI, session state, file upload |
| Numerical computing | [NumPy](https://numpy.org/) | Array operations, signal generation |
| Data manipulation | [Pandas](https://pandas.pydata.org/) | DataFrame handling, time-series indexing |
| ML — classical | [scikit-learn](https://scikit-learn.org/) | Isolation Forest, DBSCAN, preprocessing |
| ML — outlier detection | [PyOD](https://pyod.readthedocs.io/) | Extended Isolation Forest with richer diagnostics |
| Visualisation | [Plotly](https://plotly.com/python/) | Interactive charts |
| Statistical methods | [SciPy](https://scipy.org/) | Distribution utilities |
| Configuration | [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env`-based settings |
| Testing | [pytest](https://pytest.org/) | Unit test suite |

---

## Quick Start

### Prerequisites
- Python 3.10 or later
- pip / conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/anomaly-detection-dashboard.git
cd anomaly-detection-dashboard

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) configure environment variables
cp .env.example .env
# Edit .env as needed

# 5. Launch the dashboard
streamlit run src/app.py
```

The app opens automatically at `http://localhost:8501`.

### Generate sample data

```bash
python scripts/generate_sample_data.py
```

This regenerates `data/sample_sensor_data.csv` — 1 000 rows of four-channel sensor data with injected anomalies.

### Run tests

```bash
pytest tests/ -v
```

---

## Configuration Options

All settings can be overridden via environment variables or the `.env` file.

| Variable | Default | Description |
|---|---|---|
| `APP_TITLE` | `Real-Time Anomaly Detection Dashboard` | Browser tab title |
| `DEFAULT_CONTAMINATION` | `0.05` | Default Isolation Forest contamination rate |
| `DEFAULT_ZSCORE_THRESHOLD` | `3.0` | Default Z-score sigma threshold |
| `DEFAULT_WINDOW_SIZE` | `20` | Default rolling window size (data points) |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum CSV upload size |

---

## Data Format Specification

The dashboard accepts any CSV with:

1. **A timestamp column** (optional but recommended) — auto-detected by column name containing `time`, `date`, `ts`, or `index`. Parsed with `pd.to_datetime`.
2. **One or more numeric sensor columns** — any column not identified as a timestamp or boolean flag.
3. **An `is_anomaly` column** (optional) — boolean; if present and synthetic data is selected, ground-truth evaluation metrics are shown.

**Minimum viable example:**

```csv
timestamp,temperature,pressure,vibration,power
2024-01-01 00:00:00,74.85,4.52,2.73,18.41
2024-01-01 00:01:00,75.12,4.48,2.91,18.67
2024-01-01 00:02:00,74.97,4.55,2.84,18.59
```

**Supported timestamp formats:** ISO 8601 (`YYYY-MM-DD HH:MM:SS`), Unix epoch (seconds), common locale formats.

**Recommended:** 200 – 10 000 rows at 1-minute to 1-hour resolution for best performance.

---

## Injected Anomaly Types (Synthetic Generator)

The `SensorDataGenerator` class supports four anomaly archetypes that mirror real failure modes:

| Type | Description | Real-world analogy |
|---|---|---|
| **Spike** | Instantaneous extreme deviation, recovers in 1 sample | Pressure transient, inrush current, EMI noise |
| **Drift** | Gradual linear creep away from nominal | Sensor calibration drift, gradual bearing wear |
| **Stuck** | Signal freezes at a constant value | Failed sensor, disconnected transmitter |
| **Sudden Shift** | Step change in mean level | Process reconfiguration, fluid viscosity change, sensor swap |

---

## Project Structure

```
anomaly-detection-dashboard/
├── src/
│   ├── app.py                # Streamlit application entry point
│   ├── detector.py           # AnomalyDetector class (all four methods)
│   └── data_generator.py     # SensorDataGenerator + generate_sample_data()
├── tests/
│   └── test_detector.py      # pytest unit tests
├── data/
│   └── sample_sensor_data.csv
├── scripts/
│   └── generate_sample_data.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Real-World Applicability

This project directly reflects work done in manufacturing analytics:

- **Multi-channel monitoring** — simultaneous tracking of temperature, pressure, vibration, and power mirrors real SCADA/MES data streams.
- **Method selection rationale** — Z-score for fast, interpretable alerts; Isolation Forest for end-of-shift batch anomaly review; DBSCAN for cluster-based process state identification.
- **Configurable sensitivity** — adjustable hyperparameters map to real trade-offs: a maintenance team tolerates more false positives (low threshold) to catch every event; a production line prioritises specificity (high threshold) to avoid unnecessary downtime.
- **Export pipeline** — the CSV report is designed to feed downstream ticketing systems (JIRA, SAP PM) or root-cause analysis workflows.

---

## Extending the Project

- **Add a new detection method:** implement a `_my_method(self, series)` in `detector.py`, add it to `SUPPORTED_METHODS`, and the sidebar will automatically expose it.
- **Connect to live data:** replace `generate_sample_data()` in `app.py` with a streaming source (MQTT, OPC-UA, InfluxDB, Kafka) and call `st.rerun()` on a timer.
- **Multivariate detection:** the `AnomalyDetector` class accepts univariate series; extend it with a `detect_multivariate()` method accepting a DataFrame and feeding all columns to Isolation Forest or an autoencoder.
- **Alerting integration:** wrap the download button with a call to a webhook (Slack, PagerDuty) when the anomaly rate exceeds a configured threshold.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

Built with production-grade engineering practices to demonstrate applied machine learning in industrial settings.
