"""
Synthetic Industrial Sensor Data Generator
==========================================
Generates realistic multi-channel time-series data modelling a manufacturing
environment (e.g., a CNC machine, HVAC system, or pump station).

Sensor channels:
  temperature  - ambient / bearing temperature (°C)
  pressure     - hydraulic / pneumatic pressure (bar)
  vibration    - vibration RMS amplitude (mm/s)
  power        - electrical power draw (kW)

Injected anomaly types:
  spike        - instantaneous large deviation, recovers immediately
  drift        - gradual creep away from nominal, mimics sensor degradation
  stuck        - sensor output freezes at a constant value
  sudden_shift - step change in mean level, mimics process upset or recalibration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SensorConfig:
    """Physical characteristics of a single sensor channel."""

    name: str
    nominal_mean: float          # Steady-state mean value
    nominal_std: float           # Noise amplitude (1-sigma)
    sine_amplitude: float        # Amplitude of periodic oscillation
    sine_period_pts: float       # Period of oscillation in data points
    unit: str = ""               # Display unit (informational)


@dataclass
class AnomalySpec:
    """Specification for a single injected anomaly segment."""

    channel: str
    anomaly_type: str            # 'spike' | 'drift' | 'stuck' | 'sudden_shift'
    start_idx: int
    end_idx: int                 # Only relevant for non-instantaneous types
    magnitude: float = 3.0      # Multiplier relative to nominal_std


# Default sensor definitions
DEFAULT_SENSORS: List[SensorConfig] = [
    SensorConfig("temperature", nominal_mean=75.0,  nominal_std=1.2,  sine_amplitude=3.0,  sine_period_pts=200, unit="°C"),
    SensorConfig("pressure",    nominal_mean=4.5,   nominal_std=0.15, sine_amplitude=0.4,  sine_period_pts=150, unit="bar"),
    SensorConfig("vibration",   nominal_mean=2.8,   nominal_std=0.3,  sine_amplitude=0.6,  sine_period_pts=100, unit="mm/s"),
    SensorConfig("power",       nominal_mean=18.5,  nominal_std=0.8,  sine_amplitude=1.5,  sine_period_pts=300, unit="kW"),
]


class SensorDataGenerator:
    """
    Generates synthetic industrial sensor time-series with injected anomalies.

    Parameters
    ----------
    n_points : int
        Number of time-steps to generate.
    start_time : str
        ISO-8601 timestamp for the first reading.
    freq : str
        Pandas offset alias for the sampling frequency (default '1min').
    sensors : list[SensorConfig], optional
        Override default sensor definitions.
    anomaly_specs : list[AnomalySpec], optional
        Explicit anomaly injection instructions.  If None a default realistic
        set is constructed automatically.
    random_seed : int
        Reproducibility seed.
    """

    def __init__(
        self,
        n_points: int = 1000,
        start_time: str = "2024-01-01 00:00:00",
        freq: str = "1min",
        sensors: Optional[List[SensorConfig]] = None,
        anomaly_specs: Optional[List[AnomalySpec]] = None,
        random_seed: int = 42,
    ) -> None:
        self.n_points = n_points
        self.start_time = start_time
        self.freq = freq
        self.sensors = sensors or DEFAULT_SENSORS
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        if anomaly_specs is None:
            self._anomaly_specs = self._build_default_anomalies()
        else:
            self._anomaly_specs = anomaly_specs

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """
        Generate the full multi-channel sensor DataFrame.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, one column per sensor plus an 'is_anomaly' boolean
            column marking rows where at least one injected anomaly is active.
        """
        timestamps = pd.date_range(
            start=self.start_time, periods=self.n_points, freq=self.freq
        )
        data: dict[str, np.ndarray] = {}
        anomaly_mask = np.zeros(self.n_points, dtype=bool)

        for sensor in self.sensors:
            signal = self._generate_base_signal(sensor)
            specs_for_sensor = [
                s for s in self._anomaly_specs if s.channel == sensor.name
            ]
            signal, mask = self._inject_anomalies(signal, sensor, specs_for_sensor)
            data[sensor.name] = signal
            anomaly_mask |= mask

        df = pd.DataFrame(data, index=timestamps)
        df.index.name = "timestamp"
        df["is_anomaly"] = anomaly_mask

        # Round to sensible precision
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].round(4)

        logger.info(
            "Generated %d-point dataset with %d anomalous rows (%.1f%%).",
            self.n_points,
            anomaly_mask.sum(),
            100 * anomaly_mask.mean(),
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_base_signal(self, sensor: SensorConfig) -> np.ndarray:
        """
        Create a realistic normal-operating signal:
        sinusoidal trend + white noise + slow random walk component.
        """
        t = np.arange(self.n_points)

        # Primary periodic component (machine cycle)
        sine_wave = sensor.sine_amplitude * np.sin(
            2 * np.pi * t / sensor.sine_period_pts
        )

        # Secondary harmonic for realism
        harmonic = 0.3 * sensor.sine_amplitude * np.sin(
            2 * np.pi * t / (sensor.sine_period_pts * 0.47) + 1.2
        )

        # White noise
        noise = self._rng.normal(0, sensor.nominal_std, self.n_points)

        # Slow drift (random walk with mean reversion)
        slow_drift = self._mean_reverting_walk(
            n=self.n_points,
            sigma=sensor.nominal_std * 0.3,
            theta=0.05,
        )

        return sensor.nominal_mean + sine_wave + harmonic + noise + slow_drift

    def _mean_reverting_walk(
        self, n: int, sigma: float, theta: float
    ) -> np.ndarray:
        """Ornstein-Uhlenbeck process for slow background drift."""
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] - theta * x[i - 1] + self._rng.normal(0, sigma)
        return x

    def _inject_anomalies(
        self,
        signal: np.ndarray,
        sensor: SensorConfig,
        specs: List[AnomalySpec],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply each anomaly spec to the signal; return modified signal + mask."""
        signal = signal.copy()
        mask = np.zeros(self.n_points, dtype=bool)

        for spec in specs:
            s, e = spec.start_idx, min(spec.end_idx, self.n_points)
            direction = self._rng.choice([-1, 1])

            if spec.anomaly_type == "spike":
                # Single-point or very short impulse
                spike_val = direction * spec.magnitude * sensor.nominal_std * 5
                signal[s] += spike_val
                mask[s] = True

            elif spec.anomaly_type == "drift":
                # Gradual linear creep
                length = e - s
                ramp = np.linspace(0, direction * spec.magnitude * sensor.nominal_std * 4, length)
                signal[s:e] += ramp
                mask[s:e] = True

            elif spec.anomaly_type == "stuck":
                # Sensor frozen at its value at the start of the segment
                frozen_value = signal[s]
                signal[s:e] = frozen_value + self._rng.normal(0, 0.001, e - s)
                mask[s:e] = True

            elif spec.anomaly_type == "sudden_shift":
                # Step change in mean
                shift = direction * spec.magnitude * sensor.nominal_std * 3
                signal[s:e] += shift
                mask[s:e] = True

            else:
                logger.warning("Unknown anomaly type '%s' – skipped.", spec.anomaly_type)

        return signal, mask

    def _build_default_anomalies(self) -> List[AnomalySpec]:
        """
        Construct a realistic, varied set of anomaly injections spread
        across the timeline and all four sensor channels.
        """
        n = self.n_points
        specs: List[AnomalySpec] = []

        # Temperature: drift event (sensor gradually reading high)
        specs.append(AnomalySpec("temperature", "drift",        int(n * 0.12), int(n * 0.17), magnitude=2.5))
        # Temperature: sudden shift (process upset)
        specs.append(AnomalySpec("temperature", "sudden_shift", int(n * 0.55), int(n * 0.58), magnitude=2.0))

        # Pressure: spike (pressure transient)
        specs.append(AnomalySpec("pressure", "spike", int(n * 0.23), int(n * 0.23) + 1, magnitude=4.0))
        specs.append(AnomalySpec("pressure", "spike", int(n * 0.67), int(n * 0.67) + 1, magnitude=3.5))
        # Pressure: stuck sensor
        specs.append(AnomalySpec("pressure", "stuck",          int(n * 0.78), int(n * 0.82), magnitude=1.0))

        # Vibration: multiple spikes (bearing wear events)
        for frac in [0.31, 0.44, 0.61, 0.88]:
            specs.append(AnomalySpec("vibration", "spike", int(n * frac), int(n * frac) + 1, magnitude=5.0))
        # Vibration: drift (gradual bearing degradation)
        specs.append(AnomalySpec("vibration", "drift", int(n * 0.70), int(n * 0.76), magnitude=3.0))

        # Power: sudden shift (load change / reconfiguration)
        specs.append(AnomalySpec("power", "sudden_shift", int(n * 0.40), int(n * 0.43), magnitude=2.5))
        # Power: spike (inrush current event)
        specs.append(AnomalySpec("power", "spike",          int(n * 0.91), int(n * 0.91) + 1, magnitude=6.0))

        return specs


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def generate_sample_data(
    n_points: int = 1000,
    start_time: str = "2024-01-01 00:00:00",
    freq: str = "1min",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    One-call helper for generating a standard sample dataset.

    Returns a DataFrame with columns:
        timestamp (index), temperature, pressure, vibration, power, is_anomaly
    """
    gen = SensorDataGenerator(
        n_points=n_points,
        start_time=start_time,
        freq=freq,
        random_seed=random_seed,
    )
    return gen.generate()
