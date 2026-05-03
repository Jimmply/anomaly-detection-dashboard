#!/usr/bin/env python3
"""
Standalone script to regenerate data/sample_sensor_data.csv.
Run from the project root: python scripts/generate_sample_data.py
"""
import sys
import os

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generator import generate_sample_data

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sensor_data.csv")
    df = generate_sample_data(n_points=1000, random_seed=42)
    df.to_csv(output_path)
    n_anom = df["is_anomaly"].sum()
    print(f"Saved {len(df):,} rows to {output_path}")
    print(f"Anomalous rows: {n_anom} ({100 * n_anom / len(df):.1f}%)")
    print(df.describe().round(3))
