To regenerate sample_sensor_data.csv, run from the project root:

    python scripts/generate_sample_data.py

This requires the project dependencies to be installed (pip install -r requirements.txt).
The script calls src/data_generator.py with n_points=1000 and random_seed=42.
