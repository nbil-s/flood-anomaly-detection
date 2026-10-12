
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Configuration
DATA_PATH = Path("data/raw/simulated_environmental_data.csv")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

FEATURES = [
    "water_level",
    "rainfall",
    "temperature",
    "humidity",
]

WINDOW_SIZE = 12

# Chronological split: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def create_windows(data, window_size):
    """Create overlapping time-series windows."""
    if len(data) < window_size:
        raise ValueError(
            f"Need at least {window_size} rows to create a window."
        )

    return np.stack([
        data[i:i + window_size]
        for i in range(len(data) - window_size + 1)
    ]).astype(np.float32)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate the dataset
    df = pd.read_csv(DATA_PATH)

    required_columns = ["timestamp"] + FEATURES
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if df[FEATURES].isna().any().any():
        raise ValueError("Missing sensor measurements detected.")

    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite sensor values detected.")

    # Check chronological continuity.
    # This prototype expects a regular five-minute interval.
    intervals = df["timestamp"].diff().dropna()

    expected_interval = pd.Timedelta(minutes=5)

    if not (intervals == expected_interval).all():
        raise ValueError(
            "Unexpected timestamp interval or gap detected. "
            "Inspect the data before creating windows."
        )

    # 2. Extract sensor measurements
    values = df[FEATURES].to_numpy(dtype=np.float32)

    # 3. Split chronologically BEFORE fitting the scaler
    n = len(values)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_raw = values[:train_end]
    validation_raw = values[train_end:validation_end]
    test_raw = values[validation_end:]

    # 4. Fit normalization using training data only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    validation_scaled = scaler.transform(validation_raw)
    test_scaled = scaler.transform(test_raw)

    # Save the scaler for later real-time inference
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # 5. Create windows separately within each split
    X_train = create_windows(train_scaled, WINDOW_SIZE)
    X_validation = create_windows(
        validation_scaled, WINDOW_SIZE
    )
    X_test = create_windows(test_scaled, WINDOW_SIZE)

    # 6. Save processed arrays
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_validation.npy", X_validation)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)

    # Save the corresponding timestamps for later analysis
    df.iloc[:train_end]["timestamp"].to_csv(
        PROCESSED_DIR / "train_timestamps.csv", index=False
    )
    df.iloc[train_end:validation_end]["timestamp"].to_csv(
        PROCESSED_DIR / "validation_timestamps.csv",
        index=False,
    )
    df.iloc[validation_end:]["timestamp"].to_csv(
        PROCESSED_DIR / "test_timestamps.csv", index=False
    )

    print("\nPreprocessing completed successfully.")
    print(f"Total observations: {n}")
    print(f"Training observations: {len(train_raw)}")
    print(f"Validation observations: {len(validation_raw)}")
    print(f"Testing observations: {len(test_raw)}")

    print("\nWindow shapes:")
    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test:       {X_test.shape}")

    print("\nFeatures:", FEATURES)
    print("Window duration: 60 minutes")
    print("Scaler saved to:", MODEL_DIR / "scaler.joblib")
    print("Processed data saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()

