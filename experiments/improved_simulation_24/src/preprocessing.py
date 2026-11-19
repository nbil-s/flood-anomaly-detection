
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Experiment paths
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "raw"
    / "simulated_environmental_data.csv"
)

PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"

FEATURES = [
    "water_level",
    "rainfall",
    "temperature",
    "humidity",
]

WINDOW_SIZE = 24
EXPECTED_INTERVAL = pd.Timedelta(seconds=30)

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
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

    if (df["rainfall"] < 0).any():
        raise ValueError("Negative rainfall values detected.")

    intervals = df["timestamp"].diff().dropna()

    if not (intervals == EXPECTED_INTERVAL).all():
        raise ValueError(
            "Unexpected timestamp interval or gap detected."
        )

    # 2. Extract sensor measurements
    values = df[FEATURES].to_numpy(dtype=np.float32)

    # 3. Split chronologically before fitting the scaler
    n = len(values)
    train_end = int(n * TRAIN_RATIO)
    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_raw = values[:train_end]
    validation_raw = values[train_end:validation_end]
    test_raw = values[validation_end:]

    # 4. Fit scaler using training data only
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_raw)
    validation_scaled = scaler.transform(validation_raw)
    test_scaled = scaler.transform(test_raw)

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

    # Save scaler inside this experiment only
    scaler_path = PROCESSED_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Save timestamps corresponding to each split
    df.iloc[:train_end]["timestamp"].to_csv(
        PROCESSED_DIR / "train_timestamps.csv",
        index=False,
    )

    df.iloc[train_end:validation_end]["timestamp"].to_csv(
        PROCESSED_DIR / "validation_timestamps.csv",
        index=False,
    )

    df.iloc[validation_end:]["timestamp"].to_csv(
        PROCESSED_DIR / "test_timestamps.csv",
        index=False,
    )

    print("\nIMPROVED DATASET PREPROCESSING COMPLETED")
    print("=" * 50)

    print(f"Dataset: {DATA_PATH}")
    print(f"Total observations: {n}")
    print(f"Training observations: {len(train_raw)}")
    print(f"Validation observations: {len(validation_raw)}")
    print(f"Testing observations: {len(test_raw)}")

    print("\nWindow shapes:")
    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"X_test:       {X_test.shape}")

    print("\nFeatures:", FEATURES)
    print(f"Sequence length: {WINDOW_SIZE} readings")
    print(
        "Window duration: "
        f"{(WINDOW_SIZE - 1) * 30 / 60:.1f} minutes "
        "between first and last readings"
    )

    print("\nScaler saved to:", scaler_path)
    print("Processed data saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
