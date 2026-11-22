
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

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

SEQUENCE_LENGTHS = [60, 120, 240]

EXPECTED_INTERVAL = pd.Timedelta(minutes=1)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


# ============================================================
# WINDOW CREATION
# ============================================================

def create_windows(data, window_size):
    """Create overlapping time-series windows."""

    if len(data) < window_size:
        raise ValueError(
            f"Need at least {window_size} observations."
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        data,
        window_shape=window_size,
        axis=0,
    )

    # Convert from (samples, features, window)
    # to (samples, window, features).
    windows = np.moveaxis(windows, -1, 1)

    return windows.astype(np.float32)


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    required_columns = ["timestamp"] + FEATURES

    missing_columns = (
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 2. VALIDATE DATA
    # --------------------------------------------------------

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if df[FEATURES].isna().any().any():
        raise ValueError("Missing feature values detected.")

    values = df[FEATURES].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(values).all():
        raise ValueError("Non-finite values detected.")

    if (df["rainfall"] < 0).any():
        raise ValueError("Negative rainfall detected.")

    intervals = df["timestamp"].diff().dropna()

    if not (
        intervals == EXPECTED_INTERVAL
    ).all():
        raise ValueError(
            "Unexpected timestamp interval or gap detected."
        )

    # --------------------------------------------------------
    # 3. CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * TRAIN_RATIO)

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_raw = values[:train_end]

    validation_raw = values[
        train_end:validation_end
    ]

    test_raw = values[
        validation_end:
    ]

    train_timestamps = df["timestamp"].iloc[:train_end]

    validation_timestamps = (
        df["timestamp"].iloc[train_end:validation_end]
    )

    test_timestamps = (
        df["timestamp"].iloc[validation_end:]
    )

    # --------------------------------------------------------
    # 4. FIT SCALER ON TRAINING DATA ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_raw
    )

    validation_scaled = scaler.transform(
        validation_raw
    )

    test_scaled = scaler.transform(
        test_raw
    )

    # Save scaler for this shared experiment.
    scaler_path = PROCESSED_DIR / "scaler.joblib"

    joblib.dump(
        scaler,
        scaler_path,
    )

    # --------------------------------------------------------
    # 5. SAVE SPLIT TIMESTAMPS
    # --------------------------------------------------------

    train_timestamps.to_csv(
        PROCESSED_DIR / "train_timestamps.csv",
        index=False,
    )

    validation_timestamps.to_csv(
        PROCESSED_DIR / "validation_timestamps.csv",
        index=False,
    )

    test_timestamps.to_csv(
        PROCESSED_DIR / "test_timestamps.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. CREATE WINDOWS FOR EACH SEQUENCE LENGTH
    # --------------------------------------------------------

    summary = {}

    for sequence_length in SEQUENCE_LENGTHS:

        print("\n" + "=" * 65)
        print(
            f"PREPARING {sequence_length}-READING SEQUENCES"
        )
        print("=" * 65)

        X_train = create_windows(
            train_scaled,
            sequence_length,
        )

        X_validation = create_windows(
            validation_scaled,
            sequence_length,
        )

        X_test = create_windows(
            test_scaled,
            sequence_length,
        )

        experiment_output = (
            PROCESSED_DIR
            / f"sequence_{sequence_length}"
        )

        experiment_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            experiment_output / "X_train.npy",
            X_train,
        )

        np.save(
            experiment_output / "X_validation.npy",
            X_validation,
        )

        np.save(
            experiment_output / "X_test.npy",
            X_test,
        )

        # Store timestamps for the END of each window.
        window_timestamps = {
            "train": train_timestamps.iloc[
                sequence_length - 1:
            ],
            "validation": validation_timestamps.iloc[
                sequence_length - 1:
            ],
            "test": test_timestamps.iloc[
                sequence_length - 1:
            ],
        }

        for split_name, timestamps in (
            window_timestamps.items()
        ):

            timestamps.to_csv(
                experiment_output
                / f"{split_name}_window_timestamps.csv",
                index=False,
            )

        summary[str(sequence_length)] = {
            "sequence_length": sequence_length,
            "window_duration_minutes": sequence_length - 1,
            "training_windows": len(X_train),
            "validation_windows": len(X_validation),
            "testing_windows": len(X_test),
            "features": FEATURES,
        }

        print(f"Training windows:   {X_train.shape}")
        print(f"Validation windows: {X_validation.shape}")
        print(f"Testing windows:    {X_test.shape}")

        print(
            f"Window duration: {sequence_length - 1} minutes "
            "between first and last readings"
        )

        print(
            f"Saved to: {experiment_output}"
        )

    # --------------------------------------------------------
    # 7. SAVE EXPERIMENT CONFIGURATION
    # --------------------------------------------------------

    config = {
        "experiment": "improved_simulation_240",
        "total_observations": n,
        "sampling_interval_seconds": 60,
        "features": FEATURES,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": 1 - TRAIN_RATIO - VALIDATION_RATIO,
        "sequence_lengths": SEQUENCE_LENGTHS,
        "scaler": "StandardScaler",
        "scaler_fitted_on": "training_data_only",
        "windowing": "separate_within_each_split",
        "results": summary,
    }

    config_path = PROCESSED_DIR / "preprocessing_config.json"

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 8. FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 65)

    print(f"Total observations: {n:,}")

    print(
        f"Training observations: {len(train_raw):,}"
    )

    print(
        f"Validation observations: {len(validation_raw):,}"
    )

    print(
        f"Testing observations: {len(test_raw):,}"
    )

    print("\nFeatures:", FEATURES)

    print("\nSequence comparison:")

    for length, details in summary.items():

        print(
            f"{length} readings: "
            f"Train={details['training_windows']:,}, "
            f"Validation={details['validation_windows']:,}, "
            f"Test={details['testing_windows']:,}"
        )

    print("\nScaler saved:", scaler_path)

    print("Configuration saved:", config_path)

    print("\nDATA VALIDATION: PASSED")


if __name__ == "__main__":
    main()
