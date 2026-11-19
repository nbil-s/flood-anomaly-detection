
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# EXPERIMENT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import LSTMAutoencoder

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "improved_simulation_24"

DATA_PATH = EXPERIMENT_DIR / "data" / "raw" / "simulated_environmental_data.csv"
SCALER_PATH = EXPERIMENT_DIR / "data" / "processed" / "scaler.joblib"
MODEL_PATH = EXPERIMENT_DIR / "models" / "lstm_autoencoder.pth"
CONFIG_PATH = EXPERIMENT_DIR / "models" / "model_config.json"
THRESHOLD_PATH = EXPERIMENT_DIR / "models" / "threshold_config.json"
OUTPUT_DIR = EXPERIMENT_DIR / "results"

FEATURES = ["water_level", "rainfall", "temperature", "humidity"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# LOAD THE IMPROVED MODEL
# --------------------------------------------------

def load_model():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    model = LSTMAutoencoder(
        input_features=config["input_features"],
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
    ).to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_config = json.load(file)

    threshold = float(threshold_config["threshold"])

    return model, config, threshold


# --------------------------------------------------
# CALCULATE RECONSTRUCTION ERRORS
# --------------------------------------------------

def reconstruction_errors(model, windows, batch_size=128):
    errors = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(
                windows[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )

            reconstructed = model(batch)

            batch_errors = (
                (batch - reconstructed) ** 2
            ).mean(dim=(1, 2))

            errors.extend(batch_errors.cpu().numpy())

    return np.asarray(errors)


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        DATA_PATH,
        SCALER_PATH,
        MODEL_PATH,
        CONFIG_PATH,
        THRESHOLD_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required experiment file not found: {path}"
            )

    model, config, threshold = load_model()

    window_size = int(config["sequence_length"])

    data = pd.read_csv(DATA_PATH)

    required_columns = ["timestamp"] + FEATURES

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="raise",
    )

    data = data.sort_values("timestamp").reset_index(drop=True)

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing required values.")

    if not np.isfinite(
        data[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError("Dataset contains non-finite feature values.")

    # Recreate the chronological 70/15/15 split.
    n = len(data)
    validation_end = int(n * 0.85)

    test = data.iloc[validation_end:].copy().reset_index(drop=True)
    original_test = test.copy()

    if len(test) < window_size:
        raise ValueError("Test partition is shorter than the sequence length.")

    row_labels = np.zeros(len(test), dtype=int)

    def mark_event(start, length, event_name):
        end = start + length

        if start < 0 or end > len(test):
            raise ValueError(
                f"{event_name} falls outside the test partition."
            )

        row_labels[start:end] = 1

        print(
            f"{event_name}: test rows {start} to {end - 1}; "
            f"{test.loc[start, 'timestamp']} to "
            f"{test.loc[end - 1, 'timestamp']}"
        )

    # Event A: rapid rise in water level over six readings.
    start_a = 100
    mark_event(start_a, 6, "Event A - rapid water-level rise")

    baseline_a = original_test.loc[start_a, "water_level"]

    test.loc[start_a:start_a + 5, "water_level"] = (
        baseline_a + np.array([1, 2, 3, 4, 5, 6], dtype=float)
    )

    # Event B: intense rainfall over six readings.
    start_b = 250
    mark_event(start_b, 6, "Event B - intense rainfall")

    test.loc[start_b:start_b + 5, "rainfall"] = 12.0

    # Event C: heavy rainfall alongside rising water level.
    start_c = 400
    mark_event(start_c, 6, "Event C - combined conditions")

    baseline_c = original_test.loc[start_c, "water_level"]

    test.loc[start_c:start_c + 5, "rainfall"] = 12.0

    test.loc[start_c:start_c + 5, "water_level"] = (
        baseline_c + np.array([1, 2, 3, 4, 5, 6], dtype=float)
    )

    # Event D: one implausible water-level sensor reading.
    start_d = 550
    mark_event(start_d, 1, "Event D - simulated sensor fault")

    test.loc[start_d, "water_level"] = 100.0

    diagnostic_test = test.copy()
    diagnostic_test["injected_anomaly"] = row_labels

    diagnostic_path = OUTPUT_DIR / "controlled_test_data.csv"
    diagnostic_test.to_csv(diagnostic_path, index=False)

    print(f"\nSaved controlled test data to: {diagnostic_path}")
    print("Injected observations:", int(row_labels.sum()))

    scaler = joblib.load(SCALER_PATH)

    scaled = scaler.transform(
        test[FEATURES].to_numpy(dtype=float)
    )

    # Build overlapping sliding windows.
    # 24 readings at 30-second intervals span 11.5 minutes.
    windows = np.asarray([
        scaled[i:i + window_size]
        for i in range(len(scaled) - window_size + 1)
    ])

    window_labels = np.asarray([
        int(row_labels[i:i + window_size].any())
        for i in range(len(row_labels) - window_size + 1)
    ])

    # Calculate reconstruction errors and predictions.
    errors = reconstruction_errors(model, windows)

    predictions = (errors > threshold).astype(int)

    # Calculate evaluation metrics.
    precision = precision_score(
        window_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        window_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        window_labels,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        window_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) else 0.0
    )

    results = pd.DataFrame({
        "window_index": np.arange(len(errors)),
        "window_start": test["timestamp"].iloc[
            :len(errors)
        ].reset_index(drop=True),
        "reconstruction_error": errors,
        "threshold": threshold,
        "actual_anomaly": window_labels,
        "predicted_anomaly": predictions,
    })

    results_path = OUTPUT_DIR / "controlled_anomaly_evaluation.csv"
    results.to_csv(results_path, index=False)

    # Print evaluation summary.
    print("\nIMPROVED MODEL - CONTROLLED ANOMALY EVALUATION")
    print("=" * 55)

    print(f"Test observations: {len(test)}")
    print(f"Sequence length: {window_size} readings")
    window_duration = (window_size - 1) * 30 / 60
    print(f"Window duration: {window_duration:.1f} minutes")
    print(f"Evaluation windows: {len(windows)}")
    print(f"Threshold: {threshold:.6f}")

    print("\nCONFUSION MATRIX RESULTS")
    print("-" * 35)

    print(f"Actual positive windows: {window_labels.sum()}")
    print(f"Predicted positive windows: {predictions.sum()}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"True negatives: {tn}")
    print(f"False negatives: {fn}")

    print("\nEVALUATION METRICS")
    print("-" * 35)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False-positive rate: {false_positive_rate:.4f}")

    print("\nSAVED OUTPUTS")
    print("-" * 35)
    print(f"Controlled test data: {diagnostic_path}")
    print(f"Evaluation results: {results_path}")


if __name__ == "__main__":
    main()
