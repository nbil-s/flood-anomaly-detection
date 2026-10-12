
from pathlib import Path
import json

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

from model import LSTMAutoencoder


DATA_PATH = Path("data/raw/simulated_environmental_data.csv")
SCALER_PATH = Path("models/scaler.joblib")
MODEL_PATH = Path("models/lstm_autoencoder.pth")
CONFIG_PATH = Path("models/model_config.json")
THRESHOLD_PATH = Path("models/threshold_config.json")
OUTPUT_DIR = Path("results")

FEATURES = ["water_level", "rainfall", "temperature", "humidity"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    model = LSTMAutoencoder(
        input_features=config["input_features"],
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_config = json.load(file)

    return model, config, float(threshold_config["threshold"])


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
            batch_errors = ((batch - reconstructed) ** 2).mean(dim=(1, 2))
            errors.extend(batch_errors.cpu().numpy())

    return np.asarray(errors)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the original data and preserve a separate test copy.
    data = pd.read_csv(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    n = len(data)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    # Only inject anomalies into the held-out test period.
    test = data.iloc[validation_end:].copy().reset_index(drop=True)
    original_test = test.copy()

    # One label per observation: 1 means deliberately modified.
    row_labels = np.zeros(len(test), dtype=int)

    def mark_event(start, length, event_name):
        end = start + length
        if start < 0 or end > len(test):
            raise ValueError(f"{event_name} falls outside the test partition.")
        row_labels[start:end] = 1
        print(
            f"{event_name}: test rows {start} to {end - 1}; "
            f"{test.loc[start, 'timestamp']} to {test.loc[end - 1, 'timestamp']}"
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

    # Event C: heavy rainfall alongside a rising water level.
    start_c = 400
    mark_event(start_c, 6, "Event C - combined conditions")
    baseline_c = original_test.loc[start_c, "water_level"]
    test.loc[start_c:start_c + 5, "rainfall"] = 12.0
    test.loc[start_c:start_c + 5, "water_level"] = (
        baseline_c + np.array([1, 2, 3, 4, 5, 6], dtype=float)
    )

    # Event D: a single implausible water-level sensor reading.
    start_d = 550
    mark_event(start_d, 1, "Event D - simulated sensor fault")
    test.loc[start_d, "water_level"] = 100.0
    diagnostic_test = test.copy()
    diagnostic_test["injected_anomaly"] = row_labels

    diagnostic_path = OUTPUT_DIR / "controlled_test_data.csv"
    diagnostic_test.to_csv(diagnostic_path, index=False)

    print(f"\nSaved controlled test data to: {diagnostic_path}")
    print(
        "Injected observations:",
        int(diagnostic_test["injected_anomaly"].sum()),
    )

    # Scale using the scaler fitted on the original training data.
    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(test[FEATURES].to_numpy(dtype=float))

    # Build sliding windows; each window covers one hour.
    window_size = int(load_model()[1]["sequence_length"])
    windows = np.asarray([
        scaled[i:i + window_size]
        for i in range(len(scaled) - window_size + 1)
    ])

    # A window is positive if it contains at least one injected row.
    window_labels = np.asarray([
        int(row_labels[i:i + window_size].any())
        for i in range(len(row_labels) - window_size + 1)
    ])

    model, config, threshold = load_model()
    errors = reconstruction_errors(model, windows)
    predictions = (errors > threshold).astype(int)

    precision = precision_score(
        window_labels, predictions, zero_division=0
    )
    recall = recall_score(
        window_labels, predictions, zero_division=0
    )
    f1 = f1_score(
        window_labels, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        window_labels, predictions, labels=[0, 1]
    ).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

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

    results.to_csv(
        OUTPUT_DIR / "controlled_anomaly_evaluation.csv",
        index=False,
    )

    print("\nControlled anomaly evaluation")
    print("--------------------------------")
    print(f"Test observations: {len(test)}")
    print(f"Evaluation windows: {len(windows)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Actual positive windows: {window_labels.sum()}")
    print(f"Predicted positive windows: {predictions.sum()}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"True negatives: {tn}")
    print(f"False negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False-positive rate: {false_positive_rate:.4f}")
    print(
        "\nResults saved to "
        "results\\controlled_anomaly_evaluation.csv"
    )


if __name__ == "__main__":
    main()
