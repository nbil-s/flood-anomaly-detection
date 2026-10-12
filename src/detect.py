
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch

from model import LSTMAutoencoder


MODEL_PATH = Path("models/lstm_autoencoder.pth")
CONFIG_PATH = Path("models/model_config.json")
SCALER_PATH = Path("models/scaler.joblib")
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

    return model, config


def reconstruction_errors(model, windows, batch_size=128):
    """Calculate one mean squared reconstruction error per window."""
    errors = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(
                windows[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )

            reconstructed = model(batch)

            # Average across time steps and sensor features.
            batch_errors = (
                (batch - reconstructed) ** 2
            ).mean(dim=(1, 2))

            errors.extend(batch_errors.cpu().numpy())

    return np.asarray(errors)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, config = load_model()
    scaler = joblib.load(SCALER_PATH)

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    )
    X_test = np.load(DATA_DIR / "X_test.npy")

    # These windows were normalized by the training-fitted scaler.
    validation_errors = reconstruction_errors(
        model, X_validation
    )
    test_errors = reconstruction_errors(model, X_test)

    # Initial threshold: 95th percentile of validation errors.
    threshold = float(
        np.percentile(validation_errors, 95)
    )

    # Save threshold and its derivation for future inference.
    threshold_config = {
        "method": "validation_error_percentile",
        "percentile": 95,
        "threshold": threshold,
        "validation_window_count": len(validation_errors),
        "sequence_length": config["sequence_length"],
        "features": config["input_features"],
    }

    with open(
        MODEL_PATH.parent / "threshold_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(threshold_config, file, indent=4)

    # Flag test windows above the validation-derived threshold.
    test_anomalies = test_errors > threshold

    results = pd.DataFrame({
        "window_index": np.arange(len(test_errors)),
        "reconstruction_error": test_errors,
        "threshold": threshold,
        "is_anomaly": test_anomalies,
    })

    results.to_csv(
        RESULTS_DIR / "test_anomaly_scores.csv",
        index=False,
    )

    print("\nAnomaly threshold established.")
    print(f"Validation windows: {len(validation_errors)}")
    print(f"Test windows: {len(test_errors)}")
    print(f"95th-percentile threshold: {threshold:.6f}")
    print(f"Mean test error: {test_errors.mean():.6f}")
    print(f"Maximum test error: {test_errors.max():.6f}")
    print(f"Flagged test windows: {test_anomalies.sum()}")
    print(
        "Percentage flagged:",
        f"{100 * test_anomalies.mean():.2f}%",
    )
    print(
        "Scores saved to:",
        RESULTS_DIR / "test_anomaly_scores.csv",
    )
    print(
        "Threshold saved to:",
        MODEL_PATH.parent / "threshold_config.json",
    )


if __name__ == "__main__":
    main()
