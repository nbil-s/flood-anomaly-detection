
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch


# Experiment and project paths
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = EXPERIMENT_DIR / "models"
DATA_DIR = EXPERIMENT_DIR / "data" / "processed"
RESULTS_DIR = EXPERIMENT_DIR / "results"

MODEL_PATH = MODEL_DIR / "lstm_autoencoder.pth"
CONFIG_PATH = MODEL_DIR / "model_config.json"

# Reuse the original architecture
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import LSTMAutoencoder


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

            batch_errors = (
                (batch - reconstructed) ** 2
            ).mean(dim=(1, 2))

            errors.extend(batch_errors.cpu().numpy())

    return np.asarray(errors)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, config = load_model()

    X_validation = np.load(
        DATA_DIR / "X_validation.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    # Calculate reconstruction errors
    validation_errors = reconstruction_errors(
        model, X_validation
    )

    test_errors = reconstruction_errors(
        model, X_test
    )

    # Establish threshold using validation data only
    threshold = float(
        np.percentile(validation_errors, 95)
    )

    threshold_config = {
        "method": "validation_error_percentile",
        "percentile": 95,
        "threshold": threshold,
        "validation_window_count": len(validation_errors),
        "sequence_length": config["sequence_length"],
        "features": config["input_features"],
        "best_epoch": config["best_epoch"],
    }

    threshold_path = MODEL_DIR / "threshold_config.json"

    with open(
        threshold_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(threshold_config, file, indent=4)

    # Save validation scores
    validation_results = pd.DataFrame({
        "window_index": np.arange(len(validation_errors)),
        "reconstruction_error": validation_errors,
        "threshold": threshold,
        "is_anomaly": validation_errors > threshold,
    })

    validation_results.to_csv(
        RESULTS_DIR / "validation_anomaly_scores.csv",
        index=False,
    )

    # Apply threshold to test scores
    test_anomalies = test_errors > threshold

    test_results = pd.DataFrame({
        "window_index": np.arange(len(test_errors)),
        "reconstruction_error": test_errors,
        "threshold": threshold,
        "is_anomaly": test_anomalies,
    })

    test_results.to_csv(
        RESULTS_DIR / "test_anomaly_scores.csv",
        index=False,
    )

    print("\nIMPROVED MODEL DETECTION RESULTS")
    print("=" * 45)

    print("Validation windows:", len(validation_errors))
    print("Test windows:", len(test_errors))

    print(
        "Validation mean error:",
        f"{validation_errors.mean():.6f}",
    )

    print(
        "Validation 95th percentile:",
        f"{threshold:.6f}",
    )

    print(
        "Test mean error:",
        f"{test_errors.mean():.6f}",
    )

    print(
        "Test maximum error:",
        f"{test_errors.max():.6f}",
    )

    print("Flagged test windows:", int(test_anomalies.sum()))

    print(
        "Percentage flagged:",
        f"{100 * test_anomalies.mean():.2f}%",
    )

    print("\nThreshold saved to:", threshold_path)

    print(
        "Validation scores saved to:",
        RESULTS_DIR / "validation_anomaly_scores.csv",
    )

    print(
        "Test scores saved to:",
        RESULTS_DIR / "test_anomaly_scores.csv",
    )


if __name__ == "__main__":
    main()
