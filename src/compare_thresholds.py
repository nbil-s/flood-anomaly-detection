from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from model import LSTMAutoencoder


MODEL_PATH = Path("models/lstm_autoencoder.pth")
CONFIG_PATH = Path("models/model_config.json")
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    return model


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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    X_validation = np.load(DATA_DIR / "X_validation.npy")
    validation_errors = reconstruction_errors(model, X_validation)

    percentiles = [90, 92, 95, 97, 98, 99]

    rows = []
    for percentile in percentiles:
        threshold = float(np.percentile(validation_errors, percentile))
        predicted = validation_errors > threshold

        rows.append({
            "percentile": percentile,
            "threshold": threshold,
            "validation_windows": len(validation_errors),
            "flagged_windows": int(predicted.sum()),
            "flagged_percentage": float(100 * predicted.mean()),
        })

    results = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "threshold_comparison.csv"
    results.to_csv(output_path, index=False)

    print("\nVALIDATION THRESHOLD COMPARISON")
    print("=" * 65)
    print(results.to_string(index=False, formatters={
        "threshold": "{:.6f}".format,
        "flagged_percentage": "{:.2f}".format,
    }))

    print(f"\nSaved to: {output_path}")
    print(
        "\nNote: These results describe validation reconstruction errors. "
        "They do not measure precision or recall because the validation "
        "windows do not have verified anomaly labels."
    )


if __name__ == "__main__":
    main()