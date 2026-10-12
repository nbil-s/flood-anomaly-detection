
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = Path("results/test_anomaly_scores.csv")
OUTPUT_PATH = Path("results/anomaly_detection_plot.png")


def main():
    results = pd.read_csv(RESULTS_PATH)

    plt.figure(figsize=(14, 6))

    plt.plot(
        results["window_index"],
        results["reconstruction_error"],
        label="Reconstruction error",
        linewidth=1,
    )

    threshold = results["threshold"].iloc[0]

    plt.axhline(
        y=threshold,
        linestyle="--",
        label=f"Anomaly threshold ({threshold:.4f})",
    )

    anomalies = results[results["is_anomaly"]]

    plt.scatter(
        anomalies["window_index"],
        anomalies["reconstruction_error"],
        label="Flagged anomalies",
        s=30,
    )

    plt.title("LSTM Autoencoder Anomaly Detection Results")
    plt.xlabel("Test window index")
    plt.ylabel("Reconstruction error (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print(f"Plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
