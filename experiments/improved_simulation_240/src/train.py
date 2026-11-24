
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PATHS AND MODEL IMPORT
# ============================================================

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import LSTMAutoencoder


PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
MODEL_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

INPUT_FEATURES = 4
HIDDEN_SIZE = 32
LATENT_SIZE = 16

BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 7

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Train an LSTM Autoencoder."
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=[60, 120, 240],
        required=True,
        help="Number of readings per input sequence.",
    )

    return parser.parse_args()


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_sequences(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}"
        )

    data = np.load(path)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D data, received {data.shape}"
        )

    if data.shape[2] != INPUT_FEATURES:
        raise ValueError(
            f"Expected {INPUT_FEATURES} features, "
            f"received {data.shape[2]}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"Non-finite values detected in {path}"
        )

    print(
        f"Loaded {path.name}: {data.shape}"
    )

    return torch.from_numpy(data).float()


# ============================================================
# MAIN TRAINING PROCEDURE
# ============================================================

def main():

    args = parse_arguments()

    sequence_length = args.sequence_length

    print("\n" + "=" * 70)
    print("LSTM AUTOENCODER TRAINING")
    print("=" * 70)

    print("Sequence length:", sequence_length)
    print("Input features:", INPUT_FEATURES)
    print("Hidden size:", HIDDEN_SIZE)
    print("Latent size:", LATENT_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Early stopping patience:", PATIENCE)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nTraining device:", device)

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    model_output = MODEL_DIR / f"sequence_{sequence_length}"
    results_output = RESULTS_DIR / f"sequence_{sequence_length}"

    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    sequence_dir = (
        PROCESSED_DIR / f"sequence_{sequence_length}"
    )

    X_train = load_sequences(
        sequence_dir / "X_train.npy"
    )

    X_validation = load_sequences(
        sequence_dir / "X_validation.npy"
    )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        TensorDataset(X_validation),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = LSTMAutoencoder(
        input_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTrainable parameters:", parameter_count)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # EARLY STOPPING VARIABLES
    # --------------------------------------------------------

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_history = []

    best_model_path = (
        model_output / "lstm_autoencoder.pth"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        epoch_start = time.time()

        # TRAINING
        model.train()

        total_train_loss = 0.0

        for (batch,) in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(
                reconstruction,
                batch,
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item() * len(batch)
            )

        train_loss = (
            total_train_loss / len(X_train)
        )

        # VALIDATION
        model.eval()

        total_validation_loss = 0.0

        with torch.no_grad():

            for (batch,) in validation_loader:

                batch = batch.to(device)

                reconstruction = model(batch)

                loss = criterion(
                    reconstruction,
                    batch,
                )

                total_validation_loss += (
                    loss.item() * len(batch)
                )

        validation_loss = (
            total_validation_loss / len(X_validation)
        )

        epoch_duration = time.time() - epoch_start

        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "epoch_duration_seconds": epoch_duration,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Validation MSE: {validation_loss:.6f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # SAVE BEST MODEL
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print("  Best model saved.")

        else:

            epochs_without_improvement += 1

        # EARLY STOPPING
        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping: "
                "validation loss did not improve."
            )

            break

    total_training_time = time.time() - start_time

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history_path = (
        results_output / "training_history.csv"
    )

    pd.DataFrame(training_history).to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE MODEL CONFIGURATION
    # --------------------------------------------------------

    config = {
        "input_features": INPUT_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "sequence_length": sequence_length,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "random_seed": RANDOM_SEED,
        "trainable_parameters": parameter_count,
        "training_windows": len(X_train),
        "validation_windows": len(X_validation),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_device": str(device),
        "training_time_seconds": total_training_time,
    }

    config_path = (
        model_output / "model_config.json"
    )

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
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("Sequence length:", sequence_length)
    print("Best epoch:", best_epoch)

    print(
        "Best validation MSE:",
        f"{best_validation_loss:.6f}",
    )

    print(
        "Total training time:",
        f"{total_training_time / 60:.2f} minutes",
    )

    print("Model saved:", best_model_path)
    print("Configuration saved:", config_path)
    print("Training history saved:", history_path)


if __name__ == "__main__":
    main()
