
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Locate project and experiment directories
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Reuse the existing model architecture without modifying it
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import LSTMAutoencoder


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Experiment paths
DATA_DIR = EXPERIMENT_DIR / "data" / "processed"
MODEL_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Training configuration
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 7
SEQUENCE_LENGTH = 24

INPUT_FEATURES = 4
HIDDEN_SIZE = 32
LATENT_SIZE = 16


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)


def load_sequences(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run experiment preprocessing first."
        )

    data = np.load(path)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D sequence data, got shape {data.shape}"
        )

    if data.shape[1:] != (SEQUENCE_LENGTH, INPUT_FEATURES):
        raise ValueError(
            f"Unexpected sequence shape: {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values found in {path}")

    return torch.tensor(data, dtype=torch.float32)


def main():
    X_train = load_sequences("X_train.npy")
    X_validation = load_sequences("X_validation.npy")

    print("\nTraining data:", tuple(X_train.shape))
    print("Validation data:", tuple(X_validation.shape))

    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        TensorDataset(X_validation),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = LSTMAutoencoder(
        input_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_history = []

    model_path = MODEL_DIR / "lstm_autoencoder.pth"

    for epoch in range(1, EPOCHS + 1):

        # Training
        model.train()
        total_train_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(batch)

        train_loss = total_train_loss / len(X_train)

        # Validation
        model.eval()
        total_validation_loss = 0.0

        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device)

                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)

                total_validation_loss += (
                    loss.item() * len(batch)
                )

        validation_loss = (
            total_validation_loss / len(X_validation)
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Validation MSE: {validation_loss:.6f}"
        )

        # Save the best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), model_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                "\nEarly stopping: validation loss "
                "did not improve."
            )
            break

    # Save training history
    history_path = RESULTS_DIR / "training_history.csv"

    pd.DataFrame(training_history).to_csv(
        history_path,
        index=False,
    )

    # Save model configuration
    config = {
        "input_features": INPUT_FEATURES,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_device": str(device),
    }

    config_path = MODEL_DIR / "model_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print("\nIMPROVED DATASET TRAINING COMPLETED")
    print("=" * 45)
    print("Best epoch:", best_epoch)
    print(
        "Best validation MSE:",
        f"{best_validation_loss:.6f}",
    )
    print("Model saved to:", model_path)
    print("Configuration saved to:", config_path)
    print("Training history saved to:", history_path)


if __name__ == "__main__":
    main()
