
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMAutoencoder


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 7

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)


def load_sequences(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run preprocessing.py first."
        )

    data = np.load(path)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D sequence data, got shape {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values found in {path}")

    return torch.tensor(data, dtype=torch.float32)


def main():
    X_train = load_sequences("X_train.npy")
    X_validation = load_sequences("X_validation.npy")

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
        input_features=4,
        hidden_size=32,
        latent_size=16,
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

        # Save the best model based on validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "lstm_autoencoder.pth",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation loss did not improve.")
            break

    # Save history and configuration
    history_path = RESULTS_DIR / "training_history.csv"

    import pandas as pd
    pd.DataFrame(training_history).to_csv(
        history_path, index=False
    )

    config = {
        "input_features": 4,
        "hidden_size": 32,
        "latent_size": 16,
        "sequence_length": 12,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "training_device": str(device),
    }

    with open(
        MODEL_DIR / "model_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    print("\nTraining completed.")
    print("Best epoch:", best_epoch)
    print(
        "Best validation MSE:",
        f"{best_validation_loss:.6f}",
    )
    print("Model saved to:", MODEL_DIR / "lstm_autoencoder.pth")
    print("History saved to:", history_path)


if __name__ == "__main__":
    main()
