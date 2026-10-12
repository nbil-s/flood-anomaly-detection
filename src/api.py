
from collections import deque
from datetime import datetime
from pathlib import Path
import json
import math

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import LSTMAutoencoder


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "lstm_autoencoder.pth"
CONFIG_PATH = ROOT_DIR / "models" / "model_config.json"
SCALER_PATH = ROOT_DIR / "models" / "scaler.joblib"
THRESHOLD_PATH = ROOT_DIR / "models" / "threshold_config.json"

FEATURES = [
    "water_level",
    "rainfall",
    "temperature",
    "humidity",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Flood Anomaly Detection API",
    description="Receives environmental readings and detects unusual patterns.",
    version="0.1.0",
)

# In-memory buffer for the latest readings.
# This is suitable for local testing, not persistent storage.
reading_buffer = deque(maxlen=12)


def load_artifacts():
    required_files = [
        MODEL_PATH,
        CONFIG_PATH,
        SCALER_PATH,
        THRESHOLD_PATH,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise RuntimeError(
            "Required model artifacts are missing: " + ", ".join(missing)
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
        threshold_config = json.load(file)

    if threshold_config.get("sequence_length") != 12:
        raise RuntimeError("Expected a sequence length of 12.")

    if threshold_config.get("features") != len(FEATURES):
        raise RuntimeError("Expected exactly four input features.")

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

    scaler = joblib.load(SCALER_PATH)
    threshold = float(threshold_config["threshold"])

    return model, scaler, threshold


model, scaler, threshold = load_artifacts()


class SensorReading(BaseModel):
    timestamp: datetime
    water_level: float = Field(allow_inf_nan=False)
    rainfall: float = Field(ge=0, allow_inf_nan=False)
    temperature: float = Field(allow_inf_nan=False)
    humidity: float = Field(ge=0, le=100, allow_inf_nan=False)


@app.get("/")
def home():
    return {
        "message": "Flood Anomaly Detection API is running.",
        "model": "LSTM Autoencoder",
        "sequence_length": 12,
        "threshold": threshold,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "buffered_readings": len(reading_buffer),
        "readings_required": 12,
    }


@app.post("/readings")
def receive_reading(reading: SensorReading):
    values = reading.model_dump()

    # Convert sensor features into the model's expected order.
    feature_values = [values[name] for name in FEATURES]

    if not all(math.isfinite(value) for value in feature_values):
        raise HTTPException(
            status_code=422,
            detail="All sensor readings must be finite numbers.",
        )

    # Keep readings in chronological order.
    if reading_buffer and reading.timestamp <= reading_buffer[-1]["timestamp"]:
        raise HTTPException(
            status_code=400,
            detail="Timestamp must be later than the previous reading.",
        )

    reading_buffer.append({
        "timestamp": reading.timestamp,
        "features": feature_values,
    })

    if len(reading_buffer) < 12:
        return {
            "status": "collecting",
            "message": "Waiting for a complete 12-reading sequence.",
            "readings_received": len(reading_buffer),
            "readings_required": 12,
            "anomaly_detected": None,
        }

    sequence = np.array(
        [item["features"] for item in reading_buffer],
        dtype=np.float32,
    )

    scaled_sequence = scaler.transform(sequence)

    input_tensor = torch.tensor(
        scaled_sequence,
        dtype=torch.float32,
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        reconstructed = model(input_tensor)
        reconstruction_error = (
            (input_tensor - reconstructed) ** 2
        ).mean(dim=(1, 2)).item()

    anomaly_detected = reconstruction_error > threshold

    return {
        "status": "anomaly_detected" if anomaly_detected else "normal",
        "timestamp": reading.timestamp.isoformat(),
        "reconstruction_error": round(reconstruction_error, 6),
        "threshold": round(threshold, 6),
        "anomaly_detected": anomaly_detected,
        "interpretation": (
            "Unusual sensor pattern detected; investigate conditions."
            if anomaly_detected
            else "No anomaly detected by the current model."
        ),
        "data_source": "submitted_sensor_reading",
    }
