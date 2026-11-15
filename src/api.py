
from collections import deque
from pathlib import Path
import json
import math
import traceback

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from firebase_admin import auth as firebase_auth
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from model import LSTMAutoencoder
from firebase_db import save_sensor_reading, db

from notifications import send_anomaly_email
from datetime import datetime, timezone, timedelta


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# In-memory buffer for local testing.
reading_buffer = None

# Prevent repeated email alerts for a continuing anomaly.
last_email_alert_at = None
EMAIL_ALERT_COOLDOWN = 30  # minutes

# Firebase authentication
bearer_scheme = HTTPBearer(auto_error=False)


def require_firebase_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    Verify the Firebase ID token supplied by the dashboard.
    Reject requests without a valid token.
    """

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        decoded_token = firebase_auth.verify_id_token(
            credentials.credentials
        )

        return decoded_token

    except Exception as exc:
        print(f"Firebase authentication error: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


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

    sequence_length = config.get("sequence_length")

    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise RuntimeError("Invalid sequence length in model configuration.")

    if threshold_config.get("sequence_length") != sequence_length:
        raise RuntimeError(
            "The model and threshold configurations have different sequence lengths."
        )

    if config.get("input_features") != len(FEATURES):
        raise RuntimeError("Expected exactly four model input features.")

    if threshold_config.get("features") != len(FEATURES):
        raise RuntimeError("Expected exactly four threshold features.")

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

    return model, scaler, threshold, sequence_length


model, scaler, threshold, sequence_length = load_artifacts()

reading_buffer = deque(maxlen=sequence_length)


class SensorReading(BaseModel):
    timestamp: datetime
    water_level: float = Field(allow_inf_nan=False)
    rainfall: float = Field(ge=0, allow_inf_nan=False)
    temperature: float = Field(allow_inf_nan=False)
    humidity: float = Field(ge=0, le=100, allow_inf_nan=False)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)


@app.get("/")
def home():
    return {
        "message": "Flood Anomaly Detection API is running.",
        "model": "LSTM Autoencoder",
        "sequence_length": sequence_length,
        "threshold": threshold,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "buffered_readings": len(reading_buffer),
        "readings_required": sequence_length,
    }


@app.get("/dashboard-data")
def get_dashboard_data(
    limit: int = 20,
    current_user: dict = Depends(require_firebase_user),
):
    """Return recent sensor readings and detected anomalies."""

    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 500.",
        )

    try:
        documents = (
            db.collection("sensor_readings")
            .order_by("timestamp", direction="DESCENDING")
            .limit(limit)
            .stream()
        )

        readings = []

        for doc in documents:
            data = doc.to_dict() or {}

            timestamp = data.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()

            detection = data.get("detection")

            readings.append({
                "id": doc.id,
                "timestamp": timestamp,
                "water_level": data.get("water_level"),
                "rainfall": data.get("rainfall"),
                "temperature": data.get("temperature"),
                "humidity": data.get("humidity"),
                "data_source": data.get("data_source"),
                "detection": detection,
            })

        # Chronological order for dashboard charts.
        readings.reverse()

        anomaly_events = [
            reading
            for reading in readings
            if isinstance(reading.get("detection"), dict)
            and reading["detection"].get("anomaly_detected") is True
        ]

        latest_reading = readings[-1] if readings else None

        return {
            "readings": readings,
            "latest_reading": latest_reading,
            "anomaly_events": anomaly_events,
            "count": len(readings),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Dashboard data error: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()

        raise HTTPException(
            status_code=503,
            detail="Could not load dashboard data. Check the API terminal logs.",
        ) from exc


@app.post("/readings")
def receive_reading(reading: SensorReading):
    global last_email_alert_at

    values = reading.model_dump()
    feature_values = [values[name] for name in FEATURES]

    if not all(math.isfinite(value) for value in feature_values):
        raise HTTPException(
            status_code=422,
            detail="All sensor readings must be finite numbers.",
        )

    # Ensure readings arrive in chronological order.
    if reading_buffer and reading.timestamp <= reading_buffer[-1]["timestamp"]:
        raise HTTPException(
            status_code=400,
            detail="Timestamp must be later than the previous reading.",
        )

    # Save the reading to Firebase.
    try:
        sensor_doc_ref = save_sensor_reading({
            "timestamp": reading.timestamp.isoformat(),
            "water_level": values["water_level"],
            "rainfall": values["rainfall"],
            "temperature": values["temperature"],
            "humidity": values["humidity"],
            "data_source": "submitted_sensor_reading",
        })
    except Exception as exc:
        print(f"Firebase save error: {type(exc).__name__}: {exc}")
        traceback.print_exc()

        raise HTTPException(
            status_code=503,
            detail="Could not save the sensor reading to Firebase.",
        ) from exc

    # Add each successfully saved reading once.
    reading_buffer.append({
        "timestamp": reading.timestamp,
        "features": feature_values,
    })


    # Collect the required number of readings before running detection.
    if len(reading_buffer) < sequence_length:
        return {
            "status": "collecting",
            "message": f"Waiting for a complete {sequence_length}-reading sequence.",
            "readings_received": len(reading_buffer),
            "readings_required": sequence_length,
            "anomaly_detected": None,
            "firebase_saved": True,
        }

    # Prepare the sequence for the LSTM autoencoder.
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

    detection_result = {
        "status": (
            "anomaly_detected" if anomaly_detected else "normal"
        ),
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
        "firebase_saved": True,
    }

    # Save the detection result to the latest reading's document.
    try:
        sensor_doc_ref.update({
            "detection": detection_result
        })
        detection_result["firebase_detection_saved"] = True

    except Exception as exc:
        print(
            f"Firebase detection update error: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        detection_result["firebase_detection_saved"] = False
        
    # Send an email if an anomaly is detected and the cooldown has expired.

    notification_sent = False

    if anomaly_detected:
        now = datetime.now(timezone.utc)

        cooldown_expired = (
            last_email_alert_at is None
            or now - last_email_alert_at
            >= timedelta(minutes=EMAIL_ALERT_COOLDOWN)
        )

        if cooldown_expired:
            notification_sent = send_anomaly_email(
                timestamp=reading.timestamp.isoformat(),
                water_level=values["water_level"],
                rainfall=values["rainfall"],
                reconstruction_error=round(reconstruction_error, 6),
                threshold=round(threshold, 6),
            )

            if notification_sent:
                last_email_alert_at = now

    detection_result["notification_sent"] = notification_sent

    # Save detection and notification results to Firebase.
    try:
        sensor_doc_ref.update({
            "detection": detection_result
        })
        detection_result["firebase_detection_saved"] = True
    
    except Exception as exc:
        print(
            f"Firebase detection update error: {type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        detection_result["firebase_detection_saved"] = False

    return detection_result
