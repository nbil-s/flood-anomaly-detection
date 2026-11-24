
import argparse
import time
from datetime import timedelta

import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/readings"
HEALTH_URL = "http://127.0.0.1:8000/health"

DATASET_PATH = (
    "experiments/improved_simulation_240/data/raw/"
    "simulated_environmental_data.csv"
)

SEQUENCE_LENGTH = 240
SAMPLING_INTERVAL_SECONDS = 60

FEATURES = [
    "water_level",
    "rainfall",
    "temperature",
    "humidity",
]


def load_baseline():
    data = pd.read_csv(DATASET_PATH)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data = data.sort_values("timestamp").reset_index(drop=True)

    if len(data) < SEQUENCE_LENGTH:
        raise ValueError("Dataset contains fewer than 240 observations.")

    if data[FEATURES].isna().any().any():
        raise ValueError("Dataset contains missing sensor values.")

    # Select a consecutive 240-reading window.
    start = 1000
    readings = data.iloc[start:start + SEQUENCE_LENGTH].copy()

    if len(readings) != SEQUENCE_LENGTH:
        raise ValueError("Could not obtain 240 consecutive readings.")

    intervals = readings["timestamp"].diff().dropna()

    if not (intervals.dt.total_seconds() == 60).all():
        raise ValueError("Dataset readings are not exactly one minute apart.")

    return readings.reset_index(drop=True)


def apply_scenario(readings, scenario):
    readings = readings.copy()

    if scenario == "normal":
        return readings

    if scenario == "rising_water":
        # Gradually increase water level during the final 60 readings.
        start = 180
        baseline = readings.loc[start, "water_level"]

        for i in range(start, SEQUENCE_LENGTH):
            progress = (i - start + 1) / 60
            readings.loc[i, "water_level"] = (
                baseline + 8.0 * progress
            )

            readings.loc[i, "rainfall"] = max(
                readings.loc[i, "rainfall"],
                0.15,
            )

    elif scenario == "heavy_rain":
        for i in range(180, SEQUENCE_LENGTH):
            progress = (i - 179) / 60

            readings.loc[i, "rainfall"] = min(
                0.15 + 1.5 * progress,
                1.65,
            )

            readings.loc[i, "water_level"] += 4.0 * progress

            readings.loc[i, "humidity"] = min(
                readings.loc[i, "humidity"] + 3.0 * progress,
                100.0,
            )

    elif scenario == "sudden_spike":
        # Introduce one abrupt water-level increase.
        readings.loc[SEQUENCE_LENGTH - 1, "water_level"] += 12.0

    return readings


def main():
    parser = argparse.ArgumentParser(
        description="Simulate 240 environmental sensor readings."
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "normal",
            "rising_water",
            "heavy_rain",
            "sudden_spike",
        ],
        default="normal",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Delay between requests in seconds. Default: accelerated mode.",
    )

    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay cannot be negative.")

    if 0 < args.delay < SAMPLING_INTERVAL_SECONDS:
        parser.error("--delay must be at least 60 seconds.")

    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        health_response.raise_for_status()

        health = health_response.json()
        print("API health:", health)

        if health.get("status") != "ok":
            raise RuntimeError("API is not healthy.")

        if health.get("readings_required") != SEQUENCE_LENGTH:
            raise RuntimeError(
                "API and simulator sequence lengths do not match."
            )

        if health.get("buffered_readings") != 0:
            raise RuntimeError(
                "API buffer is not empty. Restart the API before "
                "starting a new scenario."
            )

        readings = load_baseline()
        readings = apply_scenario(readings, args.scenario)

    except (requests.RequestException, ValueError, RuntimeError) as error:
        print("Setup failed:", error)
        return

    print(f"\nScenario: {args.scenario}")
    print(f"Readings: {SEQUENCE_LENGTH}")
    print(f"Sampling interval: {SAMPLING_INTERVAL_SECONDS} seconds")
    print(f"Request delay: {args.delay} seconds")
    print("\nSending readings...\n")
    
    simulation_start = (
    pd.Timestamp.now(tz="UTC")
    - pd.Timedelta(
        seconds=(SEQUENCE_LENGTH - 1) * SAMPLING_INTERVAL_SECONDS
    )
)

    for index, row in readings.iterrows():
        reading_timestamp = (
            simulation_start
            + pd.Timedelta(
                seconds=index * SAMPLING_INTERVAL_SECONDS
            )
        )

        reading = {
            "timestamp": reading_timestamp.isoformat(),
            "water_level": float(row["water_level"]),
            "rainfall": float(row["rainfall"]),
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
        }

        try:
            response = requests.post(
                API_URL,
                json=reading,
                timeout=35,
            )

            if not response.ok:
                print(f"Request failed at reading {index + 1}")
                print(response.status_code, response.text)
                return

            result = response.json()

            print(
                f"Reading {index + 1}/{SEQUENCE_LENGTH} | "
                f"Water: {reading['water_level']:.2f} | "
                f"Rain: {reading['rainfall']:.3f} | "
                f"Status: {result.get('status')}"
            )

            if index == SEQUENCE_LENGTH - 1:
                print("\nFinal detection result")
                print("----------------------")

                for key in [
                    "status",
                    "anomaly_score",
                    "reconstruction_error",
                    "threshold",
                    "triggering_feature",
                    "anomaly_detected",
                    "interpretation",
                    "notification_sent",
                    "firebase_detection_saved",
                ]:
                    if key in result:
                        print(f"{key}: {result[key]}")

        except requests.RequestException as error:
            print("Request failed:", error)
            return

        if args.delay > 0 and index < SEQUENCE_LENGTH - 1:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
