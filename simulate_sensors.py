
import argparse
import time
from datetime import datetime, timedelta, timezone

import requests

API_URL = "http://127.0.0.1:8000/readings"

# Each scenario contains 12 readings: one complete model window.
SCENARIOS = {
    "normal": {
        "water_level": [
            30.5, 30.6, 30.5, 30.7, 30.6, 30.7,
            30.6, 30.5, 30.7, 30.6, 30.5, 30.6,
        ],
        "rainfall": [0.1] * 12,
    },
    "rising_water": {
        "water_level": [
            30.5, 30.6, 30.5, 30.7, 30.6, 30.7,
            32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
        ],
        "rainfall": [0.1] * 12,
    },
    "heavy_rain": {
        "water_level": [
            30.5, 30.5, 30.6, 30.6, 30.7, 30.8,
            30.9, 31.0, 31.2, 31.4, 31.6, 31.8,
        ],
        "rainfall": [
            0.1, 0.2, 0.2, 0.5, 1.0, 2.0,
            4.0, 6.0, 8.0, 10.0, 12.0, 15.0,
        ],
    },
    "sudden_spike": {
        "water_level": [
            30.5, 30.6, 30.5, 30.7, 30.6, 30.7,
            30.6, 30.5, 30.7, 30.6, 30.5, 45.0,
        ],
        "rainfall": [0.1] * 12,
    },
}


def build_reading(scenario, index, start_time):
    water_level = scenario["water_level"][index]
    rainfall = scenario["rainfall"][index]

    # Timestamps remain five minutes apart even in accelerated mode.
    timestamp = start_time + timedelta(minutes=5 * index)

    return {
        "timestamp": timestamp.isoformat(),
        "water_level": water_level,
        "rainfall": rainfall,
        "temperature": 25.0,
        "humidity": 68.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simulate environmental sensors for flood detection."
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS.keys(),
        default="rising_water",
        help="Sensor scenario to simulate.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Seconds to wait between readings; 0 means accelerated mode.",
    )
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay cannot be negative.")

    # Check API before submitting readings.
    try:
        health = requests.get(
            "http://127.0.0.1:8000/health", timeout=5
        )
        health.raise_for_status()
        print("API health:", health.json())
    except requests.RequestException as error:
        print("Cannot reach FastAPI. Start the API first.")
        print(error)
        return

    scenario = SCENARIOS[args.scenario]
    start_time = datetime.now(timezone.utc)

    print(f"\nScenario: {args.scenario}")
    print("Sending 12 readings to the LSTM autoencoder...\n")

    for index in range(12):
        reading = build_reading(scenario, index, start_time)

        try:
            response = requests.post(
                API_URL, json=reading, timeout=10
            )

            print(
                f"Reading {index + 1}/12 | "
                f"Water level: {reading['water_level']:.1f} | "
                f"Rainfall: {reading['rainfall']:.1f} | "
                f"HTTP {response.status_code}"
            )

            if not response.ok:
                print("API error:", response.text)
                print(
                    "If the timestamp is rejected, restart the API "
                    "to clear its in-memory buffer."
                )
                return

            result = response.json()

            if index == 11:
                print("\nDetection result")
                print("----------------")
                print("Status:", result.get("status"))
                print(
                    "Reconstruction error:",
                    result.get("reconstruction_error"),
                )
                print("Threshold:", result.get("threshold"))
                print(
                    "Anomaly detected:",
                    result.get("anomaly_detected"),
                )
                print("Interpretation:", result.get("interpretation"))
            else:
                print("Status:", result.get("status"))

        except requests.RequestException as error:
            print("Request failed:", error)
            return

        if args.delay > 0 and index < 11:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
