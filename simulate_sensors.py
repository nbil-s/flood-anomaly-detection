
import argparse
import time
from datetime import datetime, timedelta, timezone

import requests

API_URL = "http://127.0.0.1:8000/readings"
SAMPLING_INTERVAL_SECONDS = 30
SEQUENCE_LENGTH = 24

# Each scenario contains exactly 24 readings.
SCENARIOS = {
    "normal": {
        "water_level": [
            30.5, 30.6, 30.5, 30.7, 30.6, 30.7,
            30.6, 30.5, 30.7, 30.6, 30.5, 30.6,
            30.5, 30.6, 30.7, 30.6, 30.5, 30.7,
            30.6, 30.5, 30.6, 30.7, 30.6, 30.5,
        ],
        "rainfall": [0.0] * 24,
    },

    "rising_water": {
        "water_level": [
            30.5, 30.6, 30.5, 30.7, 30.6, 30.7,
            30.8, 31.0, 31.3, 31.6, 32.0, 32.5,
            33.0, 33.5, 34.0, 34.5, 35.0, 35.5,
            36.0, 36.5, 37.0, 37.5, 38.0, 39.0,
        ],
        "rainfall": [0.1] * 24,
    },

    "heavy_rain": {
        "water_level": [
            30.5, 30.5, 30.6, 30.6, 30.7, 30.8,
            30.9, 31.0, 31.2, 31.4, 31.6, 31.8,
            32.0, 32.3, 32.6, 33.0, 33.4, 33.8,
            34.2, 34.6, 35.0, 35.5, 36.0, 36.5,
        ],
        "rainfall": [
            0.1, 0.2, 0.2, 0.5, 1.0, 2.0,
            3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 15.0,
            14.0, 13.0, 12.0, 10.0, 8.0, 6.0,
        ],
    },

    "sudden_spike": {
        "water_level": [
            30.5, 30.6, 30.5, 30.7, 30.6, 30.7,
            30.6, 30.5, 30.7, 30.6, 30.5, 30.6,
            30.5, 30.6, 30.7, 30.6, 30.5, 30.7,
            30.6, 30.5, 30.6, 30.7, 30.6, 45.0,
        ],
        "rainfall": [0.1] * 24,
    },
}


def build_reading(scenario, index, start_time):
    timestamp = start_time + timedelta(
        seconds=SAMPLING_INTERVAL_SECONDS * index
    )

    return {
        "timestamp": timestamp.isoformat(),
        "water_level": scenario["water_level"][index],
        "rainfall": scenario["rainfall"][index],
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

    if args.delay > 0 and args.delay < SAMPLING_INTERVAL_SECONDS:
        parser.error(
            "For real-time simulation, --delay must be at least 30 seconds."
        )

    # Check API availability.
    try:
        health = requests.get(
            "http://127.0.0.1:8000/health",
            timeout=5,
        )
        health.raise_for_status()

        health_data = health.json()
        print("API health:", health_data)

        if health_data.get("readings_required") != SEQUENCE_LENGTH:
            print(
                f"ERROR: API requires "
                f"{health_data.get('readings_required')} readings, "
                f"but simulator is configured for {SEQUENCE_LENGTH}."
            )
            print("Check that the API is using the correct model.")
            return

    except requests.RequestException as error:
        print("Cannot reach FastAPI. Start the API first.")
        print(error)
        return

    scenario = SCENARIOS[args.scenario]
    start_time = datetime.now(timezone.utc)

    print(f"\nScenario: {args.scenario}")
    print(f"Sending {SEQUENCE_LENGTH} readings...")
    print(f"Sampling interval: {SAMPLING_INTERVAL_SECONDS} seconds")
    print(f"Delay between requests: {args.delay} seconds\n")

    for index in range(SEQUENCE_LENGTH):
        reading = build_reading(scenario, index, start_time)

        try:
            response = requests.post(
                API_URL,
                json=reading,
                timeout=35,
            )

            print(
                f"Reading {index + 1}/{SEQUENCE_LENGTH} | "
                f"Water level: {reading['water_level']:.1f} | "
                f"Rainfall: {reading['rainfall']:.1f} | "
                f"HTTP {response.status_code}"
            )

            if not response.ok:
                print("API error:", response.text)
                return

            result = response.json()

            if index == SEQUENCE_LENGTH - 1:
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

        if args.delay > 0 and index < SEQUENCE_LENGTH - 1:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
