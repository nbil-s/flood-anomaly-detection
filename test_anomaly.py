import requests
from datetime import datetime, timedelta, timezone

API_URL = "http://127.0.0.1:8000/readings"

water_levels = [
    30.5, 30.6, 30.5, 30.7, 30.6, 30.7,
    32.0, 33.0, 34.0, 35.0, 36.0, 37.0
]

start_time = datetime.now(timezone.utc)

print("Testing unusual sensor readings...\n")

for i, level in enumerate(water_levels):
    reading = {
        "timestamp": (
            start_time + timedelta(minutes=5 * i)
        ).isoformat(),
        "water_level": level,
        "rainfall": 0.1,
        "temperature": 25.0,
        "humidity": 68.0
    }

    try:
        response = requests.post(
            API_URL, json=reading, timeout=10
        )

        print(f"Reading {i + 1}: HTTP {response.status_code}")

        if not response.ok:
            print(response.text)
            break

        result = response.json()

        if i == 11:
            print("\nFinal result:")
            print("Reconstruction error:",
                  result.get("reconstruction_error"))
            print("Threshold:", result.get("threshold"))
            print("Anomaly detected:",
                  result.get("anomaly_detected"))
        else:
            print("Status:", result.get("status"))

    except requests.RequestException as error:
        print("Request failed:", error)
        break
