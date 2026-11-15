
import requests
from datetime import datetime, timedelta, timezone

API_URL = "http://127.0.0.1:8000/readings"

# Twelve simulated readings with values similar to the training data.
start_time = datetime.now(timezone.utc)

readings = []

for i in range(12):
    readings.append({
        "timestamp": (start_time + timedelta(minutes=5 * i))
            .isoformat(),
        "water_level": 31.0 + (i * 0.02),
        "rainfall": 0.1,
        "temperature": 25.0,
        "humidity": 68.0
    })

print("Testing the API with 12 simulated readings...\n")

for i, reading in enumerate(readings, start=1):
    try:
        response = requests.post(
            API_URL,
            json=reading,
            timeout=10
        )

        print(f"Reading {i}: HTTP {response.status_code}")

        if not response.ok:
            print(response.text)
            break

        result = response.json()
        print(f"Status: {result.get('status')}")

        if i == 12:
            print("\nFinal detection result:")
            print(f"Reconstruction error: "
                  f"{result.get('reconstruction_error')}")
            print(f"Threshold: {result.get('threshold')}")
            print(f"Anomaly detected: "
                  f"{result.get('anomaly_detected')}")

    except requests.RequestException as error:
        print(f"Request failed: {error}")
        break
