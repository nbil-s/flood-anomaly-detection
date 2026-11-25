import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from notifications import send_anomaly_email

recipient = os.getenv("FLOOD_ALERT_RECIPIENT")

if not recipient:
    raise RuntimeError("FLOOD_ALERT_RECIPIENT is not configured.")

result = send_anomaly_email(
    timestamp="2026-11-25T19:50:00+03:00",
    water_level=72.5,
    rainfall=85.0,
    temperature=24.6,
    humidity=81.3,
    reconstruction_error=4.10,
    threshold=1.655422,
    recipients=[recipient],
)

print("Test notification result:", result)