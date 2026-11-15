
import os
import smtplib
from email.message import EmailMessage


def send_anomaly_email(
    timestamp: str,
    water_level: float,
    rainfall: float,
    reconstruction_error: float,
    threshold: float,
):
    sender = os.getenv("FLOOD_EMAIL")
    password = os.getenv("FLOOD_EMAIL_PASSWORD")
    recipient = os.getenv("FLOOD_ALERT_RECIPIENT")

    if not sender or not password or not recipient:
        print("Email notification skipped: email configuration is incomplete.")
        return False

    message = EmailMessage()
    message["Subject"] = "Flood Monitoring Alert: Unusual Pattern Detected"
    message["From"] = sender
    message["To"] = recipient

    message.set_content(f"""
A potentially unusual environmental pattern has been detected.

Timestamp: {timestamp}
Water level reading: {water_level}
Rainfall reading: {rainfall}

LSTM reconstruction error: {reconstruction_error}
Detection threshold: {threshold}

Please inspect the sensor readings and verify the conditions.

This alert indicates an anomaly, not confirmation of flooding.
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(sender, password)
            server.send_message(message)

        print(f"Anomaly notification sent to {recipient}")
        return True

    except Exception as exc:
        print(f"Email notification failed: {type(exc).__name__}: {exc}")
        return False
