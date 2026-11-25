
import os
import smtplib
from email.message import EmailMessage


def send_anomaly_email(
    timestamp: str,
    water_level: float,
    rainfall: float,
    reconstruction_error: float,
    threshold: float,
    recipients: list[str],
):
    sender = os.getenv("FLOOD_EMAIL")
    password = os.getenv("FLOOD_EMAIL_PASSWORD")

    if not sender or not password:
        print("Email notification skipped: sender configuration is incomplete.")
        return False

    recipients = list(dict.fromkeys(
        email.strip()
        for email in recipients
        if isinstance(email, str) and email.strip()
    ))

    if not recipients:
        print("Email notification skipped: no subscribed recipients.")
        return False

    email_content = f"""
A potentially unusual environmental pattern has been detected.

Timestamp: {timestamp}
Water level reading: {water_level}
Rainfall reading: {rainfall}

LSTM reconstruction error: {reconstruction_error}
Detection threshold: {threshold}

Please inspect the sensor readings and verify the conditions.

This alert indicates an anomaly, not confirmation of flooding.
"""

    successful_sends = 0

    try:
        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, timeout=20
        ) as server:
            server.login(sender, password)

            for recipient in recipients:
                message = EmailMessage()
                message["Subject"] = (
                    "Flood Monitoring Alert: Unusual Pattern Detected"
                )
                message["From"] = sender
                message["To"] = recipient
                message.set_content(email_content)

                try:
                    server.send_message(message)
                    successful_sends += 1
                    print(f"Anomaly notification sent to {recipient}")

                except Exception as exc:
                    print(
                        f"Email failed for {recipient}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    except Exception as exc:
        print(f"Email notification service failed: {type(exc).__name__}: {exc}")
        return False

    print(
        f"Notifications delivered: {successful_sends}/{len(recipients)}"
    )

    return successful_sends > 0
