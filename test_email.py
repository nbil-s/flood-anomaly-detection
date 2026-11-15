import os
import smtplib
import ssl
from email.message import EmailMessage

sender = os.getenv("FLOOD_EMAIL")
password = os.getenv("FLOOD_EMAIL_PASSWORD")
recipient = os.getenv("FLOOD_ALERT_RECIPIENT")

if not all([sender, password, recipient]):
    raise RuntimeError("Email environment variables are missing.")

message = EmailMessage()
message["Subject"] = "Flood Detection System - Email Test"
message["From"] = sender
message["To"] = recipient
message.set_content(
    "Email test successful. The flood anomaly detection system "
    "can send notifications through Gmail."
)

try:
    print("Connecting to Gmail...")

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()

        print("TLS established. Authenticating...")

        server.login(sender, password)

        print("Authentication successful. Sending email...")

        server.send_message(message)

        print("SUCCESS: Email sent.")

except Exception as error:
    print("FAILED:", type(error).__name__, str(error))