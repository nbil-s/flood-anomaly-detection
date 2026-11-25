import os
import smtplib
from html import escape
from email.message import EmailMessage
from datetime import datetime
from zoneinfo import ZoneInfo


DASHBOARD_URL = "https://flood-anomaly-dashboard.onrender.com"


def format_timestamp(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Convert timezone-aware timestamps to East Africa Time.
        if dt.tzinfo is not None:
            dt = dt.astimezone(ZoneInfo("Africa/Nairobi"))

        return dt.strftime("%d %B %Y, %I:%M %p EAT")

    except (ValueError, TypeError):
        return str(timestamp)


def send_anomaly_email(
    timestamp: str,
    water_level: float,
    rainfall: float,
    reconstruction_error: float,
    threshold: float,
    recipients: list[str],
    temperature: float | None = None,
    humidity: float | None = None,
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

    # Calculate anomaly ratio.
    ratio = reconstruction_error / threshold if threshold > 0 else 0

    # These labels describe the anomaly magnitude, not actual flood severity.
    if ratio >= 5:
        anomaly_level = "HIGH"
        severity_color = "#b91c1c"
    elif ratio >= 2:
        anomaly_level = "ELEVATED"
        severity_color = "#ea580c"
    else:
        anomaly_level = "WARNING"
        severity_color = "#ca8a04"

    # Format sensor values.
    temperature_display = (
        f"{temperature:.1f} °C"
        if temperature is not None
        else "Unavailable"
    )

    humidity_display = (
        f"{humidity:.1f} %"
        if humidity is not None
        else "Unavailable"
    )

    # Escape values before inserting them into HTML.
    timestamp_safe = escape(format_timestamp(timestamp))
    water_level_safe = escape(str(water_level))
    rainfall_safe = escape(str(rainfall))
    temperature_safe = escape(temperature_display)
    humidity_safe = escape(humidity_display)
    error_safe = escape(f"{reconstruction_error:.6f}")
    threshold_safe = escape(f"{threshold:.6f}")
    ratio_safe = escape(f"{ratio:.2f}x")
    anomaly_level_safe = escape(anomaly_level)

    subject = f"[{anomaly_level}] Flood Anomaly Detection Alert"

    # Plain-text email fallback.
    email_content = f"""
FLOOD ANOMALY DETECTED
Flood Monitoring & Early Warning System

Anomaly level: {anomaly_level}

An unusual environmental pattern was detected by the
LSTM Autoencoder monitoring system.

DETECTION DETAILS
Timestamp: {format_timestamp(timestamp)}

SENSOR READINGS
Water level: {water_level}
Rainfall: {rainfall}
Temperature: {temperature_display}
Humidity: {humidity_display}

MODEL ANALYSIS
Reconstruction error: {reconstruction_error:.6f}
Detection threshold: {threshold:.6f}
Error/threshold ratio: {ratio:.2f}x

The reconstruction error exceeded the configured
detection threshold, indicating an unusual pattern.

The anomaly level is based on the reconstruction error
relative to the detection threshold. It is not a flood-risk
classification.

An anomaly alert does not confirm flooding.
Please verify the environmental conditions.

Live monitoring dashboard:
{DASHBOARD_URL}

Flood Anomaly Detection System
Automated monitoring notification.
"""

    # HTML email.
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flood Anomaly Alert</title>
</head>

<body style="
    margin: 0;
    padding: 20px 8px;
    background-color: #f1f5f9;
    font-family: Arial, Helvetica, sans-serif;
    color: #1e293b;
">

<div style="
    max-width: 620px;
    margin: 0 auto;
    background-color: #ffffff;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
">

    <!-- Header -->
    <div style="
        background-color: #991b1b;
        padding: 28px 24px;
        text-align: center;
        color: #ffffff;
    ">

        <div style="
            font-size: 13px;
            letter-spacing: 2px;
        ">
            FLOOD MONITORING SYSTEM
        </div>

        <h1 style="
            margin: 14px 0 8px;
            font-size: 25px;
        ">
            ANOMALY DETECTED
        </h1>

        <p style="
            margin: 0;
            font-size: 14px;
            color: #fecaca;
        ">
            Real-Time Environmental Monitoring Alert
        </p>

    </div>

    <!-- Main content -->
    <div style="padding: 24px;">

        <!-- Alert status -->
        <div style="
            background-color: #fff7ed;
            border-left: 5px solid {severity_color};
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 22px;
        ">

            <div style="
                color: {severity_color};
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 1px;
            ">
                {anomaly_level_safe} ANOMALY LEVEL
            </div>

            <p style="
                margin: 8px 0 0;
                font-size: 14px;
                line-height: 1.6;
                color: #334155;
            ">
                The system detected an unusual environmental
                pattern that exceeded the configured anomaly
                detection threshold.
            </p>

        </div>

        <!-- Timestamp -->
        <h2 style="
            font-size: 17px;
            margin-bottom: 8px;
        ">
            Detection Time
        </h2>

        <div style="
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 14px;
            border-radius: 6px;
            font-size: 14px;
        ">
            {timestamp_safe}
        </div>

        <!-- Sensor readings -->
        <h2 style="
            font-size: 17px;
            margin-top: 26px;
            margin-bottom: 12px;
        ">
            Sensor Readings
        </h2>

        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse: collapse;">

            <!-- First row -->
            <tr>

                <td width="50%" style="
                    padding: 14px;
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                ">

                    <div style="
                        font-size: 12px;
                        color: #64748b;
                    ">
                        WATER LEVEL
                    </div>

                    <div style="
                        font-size: 22px;
                        font-weight: bold;
                        margin-top: 6px;
                        color: #1e293b;
                    ">
                        {water_level_safe}
                    </div>

                </td>

                <td width="50%" style="
                    padding: 14px;
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                ">

                    <div style="
                        font-size: 12px;
                        color: #64748b;
                    ">
                        RAINFALL
                    </div>

                    <div style="
                        font-size: 22px;
                        font-weight: bold;
                        margin-top: 6px;
                        color: #1e293b;
                    ">
                        {rainfall_safe}
                    </div>

                </td>

            </tr>

            <!-- Second row -->
            <tr>

                <td width="50%" style="
                    padding: 14px;
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                ">

                    <div style="
                        font-size: 12px;
                        color: #64748b;
                    ">
                        TEMPERATURE
                    </div>

                    <div style="
                        font-size: 22px;
                        font-weight: bold;
                        margin-top: 6px;
                        color: #1e293b;
                    ">
                        {temperature_safe}
                    </div>

                </td>

                <td width="50%" style="
                    padding: 14px;
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                ">

                    <div style="
                        font-size: 12px;
                        color: #64748b;
                    ">
                        HUMIDITY
                    </div>

                    <div style="
                        font-size: 22px;
                        font-weight: bold;
                        margin-top: 6px;
                        color: #1e293b;
                    ">
                        {humidity_safe}
                    </div>

                </td>

            </tr>

        </table>

        <!-- Model analysis -->
        <h2 style="
            font-size: 17px;
            margin-top: 26px;
            margin-bottom: 12px;
        ">
            LSTM Autoencoder Analysis
        </h2>

        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse: collapse;">

            <tr>

                <td style="
                    padding: 12px;
                    border: 1px solid #e2e8f0;
                    font-size: 14px;
                ">
                    Reconstruction Error
                </td>

                <td style="
                    padding: 12px;
                    border: 1px solid #e2e8f0;
                    font-weight: bold;
                ">
                    {error_safe}
                </td>

            </tr>

            <tr>

                <td style="
                    padding: 12px;
                    border: 1px solid #e2e8f0;
                    font-size: 14px;
                ">
                    Detection Threshold
                </td>

                <td style="
                    padding: 12px;
                    border: 1px solid #e2e8f0;
                    font-weight: bold;
                ">
                    {threshold_safe}
                </td>

            </tr>

            <tr>

                <td style="
                    padding: 12px;
                    border: 1px solid #e2e8f0;
                    font-size: 14px;
                ">
                    Error / Threshold Ratio
                </td>

                <td style="
                    padding: 12px;
                    border: 1px solid #e2e8f0;
                    font-weight: bold;
                    color: {severity_color};
                ">
                    {ratio_safe}
                </td>

            </tr>

        </table>

        <!-- Explanation -->
        <p style="
            font-size: 13px;
            color: #64748b;
            line-height: 1.7;
            margin-top: 18px;
        ">
            The reconstruction error exceeded the model's
            configured threshold, indicating that the observed
            sensor pattern differed from learned normal patterns.
        </p>

        <!-- Dashboard button -->
        <div style="
            text-align: center;
            margin: 28px 0;
        ">

            <a href="{DASHBOARD_URL}"
               style="
                    display: inline-block;
                    background-color: #1d4ed8;
                    color: #ffffff;
                    text-decoration: none;
                    padding: 14px 24px;
                    border-radius: 7px;
                    font-weight: bold;
                    font-size: 14px;
               ">
                OPEN LIVE DASHBOARD
            </a>

        </div>

        <!-- Disclaimer -->
        <div style="
            background-color: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 6px;
            padding: 14px;
            font-size: 13px;
            color: #92400e;
            line-height: 1.6;
        ">

        </div>

    </div>

    <!-- Footer -->
    <div style="
        background-color: #f8fafc;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        padding: 20px;
        color: #64748b;
        font-size: 12px;
        line-height: 1.7;
    ">

        <strong>Flood Anomaly Detection System</strong><br>
        IoT-Based Environmental Monitoring<br>
        Automated notification generated by the monitoring system.

    </div>

</div>

</body>
</html>
"""

    successful_sends = 0

    try:
        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            timeout=20
        ) as server:

            server.login(sender, password)

            for recipient in recipients:

                message = EmailMessage()

                message["Subject"] = subject
                message["From"] = sender
                message["To"] = recipient

                # Plain-text fallback
                message.set_content(email_content)

                # HTML version
                message.add_alternative(
                    html_content,
                    subtype="html"
                )

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
        print(
            f"Email notification service failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    print(
        f"Notifications delivered: "
        f"{successful_sends}/{len(recipients)}"
    )

    return successful_sends > 0