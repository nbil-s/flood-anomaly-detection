import smtplib
import ssl

print("\nTesting Gmail STARTTLS on port 587...")

try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.set_debuglevel(1)
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()

    print("Port 587: SMTP and TLS handshake successful.")

except Exception as error:
    print("Port 587 failed:", type(error).__name__, str(error))


print("\nTesting Gmail SSL on port 465...")

try:
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        context=context,
        timeout=20
    ) as server:
        server.set_debuglevel(1)
        server.ehlo()

    print("Port 465: SMTP and SSL handshake successful.")

except Exception as error:
    print("Port 465 failed:", type(error).__name__, str(error))