"""
alert.py
--------
SOS alert module. Called automatically when a fall is detected.
Fill in your email credentials before running realtime_detect.py
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── Fill these in ─────────────────────────────────────────────────────────────
EMAIL_SENDER   = "sijomariyam@gmail.com"
EMAIL_PASSWORD = ""   # Google App Password, NOT login password
EMAIL_RECEIVER = "sijajoseph11c@gmail.com"

# Twilio SMS (optional — free trial at twilio.com)
TWILIO_ENABLED = False
TWILIO_SID     = "YOUR_SID"
TWILIO_TOKEN   = "YOUR_TOKEN"
TWILIO_FROM    = "+1XXXXXXXXXX"
TWILIO_TO      = "+91XXXXXXXXXX"
# ─────────────────────────────────────────────────────────────────────────────


def send_email(confidence: float):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEMultipart()
    msg['From']    = EMAIL_SENDER
    msg['To']      = EMAIL_RECEIVER
    msg['Subject'] = "FALL DETECTED — Immediate Attention Required"
    body = f"""FALL DETECTION ALERT
═══════════════════════════
Time       : {ts}
Confidence : {confidence*100:.1f}%
Action     : Please check on the person immediately.

This is an automated message from your Fall Detection System.
"""
    msg.attach(MIMEText(body, 'plain'))
    try:
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(EMAIL_SENDER, EMAIL_PASSWORD)
        s.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        s.quit()
        print(f"✅ Email sent to {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"❌ Email error: {e}")
        print("   Tip: Enable 'App Passwords' in your Google Account settings.")


def send_sms(confidence: float):
    if not TWILIO_ENABLED:
        return
    try:
        from twilio.rest import Client
        ts = datetime.now().strftime("%H:%M:%S")
        Client(TWILIO_SID, TWILIO_TOKEN).messages.create(
            body=f"FALL DETECTED at {ts}! Confidence: {confidence*100:.0f}%. Check immediately.",
            from_=TWILIO_FROM, to=TWILIO_TO
        )
        print("✅ SMS sent.")
    except Exception as e:
        print(f"❌ SMS error: {e}")


def send_alert(confidence: float):
    """Call this function whenever a fall is confirmed."""
    print(f"\n🚨 FALL CONFIRMED ({confidence*100:.1f}%) — Triggering alerts ...")
    send_email(confidence)
    send_sms(confidence)
