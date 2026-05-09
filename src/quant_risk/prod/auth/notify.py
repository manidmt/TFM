'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-04-25

@description: Fire-and-forget email notification for admin on new user signup.

Configured via environment variables (all optional — silent no-op when absent):
    QUANT_RISK_NOTIFY_EMAIL   — recipient address (admin inbox)
    QUANT_RISK_SMTP_HOST      — SMTP server hostname
    QUANT_RISK_SMTP_PORT      — SMTP port (default 587)
    QUANT_RISK_SMTP_USER      — SMTP login username
    QUANT_RISK_SMTP_PASSWORD  — SMTP login password
'''

from __future__ import annotations

import logging
import smtplib
import threading
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def _send(notify_email: str, smtp_host: str, smtp_port: int,
          smtp_user: str | None, smtp_password: str | None,
          signup_email: str) -> None:
    try:
        msg = MIMEText(
            f"New signup request from: {signup_email}\n\n"
            f"Go to /ops/users to approve or reject.",
            "plain",
            "utf-8",
        )
        msg["Subject"] = f"[quant-risk] New signup: {signup_email}"
        msg["From"] = smtp_user or notify_email
        msg["To"] = notify_email

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(msg["From"], [notify_email], msg.as_string())

        logger.info("Admin signup notification sent to %s for %s", notify_email, signup_email)
    except Exception as exc:
        logger.warning("Failed to send signup notification email: %s", exc)


def notify_admin_signup(signup_email: str, config: object) -> None:
    """Send a notification email to the admin when a new user signs up.

    Silent no-op if SMTP / notify env vars are not configured.
    Runs in a daemon thread so it never blocks the HTTP response.
    """
    notify_email: str | None = getattr(config, "notify_email", None)
    smtp_host: str | None = getattr(config, "smtp_host", None)

    if not notify_email or not smtp_host:
        return

    smtp_port: int = getattr(config, "smtp_port", 587)
    smtp_user: str | None = getattr(config, "smtp_user", None)
    smtp_password: str | None = getattr(config, "smtp_password", None)

    t = threading.Thread(
        target=_send,
        args=(notify_email, smtp_host, smtp_port, smtp_user, smtp_password, signup_email),
        daemon=True,
    )
    t.start()
