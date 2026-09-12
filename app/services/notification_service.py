"""Out-of-band alert notifications (email / Slack / generic webhook).

Implemented with the Python standard library only (smtplib + urllib) to avoid
adding dependencies. Delivery is best-effort and fire-and-forget: a failure to
reach any channel is logged and never affects the prediction path.

Mirrors the legacy ML-IDS notification capability (email, Slack, webhook) but
uses environment-based config instead of a DB table, keeping the prototype
lightweight. Gated globally by `ALERT_NOTIFICATION_ENABLED` (default off).
"""

import asyncio
import logging
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP, SMTP_SSL
from typing import Dict, Any

from ..core.config import Settings, settings

logger = logging.getLogger(__name__)

_SLACK_COLORS = {
    "low": "#36a64f",
    "medium": "#ffc107",
    "high": "#ff9800",
    "critical": "#dc3545",
}


class NotificationService:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return self.cfg.ALERT_NOTIFICATION_ENABLED

    # ------------------------------------------------------------ dispatch
    def notify_alert(self, alert: Dict[str, Any]) -> Dict[str, bool]:
        """Best-effort dispatch of a single alert to all configured channels.

        Returns {channel_name: success_bool}. Safe no-op when disabled or the
        alert only carries the id.
        """
        results: Dict[str, bool] = {}
        if not self.enabled or not alert:
            return results

        severity = str(alert.get("severity", "medium")).lower()
        color = _SLACK_COLORS.get(severity, "#6c757d")
        ts = alert.get("timestamp")
        title = f"ThreatLens AI Alert [{alert.get('id')}] - {alert.get('attack_type', 'unknown')}"
        fallback = (
            f"ThreatLens detected {alert.get('attack_type', 'an attack')} "
            f"from {alert.get('src_ip', 'unknown')} - severity {severity}"
        )

        if self.cfg.SLACK_WEBHOOK_URL.startswith("http"):
            ok = self._send_slack(self.cfg.SLACK_WEBHOOK_URL, title, fallback, color, alert)
            results["slack"] = ok
        if self.cfg.SMTP_HOST and self.cfg.SMTP_USER:
            ok = self._send_email(title, fallback, str(ts or ""))
            results["email"] = ok
        if self.cfg.NOTIFICATION_WEBHOOK_URL.startswith("http"):
            ok = self._send_webhook(self.cfg.NOTIFICATION_WEBHOOK_URL, alert)
            results["webhook"] = ok

        for ch, ok in results.items():
            logger.info("Notification to %s: %s", ch, "ok" if ok else "failed")
        return results

    # ------------------------------------------------------------ channels
    def _send_slack(self, webhook_url: str, title: str, fallback: str, color: str, alert: Dict[str, Any]) -> bool:
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "fallback": fallback,
                    "fields": [
                        {"title": "Attack Type", "value": str(alert.get("attack_type", "?")), "short": True},
                        {"title": "Severity", "value": severity_label(alert.get("severity", "medium")), "short": True},
                        {"title": "Source IP", "value": str(alert.get("src_ip", "?")), "short": True},
                        {"title": "Destination IP", "value": str(alert.get("dst_ip", "?")), "short": True},
                        {"title": "Confidence", "value": f"{float(alert.get('confidence') or 0):.1%}", "short": True},
                        {"title": "Timestamp", "value": str(alert.get("timestamp") or ""), "short": True},
                    ],
                }
            ]
        }
        return self._post_json(webhook_url, payload)

    def _send_email(self, subject: str, body_text: str, ref: str) -> bool:
        cfg = self.cfg
        if not cfg.SMTP_HOST or not cfg.SMTP_USER:
            return False
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = cfg.SMTP_FROM or cfg.SMTP_USER
        recipients = cfg.SMTP_TO or [cfg.SMTP_USER]
        msg["To"] = ", ".join(recipients)
        html = (
            f"<html><body><h3>{subject}</h3><p>{body_text}</p>"
            f"<p><small>ThreatLens AI prototype reference: {ref}</small></p></body></html>"
        )
        msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(html, "html"))
        try:
            if cfg.SMTP_USE_SSL:
                server = SMTP_SSL(cfg.SMTP_HOST, int(cfg.SMTP_PORT or 465), timeout=cfg.SMTP_TIMEOUT)
            else:
                server = SMTP(cfg.SMTP_HOST, int(cfg.SMTP_PORT or 587), timeout=cfg.SMTP_TIMEOUT)
            try:
                if cfg.SMTP_STARTTLS:
                    server.starttls(context=ssl.create_default_context())
                if cfg.SMTP_USER:
                    server.login(cfg.SMTP_USER, cfg.SMTP_PASSWORD or "")
                server.sendmail(cfg.SMTP_FROM or cfg.SMTP_USER, recipients, msg.as_string())
            finally:
                server.quit()
            return True
        except Exception as exc:
            logger.warning("Email notification failed: %s", exc)
            return False

    def _send_webhook(self, url: str, alert: Dict[str, Any]) -> bool:
        return self._post_json(url, alert)

    # ------------------------------------------------------------ transport
    @staticmethod
    def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str] | None = None) -> bool:
        import json
        import urllib.request

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **(headers or {})},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                resp.read()
            return resp.status < 400
        except Exception as exc:
            logger.warning("Webhook/Slack notification failed: %s", exc)
            return False

    # ------------------------------------------------------------ async glue
    async def notify_alert_async(self, alert: Dict[str, Any]) -> Dict[str, bool]:
        """Fire-and-forget wrapper callable from async handlers."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self.notify_alert, alert)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Async notification failed: %s", exc)
            return {}


def severity_label(value: Any) -> str:
    return str(value).upper() if value else "UNKNOWN"


# Global singleton
notification_service = NotificationService()