"""Delivery channels. Default is DRY RUN: messages are recorded, never sent.

Live sending needs OUTREACH_DRY_RUN=false *and* provider credentials.
"""
from __future__ import annotations

from typing import Any

import httpx

from .config import Settings

CHANNEL_LIMITS = {"email": 5000, "whatsapp": 1000, "sms": 320}


async def send_message(channel: str, to: str, subject: str, body: str, s: Settings) -> dict[str, Any]:
    if s.outreach_dry_run:
        return {"status": "dry_run", "detail": "OUTREACH_DRY_RUN=true — recorded, not sent"}
    try:
        if channel == "email":
            if not (s.sendgrid_api_key and s.email_from):
                return {"status": "dry_run", "detail": "SendGrid not configured"}
            async with httpx.AsyncClient(timeout=s.http_timeout_s) as c:
                r = await c.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    headers={"Authorization": f"Bearer {s.sendgrid_api_key}"},
                    json={
                        "personalizations": [{"to": [{"email": to}]}],
                        "from": {"email": s.email_from},
                        "subject": subject or "(no subject)",
                        "content": [{"type": "text/plain", "value": body}],
                    },
                )
            ok = r.status_code < 300
            return {"status": "sent" if ok else "failed", "detail": r.headers.get("x-message-id") or r.text[:200]}

        sender = s.twilio_whatsapp_from if channel == "whatsapp" else s.twilio_sms_from
        if not (s.twilio_account_sid and s.twilio_auth_token and sender):
            return {"status": "dry_run", "detail": "Twilio not configured"}
        prefix = "whatsapp:" if channel == "whatsapp" else ""
        async with httpx.AsyncClient(timeout=s.http_timeout_s) as c:
            r = await c.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{s.twilio_account_sid}/Messages.json",
                auth=(s.twilio_account_sid, s.twilio_auth_token),
                data={"From": prefix + sender, "To": prefix + to, "Body": body},
            )
        ok = r.status_code < 300
        return {"status": "sent" if ok else "failed", "detail": r.json().get("sid") if ok else r.text[:200]}
    except httpx.HTTPError as exc:
        return {"status": "failed", "detail": str(exc)[:200]}
