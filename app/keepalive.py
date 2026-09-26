"""Keep-alive: stop Render's free tier from sleeping, without overdoing it.

Render puts a free web service to sleep after ~15 minutes with no *inbound* traffic.
A localhost ping doesn't count, so this pings the service's public URL (Render sets
RENDER_EXTERNAL_URL automatically) on a fixed interval.

Limits that keep it light:
  * runs only when a public URL is known (i.e. on Render), never locally or in tests
  * interval can't go below 4 minutes (default 5)
  * one tiny GET to /ping (no DB, no LLM, no logging noise), 10 s timeout, no retries
  * optional active-hours window (e.g. 7-23 IST) so it can sleep overnight and save
    free-tier instance hours
  * after 3 failures in a row it backs off to every 30 minutes until a ping succeeds
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx

from .config import Settings

MIN_INTERVAL_S = 240
BACKOFF_INTERVAL_S = 1800


def parse_hours(spec: str) -> Optional[tuple[int, int]]:
    """'7-23' -> (7, 23). Empty -> None (always on)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    start, end = (int(x) for x in spec.split("-", 1))
    if not (0 <= start <= 23 and 0 <= end <= 24):
        raise ValueError("KEEPALIVE_ACTIVE_HOURS must look like 7-23")
    return start, end


class KeepAlive:
    def __init__(self, settings: Settings):
        self.s = settings
        base = (settings.keepalive_url or settings.render_external_url or "").rstrip("/")
        self.url = f"{base}/ping" if base else ""
        self.enabled = bool(settings.keepalive_enabled and self.url)
        self.interval_s = max(settings.keepalive_interval_seconds, MIN_INTERVAL_S)
        self.hours = parse_hours(settings.keepalive_active_hours)
        self.tz = ZoneInfo(settings.keepalive_timezone)
        self.pings = 0
        self.failures_in_row = 0
        self.last_ping_at: Optional[str] = None
        self.last_result: Optional[str] = None
        self._last_attempt: Optional[datetime] = None

    def in_active_window(self, now: Optional[datetime] = None) -> bool:
        if not self.hours:
            return True
        local = (now or datetime.now(timezone.utc)).astimezone(self.tz)
        start, end = self.hours
        if start <= end:
            return start <= local.hour < end
        return local.hour >= start or local.hour < end  # window crosses midnight, e.g. 20-6

    def _backing_off(self, now: datetime) -> bool:
        if self.failures_in_row < 3 or not self._last_attempt:
            return False
        return (now - self._last_attempt).total_seconds() < BACKOFF_INTERVAL_S

    async def tick(self) -> str:
        now = datetime.now(timezone.utc)
        if not self.enabled:
            return "disabled"
        if not self.in_active_window(now):
            return "outside_active_hours"  # let Render sleep; the next visitor wakes it
        if self._backing_off(now):
            return "backing_off"
        self._last_attempt = now
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(self.url, headers={"User-Agent": "keepalive"})
            ok = r.status_code < 400
        except httpx.HTTPError:
            ok = False
        self.pings += 1
        self.failures_in_row = 0 if ok else self.failures_in_row + 1
        self.last_ping_at = now.isoformat()
        self.last_result = "ok" if ok else "failed"
        return self.last_result

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_s,
            "active_hours": self.s.keepalive_active_hours or "always",
            "timezone": self.s.keepalive_timezone,
            "pings": self.pings,
            "last_ping_at": self.last_ping_at,
            "last_result": self.last_result,
        }
