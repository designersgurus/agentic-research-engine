"""HTTP-layer security: rate limiting, client IP resolution, security headers."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request

# Strict CSP for the demo page (scripts only from self/inline and the two pinned CDN libs)
PAGE_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)

BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


def client_ip(request: Request) -> str:
    """Best-effort real client IP behind Render/Cloudflare.

    CF-Connecting-IP is set by Cloudflare and can't be forged through it. Otherwise use the
    right-most X-Forwarded-For entry (added by the platform proxy), never the left-most
    (which the client controls).
    """
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding-window limiter: at most `limit` requests per `window_s` per key (in-memory)."""

    def __init__(self, limit: int, window_s: float = 60.0, max_keys: int = 10_000):
        self.limit, self.window_s, self.max_keys = limit, window_s, max_keys
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        if len(self._hits) > self.max_keys:  # bound memory under IP-spraying
            for k in list(self._hits)[: len(self._hits) // 2]:
                del self._hits[k]
        return True
