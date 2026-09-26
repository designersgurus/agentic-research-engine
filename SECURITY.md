# Security

## Threat model and controls

| Threat | Control | Where |
|---|---|---|
| Runaway API spend | Pre-call token reservation, per-job caps on LLM calls, searches and fetches, loop and recursion limits, graceful stop | `app/guardrails.py`, `app/research.py` |
| Strangers spending your credits or sending messages from your accounts | **Fail-safe:** with any live key or live sending enabled, every write endpoint returns `503` until `API_KEY` is set | `require_key` in `app/main.py` |
| Unauthorised access | `X-API-Key` on all write endpoints (constant-time compare). Provider webhooks may pass `?key=` instead | `app/main.py` |
| Data exposure across users | Listing endpoints (`GET /jobs`, `GET /outreach/campaigns`) are **disabled** unless `API_KEY` is set, and then require it. Records are fetched by random, unguessable IDs | `require_admin` |
| Prompt injection from scraped pages or contact notes | Control and bidi characters stripped, instruction-like phrases neutralised, content fenced as `<untrusted_data>`, and a system policy that treats it as data only. Research agents have no access to outreach tools | `sanitize_untrusted`, `wrap_untrusted` |
| Invented citations | Any `[n]` in the report without a matching collected source is removed. Claims without a valid source are dropped | `synthesize_node`, `research_node` |
| SSRF via scraping or callbacks | Only http(s). Private, loopback, link-local, reserved, multicast, unspecified and IPv4-mapped addresses are blocked. **Every redirect hop is re-validated** (max 5) | `validate_public_url`, `fetch_page` |
| Oversized or hostile pages | 2 MB read cap, text/HTML only, 15 s timeout, scripts, iframes and forms stripped | `fetch_page`, `html_to_text` |
| Flooding and denial of service | Per-IP rate limit on writes (30/min by default, real client IP taken from `CF-Connecting-IP` or the right-most `X-Forwarded-For` entry), job backlog cap (429), concurrency cap | `app/security.py` |
| Stale personal data | Jobs and campaigns are purged after `DATA_RETENTION_HOURS` (default 24) | `Store.purge_older_than` |
| Outreach abuse | Dry-run by default. Server-side follow-up cap. STOP / unsubscribe honoured. Address format validated. Channel length limits | `app/outreach.py` |
| XSS and clickjacking on the demo page | Strict CSP, `X-Frame-Options: DENY`, `nosniff`, HSTS, `no-referrer`. The report is rendered with DOMPurify, or an escape-first fallback renderer | `app/security.py`, `index.html` |
| Webhook spoofing to your backend | Callbacks are HMAC-SHA256 signed (`X-Signature-256`) with `WEBHOOK_SECRET` | `_callback` |
| Secrets in the repo | `.env` is git-ignored. Keys are read from the environment only | `.gitignore`, `app/config.py` |
| Vulnerable dependencies | `pip-audit` is clean. Test tooling is kept out of the production install | `requirements-dev.txt` |

## Before going live

1. Set `API_KEY` to a long random value. Live mode refuses writes without it.
2. Set `WEBHOOK_SECRET` and verify `X-Signature-256` on your backend.
3. Keep `OUTREACH_DRY_RUN=true` until sender domains and numbers are verified and consent is recorded.
4. For production, put the service behind your own gateway and move storage to Postgres.

## Known limits

- The SSRF check resolves DNS before connecting, so a DNS-rebinding attacker could in theory switch the address between the two steps. For hostile environments, add egress filtering at the network layer.
- The rate limiter is in-memory and per-instance. Use Redis for multi-instance deployments.

## Reporting

Please open a private security advisory on GitHub rather than a public issue.
