# Developer guide

[← Back to README](../README.md)

## Quick start

```bash
git clone https://github.com/designersgurus/agentic-research-engine
cd agentic-research-engine
pip install -r requirements-dev.txt              # runtime deps + test tools
pip install --no-deps -r requirements-ocr.txt    # OCR engine (see note below)
cp .env.example .env                             # optional: works without keys
uvicorn app.main:app --reload
# open http://localhost:8000 (demos) · http://localhost:8000/docs (Swagger)
pytest -q                                        # 58 tests, all offline
```

The OCR package is installed with `--no-deps` because it declares the desktop build of OpenCV, which needs graphics libraries servers don't have. Its real dependencies, including `opencv-python-headless`, are in `requirements.txt`. Peak memory for a photo upload is about 360 MB, so it fits a 512 MB instance.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/jobs` | Start a research job → `202 {job_id}` |
| `GET` | `/jobs/{id}` | Status, report, findings, verification log, usage |
| `GET` | `/jobs/{id}/report` | Cited report as `text/markdown` |
| `GET` | `/jobs` | Recent jobs (requires `API_KEY`; disabled in open demo mode) |
| `POST` | `/outreach/campaigns` | Draft and send first messages, schedule capped follow-ups |
| `GET` | `/outreach/campaigns/{id}` | Campaign with per-contact status and message history |
| `POST` | `/outreach/campaigns/{id}/contacts/{cid}/reply` | Record a reply or opt-out (stops follow-ups) |
| `POST` | `/outreach/webhooks/inbound` | Normalised inbound webhook from your email/SMS provider |
| `POST` | `/outreach/process-due` | Run the follow-up scheduler now (`force=true` for demos) |
| `GET` | `/health` | Active providers and configured caps |
| `POST` | `/demos/api/support/chat` | Support agent: one conversation turn |
| `POST` | `/demos/api/extract` | Document extraction from `text` or `file_base64` (PDF, PNG, JPEG, WebP; max 5 MB) |
| `POST` | `/demos/api/monitor/check` | Web monitor: scrape, compare and alert for a simulated day |

```bash
curl -X POST $URL/jobs -H 'Content-Type: application/json' \
  -d '{"query":"AI agents for customer support","max_passes":2,"token_budget":20000,
       "callback_url":"https://your-backend.example/hooks/research"}'
```

Callbacks are sent as `POST` with `{"event":"job.finished","job":{...}}`. When `WEBHOOK_SECRET` is set, they are signed with `X-Signature-256: sha256=<hmac>`. When `API_KEY` is set, all write endpoints require the `X-API-Key` header.

## Configuration

All settings are environment variables. See [`.env.example`](../.env.example) for the full list.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `openai` / `anthropic` / `mock`. `auto` picks by which key is present |
| `SEARCH_PROVIDER` | `auto` | `serper` / `mock` |
| `JOB_TOKEN_BUDGET` | `60000` | Hard token cap per job |
| `MAX_LLM_CALLS` / `MAX_SEARCH_CALLS` / `MAX_SCRAPE_CALLS` | `30` / `15` / `20` | Per-job counters |
| `MAX_VERIFY_PASSES` | `2` | Follow-up research rounds the verifier may trigger |
| `GRAPH_RECURSION_LIMIT` | `40` | LangGraph super-step ceiling |
| `OUTREACH_DRY_RUN` | `true` | Record messages without sending them |
| `OUTREACH_MAX_FOLLOWUPS_CAP` | `3` | Server-side ceiling on follow-ups per contact |
| `ANTHROPIC_MODEL` | *(empty)* | Required when using an Anthropic key: the model id from your Anthropic console |

## Deploy on Render

1. Click **Deploy to Render** above, or go to Render → **New → Blueprint** and select this repo. `render.yaml` configures everything.
2. Leave the key fields empty for mock mode, or add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and `SERPER_API_KEY` for live research.
3. **Set `API_KEY` before adding real keys**, so a public URL can't spend your credits.

**Keep-alive.** Render's free tier sleeps after about 15 minutes without traffic. To prevent that, the app pings its own public URL (`RENDER_EXTERNAL_URL/ping`) every 5 minutes. It is built to stay light:
- It only runs when a public URL exists, so never locally or in tests.
- The interval can't be set below 4 minutes.
- Each ping is a single tiny request to `/ping`, with no database or LLM work.
- After 3 failures in a row, it backs off to one ping every 30 minutes.
- You can set `KEEPALIVE_ACTIVE_HOURS=7-23` to let it sleep overnight and save free instance hours.

Its status is shown under `keepalive` in `/health`.

Note: Render's disk resets on each deploy. For production, use Postgres and a worker. The engine's store and scheduler are isolated modules, so they can be swapped.
