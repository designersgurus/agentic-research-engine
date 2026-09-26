"""FastAPI entry point: `uvicorn app.main:app`"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from .config import Settings, get_settings
from .guardrails import validate_public_url
from .outreach import OutreachService
from .research import run_research
from .schemas import CampaignCreate, InboundMessage, Job, JobAccepted, JobCreate, ReplyEvent
from .store import Store

STATIC = Path(__file__).parent / "static"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(settings: Optional[Settings] = None, start_scheduler: bool = True) -> FastAPI:
    s = settings or get_settings()
    store = Store(s.db_path)
    outreach = OutreachService(store, s)
    job_slots = asyncio.Semaphore(s.max_concurrent_jobs)
    scheduler = AsyncIOScheduler(timezone="UTC")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_scheduler:
            scheduler.add_job(outreach.process_due, "interval", seconds=s.scheduler_tick_seconds, max_instances=1)
            scheduler.start()
        yield
        if scheduler.running:
            scheduler.shutdown(wait=False)

    app = FastAPI(
        title=s.app_name,
        version=s.version,
        lifespan=lifespan,
        description=(
            "Standalone multi-agent engine: parallel research → self-verification loop → cited markdown "
            "reports, plus personalised outreach with hard-capped follow-ups. "
            "Runs fully in **mock mode** without API keys."
        ),
    )
    app.state.settings, app.state.store, app.state.outreach = s, store, outreach

    def require_key(x_api_key: Optional[str] = Header(None)) -> None:
        if s.api_key and not hmac.compare_digest(x_api_key or "", s.api_key):
            raise HTTPException(401, "missing or invalid X-API-Key")

    # ------------------------------------------------------------------ misc
    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": s.version,
            "llm": s.resolved_llm_provider,
            "search": s.resolved_search_provider,
            "outreach_dry_run": s.outreach_dry_run,
            "auth_required": bool(s.api_key),
            "caps": {
                "job_token_budget": s.job_token_budget,
                "max_llm_calls": s.max_llm_calls,
                "max_search_calls": s.max_search_calls,
                "max_scrape_calls": s.max_scrape_calls,
                "max_verify_passes": s.max_verify_passes,
                "graph_recursion_limit": s.graph_recursion_limit,
                "outreach_max_followups_cap": s.outreach_max_followups_cap,
            },
        }

    # ------------------------------------------------------------------ research jobs
    async def _execute(job_id: str, req: JobCreate) -> None:
        job = store.get("jobs", job_id)
        async with job_slots:  # concurrency cap
            job.update(status="running", updated_at=_now())
            store.put("jobs", job_id, job)
            try:
                result = await run_research(req.query, s, max_passes=req.max_passes, token_budget=req.token_budget)
                job.update(status="completed", result=result, updated_at=_now())
            except Exception as exc:  # never leave a job stuck in "running"
                job.update(status="failed", error=str(exc)[:500], updated_at=_now())
            store.put("jobs", job_id, job)
        url = req.callback_url or s.default_callback_url
        if url:
            await _callback(url, {"event": "job.finished", "job": job})

    async def _callback(url: str, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        headers = {"Content-Type": "application/json"}
        if s.webhook_secret:
            sig = hmac.new(s.webhook_secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Signature-256"] = f"sha256={sig}"
        try:
            await validate_public_url(url)
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(url, content=body, headers=headers)
        except Exception:
            pass  # delivery is best-effort; results stay available via GET /jobs/{id}

    @app.post("/jobs", status_code=202, response_model=JobAccepted, tags=["research"],
              dependencies=[Depends(require_key)])
    async def create_job(req: JobCreate, bg: BackgroundTasks, request: Request):
        """Start a research job. Poll `GET /jobs/{id}` or receive the result at `callback_url`."""
        job_id = uuid.uuid4().hex[:12]
        store.put(
            "jobs",
            job_id,
            {"id": job_id, "status": "queued", "query": req.query, "created_at": _now(), "updated_at": _now(),
             "result": None, "error": None},
        )
        bg.add_task(_execute, job_id, req)
        return {
            "job_id": job_id,
            "status": "queued",
            "links": {"self": f"/jobs/{job_id}", "report": f"/jobs/{job_id}/report"},
        }

    @app.get("/jobs", tags=["research"])
    async def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
        return [
            {k: j[k] for k in ("id", "status", "query", "created_at", "updated_at")}
            for j in store.list("jobs", min(limit, 100))
        ]

    @app.get("/jobs/{job_id}", response_model=Job, tags=["research"])
    async def get_job(job_id: str):
        job = store.get("jobs", job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/jobs/{job_id}/report", response_class=PlainTextResponse, tags=["research"])
    async def get_report(job_id: str):
        """The cited markdown report."""
        job = store.get("jobs", job_id)
        if not job:
            raise HTTPException(404, "job not found")
        if job["status"] != "completed":
            raise HTTPException(409, f"job is {job['status']}")
        return PlainTextResponse(job["result"]["report"], media_type="text/markdown")

    # ------------------------------------------------------------------ outreach
    @app.post("/outreach/campaigns", status_code=201, tags=["outreach"], dependencies=[Depends(require_key)])
    async def create_campaign(req: CampaignCreate):
        """Draft + send (dry-run by default) personalised first messages and schedule capped follow-ups."""
        try:
            return await outreach.create_campaign(req)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/outreach/campaigns", tags=["outreach"])
    async def list_campaigns(limit: int = 20):
        return [
            {"id": c["id"], "name": c["name"], "channel": c["channel"], "contacts": len(c["contacts"]),
             "created_at": c["created_at"]}
            for c in store.list("campaigns", min(limit, 100))
        ]

    @app.get("/outreach/campaigns/{campaign_id}", tags=["outreach"])
    async def get_campaign(campaign_id: str):
        camp = store.get("campaigns", campaign_id)
        if not camp:
            raise HTTPException(404, "campaign not found")
        return camp

    @app.post("/outreach/campaigns/{campaign_id}/contacts/{contact_id}/reply", tags=["outreach"],
              dependencies=[Depends(require_key)])
    async def mark_reply(campaign_id: str, contact_id: str, ev: ReplyEvent):
        """Record a reply or opt-out. Follow-ups for this contact stop immediately."""
        try:
            return await outreach.record_reply(campaign_id, contact_id, ev.opt_out, ev.text)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/outreach/webhooks/inbound", tags=["outreach"], dependencies=[Depends(require_key)])
    async def inbound(msg: InboundMessage):
        """Point your email/SMS/WhatsApp provider's inbound webhook here (normalised payload)."""
        return {"matched": await outreach.inbound(msg.address, msg.text)}

    @app.post("/outreach/process-due", tags=["outreach"], dependencies=[Depends(require_key)])
    async def process_due(campaign_id: Optional[str] = None, force: bool = False):
        """Run the follow-up scheduler now. `force=true` treats every open contact as due (demo/testing)."""
        return await outreach.process_due(campaign_id, force=force)

    return app


app = create_app()
