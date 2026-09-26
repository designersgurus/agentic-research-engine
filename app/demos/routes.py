"""HTTP routes for the three showcase demos (support agent, document extraction, web monitor)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..tools import fetch_html
from . import extract, monitor, support

STATIC = Path(__file__).resolve().parent.parent / "static"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., max_length=2000)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, examples=["Can I return order NW-10231?"])
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    pending: Optional[dict[str, Any]] = None


class ExtractIn(BaseModel):
    text: Optional[str] = Field(None, max_length=extract.MAX_CHARS)
    pdf_base64: Optional[str] = Field(None, max_length=int(extract.MAX_PDF_BYTES * 1.4))


class MonitorIn(BaseModel):
    day: int = Field(1, ge=1, le=monitor.MAX_DAY)
    threshold_pct: float = Field(10, ge=1, le=90)


class MonitorUrlIn(BaseModel):
    url: str = Field(..., max_length=500)
    selectors: dict[str, str] = Field(default_factory=lambda: dict(monitor.DEFAULT_SELECTORS))


def build_router(settings: Settings, require_key: Callable, require_admin: Callable) -> APIRouter:
    r = APIRouter(prefix="/demos")

    # ---------------------------------------------------------------- pages
    for path, page in {"": "demos.html", "/support": "support.html", "/extract": "extract.html",
                       "/monitor": "monitor.html"}.items():
        def make(p: str):
            async def serve():
                return FileResponse(STATIC / p)
            return serve
        r.add_api_route(path or "/", make(page), methods=["GET"], include_in_schema=False)

    @r.get("/site", response_class=HTMLResponse, include_in_schema=False)
    async def demo_site(day: int = Query(0, ge=0, le=monitor.MAX_DAY)):
        """The simulated competitor store scraped by the monitor demo."""
        return HTMLResponse(monitor.render_site(day))

    # ---------------------------------------------------------------- support agent
    @r.post("/api/support/chat", tags=["demo: support agent"], dependencies=[Depends(require_key)])
    async def support_chat(body: ChatIn):
        """One turn of the AI support agent (RAG + tools + guardrails). Pass back `pending` for multi-turn slot filling."""
        return await support.chat(body.message, [t.model_dump() for t in body.history], body.pending, settings)

    # ---------------------------------------------------------------- document extraction
    @r.get("/api/extract/samples", tags=["demo: document extraction"])
    async def extract_samples():
        return {k: v for k, v in extract.SAMPLES.items()}

    @r.post("/api/extract", tags=["demo: document extraction"], dependencies=[Depends(require_key)])
    async def run_extract(body: ExtractIn):
        """Extract + validate an invoice, receipt or purchase order (plain text or a text-based PDF as base64)."""
        if body.pdf_base64:
            try:
                text = extract.pdf_to_text(body.pdf_base64)
            except Exception as exc:  # malformed base64, not a PDF, too big, no text layer
                raise HTTPException(422, f"could not read PDF: {str(exc)[:120]}")
        elif body.text and body.text.strip():
            text = body.text
        else:
            raise HTTPException(422, "provide text or pdf_base64")
        result = await extract.run_extraction(text, settings)
        result["text"] = text[: extract.MAX_CHARS]
        return result

    # ---------------------------------------------------------------- web monitor
    @r.post("/api/monitor/check", tags=["demo: web monitor"], dependencies=[Depends(require_key)])
    async def monitor_check(body: MonitorIn, request: Request):
        """Scrape the demo store for `day`, diff against the previous day, and build alerts."""
        return monitor.run_check(body.day, body.threshold_pct, str(request.base_url).rstrip("/"))

    @r.post("/api/monitor/check-url", tags=["demo: web monitor"], dependencies=[Depends(require_admin)])
    async def monitor_check_url(body: MonitorUrlIn):
        """Scrape any public URL with CSS selectors (requires API_KEY; SSRF-protected)."""
        if not body.selectors.get("item"):
            raise HTTPException(422, "selectors.item is required")
        sel = {**monitor.DEFAULT_SELECTORS, **body.selectors}
        try:
            final_url, html = await fetch_html(body.url, settings)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            raise HTTPException(502, f"fetch failed: {str(exc)[:120]}")
        return {"url": final_url, "items": monitor.scrape(html, sel)}

    return r
