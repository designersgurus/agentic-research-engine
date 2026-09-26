"""HTTP routes for the demo pages and the showcase demo APIs."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..security import RateLimiter, client_ip
from ..tools import fetch_html
from . import extract, monitor, ocr, support

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD_B64 = ocr.MAX_UPLOAD_BYTES * 4 // 3 + 16


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., max_length=2000)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, examples=["Can I return order NW-10231?"])
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    pending: Optional[dict[str, Any]] = None


class ExtractIn(BaseModel):
    text: Optional[str] = Field(None, max_length=extract.MAX_CHARS, description="Plain-text document")
    file_base64: Optional[str] = Field(
        None, max_length=MAX_UPLOAD_B64,
        description="PDF, PNG, JPEG or WebP (max 5 MB), base64-encoded. Type is detected from the file contents.",
    )


class MonitorIn(BaseModel):
    day: int = Field(1, ge=1, le=monitor.MAX_DAY)
    threshold_pct: float = Field(10, ge=1, le=90)


class MonitorUrlIn(BaseModel):
    url: str = Field(..., max_length=500)
    selectors: dict[str, str] = Field(default_factory=lambda: dict(monitor.DEFAULT_SELECTORS))


def _page(name: str) -> Callable:
    async def serve() -> FileResponse:
        return FileResponse(STATIC / name)
    return serve


def build_router(settings: Settings, require_key: Callable, require_admin: Callable) -> APIRouter:
    r = APIRouter()
    ocr_limiter = RateLimiter(limit=8)          # OCR is CPU-heavy: stricter per-IP limit

    # ---------------------------------------------------------------- pages
    for path, page in {"/": "home.html", "/demos/research": "research.html", "/demos/support": "support.html",
                       "/demos/extract": "extract.html", "/demos/monitor": "monitor.html"}.items():
        r.add_api_route(path, _page(page), methods=["GET"], include_in_schema=False)

    @r.get("/demos", include_in_schema=False)
    @r.get("/demos/", include_in_schema=False)
    async def demos_index() -> RedirectResponse:
        return RedirectResponse("/", status_code=308)

    @r.get("/demos/site", response_class=HTMLResponse, include_in_schema=False)
    async def demo_site(day: int = Query(0, ge=0, le=monitor.MAX_DAY)) -> HTMLResponse:
        return HTMLResponse(monitor.render_site(day))

    # ---------------------------------------------------------------- support agent
    @r.post("/demos/api/support/chat", tags=["demo: support agent"], dependencies=[Depends(require_key)])
    async def support_chat(body: ChatIn) -> dict[str, Any]:
        """One turn of the support agent (retrieval + tools + guardrails). Pass `pending` back for multi-turn."""
        return await support.chat(body.message, [t.model_dump() for t in body.history], body.pending, settings)

    # ---------------------------------------------------------------- document extraction
    @r.get("/demos/api/extract/samples", tags=["demo: document extraction"])
    async def extract_samples() -> dict[str, dict[str, str]]:
        return extract.SAMPLES

    @r.post("/demos/api/extract", tags=["demo: document extraction"], dependencies=[Depends(require_key)])
    async def run_extract(body: ExtractIn, request: Request) -> dict[str, Any]:
        """Extract and validate an invoice, receipt or purchase order from text, a PDF or a photo.

        Files are processed in memory and never stored.
        """
        source, ocr_conf, seconds = "text", None, 0.0
        if body.file_base64:
            if not ocr_limiter.allow(client_ip(request)):
                raise HTTPException(429, "too many uploads; try again in a minute")
            try:
                result = await ocr.file_to_text(body.file_base64)
            except ocr.UnsupportedFile as exc:
                raise HTTPException(422, str(exc)) from exc
            except TimeoutError as exc:
                raise HTTPException(504, "reading the file took too long") from exc
            except Exception as exc:  # malformed or hostile files must never crash the service
                raise HTTPException(422, "could not read this file") from exc
            text, source, ocr_conf, seconds = result.text, result.source, result.ocr_confidence, result.seconds
            if not text.strip():
                raise HTTPException(422, "no text found in this file")
        elif body.text and body.text.strip():
            text = body.text
        else:
            raise HTTPException(422, "provide text or file_base64")
        out = await extract.run_extraction(text, settings, ocr_conf)
        out.update(text=text[: extract.MAX_CHARS], input={"source": source, "ocr_confidence": ocr_conf,
                                                            "seconds": round(seconds, 2)})
        return out

    # ---------------------------------------------------------------- web monitor
    @r.post("/demos/api/monitor/check", tags=["demo: web monitor"], dependencies=[Depends(require_key)])
    async def monitor_check(body: MonitorIn, request: Request) -> dict[str, Any]:
        """Scrape the demo store for `day`, compare with the previous day, and build alerts."""
        return monitor.run_check(body.day, body.threshold_pct, str(request.base_url).rstrip("/"))

    @r.post("/demos/api/monitor/check-url", tags=["demo: web monitor"], dependencies=[Depends(require_admin)])
    async def monitor_check_url(body: MonitorUrlIn) -> dict[str, Any]:
        """Scrape any public URL with CSS selectors (requires API_KEY; SSRF-protected)."""
        if not body.selectors.get("item"):
            raise HTTPException(422, "selectors.item is required")
        selectors = {**monitor.DEFAULT_SELECTORS, **body.selectors}
        try:
            final_url, html = await fetch_html(body.url, settings)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, "could not fetch that page") from exc
        return {"url": final_url, "items": monitor.scrape(html, selectors)}

    return r
