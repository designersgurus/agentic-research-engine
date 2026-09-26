"""Request/response models (these drive the Swagger / OpenAPI docs)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, examples=["AI agents for customer support in 2026"])
    max_passes: Optional[int] = Field(None, ge=0, le=5, description="Follow-up research rounds (capped by server)")
    token_budget: Optional[int] = Field(None, ge=500, description="Per-job token cap (can only lower the server cap)")
    callback_url: Optional[str] = Field(None, description="POSTed the finished job (HMAC-signed if WEBHOOK_SECRET set)")


class JobAccepted(BaseModel):
    job_id: str
    status: str
    links: dict[str, str]


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"]
    query: str
    created_at: str
    updated_at: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class ContactIn(BaseModel):
    name: str = Field(..., max_length=120)
    address: str = Field(..., max_length=200, description="Email address, or phone in E.164 for SMS/WhatsApp")
    company: Optional[str] = Field(None, max_length=120)
    notes: Optional[str] = Field(None, max_length=1000, description="Personalisation hints (treated as untrusted)")


class CampaignCreate(BaseModel):
    name: str = Field(..., max_length=120, examples=["Q4 partner outreach"])
    channel: Literal["email", "whatsapp", "sms"]
    goal: str = Field(..., max_length=500, examples=["Book a 20-minute intro call about our AI research engine"])
    sender_name: str = Field("The Team", max_length=80)
    research_job_id: Optional[str] = Field(None, description="Use a finished research report as context")
    max_followups: int = Field(2, ge=0, le=10, description="Capped by OUTREACH_MAX_FOLLOWUPS_CAP")
    followup_interval_minutes: int = Field(60 * 24 * 2, ge=1)
    contacts: list[ContactIn] = Field(..., min_length=1, max_length=100)


class ReplyEvent(BaseModel):
    opt_out: bool = False
    text: Optional[str] = None


class InboundMessage(BaseModel):
    address: str = Field(..., description="Sender email/phone as received by your provider webhook")
    text: str = ""
