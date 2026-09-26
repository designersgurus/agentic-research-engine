"""Outreach: personalised drafts + scheduled, hard-capped follow-ups.

Contact lifecycle
    awaiting_reply ──(due, followups_sent < max)──► send follow-up ──► awaiting_reply
          │
          ├──(due, followups_sent == max)──► closed_no_reply   (cap reached — never messaged again)
          ├──(reply webhook)──────────────► replied            (follow-ups stop immediately)
          └──(STOP / opt-out)─────────────► opted_out
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from .channels import CHANNEL_LIMITS, send_message
from .config import Settings
from .guardrails import (
    UNTRUSTED_POLICY,
    Budget,
    BudgetExceeded,
    sanitize_untrusted,
    wrap_untrusted,
)
from .llm import LLM, parse_json
from .schemas import CampaignCreate
from .store import Store

_lock = asyncio.Lock()
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")
STOP_WORDS = {"stop", "unsubscribe", "opt out", "optout", "cancel"}


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _validate_address(channel: str, address: str) -> None:
    if channel == "email" and not EMAIL_RE.match(address):
        raise ValueError(f"invalid email address: {address}")
    if channel in ("sms", "whatsapp") and not PHONE_RE.match(address):
        raise ValueError(f"phone must be E.164 like +919876543210: {address}")


async def _draft(
    llm: LLM, budget: Budget, s: Settings, campaign: dict, contact: dict, kind: str, n: int
) -> dict[str, str]:
    channel = campaign["channel"]
    limit = CHANNEL_LIMITS[channel]
    notes = sanitize_untrusted(contact.get("notes") or "", 800, budget, f"contact {contact['id']}")
    first = contact["name"].split()[0]
    company = contact.get("company") or "your team"
    goal_phrase = campaign["goal"][:1].lower() + campaign["goal"][1:].rstrip(".")

    def mock() -> str:
        if kind == "initial":
            body = (
                f"Hi {first},\n\nI'm reaching out from {campaign['sender_name']}. "
                f"Given what {company} is working on, I wanted to reach out with one goal in mind: "
                f"{goal_phrase}.\n\nWould you be open to a quick chat this week?"
            )
            subject = f"Quick idea for {company}"
        else:
            body = (
                f"Hi {first}, just following up on my earlier note ({goal_phrase}). "
                "Happy to share details whenever suits you."
            )
            subject = f"Re: Quick idea for {company}"
        return json.dumps({"subject": subject, "body": body})

    context = campaign.get("context") or ""
    user = (
        f"Channel: {channel} (max {limit} characters)\nMessage type: {kind}"
        + (f" #{n} of {campaign['max_followups']}" if kind == "followup" else "")
        + f"\nSender: {campaign['sender_name']}\nGoal: {campaign['goal']}\n"
        f"Recipient: {contact['name']} at {company}\n"
        f"Recipient notes:\n{wrap_untrusted(notes or 'none', 'contact notes')}\n"
        + (f"Research context:\n{wrap_untrusted(context, 'research report')}\n" if context else "")
    )
    try:
        raw = await llm.complete(
            system=(
                "You write short, honest, personalised outreach messages. No false claims, no pressure tactics, "
                "no fake familiarity. Follow-ups must be briefer than the first message and reference it. "
                'Return JSON: {"subject": "...", "body": "..."} (subject only matters for email).\n'
                + UNTRUSTED_POLICY
            ),
            user=user,
            budget=budget,
            max_tokens=500,
            json_mode=True,
            mock=mock,
        )
        data = parse_json(raw)
    except BudgetExceeded:
        data = json.loads(mock())  # zero-cost template fallback
    body = str(data.get("body") or json.loads(mock())["body"]).strip()
    footer = "\n\nReply STOP to opt out." if channel != "email" else "\n\nIf this isn't relevant, just reply 'unsubscribe'."
    body = body[: limit - len(footer)] + footer
    return {"subject": str(data.get("subject", ""))[:150], "body": body}


class OutreachService:
    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.s = settings

    def _budget(self) -> Budget:
        return Budget(
            token_limit=self.s.outreach_token_budget,
            llm_call_limit=self.s.max_llm_calls * 4,
            search_limit=0,
            scrape_limit=0,
        )

    async def create_campaign(self, req: CampaignCreate) -> dict[str, Any]:
        for c in req.contacts:
            _validate_address(req.channel, c.address)
        context = ""
        if req.research_job_id:
            job = self.store.get("jobs", req.research_job_id)
            if not job or job.get("status") != "completed":
                raise ValueError("research_job_id must reference a completed job")
            context = (job["result"] or {}).get("report", "")[:3000]

        capped = min(req.max_followups, self.s.outreach_max_followups_cap)
        campaign: dict[str, Any] = {
            "id": uuid.uuid4().hex[:10],
            "name": req.name,
            "channel": req.channel,
            "goal": req.goal,
            "sender_name": req.sender_name,
            "context": context,
            "max_followups": capped,
            "max_followups_requested": req.max_followups,
            "interval_minutes": max(req.followup_interval_minutes, self.s.outreach_min_interval_minutes),
            "dry_run": self.s.outreach_dry_run,
            "created_at": _iso(_now()),
            "contacts": [],
        }
        llm, budget = LLM(self.s), self._budget()
        for c in req.contacts:
            contact = {
                "id": uuid.uuid4().hex[:8],
                **c.model_dump(),
                "status": "pending",
                "followups_sent": 0,
                "next_followup_at": None,
                "messages": [],
            }
            campaign["contacts"].append(contact)
            await self._send(campaign, contact, llm, budget, "initial", 0)
        campaign["usage"] = budget.snapshot()
        async with _lock:
            self.store.put("campaigns", campaign["id"], campaign)
        return campaign

    async def _send(self, campaign: dict, contact: dict, llm: LLM, budget: Budget, kind: str, n: int) -> None:
        msg = await _draft(llm, budget, self.s, campaign, contact, kind, n)
        delivery = await send_message(campaign["channel"], contact["address"], msg["subject"], msg["body"], self.s)
        contact["messages"].append({"kind": kind, "n": n, **msg, "sent_at": _iso(_now()), "delivery": delivery})
        if delivery["status"] == "failed":
            contact["status"] = "failed"
            contact["next_followup_at"] = None
            return
        contact["status"] = "awaiting_reply"
        contact["next_followup_at"] = _iso(_now() + timedelta(minutes=campaign["interval_minutes"]))

    async def process_due(self, campaign_id: Optional[str] = None, force: bool = False) -> dict[str, int]:
        """Send due follow-ups. Called by the scheduler, or manually via the API."""
        stats = {"followups_sent": 0, "closed_no_reply": 0}
        campaigns = [self.store.get("campaigns", campaign_id)] if campaign_id else self.store.list("campaigns", 500)
        llm = LLM(self.s)
        for camp in filter(None, campaigns):
            changed = False
            budget = self._budget()
            for contact in camp["contacts"]:
                if contact["status"] != "awaiting_reply" or not contact["next_followup_at"]:
                    continue
                due = datetime.fromisoformat(contact["next_followup_at"]) <= _now()
                if not (due or force):
                    continue
                if contact["followups_sent"] >= camp["max_followups"]:
                    contact["status"] = "closed_no_reply"  # hard cap: never contacted again
                    contact["next_followup_at"] = None
                    stats["closed_no_reply"] += 1
                else:
                    contact["followups_sent"] += 1
                    await self._send(camp, contact, llm, budget, "followup", contact["followups_sent"])
                    stats["followups_sent"] += 1
                changed = True
            if changed:
                async with _lock:
                    latest = self.store.get("campaigns", camp["id"]) or camp
                    # a reply may have arrived while we were drafting; it wins
                    stopped = {c["id"]: c for c in latest["contacts"] if c["status"] in ("replied", "opted_out")}
                    camp["contacts"] = [stopped.get(c["id"], c) for c in camp["contacts"]]
                    self.store.put("campaigns", camp["id"], camp)
        return stats

    async def record_reply(self, campaign_id: str, contact_id: str, opt_out: bool, text: str | None) -> dict:
        async with _lock:
            camp = self.store.get("campaigns", campaign_id)
            if not camp:
                raise KeyError("campaign not found")
            for c in camp["contacts"]:
                if c["id"] == contact_id:
                    is_stop = opt_out or (text or "").strip().lower() in STOP_WORDS
                    c["status"] = "opted_out" if is_stop else "replied"
                    c["next_followup_at"] = None
                    c["reply"] = {"text": (text or "")[:1000], "at": _iso(_now())}
                    self.store.put("campaigns", campaign_id, camp)
                    return c
        raise KeyError("contact not found")

    async def inbound(self, address: str, text: str) -> list[dict]:
        """Match an inbound provider webhook to every open contact with that address."""
        matched = []
        for camp in self.store.list("campaigns", 500):
            for c in camp["contacts"]:
                if c["address"].lower() == address.lower() and c["status"] == "awaiting_reply":
                    updated = await self.record_reply(camp["id"], c["id"], False, text)
                    matched.append({"campaign_id": camp["id"], "contact_id": c["id"], "status": updated["status"]})
        return matched
