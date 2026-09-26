"""AI Support Agent demo: retrieval-augmented answers + tool use + guardrails, as a LangGraph.

    guard ──► route ──► act (retrieve | lookup_order | check_refund | create_ticket) ──► respond

* guard   – blocks prompt-injection attempts and payment-card numbers, redacts PII in the trace
* route   – intent detection (FAQ, order status, refund eligibility, human handoff) with slot filling
* act     – BM25 retrieval over the knowledge base, or a tool call against the order system
* respond – grounded answer with citations; low-confidence answers hand off to a human

Mock mode answers extractively from the retrieved passages (no API key needed). With an LLM
key the respond step writes the answer, constrained to the passages and tool results, and
the citation check strips any source number it invents.
"""
from __future__ import annotations

import hashlib
import json
import math
import operator
import re
from collections import Counter
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..guardrails import INJECTION_MARKER, Budget, BudgetExceeded, sanitize_untrusted, wrap_untrusted, UNTRUSTED_POLICY
from ..llm import LLM
from . import kb

ORDER_RE = re.compile(r"\bNW-?(\d{5})\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d -]{8,13}\d)(?!\d)")
STOP = set(
    "a an the is are was were be to of in on for and or my i me we you your it this that with do does can "
    "how what when where which who will would should could please hi hello thanks thank there any about".split()
)
CONFIDENCE_THRESHOLD = 0.34


def _stem(t: str) -> str:
    """Tiny suffix stripper: shipping/ships → ship, internationally → international, covers → cover."""
    for suf, rep in (("ally", "al"), ("ly", ""), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(t) > len(suf) + 3 and t.endswith(suf) and not t.endswith("ss"):
            t = t[: -len(suf)] + rep
            if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "aeiousl":
                t = t[:-1]  # shipp → ship
            break
    return t


def tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9₹]+", text.lower()) if t not in STOP]


def redact(text: str) -> str:
    text = CARD_RE.sub("[card number removed]", text)
    text = EMAIL_RE.sub(lambda m: m.group(0)[0] + "***@" + m.group(0).split("@")[-1], text)
    return PHONE_RE.sub(lambda m: m.group(0)[:3] + "*****" + m.group(0)[-2:], text)


# ---------------------------------------------------------------------------
# BM25 retrieval (dependency-free)
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[dict[str, str]], k1: float = 1.4, b: float = 0.75):
        self.docs = docs
        self.toks = [tokens(d["title"] + " " + d["text"]) for d in docs]
        self.avg = sum(map(len, self.toks)) / len(self.toks)
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float, float]]:
        q = tokens(query)
        out = []
        for doc, dt in zip(self.docs, self.toks):
            tf = Counter(dt)
            score = sum(
                self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(dt) / self.avg))
                for w in q
            )
            coverage = (sum(1 for w in set(q) if w in tf) / len(set(q))) if q else 0.0
            out.append((doc, score, coverage))
        out.sort(key=lambda x: x[1], reverse=True)
        return [o for o in out[:k] if o[1] > 0]


INDEX = BM25(kb.KB_DOCS)


def _add(a: list, b: list) -> list:
    return a + b


class SupportState(TypedDict, total=False):
    message: str
    history: list[dict[str, str]]
    pending: Optional[dict[str, Any]]
    trace: Annotated[list[dict[str, str]], operator.add]
    blocked: str
    intent: str
    order_id: str
    email: str
    passages: list[dict[str, Any]]
    tool: dict[str, Any]
    reply: str
    citations: list[dict[str, Any]]
    confidence: float
    handoff: bool
    next_pending: Optional[dict[str, Any]]


def _ctx(config: RunnableConfig):
    return config["configurable"]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def guard_node(state: SupportState, config: RunnableConfig) -> dict:
    msg = state["message"]
    budget: Budget = _ctx(config)["budget"]
    if CARD_RE.search(msg):
        return {"blocked": "card", "trace": [{"step": "guard", "detail": "Payment card number detected: message not stored, user warned"}]}
    clean = sanitize_untrusted(msg, 1000, budget, "user message")
    if INJECTION_MARKER in clean:
        return {"blocked": "injection", "message": clean,
                "trace": [{"step": "guard", "detail": "Prompt-injection attempt neutralised; agent stays on its task"}]}
    return {"message": clean, "trace": [{"step": "guard", "detail": f"Input checked · logged as: “{redact(clean)[:120]}”"}]}


def route_node(state: SupportState) -> dict:
    if state.get("blocked"):
        return {"intent": "blocked"}
    msg = state["message"]
    low = msg.lower()
    pending = state.get("pending") or {}
    m = ORDER_RE.search(msg)
    order_id = f"NW-{m.group(1)}" if m else pending.get("order_id", "")
    em = EMAIL_RE.search(msg)
    email = em.group(0) if em else ""

    if pending.get("need") == "order_id" and order_id:
        intent = pending["intent"]
    elif pending.get("need") == "email" and email:
        intent = "handoff"
    elif pending.get("need") == "confirm_handoff" and re.match(r"\s*(yes|yeah|yep|sure|ok|okay|please)\b", low):
        intent = "handoff"
    elif re.search(r"\b(human|agent|person|someone|complain|complaint|manager|escalate|call me)\b", low):
        intent = "handoff"
    elif re.search(r"\b(refund|return)\b", low) and (order_id or re.search(r"\bmy (order|item)|\bi (want|need) to\b", low)):
        intent = "refund_check"
    elif re.search(r"\b(where|track|tracking|status|arrive|arriving|delivered|shipped|when will)\b", low) and (
        order_id or re.search(r"\bmy (order|package|parcel)\b", low)
    ):
        intent = "order_status"
    else:
        intent = "faq"
    detail = {"faq": "Knowledge-base question", "order_status": "Order status → needs order-lookup tool",
              "refund_check": "Refund eligibility → needs order-lookup + policy check",
              "handoff": "Wants a human → create support ticket"}[intent]
    if order_id and intent != "faq":
        detail += f" · order {order_id}"
    return {"intent": intent, "order_id": order_id, "email": email, "trace": [{"step": "route", "detail": detail}]}


def act_node(state: SupportState) -> dict:
    intent = state["intent"]
    if intent == "blocked":
        return {}
    if intent == "faq":
        hits = INDEX.search(state["message"], k=3)
        passages = [{"n": i + 1, "id": d["id"], "title": d["title"], "text": d["text"], "score": round(s, 2),
                     "coverage": round(c, 2)} for i, (d, s, c) in enumerate(hits)]
        conf = passages[0]["coverage"] if passages else 0.0
        top = ", ".join(f"{p['title']} ({p['score']})" for p in passages) or "no matches"
        return {"passages": passages, "confidence": conf,
                "trace": [{"step": "retrieve", "detail": f"BM25 over {len(kb.KB_DOCS)} docs → {top}"}]}

    if intent in ("order_status", "refund_check"):
        oid = state.get("order_id")
        if not oid:
            return {"tool": {"name": "lookup_order", "status": "missing_order_id"},
                    "trace": [{"step": "slot", "detail": "Order number missing → ask the customer"}]}
        order = kb.get_order(oid)
        res: dict[str, Any] = {"name": "lookup_order", "order_id": oid, "found": bool(order)}
        if order:
            res["order"] = {k: str(v) for k, v in order.items()}
            if intent == "refund_check":
                policy = INDEX.search("return refund days delivery final sale", k=1)[0][0]
                ok, why = _refund_eligible(order)
                res.update(eligible=ok, reason=why, policy_title=policy["title"])
        tr = f"lookup_order({oid}) → {'found: ' + order['status'] if order else 'not found'}"
        if intent == "refund_check" and order:
            tr += f" · check_refund → {'eligible' if res['eligible'] else 'not eligible'}"
        return {"tool": res, "trace": [{"step": "tool", "detail": tr}]}

    if intent == "handoff":
        email = state.get("email")
        if not email:
            return {"tool": {"name": "create_ticket", "status": "missing_email"},
                    "trace": [{"step": "slot", "detail": "Email missing → ask the customer"}]}
        tid = "TKT-" + hashlib.sha256((email + state["message"]).encode()).hexdigest()[:6].upper()
        return {"tool": {"name": "create_ticket", "ticket_id": tid, "email": email},
                "trace": [{"step": "tool", "detail": f"create_ticket(email={redact(email)}) → {tid} (dry run)"}]}
    return {}


def _refund_eligible(order: dict) -> tuple[bool, str]:
    from datetime import date

    if order.get("final_sale"):
        return False, "This item was sold as final sale, which can't be returned."
    if order["status"] != "delivered":
        return False, f"The order is still {order['status']}. It can be cancelled before dispatch, or returned after delivery."
    days = (date.today() - order["delivered_on"]).days
    if days > 30:
        return False, f"It was delivered {days} days ago, outside the 30-day return window."
    return True, f"It was delivered {days} days ago, within the 30-day return window."


async def respond_node(state: SupportState, config: RunnableConfig) -> dict:
    c = _ctx(config)
    intent = state["intent"]
    tool = state.get("tool") or {}

    if intent == "blocked":
        if state.get("blocked") == "card":
            reply = ("For your safety, please don't share card numbers here. Our team will never ask for your card "
                     "number, CVV or OTP. How else can I help with your order?")
        else:
            reply = (f"I can only help with {kb.COMPANY} orders, shipping, returns, warranty and payments. "
                     "What would you like to know?")
        return {"reply": reply, "confidence": 1.0, "citations": [], "handoff": False, "next_pending": None,
                "trace": [{"step": "respond", "detail": "Safe refusal"}]}

    if tool.get("status") == "missing_order_id":
        return {"reply": "Sure. What's your order number? It looks like NW-10231.",
                "next_pending": {"need": "order_id", "intent": intent}, "confidence": 1.0, "citations": [],
                "handoff": False, "trace": [{"step": "respond", "detail": "Asked for order number (slot filling)"}]}
    if tool.get("status") == "missing_email":
        return {"reply": "I'll connect you with our support team. What email should they reply to?",
                "next_pending": {"need": "email"}, "confidence": 1.0, "citations": [], "handoff": False,
                "trace": [{"step": "respond", "detail": "Asked for email (slot filling)"}]}

    if intent == "handoff":
        reply = (f"Done. I've created ticket {tool['ticket_id']} and our team will email you within one business day "
                 "(support hours: Mon–Sat, 9 am–7 pm IST).")
        return {"reply": reply, "handoff": True, "confidence": 1.0, "citations": [], "next_pending": None,
                "trace": [{"step": "respond", "detail": "Handed off to human support"}]}

    if intent in ("order_status", "refund_check"):
        if not tool.get("found"):
            return {"reply": f"I couldn't find order {tool.get('order_id')}. Could you double-check the number? "
                             f"(Demo orders: {', '.join(kb.sample_order_ids())})",
                    "next_pending": {"need": "order_id", "intent": intent}, "confidence": 1.0, "citations": [],
                    "handoff": False, "trace": [{"step": "respond", "detail": "Order not found → re-ask"}]}
        o = tool["order"]
        if intent == "order_status":
            reply = {
                "delivered": f"Your {o['item']} ({tool['order_id']}) was delivered on {o.get('delivered_on')}.",
                "shipped": f"Your {o['item']} ({tool['order_id']}) has shipped and should arrive by {o.get('eta')}. "
                           f"Tracking number: {o.get('tracking')}.",
                "processing": f"Your {o['item']} ({tool['order_id']}) is being prepared and will be dispatched soon. "
                              "You'll get a tracking link by email.",
            }.get(o["status"], f"Order {tool['order_id']} is {o['status']}.")
            return {"reply": reply, "confidence": 1.0, "citations": [], "handoff": False, "next_pending": None,
                    "trace": [{"step": "respond", "detail": "Answered from order system"}]}
        policy_doc = next(d for d in kb.KB_DOCS if d["id"] == "returns")
        cites = [{"n": 1, "title": policy_doc["title"], "snippet": policy_doc["text"][:140] + "…"}]
        if tool["eligible"]:
            reply = (f"Good news: your {o['item']} is eligible for a return. {tool['reason']} Refunds go to your original "
                     "payment method within 5 to 7 business days after inspection [1]. Shall I arrange a pickup?")
        else:
            reply = f"I'm sorry, your {o['item']} isn't eligible for a return. {tool['reason']} [1]"
        return {"reply": reply, "citations": cites, "confidence": 1.0, "handoff": False, "next_pending": None,
                "trace": [{"step": "respond", "detail": "Policy applied to order data, cited"}]}

    # ---- FAQ ------------------------------------------------------------
    passages = state.get("passages") or []
    conf = state.get("confidence", 0.0)
    if not passages or conf < CONFIDENCE_THRESHOLD:
        return {"reply": "I'm not sure about that one, and I'd rather not guess. Would you like me to connect you "
                         "with our support team?",
                "confidence": conf, "citations": [], "handoff": False,
                "next_pending": {"need": "confirm_handoff"},
                "trace": [{"step": "respond", "detail": f"Low confidence ({conf:.2f} < {CONFIDENCE_THRESHOLD}) → no guess, offer human"}]}

    use = passages[:2]
    fallback = _extractive_answer(state["message"], use)
    llm: LLM = c["llm"]
    blocks = "\n".join(f"[{p['n']}] {p['title']}: {wrap_untrusted(p['text'], p['title'])}" for p in use)
    try:
        answer = await llm.complete(
            system=(f"You are a friendly support agent for {kb.COMPANY}. Answer ONLY from the numbered sources. "
                    "Cite with [n]. If the sources don't answer the question, say you're not sure and offer a human. "
                    "Keep it under 80 words.\n" + UNTRUSTED_POLICY),
            user=f"Customer: {state['message']}\n\nSources:\n{blocks}",
            budget=c["budget"], max_tokens=250, mock=lambda: fallback,
        )
    except BudgetExceeded:
        answer = fallback
    valid = {p["n"] for p in use}
    answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)
    cites = [{"n": p["n"], "title": p["title"], "snippet": p["text"][:140] + "…"} for p in use
             if f"[{p['n']}]" in answer]
    return {"reply": answer, "citations": cites, "confidence": conf, "handoff": False, "next_pending": None,
            "trace": [{"step": "respond", "detail": f"Grounded answer · confidence {conf:.2f} · {len(cites)} source(s) cited"}]}


def _extractive_answer(question: str, passages: list[dict]) -> str:
    q = set(tokens(question))
    scored = []
    for p in passages:
        for i, sent in enumerate(re.split(r"(?<=[.!?])\s+", p["text"])):
            overlap = round(sum(INDEX.idf.get(w, 0) for w in q & set(tokens(sent))), 2)  # rarer words count more
            if overlap:
                scored.append((overlap, -p["n"], -i, sent, p["n"]))
    scored.sort(reverse=True)
    if scored:  # keep only the best-matching sentences (ties allowed), max two
        best = scored[0][0]
        scored = [x for x in scored if x[0] >= best * 0.8]
    picked = sorted(scored[:2], key=lambda x: (-x[1], -x[2]))
    return " ".join(f"{s} [{n}]" for *_, s, n in picked) or passages[0]["text"].split(". ")[0] + ". [1]"


def build_graph():
    g = StateGraph(SupportState)
    g.add_node("guard", guard_node)
    g.add_node("route", route_node)
    g.add_node("act", act_node)
    g.add_node("respond", respond_node)
    g.add_edge(START, "guard")
    g.add_edge("guard", "route")
    g.add_edge("route", "act")
    g.add_edge("act", "respond")
    g.add_edge("respond", END)
    return g.compile()


GRAPH = build_graph()


async def chat(message: str, history: list[dict], pending: Optional[dict], settings: Settings) -> dict[str, Any]:
    budget = Budget(token_limit=min(4000, settings.job_token_budget), llm_call_limit=2, search_limit=0, scrape_limit=0)
    llm = LLM(settings)
    final = await GRAPH.ainvoke(
        {"message": message, "history": history[-6:], "pending": pending},
        config={"recursion_limit": 10, "configurable": {"budget": budget, "llm": llm}},
    )
    return {
        "reply": final.get("reply", ""),
        "citations": final.get("citations", []),
        "intent": final.get("intent", ""),
        "confidence": round(float(final.get("confidence", 0.0)), 2),
        "handoff": final.get("handoff", False),
        "pending": final.get("next_pending"),
        "trace": final.get("trace", []),
        "usage": {"tokens_used": budget.tokens_used, "token_limit": budget.token_limit, "llm_calls": budget.llm_calls},
        "model": llm.model_name,
    }
