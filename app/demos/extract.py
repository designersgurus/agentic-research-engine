"""Document AI extraction demo: invoices / receipts / purchase orders → validated structured data.

Pipeline: classify → extract fields + line items → validate (arithmetic, required fields, dates)
→ per-field confidence → decision (auto-approve vs human review).

The validators are the point: whether fields come from the rule-based extractor (mock mode) or
from an LLM (live mode), nothing is approved unless the maths and the required fields check out.
"""
from __future__ import annotations

import base64
import io
import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

from ..config import Settings
from ..guardrails import UNTRUSTED_POLICY, Budget, BudgetExceeded, sanitize_untrusted, wrap_untrusted
from ..llm import LLM, parse_json

MAX_CHARS = 20_000
MAX_PDF_BYTES = 2_000_000

SAMPLES: dict[str, dict[str, str]] = {
    "invoice_clean": {
        "label": "Invoice (clean)",
        "text": """ACME OFFICE SUPPLIES LTD
123 Market Street, Springfield
INVOICE
Invoice No: INV-2026-0142
Invoice Date: 2026-09-12
Due Date: 2026-10-12
Bill To: Northwind Gadgets Pvt Ltd

Description                 Qty   Unit Price    Amount
A4 Paper (box)               10       24.50     245.00
Ink Cartridge Black           4       38.00     152.00
Stapler Heavy Duty            2       19.75      39.50

Subtotal: 436.50
GST 18%: 78.57
Total: 515.07
Currency: USD""",
    },
    "invoice_errors": {
        "label": "Invoice (with errors)",
        "text": """BRIGHTLINE LOGISTICS
Invoice #: BL-7781
Invoice Date: 14/09/2026
Due Date: 01/09/2026
Bill To: Northwind Gadgets Pvt Ltd

Description                 Qty   Unit Price    Amount
Freight Chennai-Mumbai        3      450.00    1450.00
Loading charges               1      120.00     120.00

Subtotal: 1570.00
VAT 5%: 78.50
Total: 1700.00
Currency: INR""",
    },
    "receipt": {
        "label": "Receipt",
        "text": """CAFE BLOOM
Receipt No: R-20931
Date: Sep 21, 2026

Cappuccino                    2        3.50       7.00
Blueberry Muffin              1        2.90       2.90

Subtotal: 9.90
Tax 5%: 0.50
TOTAL: 10.40
Paid by VISA ****4417
Currency: EUR""",
    },
    "purchase_order": {
        "label": "Purchase order (incomplete)",
        "text": """NORTHWIND GADGETS PVT LTD
PURCHASE ORDER
PO Number: PO-55120
Supplier: Delta Components

Description                 Qty   Unit Price    Amount
Li-ion Cell 18650           200        1.85     370.00
Battery Holder 2S            50        0.60      30.00

Subtotal: 400.00
Total: 400.00""",
    },
}

MONEY = r"([\d,]+\.\d{2})"
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
DATE_PATTERNS = [
    ("%Y-%m-%d", r"\d{4}-\d{2}-\d{2}"),
    ("%d/%m/%Y", r"\d{2}/\d{2}/\d{4}"),
    ("%b %d, %Y", r"[A-Z][a-z]{2} \d{1,2}, \d{4}"),
]
CURRENCIES = {"USD": "USD", "INR": "INR", "EUR": "EUR", "GBP": "GBP", "$": "USD", "₹": "INR", "€": "EUR", "£": "GBP"}


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _parse_date(s: str) -> Optional[str]:
    for fmt, _ in DATE_PATTERNS:
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _labeled(text: str, labels: str, value_re: str) -> Optional[str]:
    m = re.search(rf"^\s*(?:{labels})\s*[:#]?\s*[:#]?\s*({value_re})", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def field(value: Any, confidence: float, source: str) -> dict[str, Any]:
    return {"value": value, "confidence": round(confidence, 2), "source": source}


# ---------------------------------------------------------------------------
# Rule-based extractor (mock mode, and the fallback in live mode)
# ---------------------------------------------------------------------------
def extract_rules(text: str) -> dict[str, Any]:
    low = text.lower()
    doc_type = ("purchase_order" if "purchase order" in low or re.search(r"\bpo number\b", low)
                else "receipt" if "receipt" in low else "invoice" if "invoice" in low else "unknown")
    lines = [l for l in text.splitlines() if l.strip()]
    f: dict[str, Any] = {"document_type": field(doc_type, 0.9 if doc_type != "unknown" else 0.3, "keywords")}

    first = lines[0].strip() if lines else ""
    looks_like_company = bool(re.search(r"\b(ltd|llc|inc|pvt|gmbh|co|corp|limited)\b", first, re.I)) or first.isupper()
    f["vendor"] = field(first.title() or None, (0.85 if looks_like_company else 0.6) if first else 0, "letterhead")
    sup = _labeled(text, r"supplier|vendor|from", r"[^\n]+")
    if sup:
        f["vendor"] = field(sup.title(), 0.95, "labeled 'Supplier'")

    num = _labeled(text, r"invoice\s*(?:no\.?|number|#)|receipt\s*(?:no\.?|number|#)|po\s*(?:no\.?|number|#)", r"[A-Z0-9][A-Z0-9\-/]+")
    f["document_number"] = field(num, 0.95 if num else 0, "labeled number" if num else "not found")

    date_re = "|".join(p for _, p in DATE_PATTERNS)
    d = _labeled(text, r"invoice\s*date|date", date_re)
    f["issue_date"] = field(_parse_date(d) if d else None, 0.95 if d else 0, "labeled date" if d else "not found")
    due = _labeled(text, r"due\s*date|payment\s*due", date_re)
    f["due_date"] = field(_parse_date(due) if due else None, 0.95 if due else 0, "labeled due date" if due else "not found")

    bill = _labeled(text, r"bill\s*to|customer", r"[^\n]+")
    f["bill_to"] = field(bill, 0.9 if bill else 0, "labeled 'Bill To'" if bill else "not found")

    cur = _labeled(text, r"currency", r"[A-Z]{3}")
    if not cur:
        sym = next((s for s in ("₹", "€", "£", "$") if s in text), None)
        cur = CURRENCIES.get(sym) if sym else None
    f["currency"] = field(cur, 0.95 if cur else 0, "labeled/symbol" if cur else "not found")

    items = []
    for l in lines:
        m = re.match(rf"^\s*(.+?)\s+(\d+(?:\.\d+)?)\s+{MONEY}\s+{MONEY}\s*$", l)
        if m:
            items.append({"description": m.group(1).strip(), "quantity": _num(m.group(2)),
                          "unit_price": _num(m.group(3)), "amount": _num(m.group(4))})
    f["line_items"] = field(items, 0.9 if items else 0, f"{len(items)} table rows" if items else "no table found")

    sub = _labeled(text, r"sub\s*-?\s*total", MONEY)
    f["subtotal"] = field(_num(sub) if sub else None, 0.95 if sub else 0, "labeled" if sub else "not found")
    tm = re.search(rf"^\s*((?:gst|vat|tax|sales tax)[^:\n]*?)(\d+(?:\.\d+)?)?\s*%?\s*:\s*{MONEY}", text, re.I | re.M)
    f["tax_rate"] = field(float(tm.group(2)) if tm and tm.group(2) else None, 0.9 if tm and tm.group(2) else 0,
                          "from tax label" if tm else "not found")
    f["tax"] = field(_num(tm.group(3)) if tm else None, 0.95 if tm else 0, "labeled" if tm else "not found")
    tot = _labeled(text, r"grand\s*total|total\s*due|amount\s*due|total", MONEY)
    f["total"] = field(_num(tot) if tot else None, 0.95 if tot else 0, "labeled" if tot else "not found")
    return f


# ---------------------------------------------------------------------------
# Validation — applied to any extractor's output
# ---------------------------------------------------------------------------
def validate(f: dict[str, Any]) -> list[dict[str, Any]]:
    v = lambda k: (f.get(k) or {}).get("value")  # noqa: E731
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: Optional[bool], detail: str, severity: str = "error"):
        checks.append({"check": name, "status": "pass" if ok else ("skip" if ok is None else "fail"),
                       "detail": detail, "severity": severity})

    required = ["vendor", "document_number", "issue_date", "total", "currency"]
    missing = [k for k in required if v(k) in (None, "", [])]
    add("Required fields", not missing, "all present" if not missing else "missing: " + ", ".join(missing))

    items = v("line_items") or []
    bad = [i["description"] for i in items if abs(i["quantity"] * i["unit_price"] - i["amount"]) > 0.01]
    add("Line maths (qty × price)", None if not items else not bad,
        "no line items" if not items else ("all rows correct" if not bad else "wrong amount on: " + ", ".join(bad)))

    if items and v("subtotal") is not None:
        s = round(sum(i["amount"] for i in items), 2)
        add("Lines add up to subtotal", abs(s - v("subtotal")) <= 0.01, f"lines {s:.2f} vs subtotal {v('subtotal'):.2f}")
    else:
        add("Lines add up to subtotal", None, "not enough data")

    if v("subtotal") is not None and v("total") is not None:
        exp = round(v("subtotal") + (v("tax") or 0), 2)
        add("Subtotal + tax = total", abs(exp - v("total")) <= 0.01, f"expected {exp:.2f}, document says {v('total'):.2f}")
    else:
        add("Subtotal + tax = total", None, "not enough data")

    if v("tax_rate") is not None and v("subtotal") is not None and v("tax") is not None:
        exp = float(Decimal(str(v("subtotal") * v("tax_rate") / 100)).quantize(Decimal("0.01"), ROUND_HALF_UP))
        add("Tax matches stated rate", abs(exp - v("tax")) <= 0.02, f"{v('tax_rate')}% of subtotal = {exp:.2f}, document says {v('tax'):.2f}")
    else:
        add("Tax matches stated rate", None, "no tax rate stated", "warning")

    if v("issue_date") and v("due_date"):
        add("Due date after issue date", v("due_date") >= v("issue_date"), f"issued {v('issue_date')}, due {v('due_date')}")
    else:
        add("Due date after issue date", None, "no due date", "warning")

    if v("issue_date"):
        add("Issue date not in the future", v("issue_date") <= date.today().isoformat(), f"issued {v('issue_date')}", "warning")
    return checks


def decide(fields: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = [f"{c['check']}: {c['detail']}" for c in checks if c["status"] == "fail" and c["severity"] == "error"]
    low_conf = [k for k, x in fields.items() if x["value"] not in (None, [], "") and x["confidence"] < 0.8]
    if low_conf:
        reasons.append("low-confidence fields: " + ", ".join(low_conf))
    return {"status": "needs_review" if reasons else "auto_approved", "reasons": reasons}


# ---------------------------------------------------------------------------
# LLM extractor (live mode)
# ---------------------------------------------------------------------------
LLM_SCHEMA = ('{"document_type": "invoice|receipt|purchase_order|unknown", "vendor": str, "document_number": str, '
              '"issue_date": "YYYY-MM-DD", "due_date": "YYYY-MM-DD|null", "bill_to": str|null, "currency": "ISO code", '
              '"line_items": [{"description": str, "quantity": number, "unit_price": number, "amount": number}], '
              '"subtotal": number, "tax_rate": number|null, "tax": number|null, "total": number}')


async def extract_llm(text: str, llm: LLM, budget: Budget) -> Optional[dict[str, Any]]:
    raw = await llm.complete(
        system=("Extract structured data from the business document. Copy numbers exactly as printed; do not "
                "correct arithmetic. Use null for anything not present. Return JSON matching: " + LLM_SCHEMA
                + "\n" + UNTRUSTED_POLICY),
        user=wrap_untrusted(sanitize_untrusted(text, MAX_CHARS, budget, "document"), "uploaded document"),
        budget=budget, max_tokens=1200, json_mode=True,
        mock=lambda: "{}",
    )
    data = parse_json(raw)
    if not data:
        return None
    keys = ["document_type", "vendor", "document_number", "issue_date", "due_date", "bill_to", "currency",
            "line_items", "subtotal", "tax_rate", "tax", "total"]
    return {k: field(data.get(k), 0.9 if data.get(k) not in (None, "", []) else 0, "LLM") for k in keys}


def pdf_to_text(b64: str) -> str:
    from pypdf import PdfReader

    raw = base64.b64decode(b64, validate=True)
    if len(raw) > MAX_PDF_BYTES:
        raise ValueError("PDF larger than 2 MB")
    if not raw.startswith(b"%PDF"):
        raise ValueError("not a PDF file")
    reader = PdfReader(io.BytesIO(raw))
    if len(reader.pages) > 10:
        raise ValueError("PDF has more than 10 pages")
    text = "\n".join((p.extract_text(extraction_mode="layout") or "") for p in reader.pages)
    if not text.strip():
        raise ValueError("no text layer found (scanned PDFs need OCR, available in the full version)")
    return text


async def run_extraction(text: str, settings: Settings) -> dict[str, Any]:
    text = CONTROL_CHARS.sub("", text)[:MAX_CHARS]  # keep layout: table columns matter
    budget = Budget(token_limit=min(8000, settings.job_token_budget), llm_call_limit=1, search_limit=0, scrape_limit=0)
    llm = LLM(settings)
    engine = "rules"
    fields = None
    if llm.provider != "mock":
        try:
            fields = await extract_llm(text, llm, budget)
            engine = "llm" if fields else "rules (LLM output unusable)"
        except BudgetExceeded:
            engine = "rules (budget cap)"
    if not fields:
        fields = extract_rules(text)
    checks = validate(fields)
    decision = decide(fields, checks)
    flat = {k: x["value"] for k, x in fields.items()}
    return {"engine": engine, "fields": fields, "checks": checks, "decision": decision, "data": flat,
            "usage": {"tokens_used": budget.tokens_used, "llm_calls": budget.llm_calls}}
