"""Document AI extraction demo: invoices / receipts / purchase orders → validated structured data.

Pipeline: text (typed, PDF text layer, or OCR) → extract fields + line items → validate
(arithmetic, required fields, dates) → per-field confidence → auto-approve or human review.

The validators are the point: whether fields come from the rule-based extractor (demo mode) or
from an LLM (live mode), nothing is approved unless the maths and the required fields check out.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

from ..config import Settings
from ..guardrails import (
    CONTROL_RE,
    UNTRUSTED_POLICY,
    Budget,
    BudgetExceeded,
    sanitize_untrusted,
    wrap_untrusted,
)
from ..llm import LLM, parse_json

MAX_CHARS = 20_000
OCR_REVIEW_THRESHOLD = 0.85

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

SYMBOL = r"(?:[$€£₹]|rs\.?|inr|usd|eur|gbp)?\s?"
MONEY = SYMBOL + r"(-?[\d,]*\d\.\d{2})"
FIELD_START = r"(?:^|\s{2,}|\|\s*)"                     # a label starts a line or a column
DATE_FORMATS = {
    "%Y-%m-%d": r"\d{4}-\d{2}-\d{2}",
    "%d/%m/%Y": r"\d{1,2}/\d{1,2}/\d{4}",
    "%d-%m-%Y": r"\d{1,2}-\d{1,2}-\d{4}",
    "%d.%m.%Y": r"\d{1,2}\.\d{1,2}\.\d{4}",
    "%b %d, %Y": r"[A-Z][a-z]{2} \d{1,2}, \d{4}",
    "%B %d, %Y": r"[A-Z][a-z]{3,8} \d{1,2}, \d{4}",
    "%d %b %Y": r"\d{1,2} [A-Z][a-z]{2} \d{4}",
    "%d %B %Y": r"\d{1,2} [A-Z][a-z]{3,8} \d{4}",
}
DATE_VALUE = "|".join(f"(?:{p})" for p in DATE_FORMATS.values())
ISO_CURRENCIES = {"USD", "INR", "EUR", "GBP", "AED", "SGD", "AUD", "CAD", "JPY", "CHF", "CNY", "SAR", "NZD", "ZAR"}
SYMBOL_CURRENCY = {"₹": "INR", "€": "EUR", "£": "GBP", "$": "USD"}
DOC_TITLE = re.compile(r"^(tax\s+)?(invoice|receipt|purchase\s+order|bill|statement|quotation|estimate)\s*$", re.I)
COMPANY_HINT = re.compile(
    r"\b(ltd|llc|llp|inc|pvt|gmbh|co|corp|limited|plc|studio|design|solutions|services|technologies|"
    r"traders|enterprises|industries|logistics|supplies|systems|labs|group|agency|consulting|cafe|store)\b", re.I)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _parse_date(s: str) -> Optional[str]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _labeled(text: str, labels: str, value_re: str) -> Optional[str]:
    """Value that follows a label, where the label begins a line or a column."""
    m = re.search(rf"{FIELD_START}(?:{labels})\s*[:#.\-]?\s*[:#]?\s*({value_re})", text, re.I | re.M)
    if not m:
        return None
    return [g for g in m.groups() if g is not None][-1].strip()  # innermost group: the number without its symbol


def field(value: Any, confidence: float, source: str) -> dict[str, Any]:
    return {"value": value, "confidence": round(confidence, 2), "source": source}


# ---------------------------------------------------------------------------
# Rule-based extractor (demo mode, and the fallback in live mode)
# ---------------------------------------------------------------------------
def _vendor(text: str, lines: list[str]) -> dict[str, Any]:
    labeled = _labeled(text, r"supplier|vendor|seller|from", r"[^\n]+?(?=\s{2,}|$)")
    if labeled:
        return field(labeled.title() if labeled.isupper() else labeled, 0.95, "labeled 'Supplier'")
    for line in lines[:6]:
        first_col = re.split(r"\s{2,}", line.strip())[0]
        if not first_col or DOC_TITLE.match(first_col) or re.search(r"\d{3,}|:", first_col):
            continue
        name = first_col.title() if first_col.isupper() else first_col
        return field(name, 0.85 if COMPANY_HINT.search(first_col) or first_col.isupper() else 0.6, "letterhead")
    return field(None, 0, "not found")


def extract_rules(text: str) -> dict[str, Any]:
    low = text.lower()
    doc_type = ("purchase_order" if "purchase order" in low or re.search(r"\bpo\s*(number|no|#)", low)
                else "receipt" if "receipt" in low else "invoice" if "invoice" in low else "unknown")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    f: dict[str, Any] = {"document_type": field(doc_type, 0.9 if doc_type != "unknown" else 0.3, "keywords")}
    f["vendor"] = _vendor(text, lines)

    num = _labeled(text, r"invoice\s*(?:no\.?|number|num|#)|receipt\s*(?:no\.?|number|#)|po\s*(?:no\.?|number|#)|"
                         r"bill\s*(?:no\.?|number)", r"[A-Z0-9][A-Z0-9\-/]*\d[A-Z0-9\-/]*")
    f["document_number"] = field(num, 0.95 if num else 0, "labeled number" if num else "not found")

    issued = _labeled(text, r"invoice\s*date|issue\s*date|date\s*of\s*issue|bill\s*date|receipt\s*date|date", DATE_VALUE)
    f["issue_date"] = field(_parse_date(issued) if issued else None, 0.95 if issued else 0,
                            "labeled date" if issued else "not found")
    due = _labeled(text, r"due\s*date|payment\s*due|due", DATE_VALUE)
    f["due_date"] = field(_parse_date(due) if due else None, 0.95 if due else 0, "labeled due date" if due else "not found")

    bill = _labeled(text, r"bill(?:ed)?\s*to|customer|client|sold\s*to", r"[^\n]+?(?=\s{2,}|$)")
    f["bill_to"] = field(bill, 0.9 if bill else 0, "labeled 'Bill To'" if bill else "not found")

    cur = (_labeled(text, r"currency", r"[A-Za-z]{3}\b") or "").upper()
    if cur in ISO_CURRENCIES:
        f["currency"] = field(cur, 0.95, "labeled")
    else:
        sym = next((s for s in SYMBOL_CURRENCY if s in text), None)
        f["currency"] = field(SYMBOL_CURRENCY[sym], 0.85, f"symbol '{sym}'") if sym else field(None, 0, "not found")

    items = []
    for ln in lines:
        m = re.match(rf"^\s*(.+?)\s+(\d+(?:\.\d+)?)\s+{MONEY}\s+{MONEY}\s*$", ln, re.I)
        if m and not re.search(r"total|tax|gst|vat", m.group(1), re.I):
            items.append({"description": re.sub(r"\s{2,}", " ", m.group(1).strip()), "quantity": _num(m.group(2)),
                          "unit_price": _num(m.group(3)), "amount": _num(m.group(4))})
    f["line_items"] = field(items, 0.9 if items else 0, f"{len(items)} table rows" if items else "no table found")

    sub = _labeled(text, r"sub\s*-?\s*total", MONEY)
    f["subtotal"] = field(_num(sub) if sub else None, 0.95 if sub else 0, "labeled" if sub else "not found")
    tm = re.search(rf"{FIELD_START}(?:gst|vat|sales\s*tax|tax)\b[^:\n\d%]*?(?:\(?\s*(\d+(?:\.\d+)?)\s*%\s*\)?)?"
                   rf"[^:\n\d]*?[:\-]?\s*{MONEY}", text, re.I | re.M)
    f["tax_rate"] = field(float(tm.group(1)) if tm and tm.group(1) else None, 0.9 if tm and tm.group(1) else 0,
                          "from tax label" if tm and tm.group(1) else "not found")
    f["tax"] = field(_num(tm.group(2)) if tm else None, 0.95 if tm else 0, "labeled" if tm else "not found")
    tot = _labeled(text, r"grand\s*total|total\s*due|amount\s*due|balance\s*due|total\s*amount|total", MONEY)
    f["total"] = field(_num(tot) if tot else None, 0.95 if tot else 0, "labeled" if tot else "not found")
    return f


# ---------------------------------------------------------------------------
# Validation — applied to any extractor's output
# ---------------------------------------------------------------------------
def _round2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def validate(f: dict[str, Any]) -> list[dict[str, Any]]:
    def v(key: str) -> Any:
        return (f.get(key) or {}).get("value")

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: Optional[bool], detail: str, severity: str = "error") -> None:
        status = "skip" if ok is None else ("pass" if ok else "fail")
        checks.append({"check": name, "status": status, "detail": detail, "severity": severity})

    missing = [k for k in ("vendor", "document_number", "issue_date", "total", "currency") if v(k) in (None, "", [])]
    add("Required fields", not missing, "all present" if not missing else "missing: " + ", ".join(missing))

    items = v("line_items") or []
    bad = [i["description"] for i in items if abs(i["quantity"] * i["unit_price"] - i["amount"]) > 0.01]
    add("Line maths (qty × price)", None if not items else not bad,
        "no line items" if not items else ("all rows correct" if not bad else "wrong amount on: " + ", ".join(bad)))

    if items and v("subtotal") is not None:
        total_lines = _round2(sum(i["amount"] for i in items))
        add("Lines add up to subtotal", abs(total_lines - v("subtotal")) <= 0.01,
            f"lines {total_lines:.2f} vs subtotal {v('subtotal'):.2f}")
    else:
        add("Lines add up to subtotal", None, "not enough data")

    if v("subtotal") is not None and v("total") is not None:
        expected = _round2(v("subtotal") + (v("tax") or 0))
        add("Subtotal + tax = total", abs(expected - v("total")) <= 0.01,
            f"expected {expected:.2f}, document says {v('total'):.2f}")
    else:
        add("Subtotal + tax = total", None, "not enough data")

    if None not in (v("tax_rate"), v("subtotal"), v("tax")):
        expected = _round2(v("subtotal") * v("tax_rate") / 100)
        add("Tax matches stated rate", abs(expected - v("tax")) <= 0.02,
            f"{v('tax_rate')}% of subtotal = {expected:.2f}, document says {v('tax'):.2f}")
    else:
        add("Tax matches stated rate", None, "no tax rate stated", "warning")

    if v("issue_date") and v("due_date"):
        add("Due date after issue date", v("due_date") >= v("issue_date"), f"issued {v('issue_date')}, due {v('due_date')}")
    else:
        add("Due date after issue date", None, "no due date", "warning")

    if v("issue_date"):
        add("Issue date not in the future", v("issue_date") <= date.today().isoformat(),
            f"issued {v('issue_date')}", "warning")
    return checks


def decide(fields: dict[str, Any], checks: list[dict[str, Any]], ocr_confidence: Optional[float] = None) -> dict:
    reasons = [f"{c['check']}: {c['detail']}" for c in checks if c["status"] == "fail" and c["severity"] == "error"]
    low_conf = [k for k, x in fields.items() if x["value"] not in (None, [], "") and x["confidence"] < 0.8]
    if low_conf:
        reasons.append("low-confidence fields: " + ", ".join(low_conf))
    if ocr_confidence is not None and ocr_confidence < OCR_REVIEW_THRESHOLD:
        reasons.append(f"image quality: OCR confidence {ocr_confidence:.2f} is below {OCR_REVIEW_THRESHOLD}")
    return {"status": "needs_review" if reasons else "auto_approved", "reasons": reasons}


# ---------------------------------------------------------------------------
# LLM extractor (live mode)
# ---------------------------------------------------------------------------
LLM_KEYS = ["document_type", "vendor", "document_number", "issue_date", "due_date", "bill_to", "currency",
            "line_items", "subtotal", "tax_rate", "tax", "total"]
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
    return {k: field(data.get(k), 0.9 if data.get(k) not in (None, "", []) else 0, "LLM") for k in LLM_KEYS}


async def run_extraction(text: str, settings: Settings, ocr_confidence: Optional[float] = None) -> dict[str, Any]:
    text = CONTROL_RE.sub("", text)[:MAX_CHARS]      # keep layout: table columns matter
    budget = Budget(token_limit=min(8000, settings.job_token_budget), llm_call_limit=1, search_limit=0, scrape_limit=0)
    llm = LLM(settings)
    engine, fields = "rules", None
    if llm.provider != "mock":
        try:
            fields = await extract_llm(text, llm, budget)
            engine = "llm" if fields else "rules (LLM output unusable)"
        except BudgetExceeded:
            engine = "rules (budget cap)"
    if not fields:
        fields = extract_rules(text)
    checks = validate(fields)
    return {
        "engine": engine,
        "fields": fields,
        "checks": checks,
        "decision": decide(fields, checks, ocr_confidence),
        "data": {k: x["value"] for k, x in fields.items()},
        "usage": {"tokens_used": budget.tokens_used, "llm_calls": budget.llm_calls},
    }
