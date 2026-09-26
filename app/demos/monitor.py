"""Web monitor automation demo: scrape → snapshot → diff → rule-based alerts.

A fictional competitor store ("Gadget Hub") is served by this app at /demos/site?day=N, with a
scripted 30-day history of price changes, stock-outs, launches and delistings. The monitor
scrapes that real HTML with CSS selectors, compares it to the previous day's snapshot and
turns meaningful changes into alerts (Slack / email / webhook payloads, dry run).

The same scraper works on any public page: `scrape()` takes raw HTML plus CSS selectors, and
the API-key-protected /demos/monitor/check-url endpoint fetches a live URL through the
SSRF-safe fetcher.
"""
from __future__ import annotations

import html as html_lib
import re
from typing import Any, Optional

from bs4 import BeautifulSoup

MAX_DAY = 30
STORE = "Gadget Hub"
DEFAULT_SELECTORS = {"item": ".product", "sku": "data-sku", "name": ".name", "price": ".price", "stock": ".stock"}

BASE = {
    "wb-pro": ("Wireless Earbuds Pro", 3299),
    "chg-65": ("USB-C Charger 65W", 1799),
    "spk-mini": ("Bluetooth Speaker Mini", 1199),
    "sw-s2": ("Smart Watch S2", 4499),
    "stand-al": ("Laptop Stand Aluminium", 1499),
    "pb-20k": ("Power Bank 20000mAh", 1999),
}
# day -> list of (sku, action, value)
EVENTS: dict[int, list[tuple[str, str, Any]]] = {
    1: [("wb-pro", "price_pct", -12)],
    2: [("chg-65", "price_pct", 5)],
    3: [("spk-mini", "stock", False)],
    5: [("cam-hd", "launch", ("Webcam HD 1080p", 2299))],
    6: [("spk-mini", "stock", True), ("spk-mini", "price_pct", -10)],
    8: [("sw-s2", "price_pct", -20)],
    10: [("pb-20k", "price_pct", 8)],
    12: [("stand-al", "delist", None)],
    14: [("sw-s2", "price_pct", 25)],
    17: [("wb-pro", "price_pct", -5), ("chg-65", "price_pct", -3)],
    20: [("cam-hd", "price_pct", -15)],
    23: [("chg-65", "stock", False)],
    25: [("chg-65", "stock", True)],
    28: [("pb-20k", "price_pct", -10), ("wb-pro", "price_pct", 4)],
    30: [("spk-mini", "price_pct", -15), ("cam-hd", "stock", False)],
}


def catalog(day: int) -> dict[str, dict[str, Any]]:
    """Ground-truth store state on a given day (what the simulated website renders)."""
    items = {sku: {"name": n, "price": p, "in_stock": True} for sku, (n, p) in BASE.items()}
    for d in range(1, max(0, min(day, MAX_DAY)) + 1):
        for sku, action, val in EVENTS.get(d, []):
            if action == "launch":
                items[sku] = {"name": val[0], "price": val[1], "in_stock": True}
            elif action == "delist":
                items.pop(sku, None)
            elif sku in items and action == "price_pct":
                items[sku]["price"] = int(round(items[sku]["price"] * (1 + val / 100)))
            elif sku in items and action == "stock":
                items[sku]["in_stock"] = val
    return items


def render_site(day: int) -> str:
    rows = "".join(
        f'<div class="product" data-sku="{sku}"><h3 class="name">{html_lib.escape(i["name"])}</h3>'
        f'<span class="price">₹{i["price"]:,}</span>'
        f'<span class="stock {"in" if i["in_stock"] else "out"}">{"In stock" if i["in_stock"] else "Out of stock"}</span></div>'
        for sku, i in catalog(day).items()
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{STORE} (demo store) · day {day}</title>"
        "<style>body{font:15px system-ui;margin:0;background:#fafafa;color:#222}header{background:#3b3b98;color:#fff;padding:14px 16px}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;padding:16px}"
        ".product{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px}.name{font-size:15px;margin:0 0 8px}"
        ".price{display:block;font-weight:700;font-size:18px}.stock.in{color:#1f7a3d}.stock.out{color:#b3261e}"
        ".note{padding:0 16px 16px;color:#666;font-size:13px}</style></head><body>"
        f"<header><strong>{STORE}</strong> · fictional demo store · simulated day {day}</header>"
        f"<main>{rows}</main><p class='note'>This page is a simulated competitor website used to demonstrate "
        "the web-monitor automation. Products and prices are invented.</p></body></html>"
    )


# ---------------------------------------------------------------------------
# Generic scraper + change detection
# ---------------------------------------------------------------------------
def parse_price(text: str) -> Optional[float]:
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)", text or "")
    return float(m.group(1).replace(",", "")) if m else None


def scrape(page_html: str, sel: dict[str, str]) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    out = []
    for i, node in enumerate(soup.select(sel["item"])[:500]):
        name_el = node.select_one(sel["name"])
        price_el = node.select_one(sel["price"])
        stock_el = node.select_one(sel["stock"]) if sel.get("stock") else None
        name = name_el.get_text(strip=True) if name_el else f"item {i + 1}"
        sku = node.get(sel.get("sku", "data-sku")) or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        stock_text = stock_el.get_text(strip=True).lower() if stock_el else ""
        out.append({
            "sku": sku,
            "name": name,
            "price": parse_price(price_el.get_text() if price_el else ""),
            "in_stock": not re.search(r"out of stock|sold out|unavailable", stock_text),
        })
    return out


def diff(prev: list[dict], curr: list[dict], threshold_pct: float) -> tuple[list[dict], list[dict]]:
    p = {i["sku"]: i for i in prev}
    c = {i["sku"]: i for i in curr}
    changes: list[dict] = []
    for sku, now in c.items():
        was = p.get(sku)
        if not was:
            changes.append({"type": "new_product", "sku": sku, "name": now["name"], "price": now["price"]})
            continue
        if was["price"] and now["price"] and now["price"] != was["price"]:
            pct = round((now["price"] - was["price"]) / was["price"] * 100, 1)
            changes.append({"type": "price_drop" if pct < 0 else "price_rise", "sku": sku, "name": now["name"],
                            "old": was["price"], "new": now["price"], "pct": pct})
        if was["in_stock"] != now["in_stock"]:
            changes.append({"type": "back_in_stock" if now["in_stock"] else "out_of_stock", "sku": sku, "name": now["name"]})
    for sku, was in p.items():
        if sku not in c:
            changes.append({"type": "removed", "sku": sku, "name": was["name"]})
    alerts = [ch for ch in changes if not ch["type"].startswith("price_") or abs(ch["pct"]) >= threshold_pct]
    return changes, alerts


def describe(ch: dict) -> str:
    t = ch["type"]
    if t == "price_drop":
        return f"📉 {ch['name']}: ₹{ch['old']:,.0f} → ₹{ch['new']:,.0f} ({ch['pct']}%)"
    if t == "price_rise":
        return f"📈 {ch['name']}: ₹{ch['old']:,.0f} → ₹{ch['new']:,.0f} (+{ch['pct']}%)"
    return {"out_of_stock": f"⛔ {ch['name']} is out of stock",
            "back_in_stock": f"✅ {ch['name']} is back in stock",
            "new_product": f"🆕 New product: {ch['name']} at ₹{ch.get('price') or 0:,.0f}",
            "removed": f"🗑️ {ch['name']} was removed from the store"}[t]


def run_check(day: int, threshold_pct: float, base_url: str = "") -> dict[str, Any]:
    day = max(1, min(int(day), MAX_DAY))
    prev = scrape(render_site(day - 1), DEFAULT_SELECTORS)
    curr = scrape(render_site(day), DEFAULT_SELECTORS)
    changes, alerts = diff(prev, curr, threshold_pct)
    lines = [describe(a) for a in alerts]
    history: dict[str, dict[str, Any]] = {}
    for d in range(0, day + 1):
        for sku, item in catalog(d).items():
            h = history.setdefault(sku, {"name": item["name"], "prices": [None] * (day + 1)})
            h["prices"][d] = item["price"]
    return {
        "day": day,
        "source_url": f"{base_url}/demos/site?day={day}",
        "selectors": DEFAULT_SELECTORS,
        "scraped": curr,
        "changes": [{**c, "text": describe(c), "alert": c in alerts} for c in changes],
        "alerts": lines,
        "notifications": {
            "status": "dry_run",
            "slack": (f"*{STORE} monitor · day {day}* — {len(lines)} alert(s)\n" + "\n".join(f"• {l}" for l in lines))
            if lines else None,
            "email_subject": f"[Price monitor] {len(lines)} change(s) at {STORE}" if lines else None,
            "webhook": {"event": "monitor.changes", "store": STORE, "day": day, "alerts": alerts} if lines else None,
        },
        "history": history,
        "schedule": "In production this check runs on a schedule (e.g. every 6 hours) with the scheduler the app already uses; alerts go to Slack, email or a webhook.",
    }
