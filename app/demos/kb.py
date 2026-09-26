"""Fictional knowledge base and order data for the support-agent demo ("Northwind Gadgets").

Everything here is invented demo data. In a client deployment this module is replaced by
their docs (Notion, Zendesk, PDFs, a vector DB) and their order system's API.
"""
from __future__ import annotations

from datetime import date, timedelta

COMPANY = "Northwind Gadgets"

KB_DOCS: list[dict[str, str]] = [
    {
        "id": "shipping",
        "title": "Shipping policy",
        "text": (
            "Standard shipping takes 3 to 5 business days within India and is free on orders above ₹999. "
            "Express shipping delivers in 1 to 2 business days for a flat fee of ₹149. "
            "Orders placed before 2 pm on a business day are dispatched the same day. "
            "International shipping is available to 40 countries and takes 7 to 14 business days. "
            "A tracking link is emailed as soon as the order is dispatched."
        ),
    },
    {
        "id": "returns",
        "title": "Returns and refunds",
        "text": (
            "Most items can be returned within 30 days of delivery if they are unused and in original packaging. "
            "Items marked final sale cannot be returned. "
            "Refunds are issued to the original payment method within 5 to 7 business days after the return is inspected. "
            "Return pickup is free for defective items; for other returns a ₹99 pickup fee is deducted from the refund. "
            "To start a return, share your order number and we will arrange a pickup."
        ),
    },
    {
        "id": "warranty",
        "title": "Warranty",
        "text": (
            "All electronics carry a 12-month manufacturer warranty from the date of delivery. "
            "Accessories such as cables and cases carry a 6-month warranty. "
            "The warranty covers manufacturing defects but not physical or water damage. "
            "Warranty claims need the order number and a short video or photo of the issue."
        ),
    },
    {
        "id": "payments",
        "title": "Payments",
        "text": (
            "We accept UPI, credit and debit cards, net banking and cash on delivery for orders up to ₹5,000. "
            "No-cost EMI is available on orders above ₹3,000 with selected banks. "
            "Our team will never ask for your card number, CVV or OTP over chat, email or phone."
        ),
    },
    {
        "id": "cancellation",
        "title": "Order cancellation",
        "text": (
            "Orders can be cancelled free of charge until they are dispatched. "
            "Once an order has been dispatched it cannot be cancelled, but it can be returned after delivery. "
            "Cancelled prepaid orders are refunded within 3 business days."
        ),
    },
    {
        "id": "account",
        "title": "Account and support hours",
        "text": (
            "You can change your delivery address until the order is dispatched. "
            "Our human support team is available Monday to Saturday, 9 am to 7 pm IST. "
            "Support tickets are answered within one business day."
        ),
    },
]


def _orders() -> dict[str, dict]:
    t = date.today()
    return {
        "NW-10231": {"item": "Wireless Earbuds Pro", "status": "delivered", "delivered_on": t - timedelta(days=6),
                     "final_sale": False, "total": 3499},
        "NW-10232": {"item": "USB-C Fast Charger 65W", "status": "shipped", "eta": t + timedelta(days=2),
                     "tracking": "DLV-88412093", "final_sale": False, "total": 1899},
        "NW-10233": {"item": "Smart Watch S2 (clearance)", "status": "delivered", "delivered_on": t - timedelta(days=4),
                     "final_sale": True, "total": 2999},
        "NW-10234": {"item": "Bluetooth Speaker Mini", "status": "processing", "final_sale": False, "total": 1299},
        "NW-10235": {"item": "Laptop Stand Aluminium", "status": "delivered", "delivered_on": t - timedelta(days=41),
                     "final_sale": False, "total": 1599},
    }


def get_order(order_id: str) -> dict | None:
    return _orders().get(order_id.upper())


def sample_order_ids() -> list[str]:
    return list(_orders().keys())
