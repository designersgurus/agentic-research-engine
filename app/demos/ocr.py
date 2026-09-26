"""Turn uploaded files (PDF, PNG, JPEG, WebP) into layout-preserving text for extraction.

* Text PDFs: the embedded text layer is used directly (fast, exact).
* Images and scanned PDFs: on-device OCR (RapidOCR / ONNX Runtime), then the detected text
  boxes are deskewed and regrouped into lines so table columns survive for the parser.

Safety: file type is decided by magic bytes (never by name or declared type), size and pixel
counts are capped before decoding, and only one OCR job runs at a time to bound memory.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import math
import re
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any

MAX_UPLOAD_BYTES = 5_000_000
MAX_PDF_PAGES = 10
MAX_OCR_PAGES = 2
MAX_PIXELS = 25_000_000        # reject decompression bombs before decoding
OCR_MAX_SIDE = 1100            # keeps peak memory well inside a 512 MB instance
OCR_TIMEOUT_S = 45

_engine: Any = None
_engine_lock = threading.Lock()
_ocr_slot = asyncio.Semaphore(1)


class UnsupportedFile(ValueError):
    pass


@dataclass
class TextResult:
    text: str
    source: str                 # "pdf_text" | "ocr_image" | "ocr_pdf"
    ocr_confidence: float | None = None
    seconds: float = 0.0


def sniff(raw: bytes) -> str:
    if raw.startswith(b"%PDF"):
        return "pdf"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    raise UnsupportedFile("unsupported file type (use PDF, PNG, JPEG or WebP)")


def decode_upload(b64: str) -> bytes:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UnsupportedFile("file is not valid base64") from exc
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UnsupportedFile("file is larger than 5 MB")
    return raw


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import (
                RapidOCR,  # imported lazily: keeps idle memory low
            )

            _engine = RapidOCR(use_cls=False, intra_op_num_threads=1, inter_op_num_threads=1,
                               max_side_len=OCR_MAX_SIDE)
        return _engine


def _load_image(raw: bytes):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as probe:
        w, h = probe.size
    if w * h > MAX_PIXELS:
        raise UnsupportedFile("image resolution is too large")
    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im)           # phone photos are often stored rotated
    im = im.convert("RGB")
    im.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
    return im


_CAMEL_JOIN = re.compile(r"(?<=[a-z]{3})(?=[A-Z][a-z]{2})")


def split_merged_words(text: str) -> str:
    """OCR at low resolution can drop spaces: 'PinecrestDesignStudio' -> 'Pinecrest Design Studio'."""
    return _CAMEL_JOIN.sub(" ", text)


def boxes_to_lines(results: list) -> str:
    """Group OCR boxes into text lines, correcting for a slightly rotated photo."""
    if not results:
        return ""
    angles = []
    for box, _, _ in results:
        (x0, y0), (x1, y1) = box[0], box[1]
        if x1 - x0 > 40:
            angles.append(math.atan2(y1 - y0, x1 - x0))
    tilt = statistics.median(angles) if angles else 0.0
    tan = math.tan(tilt)

    items = []
    for box, text, _ in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x = min(xs)
        cy = sum(ys) / 4 - x * tan                  # de-rotated vertical centre
        items.append((cy, x, max(ys) - min(ys), split_merged_words(text.strip())))
    median_h = statistics.median(i[2] for i in items) or 10
    items.sort(key=lambda i: i[0])

    lines: list[list[tuple]] = []
    for it in items:
        if lines and abs(it[0] - lines[-1][0][0]) <= median_h * 0.6:
            lines[-1].append(it)
        else:
            lines.append([it])
    return "\n".join("   ".join(t for *_, t in sorted(line, key=lambda i: i[1])) for line in lines)


def _ocr_image_sync(im) -> tuple[str, float]:
    import numpy as np

    results, _ = _get_engine()(np.asarray(im))
    results = results or []
    conf = statistics.mean(r[2] for r in results) if results else 0.0
    return boxes_to_lines(results), float(conf)


async def _ocr(images: list) -> tuple[str, float]:
    async with _ocr_slot:                           # one OCR job at a time
        texts, confs = [], []
        for im in images:
            text, conf = await asyncio.wait_for(asyncio.to_thread(_ocr_image_sync, im), OCR_TIMEOUT_S)
            texts.append(text)
            confs.append(conf)
    return "\n\n".join(t for t in texts if t), (statistics.mean(confs) if confs else 0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _pdf_text_and_images(raw: bytes) -> tuple[str, list]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise UnsupportedFile(f"PDF has more than {MAX_PDF_PAGES} pages")
    text = "\n".join((p.extract_text(extraction_mode="layout") or "") for p in reader.pages)
    if text.strip():
        return text, []
    images = []                                     # scanned PDF: OCR the page images
    for page in reader.pages[:MAX_OCR_PAGES]:
        page_images = sorted(page.images, key=lambda i: len(i.data), reverse=True)
        if page_images:
            images.append(_load_image(page_images[0].data))
    if not images:
        raise UnsupportedFile("PDF has no text or images to read")
    return "", images


async def file_to_text(b64: str) -> TextResult:
    raw = decode_upload(b64)
    kind = sniff(raw)
    start = time.monotonic()
    if kind == "pdf":
        text, images = await asyncio.to_thread(_pdf_text_and_images, raw)
        if text:
            return TextResult(text, "pdf_text", None, time.monotonic() - start)
        ocr_text, conf = await _ocr(images)
        return TextResult(ocr_text, "ocr_pdf", round(conf, 2), time.monotonic() - start)
    im = await asyncio.to_thread(_load_image, raw)
    ocr_text, conf = await _ocr([im])
    return TextResult(ocr_text, "ocr_image", round(conf, 2), time.monotonic() - start)
