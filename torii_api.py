"""
Torii Image Translator API Integration
======================================

Wrapper for Torii's professional comic/manga OCR + translation API,
written against the REAL API schema (https://toriitranslate.com/api):

    POST /api/v2/upload   -- full translation. Returns the translated image,
                             the INPAINTED (clean) image, and an array of
                             translated text boxes with coordinates.
    POST /api/v2/ocr      -- OCR. Returns a JSON *array* of paragraph objects,
                             each with a 4-point ``polygon`` (not a bbox dict).
    POST /api/inpaint     -- takes an image + a MASK IMAGE (white = remove).
    POST /api/typeset     -- takes ``text_boxes``: a JSON-stringified array of
                             {x, y, width, height, text, alignment,
                              text_color, stroke_color} (x/y = TOP-LEFT).
    GET  /api/credits     -- credit balance.

To use:
    1. Sign up at https://toriitranslate.com and get an API key
    2. Set the TORII_API_KEY environment variable
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_ROOT = "https://api.toriitranslate.com/api"


def _get_api_key() -> str:
    """Return the Torii API key from environment or raise."""
    key = os.environ.get("TORII_API_KEY", "")
    if not key:
        raise ValueError(
            "TORII_API_KEY environment variable not set.\n"
            "Get a free key at: https://toriitranslate.com\n"
            "Then set: export TORII_API_KEY='your_key_here'"
        )
    return key


def _headers() -> Dict[str, str]:
    """Build request headers with Bearer token."""
    return {"Authorization": f"Bearer {_get_api_key()}"}


def _image_to_png_buffer(image: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _decode_data_url(data_url: str) -> Image.Image:
    """Decode a 'data:image/png;base64,....' Data URL into a PIL image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_data = base64.b64decode(data_url)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def _check_success(resp: requests.Response) -> None:
    """Raise a helpful error if the API signalled failure."""
    resp.raise_for_status()
    if resp.headers.get("success", "true").lower() == "false":
        raise RuntimeError(f"Torii API error: {resp.text[:500]}")


def _polygon_to_bbox(polygon: List[List[float]]) -> Tuple[int, int, int, int]:
    """Convert a 4-point polygon to an axis-aligned (x1, y1, x2, y2) bbox."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# ---------------------------------------------------------------------------
# Full Translation Endpoint (RECOMMENDED)
# ---------------------------------------------------------------------------

def translate_full(
    image: Image.Image,
    target_lang: str = "he",
    translator: str = "gemini-3.1-flash-lite",
    font: str = "NotoSans",
    text_align: str = "auto",
    min_font_size: int = 8,
    bubbles_only: bool = False,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Full comic translation via POST /api/v2/upload.

    Returns a dict with:
        'image'      : PIL image  -- final translated image (text rendered)
        'inpainted'  : PIL image  -- CLEAN image with source text removed
        'text_boxes' : list[dict] -- translated text boxes. Each has (per the
                       API docs): x, y = CENTER point of the box; width,
                       height; text (translated); originalText; textAlign;
                       fillColor; strokeColor; font; etc.

    Notes
    -----
    * 'NotoSans' covers 130 languages including Hebrew; 'WildWords' may not
      include Hebrew glyphs, so NotoSans is the safe default here.
    * Costs ~1+ credit per image.
    """
    data = {
        "target_lang": target_lang,
        "translator": translator,
        "font": font,
        "text_align": text_align,
        "min_font_size": str(min_font_size),
        "bubbles_only": str(bubbles_only).lower(),
    }
    if custom_prompt:
        data["custom_prompt"] = custom_prompt

    files = {"file": ("image.png", _image_to_png_buffer(image), "image/png")}

    resp = requests.post(
        f"{API_ROOT}/v2/upload",
        headers=_headers(),
        data=data,
        files=files,
        timeout=180,
    )
    _check_success(resp)
    result = resp.json()

    out: Dict[str, Any] = {
        "image": _decode_data_url(result["image"]) if result.get("image") else None,
        "inpainted": _decode_data_url(result["inpainted"]) if result.get("inpainted") else None,
        "text_boxes": result.get("text", []) or [],
        "context": result.get("context", ""),
    }
    return out


def translate_image(
    image: Image.Image,
    target_lang: str = "he",
    translator: str = "gemini-3.1-flash-lite",
    font: str = "NotoSans",
    text_align: str = "auto",
    min_font_size: int = 8,
    bubbles_only: bool = False,
) -> Image.Image:
    """Backward-compatible helper: return only the final translated image."""
    result = translate_full(
        image,
        target_lang=target_lang,
        translator=translator,
        font=font,
        text_align=text_align,
        min_font_size=min_font_size,
        bubbles_only=bubbles_only,
    )
    if result["image"] is None:
        raise RuntimeError("Torii API did not return a translated image.")
    return result["image"]


# ---------------------------------------------------------------------------
# OCR Endpoint
# ---------------------------------------------------------------------------

def ocr_detect(image: Image.Image, include_removed: bool = False) -> List[Dict]:
    """Run OCR via POST /api/v2/ocr.

    The API returns a JSON *array* of paragraph objects (NOT a dict), each
    with a 4-point 'polygon'. This function normalises them to:

        {'text': str, 'bbox': (x1, y1, x2, y2), 'confidence': float,
         'polygon': [[x,y]*4], 'direction': str, 'language': str}

    Paragraphs flagged 'removed' (noise/furigana) are skipped unless
    *include_removed* is True.

    Costs 1 credit per image.
    """
    files = {"file": ("image.png", _image_to_png_buffer(image), "image/png")}

    resp = requests.post(
        f"{API_ROOT}/v2/ocr",
        headers=_headers(),
        files=files,
        timeout=120,
    )
    _check_success(resp)

    paragraphs = resp.json()
    if not isinstance(paragraphs, list):
        # Defensive: some error payloads may be dicts
        raise RuntimeError(f"Unexpected OCR response: {str(paragraphs)[:300]}")

    detections: List[Dict] = []
    for p in paragraphs:
        if p.get("removed") and not include_removed:
            continue
        polygon = p.get("polygon", [])
        if not polygon:
            continue
        detections.append({
            "text": p.get("text", ""),
            "bbox": _polygon_to_bbox(polygon),
            "polygon": polygon,
            "confidence": float(p.get("confidence", 0.0)),
            "direction": p.get("direction", "left_to_right"),
            "language": (p.get("language_details") or {}).get("code", ""),
        })
    return detections


# ---------------------------------------------------------------------------
# Inpaint Endpoint
# ---------------------------------------------------------------------------

def inpaint_text(
    image: Image.Image,
    bboxes: List[Tuple[int, int, int, int]],
    dilate_px: int = 4,
) -> Image.Image:
    """Remove text via POST /api/inpaint.

    The real API expects TWO files: the image and a MASK image where white
    areas are inpainted. This builds the mask from the given bboxes
    (slightly dilated so text edges are fully covered).

    Costs 0.02 credits per image.
    """
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for (x1, y1, x2, y2) in bboxes:
        draw.rectangle(
            [x1 - dilate_px, y1 - dilate_px, x2 + dilate_px, y2 + dilate_px],
            fill=255,
        )

    files = {
        "image": ("image.png", _image_to_png_buffer(image), "image/png"),
        "mask": ("mask.png", _image_to_png_buffer(mask), "image/png"),
    }

    resp = requests.post(
        f"{API_ROOT}/inpaint",
        headers=_headers(),
        files=files,
        timeout=120,
    )
    _check_success(resp)
    return _decode_data_url(resp.json()["image"])


# ---------------------------------------------------------------------------
# Typeset Endpoint
# ---------------------------------------------------------------------------

def typeset_text(
    image: Image.Image,
    text_boxes: List[Dict[str, Any]],
    font: str = "NotoSans",
    min_font_size: int = 8,
    stroke_disabled: bool = False,
) -> Image.Image:
    """Render text boxes via POST /api/typeset.

    Parameters
    ----------
    image:
        PIL RGB image (already inpainted / clean).
    text_boxes:
        List of dicts, each with TOP-LEFT coordinates:
            {'x': int, 'y': int, 'width': int, 'height': int,
             'text': str, 'alignment': 'left'|'center'|'right',
             'text_color': '#000000', 'stroke_color': '#ffffff'}
    font:
        Torii font name. 'NotoSans' covers Hebrew; 'WildWords' may not.

    Costs 0.02 credits per image.
    """
    data = {
        "text_boxes": json.dumps(text_boxes, ensure_ascii=False),
        "font": font,
        "min_font_size": str(min_font_size),
        "stroke_disabled": str(stroke_disabled).lower(),
    }
    files = {"file": ("image.png", _image_to_png_buffer(image), "image/png")}

    resp = requests.post(
        f"{API_ROOT}/typeset",
        headers=_headers(),
        data=data,
        files=files,
        timeout=120,
    )
    _check_success(resp)
    return _decode_data_url(resp.json()["image"])


# ---------------------------------------------------------------------------
# Credits check
# ---------------------------------------------------------------------------

def get_credits() -> Dict:
    """Return current credit balance via GET /api/credits."""
    resp = requests.get(
        f"{API_ROOT}/credits",
        headers=_headers(),
        timeout=30,
    )
    _check_success(resp)
    return resp.json()
