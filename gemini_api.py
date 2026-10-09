"""
Gemini API (Free Tier) -- OCR + Hebrew Translation for Comics
==============================================================

Uses Google's Gemini vision models via the free Google AI Studio tier to
detect comic text (including handwritten / stylised lettering) and translate
it to Hebrew, in a SINGLE API call per page.

The heavy lifting (inpainting, page mirroring, Hebrew RTL rendering) stays
in the existing local pipeline (image_processor.py) -- Gemini only replaces
EasyOCR + Google Translate.

Setup
-----
1. Get a free API key at https://aistudio.google.com  (no credit card)
2. export GEMINI_API_KEY='your_key_here'
3. Optional: export GEMINI_MODEL='gemini-2.5-flash'  (default)

Free-tier notes
---------------
* ~10-15 requests/minute, ~1,500 requests/day (one page = one request).
  This module sleeps + retries automatically on 429 rate-limit errors.
* Google may use free-tier prompts for model training.
* Do NOT enable billing on the Google Cloud project you use for this key,
  or the free tier disappears entirely.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Model fallback chain, ordered QUALITY-FIRST (past mistake: the weakest
# model answered first and silently produced the poor translations the user
# spent days chasing). Free-tier quotas, mid-2026:
#   gemini-2.5-flash      : best quality here; ~10 RPM / ~250 requests per
#                           day - enough for ~4 full albums a day
#   gemini-3.1-flash-lite : newer generation, generous quota - solid fallback
#   gemini-2.5-flash-lite : last resort only
# If GEMINI_MODEL is set it is tried FIRST, then the chain.
# The model that actually answered is published in LAST_MODEL_USED and the
# app writes it into the visible log after every page - no silent switches.
_MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]

# The model that produced the LAST successful page. The windowed exe has no
# console, so print() is invisible there - the app reads this instead.
LAST_MODEL_USED: "str | None" = None

_RETRIES_PER_MODEL = 3
_RETRY_BASE_SLEEP = 10.0

# ---------------------------------------------------------------------------
# Visibility + cancellation (root cause of the "stuck on page 1" report)
# ---------------------------------------------------------------------------
# The windowed exe has NO console, so every print() in this module - rate
# limit waits, model switches, 503 retries - was invisible. Worst case the
# retry chain (3 models x 3 attempts x up to 180s timeout + 90s waits) ran
# for 20+ minutes with the log frozen on "=== page 1/1 ===". Now:
#   * LOG_HOOK   - the app points this at its log window
#   * CANCEL_EVENT - the app's stop flag; every wait / retry checks it
#   * a hard per-page deadline, so a page can never hang forever
LOG_HOOK = print
CANCEL_EVENT = None            # threading.Event set by the app, or None
PAGE_DEADLINE_S = float(os.environ.get("GEMINI_PAGE_DEADLINE", "420"))
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 150

# Images are downscaled before upload. A 2x-rendered PDF page as PNG can be
# 10-25 MB: slow to upload, and above ~20 MB Google rejects it with a 400
# that was then retried 9 times silently. Gemini resizes internally anyway,
# and boxes come back normalized (0-1000), so accuracy is unaffected.
MAX_SIDE = int(os.environ.get("GEMINI_MAX_SIDE", "2048"))


class GeminiCancelled(Exception):
    """Raised when the user pressed Stop while a request was pending."""


def _log(msg: str) -> None:
    try:
        (LOG_HOOK or print)(msg)
    except Exception:  # noqa: BLE001 - logging must never break a page
        print(msg)


def _cancelled() -> bool:
    return bool(CANCEL_EVENT is not None and CANCEL_EVENT.is_set())


def _sleep(seconds: float) -> None:
    """Interruptible sleep: returns early (raising) when Stop is pressed."""
    if seconds <= 0:
        return
    if CANCEL_EVENT is not None:
        if CANCEL_EVENT.wait(seconds):
            raise GeminiCancelled()
    else:
        time.sleep(seconds)


def _encode_for_upload(image: Image.Image) -> "tuple[str, str, int, int]":
    """Downscale (long side <= MAX_SIDE) and JPEG-encode. Returns
    (base64, mime, sent_w, sent_h)."""
    img = image.convert("RGB")
    long_side = max(img.size)
    if long_side > MAX_SIDE:
        f = MAX_SIDE / float(long_side)
        img = img.resize((max(1, round(img.width * f)),
                          max(1, round(img.height * f))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return (base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg",
            img.width, img.height)

# Free tier is ~10-15 requests/minute (rolling window). Pace ourselves so
# multi-page files never trip the per-minute limit in the first place.
_MIN_REQUEST_INTERVAL = 6.5  # seconds between requests (~9/min)
_last_request_ts = 0.0


_last_request_by_key: Dict[int, float] = {}


def _pace(key_idx: int = 0) -> None:
    """Sleep just enough to stay under the free-tier per-minute limit.
    Limits are per PROJECT (= per key), so each key is paced on its own."""
    last = _last_request_by_key.get(key_idx, 0.0)
    wait = _MIN_REQUEST_INTERVAL - (time.time() - last)
    if wait > 0:
        _sleep(wait)
    _last_request_by_key[key_idx] = time.time()


# ---------------------------------------------------------------------------
# Session state: which (key, model) slots are usable right now
# ---------------------------------------------------------------------------
# Lessons from a real 390-page run log and from the Calibre Ebook-Translator
# plugin (which "never chokes" on long books):
# * A model whose DAILY quota is gone was re-tried on EVERY page (a wasted
#   3-13 s 429 per page - over an hour across an album). Exhausted slots are
#   now remembered until the daily reset.
# * Calibre rotates through SEVERAL API keys on RESOURCE_EXHAUSTED instead of
#   degrading. Quotas are per Google project, so N keys = N x the daily
#   quota of the BEST model. The key field accepts several keys.
# * An overloaded (503) model gets a short cool-down instead of being
#   hammered again on the very next page.
_DAILY_EXHAUSTED: Dict[tuple, float] = {}   # (key_idx, model) -> reset ts
_COOLDOWN: Dict[tuple, float] = {}          # (key_idx, model) -> until ts
_UNUSABLE: set = set()                      # (key_idx, model): 404 / 400
_BAD_KEYS: set = set()                      # key_idx rejected as invalid
_OVERLOAD_COOLDOWN_S = 90.0


def _next_daily_reset() -> float:
    """Free-tier daily quotas reset at midnight Pacific time. 08:00 UTC is
    that moment in winter and one hour AFTER it in summer - never early,
    and no tz database needed inside the frozen exe."""
    now = time.time()
    day = 86400.0
    reset = (now // day) * day + 8 * 3600
    return reset if reset > now else reset + day


def _slot_state(slot: tuple) -> str:
    now = time.time()
    if slot[0] in _BAD_KEYS or slot in _UNUSABLE:
        return "dead"
    if _DAILY_EXHAUSTED.get(slot, 0) > now:
        return "daily"
    if _COOLDOWN.get(slot, 0) > now:
        return "cool"
    return "ok"


def reset_session_state() -> None:
    """Forget exhausted/cool-down marks (e.g. after the user edits keys)."""
    _DAILY_EXHAUSTED.clear(); _COOLDOWN.clear(); _UNUSABLE.clear()
    _BAD_KEYS.clear(); _last_request_by_key.clear()


def quota_reset_local_str() -> str:
    return time.strftime("%H:%M", time.localtime(_next_daily_reset()))


class GeminiQuotaExhausted(RuntimeError):
    """Every key x model has used up its daily quota: nothing will work
    until the reset, so the caller should STOP (not fail page after page)."""


def _get_model_chain() -> List[str]:
    chain = list(_MODEL_CHAIN)
    env_model = os.environ.get("GEMINI_MODEL", "").strip()
    if env_model:
        chain = [env_model] + [m for m in chain if m != env_model]
    return chain


def _get_api_keys() -> List[str]:
    """One or more keys: comma / semicolon / whitespace separated."""
    raw = os.environ.get("GEMINI_API_KEY", "")
    keys = [k for k in re.split(r"[\s,;]+", raw) if k]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    if not out:
        _get_api_key()  # raises the helpful message
    return out


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Get a FREE key (no credit card) at: https://aistudio.google.com\n"
            "Then set: export GEMINI_API_KEY='your_key_here'"
        )
    return key


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT = """You are an expert comic translator. Analyze this comic page image.

Find EVERY piece of dialogue / caption / narration text (speech bubbles,
thought bubbles, caption boxes). This INCLUDES stylised, handwritten, large,
or emphasized lettering inside bubbles (e.g. a single shouted word) -- do not
skip any text that is inside a bubble. Ignore background signage, watermarks,
and artist signatures.

The source language may be English, French, or any other language. Translate
DIRECTLY from the source language to Hebrew (never via an intermediate
language) and preserve idioms naturally.

CRITICAL - HEBREW GRAMMATICAL GENDER:
Hebrew verbs, pronouns and adjectives are gendered. USE THE ARTWORK to get
this right for every bubble:
1. Identify the SPEAKER: the bubble's tail points at them. Note their
   apparent gender from the drawing.
2. Identify the ADDRESSEE: usually the other character in the panel, or
   whoever the speaker faces.
3. Conjugate accordingly: a man addressing a woman uses feminine 2nd person
   (e.g. "את", "תעשי", "בואי"); a woman addressing a man uses masculine
   ("אתה", "תעשה", "בוא"). First-person verbs/adjectives agree with the
   SPEAKER's gender (e.g. a woman says "אני בטוחה", a man "אני בטוח").
4. Keep each character's gender CONSISTENT across all bubbles on the page.
5. If gender is genuinely unclear from the art and context, prefer masculine
   (Hebrew's unmarked default) rather than guessing wildly.

For each text region return:
- "box_2d": its bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.
  The box MUST fully cover EVERY letter of that text, from the first to the
  last line, with a small margin. Double-check the box is not shifted.
- "text": the original text, exactly as written
- "hebrew": a natural, fluent Hebrew translation with correct gender
  agreement as described above. Keep the tone and register of the original
  (casual speech stays casual). Keep it CONCISE so it fits in a speech bubble.
- "style": classify the LETTERING style of this text as exactly one of:
  "regular" (normal comic lettering), "bold" (heavy/emphasized strokes),
  "handwritten" (loose, script-like or scribbled), "title" (large display
  lettering, logos, chapter titles).

If one bubble contains multiple lines of the same sentence, return it as ONE
region with one box covering all the lines.

NEVER add nikud (Hebrew vowel points / diacritics). Comics are lettered in
plain unpointed Hebrew - write only the base letters.

FIDELITY RULES (learned from real album comparisons):
- SPLIT SENTENCES: when one sentence continues across two consecutive
  bubbles (e.g. "que pensez-vous de ma..." / "...pharmacie de secours?"),
  translate the two halves as ONE continuous Hebrew sentence split at the
  same point, so each half reads naturally in its own bubble.
- SENTENCE TYPE: preserve it. A question in the source stays a question in
  Hebrew (even when the "?" is drawn as a large graphic symbol); an
  exclamation stays an exclamation.
- NAME WORDPLAY: this series deliberately has some speakers mangle the
  hero's name (e.g. a caller saying a distorted variant). PRESERVE such
  intentional manglings with an equivalent Hebrew distortion. But when the
  source spells the plain name correctly, always use the exact glossary
  transliteration.

ALBUM MEMORY:
You will also maintain a compact "album memory" used for consistency across
pages of the same album. After translating this page, return an UPDATED
memory that merges the previous memory (if given below) with what this page
adds. The memory must contain, in at most 120 words:
- Characters: name, gender, and the exact Hebrew transliteration you chose
  (reuse it verbatim on every page!).
- Recurring terms/places and the Hebrew translation you chose for them.
- One or two lines of ongoing plot context.
Keep it plain text, no JSON inside.

Respond with ONLY a JSON object, no other text:
{"regions": [{"box_2d": [ymin, xmin, ymax, xmax], "text": "...", "hebrew": "...", "style": "..."}],
 "memory": "updated album memory, max 120 words"}

If there is no text, respond with {"regions": [], "memory": "..."}."""


# ---------------------------------------------------------------------------
# Name glossary (glossary.json next to the app / exe)
# ---------------------------------------------------------------------------
# Two past mistakes drove this design:
# * Names drifted between pages (three spellings of the hero in one album),
#   so fixes must apply to BOTH the prompt (prevention) and the output
#   (correction), including the album memory - one wrong transliteration in
#   the memory re-poisons every later page.
# * A frozen exe has a different working directory, so the file is looked
#   up next to the executable first, then next to this source file.

_GLOSSARY_CACHE: "dict | None" = None


def _glossary_path() -> str:
    import sys as _sys
    base = (os.path.dirname(_sys.executable)
            if getattr(_sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "glossary.json")


def _load_glossary() -> dict:
    global _GLOSSARY_CACHE
    if _GLOSSARY_CACHE is not None:
        return _GLOSSARY_CACHE
    data = {"replacements": {}, "names_for_prompt": {}}
    try:
        with open(_glossary_path(), encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw.get("replacements"), dict):
            data["replacements"] = {str(k): str(v)
                                    for k, v in raw["replacements"].items()}
        if isinstance(raw.get("names_for_prompt"), dict):
            data["names_for_prompt"] = {str(k): str(v)
                                        for k, v in raw["names_for_prompt"].items()}
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001 - a broken glossary must not kill pages
        _log(f"  [glossary] failed to load ({exc}) - continuing without it")
    _GLOSSARY_CACHE = data
    return data


# Hebrew combining marks: nikud (U+0591-U+05BD, U+05BF-U+05C7) and the
# cantillation marks. Stripped unconditionally - a model that decides to
# point the text produces lettering no comic reader expects, and prompt
# instructions alone proved unreliable (observed after a model change).
_NIKUD = {c for c in range(0x0591, 0x05C8)} - {0x05BE, 0x05C0, 0x05C3, 0x05C6}


def _strip_nikud(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if ord(ch) not in _NIKUD)


def _apply_glossary(text: str) -> str:
    """Fixed replacements, longest key first so overlapping keys behave."""
    if not text:
        return text
    reps = _load_glossary()["replacements"]
    for wrong in sorted(reps, key=len, reverse=True):
        if wrong in text:
            text = text.replace(wrong, reps[wrong])
    return text


def _glossary_prompt_block() -> str:
    names = _load_glossary()["names_for_prompt"]
    if not names:
        return ""
    lines = "\n".join(f"- {src_name} -> {heb}" for src_name, heb in names.items())
    return (
        "\n\nNAME GLOSSARY (authoritative; use these EXACT Hebrew "
        "transliterations every time the name appears, in text and in the "
        "album memory):\n" + lines
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ocr_and_translate(
    image: Image.Image,
    model: str | None = None,
    context: str | None = None,
    memory: str | None = None,
    return_memory: bool = False,
) -> "List[Dict[str, Any]] | tuple[List[Dict[str, Any]], str]":
    """Detect + translate all comic text in *image* using Gemini (1 request).

    *context* -- optional user-provided background (character names/genders,
    style notes), authoritative for Hebrew gender agreement.

    *memory* -- the ALBUM MEMORY accumulated from previous pages of the same
    album. The model merges it with this page and returns an updated version
    (same single request -- no extra quota). Pass ``return_memory=True`` to
    receive ``(bubbles, updated_memory)`` instead of just ``bubbles``.
    """
    keys = _get_api_keys()

    prompt = _PROMPT + _glossary_prompt_block()
    if context and context.strip():
        prompt += (
            "\n\nADDITIONAL CONTEXT from the user (character names, genders, "
            "style notes) - treat this as authoritative, especially for "
            "Hebrew gender agreement and consistent name transliteration:\n"
            + context.strip()
        )
    if memory and memory.strip():
        prompt += (
            "\n\nPREVIOUS ALBUM MEMORY (accumulated from earlier pages - use "
            "it for consistent names, genders, terms and plot; then return "
            "an updated version):\n" + memory.strip()
        )

    img_b64, mime, sent_w, sent_h = _encode_for_upload(image)
    if (sent_w, sent_h) != image.size:
        _log(f"  [Gemini] תמונה הוקטנה לשליחה: {image.width}x{image.height} "
             f"-> {sent_w}x{sent_h}")

    def _payload_for(model_name: str) -> dict:
        gen = {
            "temperature": 0.2,
            "response_mime_type": "application/json",
            # Bounded output: a runaway / looping answer used to stream for
            # minutes and then fail JSON parsing anyway.
            "maxOutputTokens": 16384,
        }
        if model_name.startswith("gemini-2.5") and "lite" not in model_name:
            # 2.5-flash "thinks" with a dynamic budget by default; on busy
            # pages that alone took 1-3 minutes per request. A fixed budget
            # keeps most of the quality and makes latency predictable.
            gen["thinkingConfig"] = {"thinkingBudget": 2048}
        return {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": img_b64}},
                    {"text": prompt},
                ],
            }],
            "generationConfig": gen,
        }

    models = [model] if model else _get_model_chain()
    errors: List[str] = []
    deadline = time.time() + PAGE_DEADLINE_S
    multi = len(keys) > 1

    def label(m: str, ki: int) -> str:
        return f"{m} [מפתח {ki+1}]" if multi else m

    def finish(data: dict, m: str):
        raw_text = _extract_text(data)
        items, new_memory = _parse_response(raw_text)
        bubbles = _to_bubbles(items, sent_w, sent_h)
        if (sent_w, sent_h) != image.size:
            fx = image.width / float(sent_w)
            fy = image.height / float(sent_h)
            for b in bubbles:
                x1, y1, x2, y2 = b["merged_bbox"]
                b["merged_bbox"] = (int(round(x1 * fx)), int(round(y1 * fy)),
                                    min(image.width, int(round(x2 * fx))),
                                    min(image.height, int(round(y2 * fy))))
        for b in bubbles:
            b["hebrew_text"] = _apply_glossary(
                _strip_nikud(b.get("hebrew_text", "")))
        # Scrub the memory too - one wrong transliteration stored there
        # would re-poison every later page (past mistake).
        out_memory = _apply_glossary(_strip_nikud(new_memory or (memory or "")))
        return bubbles, out_memory

    def other_key_free(m: str, ki: int) -> bool:
        return any(_slot_state((kj, m)) == "ok"
                   for kj in range(len(keys)) if kj != ki)

    def try_slot(m: str, ki: int):
        """Up to _RETRIES_PER_MODEL attempts on one (model, key) slot.
        Returns (bubbles, memory) on success, None to move on."""
        slot = (ki, m)
        url = f"{API_ROOT}/models/{m}:generateContent"
        payload = _payload_for(m)
        n503 = 0
        for attempt in range(_RETRIES_PER_MODEL):
            if _cancelled():
                raise GeminiCancelled()
            if time.time() > deadline:
                errors.append(f"deadline {PAGE_DEADLINE_S:.0f}s exceeded")
                return None
            _pace(ki)
            _log(f"  [Gemini] שולח ל-{label(m, ki)} "
                 f"(ניסיון {attempt+1}/{_RETRIES_PER_MODEL})...")
            t0 = time.time()
            try:
                resp = requests.post(url, params={"key": keys[ki]}, json=payload,
                                     timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            except requests.RequestException as exc:
                errors.append(f"{m}: network error: {exc}")
                _log(f"  [Gemini] שגיאת רשת/זמן ({type(exc).__name__}) "
                     f"אחרי {time.time()-t0:.0f} שנ' - מנסה שוב")
                _sleep(3.0 * (attempt + 1))
                continue
            code = resp.status_code
            _log(f"  [Gemini] תשובה {code} אחרי {time.time()-t0:.0f} שנ'")
            body = resp.text or ""

            if code in (401, 403) or (code == 400 and re.search(
                    r"API_KEY_INVALID|API key not valid|PERMISSION_DENIED", body)):
                _BAD_KEYS.add(ki)
                _log(f"  [Gemini] מפתח {ki+1} לא תקין / ללא הרשאה - לא ישמש יותר")
                if len(_BAD_KEYS) >= len(keys):
                    raise RuntimeError(
                        "כל מפתחות Gemini לא תקינים או ללא הרשאה "
                        f"(HTTP {code}): {body[:200]}")
                return None

            if code in (400, 404):
                # Model not offered to this key / rejects this request shape:
                # retrying cannot help, and it won't change this session.
                _UNUSABLE.add(slot)
                errors.append(f"{m}: {code} {body[:150]}")
                _log(f"  [Gemini] {label(m, ki)} לא זמין ({code}) - מדלג עליו מעכשיו")
                return None

            if code == 429:
                is_daily, retry_s = _parse_429(resp)
                if is_daily:
                    _DAILY_EXHAUSTED[slot] = _next_daily_reset()
                    errors.append(f"{m}: daily quota exhausted")
                    _log(f"  [Gemini] מכסה יומית נגמרה ב-{label(m, ki)} - לא ינוסה "
                         f"שוב עד {quota_reset_local_str()}")
                    return None
                wait = retry_s if retry_s else _RETRY_BASE_SLEEP * (attempt + 1)
                if other_key_free(m, ki):
                    # Calibre-style rotation: another project's key has its
                    # own per-minute budget - use it now instead of waiting.
                    _COOLDOWN[slot] = time.time() + wait
                    _log(f"  [Gemini] מגבלת דקה במפתח {ki+1} - עובר למפתח אחר")
                    return None
                _log(f"  [Gemini] מגבלת קצב לדקה ב-{label(m, ki)} - ממתין {wait:.0f} שנ'")
                _sleep(wait)
                continue

            if code >= 500:
                n503 += 1
                errors.append(f"{m}: HTTP {code}")
                if n503 >= 2:
                    # Overloaded: park it briefly so the next pages don't
                    # queue behind it (it came back 503 on page after page).
                    _COOLDOWN[slot] = time.time() + _OVERLOAD_COOLDOWN_S
                    _log(f"  [Gemini] {label(m, ki)} עמוס - מושהה ל-"
                         f"{_OVERLOAD_COOLDOWN_S:.0f} שנ', עובר הלאה")
                    return None
                wait = 4.0 + 4.0 * attempt
                _log(f"  [Gemini] שרת עמוס/תקלה ({code}) - ממתין {wait:.0f} שנ'")
                _sleep(wait)
                continue

            try:
                resp.raise_for_status()
                return finish(resp.json(), m)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                errors.append(f"{m}: {exc}")
                _log(f"  [Gemini] תשובה לא תקינה מ-{label(m, ki)}: {str(exc)[:120]}")
                _sleep(2.0 * (attempt + 1))
        return None

    global LAST_MODEL_USED
    while True:
        # Quality first: the best model on EVERY key before a weaker model.
        for m in models:
            for ki in range(len(keys)):
                if _slot_state((ki, m)) != "ok":
                    continue
                res = try_slot(m, ki)
                if res is not None:
                    if m != models[0]:
                        _log(f"  [Gemini] הצליח עם מודל גיבוי '{m}'")
                    LAST_MODEL_USED = m
                    bubbles, out_memory = res
                    if return_memory:
                        return bubbles, out_memory
                    return bubbles
                if time.time() > deadline:
                    break

        states = [_slot_state((ki, m)) for m in models for ki in range(len(keys))]
        if states and all(st in ("daily", "dead") for st in states):
            raise GeminiQuotaExhausted(
                "המכסה היומית של כל המודלים" + (" בכל המפתחות" if multi else "")
                + f" נגמרה. היא מתחדשת בערך ב-{quota_reset_local_str()}. "
                "טיפ: מפתח מפרויקט Google חדש ב-AI Studio = מכסה נוספת; "
                "אפשר להזין כמה מפתחות מופרדים בפסיק.")
        cools = [_COOLDOWN[(ki, m)] for m in models for ki in range(len(keys))
                 if _slot_state((ki, m)) == "cool"]
        if not cools or time.time() > deadline:
            break
        wait = min(max(1.0, min(cools) - time.time()), deadline - time.time())
        if wait <= 0:
            break
        _log(f"  [Gemini] כל המודלים הזמינים עמוסים - ממתין {wait:.0f} שנ'")
        _sleep(wait)

    raise RuntimeError(
        "Gemini API failed on all models.\n"
        "Details: " + " | ".join(errors[-4:])
    )


def _parse_429(resp: requests.Response) -> "tuple[bool, float | None]":
    """Inspect a 429 body: is it a DAILY quota (vs per-minute), and did Google
    suggest a retryDelay?

    Google's 429 details include a quota metric id (…_requests_per_day /
    …_requests_per_minute) and often a RetryInfo entry like "retryDelay": "22s".
    """
    try:
        body = resp.text or ""
    except Exception:
        return False, None

    is_daily = bool(re.search(r"per_?day|PerDay|daily", body, re.IGNORECASE))

    retry_s = None
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    if match:
        retry_s = min(float(match.group(1)) + 1.0, 90.0)
    return is_daily, retry_s


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(data: Dict) -> str:
    """Pull the text out of a generateContent response."""
    try:
        candidates = data["candidates"]
        reason = candidates[0].get("finishReason", "")
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts
                       if not p.get("thought"))
        if reason and reason not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            # MAX_TOKENS -> truncated JSON; SAFETY/RECITATION -> blocked.
            raise RuntimeError(f"Gemini stopped early: finishReason={reason}")
        return text
    except (KeyError, IndexError, TypeError):
        # Blocked / empty responses land here
        feedback = data.get("promptFeedback", {})
        raise RuntimeError(f"Unexpected Gemini response: {str(feedback or data)[:300]}")


def _parse_response(text: str) -> "tuple[List[Dict], str]":
    """Parse Gemini output: new schema {"regions": [...], "memory": "..."}
    or the legacy bare array. Returns (items, memory)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if not match:
            raise RuntimeError(f"Gemini did not return JSON: {text[:200]}")
        parsed = json.loads(match.group(0))

    if isinstance(parsed, dict):
        items = parsed.get("regions") or parsed.get("items") or []
        memory = str(parsed.get("memory", "") or "")
        if not isinstance(items, list):
            raise RuntimeError("'regions' is not a list")
        return items, memory[:2000]
    if isinstance(parsed, list):
        return parsed, ""
    raise RuntimeError(f"Unexpected JSON type: {type(parsed).__name__}")


def _parse_json_array(text: str) -> List[Dict]:
    """Parse Gemini output into a JSON array, tolerating code fences."""
    text = text.strip()
    # Strip markdown fences if present despite response_mime_type
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find the outermost [...] block
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise RuntimeError(f"Gemini did not return JSON: {text[:200]}")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected a JSON array, got: {type(parsed).__name__}")
    return parsed


def _contains_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)


def _convert_box(box, order: str, scale: str, img_w: int, img_h: int):
    """One raw box -> pixel (x1, y1, x2, y2) under the given interpretation."""
    a, b, c, d = (float(v) for v in box)
    if order == "xyxy":
        xmin, ymin, xmax, ymax = a, b, c, d
    else:  # "yxyx" - the documented Gemini convention
        ymin, xmin, ymax, xmax = a, b, c, d
    if scale == "norm":
        xmin, xmax = min(max(xmin, 0.0), 1000.0), min(max(xmax, 0.0), 1000.0)
        ymin, ymax = min(max(ymin, 0.0), 1000.0), min(max(ymax, 0.0), 1000.0)
        x1 = xmin / 1000.0 * img_w
        y1 = ymin / 1000.0 * img_h
        x2 = xmax / 1000.0 * img_w
        y2 = ymax / 1000.0 * img_h
    else:  # already pixels
        x1, y1, x2, y2 = xmin, ymin, xmax, ymax
    return (max(0, int(x1)), max(0, int(y1)),
            min(img_w, int(x2)), min(img_h, int(y2)))


def _pick_interpretation(boxes, img_w: int, img_h: int) -> "tuple[str, str]":
    """Auto-detect the (axis order, scale) dialect of THIS response.

    Root cause of a real disaster: this module hardcoded one dialect
    ([ymin, xmin, ymax, xmax], normalized 0-1000). Different Gemini models
    speak different dialects, so switching the model transposed every box -
    Hebrew rendered as narrow columns scattered across the artwork. The
    dialect is therefore MEASURED per response, never assumed:

    * scale: if any coordinate exceeds 1000, the response is in pixels.
    * order: try both; comic text boxes are overwhelmingly WIDER than tall
      and must lie in-bounds with positive area, so the interpretation that
      yields more valid, landscape-shaped boxes wins. Decided once for the
      whole response (a model does not mix dialects mid-reply).
    """
    flat = [float(v) for bx in boxes for v in bx]
    scale = "px" if flat and max(flat) > 1100.0 else "norm"

    def score(order: str) -> int:
        s = 0
        for bx in boxes:
            x1, y1, x2, y2 = _convert_box(bx, order, scale, img_w, img_h)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            s += 1          # valid box
            if w >= h:
                s += 2      # text boxes are landscape far more often than not
        return s

    return ("yxyx" if score("yxyx") >= score("xyxy") else "xyxy"), scale


# Regions whose "hebrew" field came back without a single Hebrew letter on
# the LAST page (the model copied the source text through). They are DROPPED
# rather than typeset - rendering French through the Hebrew renderer was a
# real, observed failure. The app surfaces this count in the visible log.
LAST_NON_HEBREW: int = 0


def _to_bubbles(items: List[Dict], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Convert Gemini items to pipeline bubbles, dialect-proof."""
    global LAST_NON_HEBREW
    LAST_NON_HEBREW = 0
    raw = []
    for item in items:
        box = item.get("box_2d") or item.get("bbox")
        text = str(item.get("text", "")).strip()
        hebrew = str(item.get("hebrew", "")).strip()
        if not box or len(box) != 4 or not hebrew:
            continue
        if not _contains_hebrew(hebrew):
            LAST_NON_HEBREW += 1
            _log(f"  [Gemini] region translated WITHOUT Hebrew "
                  f"('{hebrew[:40]}') - dropped, source stays visible")
            continue
        raw.append((box, text, hebrew, item))
    if not raw:
        return []

    order, scale = _pick_interpretation([r[0] for r in raw], img_w, img_h)
    if (order, scale) != ("yxyx", "norm"):
        _log(f"  [Gemini] box dialect detected: order={order}, scale={scale}")

    bubbles: List[Dict[str, Any]] = []
    for box, text, hebrew, item in raw:
        x1, y1, x2, y2 = _convert_box(box, order, scale, img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        style = str(item.get("style", "regular")).strip().lower()
        if style not in ("regular", "bold", "handwritten", "title"):
            style = "regular"
        bubbles.append({
            "text": text,
            "hebrew_text": hebrew,
            "merged_bbox": (x1, y1, x2, y2),
            "style": style,
        })
    return bubbles


# ---------------------------------------------------------------------------
# Stand-alone CLI (quick test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gemini_api.py <comic_page_image>")
        sys.exit(1)

    img = Image.open(sys.argv[1]).convert("RGB")
    print(f"Sending {sys.argv[1]} ({img.width}x{img.height}) to Gemini...")
    result = ocr_and_translate(img)
    print(f"Found {len(result)} text region(s):")
    for b in result:
        print(f"  {b['merged_bbox']}: '{b['text'][:40]}' -> '{b['hebrew_text'][:40]}'")
