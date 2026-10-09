#!/usr/bin/env python3
"""
Comic Translator to Hebrew -- Windows Desktop App (tkinter)
===========================================================

Native window UI designed for packaging as a single .exe with PyInstaller.

Features
--------
* API key fields (Gemini / Torii) saved to %APPDATA%\\ComicTranslatorHebrew\\config.json
* Progress bar with percentage + per-page status
* Live log window
* Modes: Gemini (free) / Torii (paid). The EasyOCR local mode is intentionally
  excluded to keep the .exe small (PyTorch would add ~3GB).

Build the exe (run build_exe.bat, or manually):
    pyinstaller --onefile --noconsole --name ComicTranslatorHebrew app.py
"""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Pipeline modules (must sit next to this file / be bundled by PyInstaller)
import comic_loader
import image_processor
import gemini_api

try:
    import torii_api
    _HAS_TORII = True
except ImportError:
    _HAS_TORII = False

import cv2
import numpy as np
from PIL import Image

# ============================================================
#  שנה כאן את שם התוכנה (מופיע בכותרת החלון ובחלון האודות).
#  את שם קובץ ה-EXE משנים בקובץ ComicTranslatorHebrew.spec (EXE_NAME).
# ============================================================
APP_NAME = "Comic Translator to Hebrew"
APP_VERSION = "1.3"
CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "ComicTranslatorHebrew"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Fonts folder sits NEXT TO the exe/script - drop Hebrew .ttf/.otf files there.
import io
import re
import sys
APP_DIR = Path(sys.argv[0]).resolve().parent
FONTS_DIR = APP_DIR / "fonts"


def resource_path(name: str) -> Path:
    """Path to a bundled resource: works both when running as a script and
    inside a PyInstaller onefile exe (files unpack to sys._MEIPASS)."""
    base = Path(getattr(sys, "_MEIPASS", APP_DIR))
    return base / name

STYLE_KEYS = ["regular", "bold", "handwritten", "title"]
STYLE_LABELS = {"regular": "רגיל", "bold": "מודגש",
                "handwritten": "כתב יד", "title": "כותרת"}
AUTO_FONT = "(ברירת מחדל)"


def discover_fonts() -> dict:
    """Map font display-name -> full path for every .ttf/.otf in ./fonts."""
    fonts = {}
    if FONTS_DIR.is_dir():
        for p in sorted(FONTS_DIR.glob("*")):
            if p.suffix.lower() in (".ttf", ".otf"):
                fonts[p.stem] = str(p)
    return fonts


ABOUT_TEXT = """Comic Translator to Hebrew - גרסה {ver}

מה התוכנה עושה?
מתרגמת עמודי קומיקס לעברית מקצה לקצה: זיהוי הטקסט בבועות (OCR),
תרגום לעברית, מחיקת הטקסט המקורי, היפוך ראי של העמוד לקריאה
מימין-לשמאל, ורינדור העברית לתוך הבועות.

מנועים:
• Gemini (חינם) - מפתח חינמי מ-aistudio.google.com. מכסה של מאות
  עמודים ביום. התוכנה מווסתת קצב אוטומטית.
• Torii (בתשלום) - toriitranslate.com, איכות מחיקת-טקסט מיטבית.
  כ-1 קרדיט לעמוד.

פורמטים נתמכים: PDF, CBZ, CBR, JPG, PNG, WEBP, GIF.
פלט: תמונות / CBZ / PDF, בתיקיית output ליד קובץ המקור.

התאמת פונטים לסגנון הכתב:
שים קובצי פונט עבריים (‎.ttf / ‎.otf) בתיקיית fonts שליד התוכנה,
לחץ "רענן", ומפה כל סגנון כתב (רגיל / מודגש / כתב יד / כותרת)
לפונט המתאים. הבינה המלאכותית מזהה את סגנון הכתב המקורי של כל
בועה, והעברית תרונדר בפונט שבחרת לאותו סגנון.
המלצות לפונטים חינמיים: משפחת Noto Sans Hebrew (רגיל+מודגש),
Amatic / Gveret Levin (כתב יד), Secular One (כותרות).

טיפים:
• עמוד ראשון של אלבום ארוך לוקח ~7 שניות בגלל ויסות הקצב - זה תקין.
• אפשר לעצור באמצע: כל מה שתורגם כבר שמור בתיקיית output.
• המפתחות נשמרים במחשב שלך בלבד (%APPDATA%\\ComicTranslatorHebrew).

הפרויקט נבנה בעזרת Claude של Anthropic.
""".format(ver=APP_VERSION)


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _keep_awake(enable: bool) -> None:
    """Prevent Windows from sleeping while a translation is running.

    Uses SetThreadExecutionState; harmless no-op on other platforms or on
    failure. Display may still turn off - only system sleep is blocked.
    """
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page processing (same pipeline as main.py, without Gradio)
# ---------------------------------------------------------------------------

# The page-editor template, zlib+base64, baked into the app itself.
# Root cause fix for a real failure: the external template file went missing
# on the user's machine (partial extraction / frozen-exe temp dir) and the
# editor export silently skipped every page. An embedded copy means the
# missing-file failure class no longer exists; an external file, when
# present, still wins so the template stays user-customizable.
_EMBEDDED_EDITOR_TEMPLATE = (
    "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"
)


def _load_editor_template() -> str:
    candidates = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:  # onefile bundles unpack here, NOT next to the exe
            candidates.append(os.path.join(meipass, "editor_template.html"))
        candidates.append(os.path.join(os.path.dirname(sys.executable),
                                       "editor_template.html"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "editor_template.html"))
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            continue
    import zlib as _zlib
    import base64 as _b64
    return _zlib.decompress(
        _b64.b64decode(_EMBEDDED_EDITOR_TEMPLATE)).decode("utf-8")


def _font_family_name(font_path: str) -> str:
    """Read the FAMILY name out of a font file.

    The editor is a browser: it matches fonts by FAMILY name ("Amatic SC"),
    while the app stores a FILE path ("...\\AmaticSC-Bold.ttf"). Passing the
    file stem produced the wrong font in every exported editor page (real
    report). PIL exposes the family recorded inside the font itself, which
    is exactly the name the browser looks for. Falls back to a cleaned-up
    file stem when the file cannot be read.
    """
    try:
        from PIL import ImageFont as _IF
        family, _style = _IF.truetype(font_path, 20).getname()
        if family:
            return str(family).strip()
    except Exception:  # noqa: BLE001 - never break an export over a font name
        pass
    # Windows paths reaching a POSIX host keep their backslashes, so
    # os.path.basename does nothing - split on both separators.
    raw = str(font_path or "").replace("\\", "/")
    stem = os.path.splitext(raw.rsplit("/", 1)[-1])[0]
    stem = re.sub(r"[-_ ]+(bold|italic|regular|light|medium|black)$", "",
                  stem, flags=re.I)
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem).strip() or "Arial"


def _font_is_bold(font_path: str) -> bool:
    try:
        from PIL import ImageFont as _IF
        _family, style = _IF.truetype(font_path, 20).getname()
        return bool(style) and "bold" in str(style).lower()
    except Exception:  # noqa: BLE001
        raw = str(font_path or "").replace("\\", "/")
        return bool(re.search(r"bold", raw.rsplit("/", 1)[-1], re.I))


def write_page_editor(out_dir, page_no: int, clean_img, bubbles, log,
                      font_map: dict | None = None) -> None:
    """Write page_NNN.editor.html - a standalone, double-clickable editor.

    Design decisions (each answering a real past failure):
    * SELF-CONTAINED: the clean page is embedded as base64 and the boxes as
      JSON inside one html file. No sibling files to lose, no mixed-version
      folders, no file-picking - the exact delivery failures we hit before.
    * Exported BEFORE rendering: the embedded page is the mirrored, cleaned
      plate, so a bad automatic typeset can always be redone by hand.
    * Never fatal: any failure here logs one line and the page still saves.
    """
    try:
        import base64 as _b64
        tpl = _load_editor_template()  # external file or the embedded copy
        buf = io.BytesIO()
        clean_img.save(buf, format="PNG")
        img_url = "data:image/png;base64," + _b64.b64encode(
            buf.getvalue()).decode("ascii")
        name = f"page_{page_no:03d}"
        # style-key -> (browser family name, bold?) resolved ONCE per page
        fam_cache: dict = {}
        for style_key, path in (font_map or {}).items():
            if path:
                fam_cache[style_key] = (_font_family_name(path),
                                        _font_is_bold(path))
        def _box(b):
            # The LAYOUT box (room inside the bubble) when known, otherwise
            # the tight source-text box - same area the renderer used.
            return tuple(int(v) for v in (b.get("layout_bbox") or b["merged_bbox"]))

        def _hex(b):
            c = b.get("ink_rgb")
            return "#%02x%02x%02x" % tuple(int(v) for v in c) if c else "#000000"

        payload = {
            "page_name": name,
            "width": clean_img.width,
            "height": clean_img.height,
            "boxes": [{
                "x": _box(b)[0],
                "y": _box(b)[1],
                "w": _box(b)[2] - _box(b)[0],
                "h": _box(b)[3] - _box(b)[1],
                "color": _hex(b),
                "text": b.get("hebrew_text", ""),
                "style": b.get("style", "regular"),
                "font": fam_cache.get(b.get("style", "regular"),
                                      fam_cache.get("regular",
                                                    ("Arial", True)))[0],
                "bold": fam_cache.get(b.get("style", "regular"),
                                      fam_cache.get("regular",
                                                    ("Arial", True)))[1],
            } for b in bubbles],
        }
        js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
        html = (tpl.replace("__PAGE_TITLE__", name)
                   .replace("__PAGE_IMAGE__", img_url)
                   .replace("__PAGE_DATA__", js))
        out = Path(out_dir) / f"{name}.editor.html"
        out.write_text(html, encoding="utf-8")
        log(f"  [עורך] נשמר {out.name} - פתיחה בדפדפן לעריכה ידנית")
    except Exception as exc:  # noqa: BLE001 - editor export must never kill a page
        log(f"  (ייצוא לעורך נכשל: {exc})")


def process_page_gemini(image: Image.Image, log, font_map: dict | None = None,
                        clean_bubbles: bool = True,
                        context: str | None = None,
                        memory: str = "",
                        editor_export: dict | None = None) -> "tuple[Image.Image, str]":
    bubbles, memory = gemini_api.ocr_and_translate(
        image, context=context, memory=memory or None, return_memory=True)
    reader_model = getattr(gemini_api, "LAST_MODEL_USED", None)
    if reader_model:
        log(f"  [מודל קריאה] {reader_model}")
    non_heb = getattr(gemini_api, "LAST_NON_HEBREW", 0)
    if non_heb:
        log(f"  אזהרה: {non_heb} אזורים חזרו מהמנוע ללא עברית - "
            "נפסלו והמקור נשאר בהם")
    if not bubbles:
        log("  לא נמצא טקסט בעמוד - רק היפוך.")
        return image.transpose(Image.FLIP_LEFT_RIGHT), memory
    log(f"  {len(bubbles)} אזורי טקסט תורגמו")

    # Style-matched fonts: Gemini classified each bubble's lettering style;
    # map it to the user's chosen Hebrew font (falls back to default).
    if font_map:
        for b in bubbles:
            fp = font_map.get(b.get("style", "regular"))
            if fp:
                b["font_path"] = fp

    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    w = img_bgr.shape[1]
    refined, text_mask, interiors = image_processor.refine_text_regions(img_bgr, bubbles)
    if clean_bubbles:
        img_bgr = image_processor.clean_bubble_interiors(img_bgr, interiors)
        log("  רקע הבועות נוקה (צבע אחיד)")
    # Flat bubbles: paint the paper colour over the letters (no inpaint
    # blotches). Only what's left (textured / untrusted areas) is inpainted.
    img_bgr, rest_mask = image_processor.flat_fill_text(img_bgr, text_mask, interiors)
    cleaned = image_processor.remove_text_by_mask(img_bgr, rest_mask)
    n_lay = sum(1 for b in refined if b.get("layout_bbox"))
    log(f"  מיקום: {n_lay}/{len(refined)} בועות עם מסגרת-בועה מלאה, "
        f"{len(refined)-n_lay} לפי תיבת הטקסט המקורית")
    mirrored = image_processor.mirror_image(cleaned)
    mbubbles = image_processor.mirror_bubbles(refined, w)
    if editor_export is not None:
        # The mirrored CLEAN plate + mirrored boxes, i.e. exactly what the
        # renderer is about to draw on - so hand-editing starts from the
        # same truth the automatic pass used.
        editor_export["clean"] = Image.fromarray(
            cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB))
        editor_export["bubbles"] = [dict(b) for b in mbubbles]
    result = image_processor.render_hebrew_text(mirrored, mbubbles)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), memory


def process_page_torii(image: Image.Image, log) -> Image.Image:
    MIN_FONT, BOX_SCALE = 5, 0.88
    result = torii_api.translate_full(
        image, target_lang="he", translator="gemini-3.1-flash-lite",
        font="NotoSans", text_align="right", min_font_size=MIN_FONT,
    )
    boxes, inpainted = result["text_boxes"], result["inpainted"]
    if inpainted is None:
        log("  אזהרה: Torii לא החזיר תמונה נקייה - מחזיר ללא היפוך.")
        return result["image"] or image
    if not boxes:
        return inpainted.transpose(Image.FLIP_LEFT_RIGHT)

    mirrored = inpainted.transpose(Image.FLIP_LEFT_RIGHT)
    W = inpainted.width
    tboxes = []
    for b in boxes:
        cx, cy = float(b["x"]), float(b["y"])
        w0, h0 = float(b["width"]) * BOX_SCALE, float(b["height"]) * BOX_SCALE
        tboxes.append({
            "x": int(round((W - cx) - w0 / 2)), "y": int(round(cy - h0 / 2)),
            "width": int(round(w0)), "height": int(round(h0)),
            "text": b.get("text", ""), "alignment": "right",
            "text_color": b.get("fillColor", "#000000"),
            "stroke_color": b.get("strokeColor", "#ffffff"),
        })
    return torii_api.typeset_text(mirrored, tboxes, font="NotoSans",
                                  min_font_size=MIN_FONT)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(APP_NAME)
        root.geometry("720x640")
        root.minsize(640, 560)

        # Window + taskbar icon: uses app_icon.ico if present (bundled in the
        # exe, or sitting next to app.py when running as a script).
        for icon_candidate in (resource_path("app_icon.ico"),
                               APP_DIR / "app_icon.ico"):
            try:
                if icon_candidate.is_file():
                    root.iconbitmap(str(icon_candidate))
                    break
            except tk.TclError:
                pass

        self.cfg = load_config()
        self.msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_flag = threading.Event()

        pad = {"padx": 10, "pady": 4}
        frm = ttk.Frame(root)
        frm.pack(fill="both", expand=True)

        # --- File selection
        row = ttk.Frame(frm); row.pack(fill="x", **pad)
        ttk.Label(row, text="קובץ קומיקס:").pack(side="right")
        self.file_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.file_var).pack(side="right", fill="x",
                                                        expand=True, padx=6)
        ttk.Button(row, text="בחר...", command=self.pick_file).pack(side="right")

        # --- API keys
        keys = ttk.LabelFrame(frm, text="מפתחות API")
        keys.pack(fill="x", **pad)

        krow1 = ttk.Frame(keys); krow1.pack(fill="x", padx=8, pady=3)
        ttk.Label(krow1, text="Gemini (אפשר כמה מפתחות, מופרדים בפסיק):").pack(side="right")
        self.gemini_var = tk.StringVar(value=self.cfg.get("gemini_key", ""))
        ttk.Entry(krow1, textvariable=self.gemini_var, show="*").pack(
            side="right", fill="x", expand=True, padx=6)

        krow2 = ttk.Frame(keys); krow2.pack(fill="x", padx=8, pady=3)
        ttk.Label(krow2, text="Torii (בתשלום - toriitranslate.com):").pack(side="right")
        self.torii_var = tk.StringVar(value=self.cfg.get("torii_key", ""))
        ttk.Entry(krow2, textvariable=self.torii_var, show="*").pack(
            side="right", fill="x", expand=True, padx=6)

        # --- Translation context (character names & genders)
        ctx_frame = ttk.LabelFrame(
            frm, text="הקשר לתרגום (אופציונלי): שמות דמויות ומינן, הערות סגנון")
        ctx_frame.pack(fill="x", **pad)
        self.ctx_var = tk.StringVar(value=self.cfg.get("context", ""))
        ctx_entry = ttk.Entry(ctx_frame, textvariable=self.ctx_var, justify="right")
        ctx_entry.pack(fill="x", padx=8, pady=4)
        ttk.Label(ctx_frame, foreground="#666",
                  text='לדוגמה: מורטימר - גבר, בלייק - גבר, נדיה - אישה. עוזר לדיוק לשון זכר/נקבה.'
                  ).pack(anchor="e", padx=8, pady=(0, 4))

        # --- Mode + output format
        opts = ttk.Frame(frm); opts.pack(fill="x", **pad)
        self.mode_var = tk.StringVar(value=self.cfg.get("mode", "gemini"))
        ttk.Label(opts, text="מנוע:").pack(side="right")
        ttk.Radiobutton(opts, text="Gemini (חינם)", value="gemini",
                        variable=self.mode_var).pack(side="right", padx=4)
        ttk.Radiobutton(opts, text="Torii (איכותי, בתשלום)", value="torii",
                        variable=self.mode_var).pack(side="right", padx=4)

        self.clean_var = tk.BooleanVar(value=self.cfg.get("clean_bubbles", True))
        ttk.Checkbutton(opts, text="נקה רקע בועות (מומלץ לסריקות ישנות)",
                        variable=self.clean_var).pack(side="left", padx=4)

        # Resume skips pages already in output/ - so after changing fonts /
        # cleaning settings, re-running the same file changed NOTHING and
        # looked like the new settings were ignored. This forces a redo.
        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="תרגם מחדש גם עמודים קיימים",
                        variable=self.force_var).pack(side="left", padx=4)

        ttk.Label(opts, text="    פלט:").pack(side="right")
        self.fmt_var = tk.StringVar(value=self.cfg.get("format", "Images"))
        for fmt in ("Images", "CBZ", "PDF"):
            ttk.Radiobutton(opts, text=fmt, value=fmt,
                            variable=self.fmt_var).pack(side="right", padx=4)

        # --- Hebrew fonts by lettering style
        fonts_frame = ttk.LabelFrame(
            frm, text="פונטים בעברית לפי סגנון הכתב המקורי (תיקיית fonts)")
        fonts_frame.pack(fill="x", **pad)

        self.fonts = discover_fonts()
        saved_map = self.cfg.get("font_map", {})
        self.font_vars: dict[str, tk.StringVar] = {}

        grid = ttk.Frame(fonts_frame); grid.pack(fill="x", padx=8, pady=4)
        self.font_combos: dict[str, ttk.Combobox] = {}
        for col, key in enumerate(STYLE_KEYS):
            cell = ttk.Frame(grid); cell.grid(row=0, column=col, padx=6, sticky="ew")
            grid.columnconfigure(col, weight=1)
            ttk.Label(cell, text=STYLE_LABELS[key]).pack()
            var = tk.StringVar(value=saved_map.get(key, AUTO_FONT))
            cb = ttk.Combobox(cell, textvariable=var, state="readonly",
                              values=[AUTO_FONT] + list(self.fonts.keys()), width=16)
            cb.pack(fill="x")
            self.font_vars[key] = var
            self.font_combos[key] = cb

        frow = ttk.Frame(fonts_frame); frow.pack(fill="x", padx=8, pady=(0, 5))
        ttk.Button(frow, text="רענן פונטים", command=self.refresh_fonts).pack(side="right")
        self.fonts_status = ttk.Label(
            frow, text=f"נמצאו {len(self.fonts)} פונטים בתיקיית fonts")
        self.fonts_status.pack(side="right", padx=8)

        # --- Run / Cancel + progress
        run_row = ttk.Frame(frm); run_row.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_row, text="תרגם לעברית!", command=self.start)
        self.run_btn.pack(side="right")
        self.cancel_btn = ttk.Button(run_row, text="עצור", command=self.cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="right", padx=6)
        self.open_btn = ttk.Button(run_row, text="פתח תיקיית תוצאות",
                                   command=self.open_output, state="disabled")
        self.open_btn.pack(side="left")
        ttk.Button(run_row, text="אודות", command=self.show_about).pack(
            side="left", padx=6)

        self.progress = ttk.Progressbar(frm, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=10, pady=(2, 0))
        self.status_var = tk.StringVar(value="מוכן.")
        ttk.Label(frm, textvariable=self.status_var, anchor="e").pack(
            fill="x", padx=10)

        # --- Log
        self.log_box = tk.Text(frm, height=16, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.output_dir: Path | None = None
        # Heartbeat: while a page is in flight the status line shows elapsed
        # seconds, so a slow API call is distinguishable from a real hang.
        self._busy_since: float | None = None
        self._busy_label = ""
        root.after(100, self._poll_queue)
        root.after(1000, self._tick)

    # ------------------------------------------------------------------ UI

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="בחר קובץ קומיקס",
            filetypes=[("Comic files", "*.pdf *.cbz *.cbr *.jpg *.jpeg *.png *.webp *.gif"),
                       ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def log(self, msg: str):
        self.msg_queue.put(("log", msg))

    def set_progress(self, pct: float, status: str):
        self.msg_queue.put(("progress", pct, status))

    def _poll_queue(self):
        try:
            while True:
                item = self.msg_queue.get_nowait()
                if item[0] == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", item[1] + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif item[0] == "progress":
                    self.progress["value"] = item[1]
                    self.status_var.set(item[2])
                elif item[0] == "done":
                    self.run_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    if self.output_dir:
                        self.open_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _tick(self):
        import time as _t
        if self._busy_since is not None:
            el = int(_t.time() - self._busy_since)
            self.status_var.set(f"{self._busy_label} - {el} שנ'")
        self.root.after(1000, self._tick)

    def _busy(self, label: str | None):
        import time as _t
        self._busy_label = label or ""
        self._busy_since = _t.time() if label else None

    def refresh_fonts(self):
        self.fonts = discover_fonts()
        values = [AUTO_FONT] + list(self.fonts.keys())
        for key, cb in self.font_combos.items():
            cur = self.font_vars[key].get()
            cb.configure(values=values)
            if cur not in values:
                self.font_vars[key].set(AUTO_FONT)
        self.fonts_status.configure(
            text=f"נמצאו {len(self.fonts)} פונטים בתיקיית fonts")
        if not self.fonts:
            self.log(f"טיפ: צור תיקייה בשם fonts ליד התוכנה ({FONTS_DIR}) "
                     f"ושים בה קובצי ttf/otf עבריים, ואז לחץ 'רענן פונטים'.")

    def get_font_map(self) -> dict:
        """style-key -> font file path (only for styles with a chosen font)."""
        m = {}
        for key, var in self.font_vars.items():
            name = var.get()
            if name != AUTO_FONT and name in self.fonts:
                m[key] = self.fonts[name]
        return m

    def show_about(self):
        win = tk.Toplevel(self.root)
        win.title(f"אודות - {APP_NAME}")
        win.geometry("560x520")
        win.transient(self.root)
        txt = tk.Text(win, wrap="word", padx=12, pady=10)
        txt.insert("1.0", ABOUT_TEXT)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)
        ttk.Button(win, text="סגור", command=win.destroy).pack(pady=6)

    def open_output(self):
        if self.output_dir and self.output_dir.exists():
            os.startfile(self.output_dir)  # Windows only

    def cancel(self):
        self.cancel_flag.set()
        self.log("בקשת עצירה... עוצר בהקדם (בקשה פתוחה תסתיים או תפוג).")

    # -------------------------------------------------------------- Worker

    def start(self):
        src = self.file_var.get().strip()
        if not src or not os.path.exists(src):
            messagebox.showerror(APP_NAME, "בחר קובץ קומיקס קיים.")
            return
        mode = self.mode_var.get()
        gem_key = self.gemini_var.get().strip()
        tor_key = self.torii_var.get().strip()
        if mode == "gemini" and not gem_key:
            messagebox.showerror(APP_NAME, "הזן מפתח Gemini (חינם ב-aistudio.google.com).")
            return
        if mode == "torii" and (not tor_key or not _HAS_TORII):
            messagebox.showerror(APP_NAME, "הזן מפתח Torii כדי להשתמש במצב זה.")
            return

        # Persist settings + expose keys to the pipeline modules
        font_map = self.get_font_map()
        self.cfg.update({"gemini_key": gem_key, "torii_key": tor_key,
                         "mode": mode, "format": self.fmt_var.get(),
                         "clean_bubbles": self.clean_var.get(),
                         "context": self.ctx_var.get(),
                         "font_map": {k: v.get() for k, v in self.font_vars.items()}})
        save_config(self.cfg)
        if gem_key: os.environ["GEMINI_API_KEY"] = gem_key
        if tor_key: os.environ["TORII_API_KEY"] = tor_key

        self.cancel_flag.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.open_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("טוען קובץ...")

        # Make Gemini's waits / retries / model switches VISIBLE in the log
        # (the exe has no console) and let "Stop" interrupt a pending page.
        gemini_api.LOG_HOOK = self.log
        gemini_api.CANCEL_EVENT = self.cancel_flag
        if self.cfg.get("_last_keys") != gem_key:
            gemini_api.reset_session_state()  # new keys = fresh quotas
            self.cfg["_last_keys"] = gem_key
        n_keys = len([k for k in re.split(r"[\s,;]+", gem_key) if k])
        if mode == "gemini" and n_keys > 1:
            self.log(f"[Gemini] {n_keys} מפתחות - סבב אוטומטי כשמכסה נגמרת")

        self.worker = threading.Thread(target=self._run,
                                       args=(src, mode, font_map, self.clean_var.get(),
                                             self.ctx_var.get().strip(), self.fmt_var.get(),
                                             self.force_var.get()),
                                       daemon=True)
        self.worker.start()

    def _run(self, src: str, mode: str, font_map: dict, clean_bubbles: bool = True,
             context: str = "", fmt: str = "Images", force: bool = False):
        try:
            self.log(f"טוען: {Path(src).name}")
            pages = comic_loader.load_comic_pages(src)
            total = len(pages)
            self.log(f"נטענו {total} עמודים")
            if not total:
                self.set_progress(0, "לא נמצאו עמודים בקובץ.")
                return

            # Output dir is needed up-front for RESUME support
            stem = Path(src).stem
            out_dir = Path(src).parent / "output" / f"{stem}_hebrew"
            out_dir.mkdir(parents=True, exist_ok=True)
            mem_file = out_dir / "_album_memory.txt"
            failed_file = out_dir / "_failed_pages.json"

            results = []
            album_memory = ""
            failed_pages = set()
            if failed_file.exists():
                try:
                    failed_pages = set(json.loads(failed_file.read_text(encoding="utf-8")))
                    if failed_pages:
                        self.log(f"[המשכה] עמודים שנכשלו בריצה קודמת יתורגמו שוב: {sorted(failed_pages)}")
                except Exception:
                    failed_pages = set()
            if force:
                album_memory = ""  # fresh run: don't inherit old choices
            elif mem_file.exists():
                try:
                    album_memory = mem_file.read_text(encoding="utf-8").strip()
                    if album_memory:
                        self.log("[המשכה] נטען זיכרון אלבום מריצה קודמת")
                except Exception:
                    album_memory = ""

            _keep_awake(True)
            skipped = 0
            consecutive_fail = 0
            quota_stop = False
            MAX_CONSECUTIVE_FAIL = 5  # Calibre-style circuit breaker
            for i, page in enumerate(pages):
                if self.cancel_flag.is_set():
                    self.log(f"נעצר. תורגמו {len(results)}/{total} עמודים.")
                    break
                pct = i / total * 100
                page_file = out_dir / f"page_{i+1:03d}.png"
                if not force and page_file.exists() and (i + 1) not in failed_pages:
                    try:
                        results.append(Image.open(page_file).convert("RGB"))
                        skipped += 1
                        self.set_progress(pct, f"עמוד {i+1}/{total} כבר תורגם - מדלג")
                        self.log(f"=== עמוד {i+1}/{total}: קיים מריצה קודמת - מדלג (0 בקשות API) ===")
                        continue
                    except Exception:
                        pass  # corrupt file - retranslate it
                self.set_progress(pct, f"מתרגם עמוד {i+1}/{total} ({pct:.0f}%)")
                self._busy(f"מתרגם עמוד {i+1}/{total} ({pct:.0f}%)")
                self.log(f"\n=== עמוד {i+1}/{total} ===")
                ed = {}
                try:
                    if mode == "torii":
                        out = process_page_torii(page, self.log)
                    else:
                        out, album_memory = process_page_gemini(
                            page, self.log, font_map, clean_bubbles=clean_bubbles,
                            context=context or None, memory=album_memory,
                            editor_export=ed)
                        if album_memory:
                            self.log(f"  [זיכרון אלבום] {album_memory[:90]}...")
                    failed_pages.discard(i + 1)
                    consecutive_fail = 0
                except gemini_api.GeminiCancelled:
                    self.log(f"נעצר באמצע עמוד {i+1}. תורגמו {len(results)}/{total} עמודים.")
                    break
                except gemini_api.GeminiQuotaExhausted as exc:
                    # Stop cleanly instead of "failing" every remaining page
                    # as a flipped original. Resume later skips done pages.
                    quota_stop = True
                    self.log(f"\n{exc}")
                    self.log(f"הריצה נעצרה בעמוד {i+1}. הרץ שוב את אותו קובץ אחרי "
                             "חידוש המכסה - התרגום ימשיך מאותו עמוד (0 עלות על מה שכבר תורגם).")
                    break
                except Exception as exc:
                    traceback.print_exc()
                    consecutive_fail += 1
                    self.log(f"  שגיאה בעמוד {i+1}: {exc}")
                    self.log("  (העמוד נשמר כמקור-מהופך; ריצה חוזרת תנסה לתרגם אותו שוב)")
                    failed_pages.add(i + 1)
                    out = page.transpose(Image.FLIP_LEFT_RIGHT)
                finally:
                    self._busy(None)
                results.append(out)
                if mode != "torii" and ed.get("clean") is not None:
                    write_page_editor(out_dir, i + 1, ed["clean"],
                                      ed.get("bubbles", []), self.log,
                                      font_map=font_map)
                try:
                    out.save(out_dir / f"page_{i+1:03d}.png")
                    if album_memory:
                        mem_file.write_text(album_memory, encoding="utf-8")
                    failed_file.write_text(json.dumps(sorted(failed_pages)), encoding="utf-8")
                except Exception:
                    pass
                if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                    # Something systemic (network down, service outage):
                    # burning through hundreds of pages as failures helps
                    # nobody. Stop; a re-run resumes exactly here.
                    self.log(f"\n{consecutive_fail} עמודים ברצף נכשלו - עוצר. "
                             "בדוק חיבור/מפתחות והרץ שוב; התרגום ימשיך מכאן.")
                    quota_stop = True
                    break

            if not results:
                self.set_progress(0, "לא הופקו עמודים.")
                return

            # סבב ניסיון חוזר לעמודים שנכשלו (תקלות זמניות בדרך כלל חולפות)
            for retry in range(2):
                if not failed_pages or self.cancel_flag.is_set() or quota_stop:
                    break
                self.log(f"\n[ניסיון חוזר {retry+1}] עמודים: {sorted(failed_pages)}")
                self.set_progress(95, f"מנסה שוב עמודים שנכשלו: {sorted(failed_pages)}")
                if self.cancel_flag.wait(15):
                    break
                for pno in sorted(failed_pages):
                    if pno - 1 >= len(results):
                        continue  # never reached in the main pass (stopped)
                    ed = {}
                    self._busy(f"ניסיון חוזר לעמוד {pno}")
                    try:
                        if mode == "torii":
                            out = process_page_torii(pages[pno-1], self.log)
                        else:
                            out, album_memory = process_page_gemini(
                                pages[pno-1], self.log, font_map,
                                clean_bubbles=clean_bubbles,
                                context=context or None, memory=album_memory,
                                editor_export=ed)
                        if ed.get("clean") is not None:
                            write_page_editor(out_dir, pno, ed["clean"],
                                              ed.get("bubbles", []), self.log,
                                              font_map=font_map)
                        out.save(out_dir / f"page_{pno:03d}.png")
                        results[pno-1] = out
                        failed_pages.discard(pno)
                        self.log(f"  עמוד {pno} תורגם בהצלחה בניסיון החוזר")
                    except gemini_api.GeminiCancelled:
                        self.log("נעצר.")
                        break
                    except gemini_api.GeminiQuotaExhausted as exc:
                        self.log(str(exc)); quota_stop = True
                        break
                    except Exception as exc:
                        self.log(f"  עמוד {pno} נכשל שוב: {exc}")
                    finally:
                        self._busy(None)
                failed_file.write_text(json.dumps(sorted(failed_pages)), encoding="utf-8")
            if failed_pages:
                self.log(f"\nעמודים שנשארו כמקור-מהופך: {sorted(failed_pages)} - "
                         "הרץ שוב את אותו קובץ כדי לנסות שוב (שאר העמודים ידולגו ב-0 עלות).")

            self.set_progress(97, "שומר קבצים...")
            if skipped:
                self.log(f"\n[המשכה] {skipped} עמודים נטענו מריצה קודמת, {len(results)-skipped} תורגמו עכשיו")
            self.output_dir = out_dir

            if fmt == "CBZ":
                paths = comic_loader.save_comic_pages(results, self.output_dir,
                                                      "page", "png")
                out_file = comic_loader.create_cbz(
                    paths, self.output_dir / f"{stem}_hebrew.cbz")
                self.log(f"\nנשמר: {out_file}")
            elif fmt == "PDF":
                out_file = comic_loader.create_pdf(
                    results, self.output_dir / f"{stem}_hebrew.pdf")
                self.log(f"\nנשמר: {out_file}")
            else:
                paths = comic_loader.save_comic_pages(results, self.output_dir,
                                                      "page", "png")
                self.log(f"\nנשמרו {len(paths)} תמונות אל: {self.output_dir}")

            self.set_progress(100, f"הושלם! {len(results)}/{len(pages)} עמודים.")
        except Exception as exc:
            traceback.print_exc()
            self.log(f"שגיאה: {exc}")
            self.set_progress(0, "נכשל - ראה יומן.")
        finally:
            self._busy(None)
            _keep_awake(False)
            self.msg_queue.put(("done",))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")  # native Windows look
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
