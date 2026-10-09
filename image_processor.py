"""
Comic Book Image Processor
==========================

A module for processing comic book images including mirroring (horizontal flip),
text removal via inpainting, and Hebrew text rendering with RTL support.

Dependencies:
    - opencv-python (cv2)
    - Pillow (PIL)
    - numpy
    - arabic_reshaper
    - python-bidi

Typical workflow:
    1. mirror_image()      -- flip the page horizontally
    2. mirror_bubbles()    -- adjust bubble coordinates for mirrored page
    3. remove_text_from_bubbles() -- inpaint original text
    4. render_hebrew_text() -- draw translated Hebrew text into each bubble
"""

from __future__ import annotations

import functools
import math
import os
import platform
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Optional dependencies -- Hebrew shaping / RTL handling
# ---------------------------------------------------------------------------

try:
    import arabic_reshaper

    _HAS_ARABIC_RESHAPER = True
except ImportError:  # pragma: no cover
    _HAS_ARABIC_RESHAPER = False
    warnings.warn(
        "arabic_reshaper is not installed. Hebrew text will be rendered "
        "without proper character reshaping.",
        stacklevel=2,
    )

try:
    from bidi.algorithm import get_display

    _HAS_BIDI = True
except ImportError:  # pragma: no cover
    _HAS_BIDI = False
    warnings.warn(
        "python-bidi is not installed. RTL text will be rendered "
        "left-to-right instead of right-to-left.",
        stacklevel=2,
    )

# If Pillow was built with the Raqm layout engine, it performs BiDi
# reordering and shaping NATIVELY when drawing text. In that case we must
# NOT apply python-bidi ourselves -- doing both reverses the text twice
# and produces mirror-writing again.
try:
    from PIL import features as _pil_features

    _HAS_RAQM = bool(_pil_features.check("raqm"))
except Exception:  # pragma: no cover
    _HAS_RAQM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def _get_font_candidates() -> List[str]:
    """Return font search paths appropriate for the current OS."""
    system = platform.system()
    fonts: List[str] = []

    if system == "Windows":
        win_fonts = Path(os.environ.get("SYSTEMROOT", "C:\\Windows")) / "Fonts"
        fonts += [
            # Hebrew fonts on Windows
            str(win_fonts / "ahronbd.ttf"),     # Aharoni Bold (Hebrew)
            str(win_fonts / "david.ttf"),       # David (Hebrew)
            str(win_fonts / "davidbd.ttf"),     # David Bold (Hebrew)
            str(win_fonts / "frank.ttf"),       # FrankRuehl (Hebrew)
            str(win_fonts / "segoeui.ttf"),     # Segoe UI (has Hebrew)
            str(win_fonts / "tahoma.ttf"),      # Tahoma (has Hebrew)
            str(win_fonts / "arial.ttf"),       # Arial (has Hebrew)
            str(win_fonts / "calibri.ttf"),     # Calibri
            str(win_fonts / "verdana.ttf"),     # Verdana
        ]
    elif system == "Darwin":  # macOS
        fonts += [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]
    else:  # Linux
        fonts += [
            "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansHebrew.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]

    return fonts


# Lazily evaluated font candidates for the current platform
_DEFAULT_FONT_CANDIDATES: List[str] = _get_font_candidates()

# Inpainting parameters
_INPAINT_DILATE_KERNEL_SIZE: int = 7  # pixels
_INPAINT_RADIUS: int = 5  # inpaint neighbourhood radius

# Text rendering parameters
_TEXT_PADDING_RATIO: float = 0.15  # 15% padding inside bubble
_MIN_FONT_SIZE: int = 6
_MAX_FONT_SIZE: int = 120
_FONT_SIZE_FIT_ITERATIONS: int = 15
_LINE_SPACING: float = 1.12
# Reference string spanning Hebrew ascender (ל) and descenders (ק ך ן ף ץ)
_HEB_METRIC_REF = "לקךןףץאבגש"

# BGR colour used for text (black)
_TEXT_COLOUR: Tuple[int, int, int] = (0, 0, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mirror_image(image: np.ndarray) -> np.ndarray:
    """Return a horizontally-flipped copy of *image*.

    Parameters
    ----------
    image:
        Input image as a NumPy array (H x W x C) or (H x W).  The array
        is not modified in place.

    Returns
    -------
    np.ndarray
        Horizontally mirrored image with the same dtype and shape as *image*.

    Examples
    --------
    >>> mirrored = mirror_image(original)
    """
    return cv2.flip(image, 1)  # 1 == horizontal flip


def mirror_bubbles(
    bubbles: List[Dict],
    image_width: int,
) -> List[Dict]:
    """Adjust bubble coordinates for a horizontally-mirrored image.

    For each bubble with ``merged_bbox`` ``(x1, y1, x2, y2)`` the new
    coordinates after a horizontal flip are::

        new_x1 = image_width - x2
        new_x2 = image_width - x1
        y1, y2  unchanged

    Parameters
    ----------
    bubbles:
        List of bubble dictionaries.  Each dict must contain the key
        ``"merged_bbox"`` with value ``(x1, y1, x2, y2)``.
    image_width:
        Width of the mirrored image in pixels.

    Returns
    -------
    List[Dict]
        New list of bubble dictionaries with updated ``merged_bbox`` values.
        The original list is **not** modified.

    Raises
    ------
    ValueError
        If ``image_width`` is not positive or if a bubble is missing the
        ``merged_bbox`` key.
    """
    if image_width <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}")

    result: List[Dict] = []
    for bubble in bubbles:
        if "merged_bbox" not in bubble:
            raise ValueError(
                f"Bubble dictionary missing required key 'merged_bbox': {bubble}"
            )

        new_bubble = dict(bubble)  # shallow copy
        # BUGFIX: only merged_bbox used to be mirrored. layout_bbox /
        # _orig_bbox stayed in UNmirrored coordinates, so anything using
        # them after the flip pointed at the opposite side of the page.
        for key in ("merged_bbox", "layout_bbox", "_orig_bbox"):
            box = bubble.get(key)
            if box:
                x1, y1, x2, y2 = box
                new_bubble[key] = (image_width - x2, y1, image_width - x1, y2)
        result.append(new_bubble)

    return result


def refine_text_regions(
    image: np.ndarray,
    bubbles: List[Dict],
    expand_ratio: float = 1.0,
    dark_thresh: int = 110,
    flood_tolerance: int = 60,
    min_component_area: int = 3,
) -> Tuple[List[Dict], np.ndarray]:
    """Snap approximate text boxes to the ACTUAL dark text pixels.

    AI-provided bounding boxes (e.g. from Gemini) are often shifted or only
    partially cover the text. Rectangle-inpainting such boxes smears bubble
    outlines / panel borders into black smudges, leaves stray letters behind,
    and misplaces the rendered Hebrew.

    Strategy per bubble (works no matter how imprecise the input box is, as
    long as its center lands inside the right bubble):

    1. FLOOD FILL the light bubble background from the box center. The dark
       bubble outline stops the fill, so the filled region == the bubble
       interior. (A search window of the box expanded by *expand_ratio*
       bounds the fill in case the outline has gaps.)
    2. Dark "ink" pixels INSIDE that interior are text -- the outline itself
       is excluded by construction, and letters the box missed are included.
    3. The text-pixel bounding box replaces ``merged_bbox`` so the Hebrew is
       rendered where the original text really was.
    4. Fallback (flood failed / unusual background): dark pixels inside the
       window, minus connected components touching the window border (those
       are bubble outlines / panel frame lines).

    Returns
    -------
    (refined_bubbles, text_mask, interior_mask)
        *refined_bubbles* -- new list with snapped ``merged_bbox`` values.
        *text_mask* -- full-page uint8 mask (255 = text pixel to remove),
        slightly dilated, ready for :func:`remove_text_by_mask`.
        *interior_mask* -- full-page uint8 mask of TRUSTED bubble interiors,
        for optional background cleaning via :func:`clean_bubble_interiors`.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {image.shape}")

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    refined: List[Dict] = []

    for bubble in bubbles:
        x1, y1, x2, y2 = map(int, bubble["merged_bbox"])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            refined.append(dict(bubble))
            continue

        # Search window: generous expansion, bounded to the page
        # ISOTROPIC expansion: a wide 1-2 line caption box expanded by its
        # own (small) height never reached the bubble's top/bottom outline,
        # so the flood always "touched the window" and the bubble was
        # rejected as untrusted -> no clean fill, no layout room.
        ex = ey = int(max(bw, bh) * expand_ratio)
        cx1 = max(0, x1 - ex)
        cy1 = max(0, y1 - ey)
        cx2 = min(w, x2 + ex)
        cy2 = min(h, y2 + ey)
        crop = gray[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            refined.append(dict(bubble))
            continue

        # ADAPTIVE ink threshold. A fixed 110 missed the grey anti-aliased
        # edge of every letter (and faded grey ink on old scans entirely):
        # those pixels survived the inpaint as a visible "ghost" of the
        # original lettering - the main source of dirty-looking bubbles.
        # The bar is set relative to THIS bubble's own paper brightness.
        ob_px = crop[max(0, y1 - cy1):max(0, y2 - cy1),
                     max(0, x1 - cx1):max(0, x2 - cx1)]
        paper = float(np.percentile(ob_px, 90)) if ob_px.size else 255.0
        thr = int(np.clip(paper - 70, dark_thresh - 20, dark_thresh + 45))
        dark = (crop < thr).astype(np.uint8) * 255

        # Gap-closing barrier: scanned outlines have 1-2 px breaks through
        # which the flood leaked into the art - the bubble was then rejected
        # as "untrusted" and fell back to rough inpainting. Thickening the
        # ink by 2 px for the FLOOD ONLY seals gaps up to ~4 px.
        barrier = cv2.dilate(dark, np.ones((5, 5), np.uint8))
        flood_src = crop.copy()
        flood_src[barrier > 0] = 0

        # --- Step 1: flood-fill bubble interior from a light seed near center
        seed = _find_light_seed(flood_src, (x1 + x2) // 2 - cx1, (y1 + y2) // 2 - cy1,
                                thr)
        text = None
        flood_interior = None
        if seed is not None:
            flood_mask = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
            flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
            try:
                cv2.floodFill(
                    flood_src, flood_mask, seed, 0,
                    loDiff=flood_tolerance, upDiff=flood_tolerance, flags=flags,
                )
                interior = flood_mask[1:-1, 1:-1]
                # The text letters are HOLES in the flooded background.
                # Fill every enclosed hole: any unfilled component that does
                # NOT touch the crop border is inside the bubble (the outline
                # + exterior form one border-touching component and stay out).
                inv = (interior == 0).astype(np.uint8)
                inum, ilabels, istats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
                # Vectorised (the per-label full-array assignment was
                # O(letters x window) - slow on text-heavy pages).
                border = np.unique(np.concatenate([
                    ilabels[0, :], ilabels[-1, :], ilabels[:, 0], ilabels[:, -1]]))
                interior[(ilabels > 0) & ~np.isin(ilabels, border)] = 255

                # RESCUE letters the gap-closing barrier glued to the bubble
                # outline (a letter within ~2px of the outline became part of
                # the outline component and was left on the page). A dark
                # component (un-thickened) whose surrounding ring is mostly
                # bubble interior is a letter; the outline's ring is only
                # ~half interior, so it stays out.
                dnum, dlab, dst, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
                for lbl in range(1, dnum):
                    lx, ly, lw, lh, area = dst[lbl]
                    if area < min_component_area or lw > 0.5 * cw or lh > 0.5 * ch:
                        continue
                    sx1, sy1 = max(0, lx - 6), max(0, ly - 6)
                    sx2, sy2 = min(cw, lx + lw + 6), min(ch, ly + lh + 6)
                    comp = (dlab[sy1:sy2, sx1:sx2] == lbl).astype(np.uint8)
                    if interior[sy1:sy2, sx1:sx2][comp > 0].all():
                        continue  # already inside
                    # ring 3-4 px out: just beyond the barrier's 2px halo
                    ring = (cv2.dilate(comp, np.ones((9, 9), np.uint8))
                            & (1 - cv2.dilate(comp, np.ones((5, 5), np.uint8))))
                    rin = interior[sy1:sy2, sx1:sx2][ring > 0]
                    if rin.size and (rin > 0).mean() >= 0.62:
                        interior[sy1:sy2, sx1:sx2][comp > 0] = 255
                candidate = cv2.bitwise_and(dark, interior)
                # A leaked flood (interior touching the search-window border)
                # is untrustworthy for TEXT too, not just for cleaning: its
                # dark pixels include art outside the bubble. Reject the
                # whole flood path and let the clamped fallback handle it.
                touches_window = (interior[0, :].any() or interior[-1, :].any()
                                  or interior[:, 0].any() or interior[:, -1].any())
                if cv2.countNonZero(candidate) > 0 and not touches_window:
                    text = candidate
                    flood_interior = interior
            except cv2.error:
                text = None

        # --- Step 4: fallback -- dark pixels near the ORIGINAL box only.
        # (The generous window exists for the flood path; when the flood is
        # unusable, searching the whole window grabs neighbouring art.)
        if text is None or cv2.countNonZero(text) == 0:
            fx = int(bw * 0.25)
            fy = int(bh * 0.25)
            fx1 = max(0, (x1 - fx) - cx1)
            fy1 = max(0, (y1 - fy) - cy1)
            fx2 = min(cw, (x2 + fx) - cx1)
            fy2 = min(ch, (y2 + fy) - cy1)
            region = dark[fy1:fy2, fx1:fx2]
            region_gray = crop[fy1:fy2, fx1:fx2]
            # Adaptive halo bar: measure the ORIGINAL box's own background
            # brightness; letters must sit on a background close to it.
            # (A fixed bar fails when e.g. a purple caption is at 164 and the
            # sky just below is at ~135 - both above a low fixed threshold.)
            ob = crop[max(0, y1 - cy1):max(0, y2 - cy1), max(0, x1 - cx1):max(0, x2 - cx1)]
            ob_light = ob[ob >= thr]
            bg_level = float(np.median(ob_light)) if ob_light.size else 255.0
            halo_bar = max(thr + 25, bg_level - 15)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
            rh, rw = region.shape
            keep = np.zeros_like(dark)
            for lbl in range(1, num):
                lx, ly, lw, lh, area = stats[lbl]
                if area < min_component_area:
                    continue
                if lx == 0 or ly == 0 or lx + lw >= rw or ly + lh >= rh:
                    continue  # touches fallback-region border: outline / frame
                # A component spanning most of the region is a caption FRAME
                # fully enclosed in the region, not a letter.
                if lw > 0.7 * rw or lh > 0.7 * rh:
                    continue
                # LIGHT-HALO test: comic lettering always sits on a light,
                # flat background; art fragments (hatching, shadows, sky
                # details) do not. Sample the ring around the component's
                # bbox and require it to be clearly light.
                m = 3
                hx1, hy1 = max(0, lx - m), max(0, ly - m)
                hx2, hy2 = min(rw, lx + lw + m), min(rh, ly + lh + m)
                patch = region_gray[hy1:hy2, hx1:hx2]
                patch_dark = region[hy1:hy2, hx1:hx2]
                ring = patch[patch_dark == 0]
                if ring.size == 0 or np.median(ring) < halo_bar:
                    continue
                keep[fy1:fy2, fx1:fx2][labels == lbl] = 255
            text = keep

        if cv2.countNonZero(text) == 0:
            refined.append(dict(bubble))
            continue

        # ------------------------------------------------------------------
        # CLUSTER SELECTION: the mask may legitimately contain letters of a
        # NEIGHBOURING bubble caught inside the search region. Group ink into
        # text clusters (dilating by ~letter height merges letters, words and
        # lines, but not the gap to another bubble), then keep only the
        # cluster that overlaps Gemini's original box the most.
        # ------------------------------------------------------------------
        num, labels, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
        comp_hs = [int(stats[lbl, cv2.CC_STAT_HEIGHT]) for lbl in range(1, num)
                   if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area]
        if not comp_hs:
            refined.append(dict(bubble))
            continue
        med_h = float(np.median(comp_hs))

        k = max(7, int(med_h * 1.8))
        clusters = cv2.dilate(text, np.ones((k, k), dtype=np.uint8))
        cnum, clab, cstats, _ = cv2.connectedComponentsWithStats(clusters, connectivity=8)

        # Original box in crop coords
        obx1, oby1 = max(0, x1 - cx1), max(0, y1 - cy1)
        obx2, oby2 = min(cw, x2 - cx1), min(ch, y2 - cy1)

        def _inter(l, t_, r, b):
            iw = min(r, obx2) - max(l, obx1)
            ih = min(b, oby2) - max(t_, oby1)
            return max(0, iw) * max(0, ih)

        # Keep: (a) every cluster that lies MOSTLY inside Gemini's box (multi-
        # paragraph bubbles split into several clusters - all are ours), and
        # (b) the single best-overlapping cluster (partial-coverage cases).
        best_lbl, best_score = 0, -1.0
        keep_lbls = set()
        for lbl in range(1, cnum):
            lx, ly = cstats[lbl, cv2.CC_STAT_LEFT], cstats[lbl, cv2.CC_STAT_TOP]
            lw, lh = cstats[lbl, cv2.CC_STAT_WIDTH], cstats[lbl, cv2.CC_STAT_HEIGHT]
            area = max(1, lw * lh)
            ov = _inter(lx, ly, lx + lw, ly + lh)
            if ov / area >= 0.5:
                keep_lbls.add(lbl)
            score = ov + area * 1e-6
            if score > best_score:
                best_score, best_lbl = score, lbl
        if best_lbl > 0:
            keep_lbls.add(best_lbl)

        if keep_lbls:
            sel = np.isin(clab, list(keep_lbls)).astype(np.uint8) * 255
            text = cv2.bitwise_and(text, sel)
        if cv2.countNonZero(text) == 0:
            refined.append(dict(bubble))
            continue

        # ------------------------------------------------------------------
        # SANITY GUARDS -- the flood fill can leak out of open/low-contrast
        # bubbles and swallow neighbouring art or other bubbles' text. If the
        # detected region is much bigger than Gemini's box, or its centre
        # drifted far away, we don't trust it: clamp everything back to the
        # original box (slightly expanded).
        # ------------------------------------------------------------------
        ys, xs = np.nonzero(text)
        det_w = int(xs.max() - xs.min())
        det_h = int(ys.max() - ys.min())
        det_cx = cx1 + (int(xs.min()) + int(xs.max())) / 2.0
        det_cy = cy1 + (int(ys.min()) + int(ys.max())) / 2.0
        orig_cx, orig_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        inflated = det_w > bw * 1.8 + 30 or det_h > bh * 1.8 + 30
        # Per-AXIS drift check: a wide, short caption has a huge diagonal, so
        # a radial test misses text that drifted a full box-height downward.
        drifted = (abs(det_cx - orig_cx) > 0.6 * bw + 20
                   or abs(det_cy - orig_cy) > 0.6 * bh + 20)

        if inflated or drifted:
            flood_interior = None  # can't trust the interior either
            gx, gy = int(bw * 0.12), int(bh * 0.12)
            ix1 = max(0, (x1 - gx) - cx1)
            iy1 = max(0, (y1 - gy) - cy1)
            ix2 = min(cw, (x2 + gx) - cx1)
            iy2 = min(ch, (y2 + gy) - cy1)
            inner = np.zeros_like(dark)
            inner[iy1:iy2, ix1:ix2] = 255
            clamped = cv2.bitwise_and(dark, inner)
            if cv2.countNonZero(clamped) > 0:
                text = clamped
            else:
                # No ink near the original box - keep Gemini's box untouched
                nb = dict(bubble)
                nb["_orig_bbox"] = (x1, y1, x2, y2)
                refined.append(nb)
                continue

        # Measure the ORIGINAL lettering height: median height of the text's
        # connected components (approx. the source font size). Used later to
        # cap the Hebrew font so it never balloons.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
        comp_hs = [int(stats[lbl, cv2.CC_STAT_HEIGHT]) for lbl in range(1, num)
                   if stats[lbl, cv2.CC_STAT_AREA] >= 8]
        src_letter_h = int(np.median(comp_hs)) if comp_hs else 0

        # Snap bbox to the true text extent
        ys, xs = np.nonzero(text)
        pad = 4
        nx1 = max(0, cx1 + int(xs.min()) - pad)
        ny1 = max(0, cy1 + int(ys.min()) - pad)
        nx2 = min(w, cx1 + int(xs.max()) + pad)
        ny2 = min(h, cy1 + int(ys.max()) + pad)

        new_bubble = dict(bubble)
        new_bubble["merged_bbox"] = (nx1, ny1, nx2, ny2)
        new_bubble["_orig_bbox"] = (x1, y1, x2, y2)
        if src_letter_h:
            new_bubble["src_letter_h"] = src_letter_h

        # Original INK COLOUR (red shouts, white-on-dark captions, brown ink
        # of old scans): the darkest part of the letters, so blur/halo
        # doesn't wash it out. Rendering everything pure black was wrong.
        crop_bgr = image[cy1:cy2, cx1:cx2]
        core = text > 0
        if core.sum() >= 20:
            g = crop[core]
            cut = np.percentile(g, 35)
            sel_px = crop_bgr[core][g <= cut]
            if sel_px.size:
                b_, g_, r_ = np.median(sel_px, axis=0)
                new_bubble["ink_rgb"] = (int(r_), int(g_), int(b_))

        # Accumulate into the full-page mask (dilated to cover anti-aliasing).
        # Dilation is sized to the lettering (bigger letters = wider grey
        # halo) and, when the bubble interior is known, CLIPPED to it so the
        # mask can never eat into the bubble outline (outline smears).
        k = 5 if src_letter_h < 18 else 7
        text_d = cv2.dilate(text, np.ones((k, k), np.uint8), iterations=1)
        if flood_interior is not None:
            guard = cv2.dilate(flood_interior, np.ones((3, 3), np.uint8))
            text_d = cv2.bitwise_and(text_d, guard)
        full_mask[cy1:cy2, cx1:cx2] = cv2.bitwise_or(
            full_mask[cy1:cy2, cx1:cx2], text_d
        )

        # Collect the bubble interior for background cleaning - only when it
        # is trustworthy: the flood path was used, guards passed, and the
        # interior is FULLY enclosed inside the search window (touching the
        # window border means the bubble extends beyond it, and flat-filling
        # a partial interior would leave a visible seam).
        if flood_interior is not None:
            touches = (flood_interior[0, :].any() or flood_interior[-1, :].any()
                       or flood_interior[:, 0].any() or flood_interior[:, -1].any())
            if not touches:
                interior_mask[cy1:cy2, cx1:cx2] = cv2.bitwise_or(
                    interior_mask[cy1:cy2, cx1:cx2], flood_interior
                )
                # LAYOUT BOX: the room the Hebrew may use = the largest
                # rectangle grown from the source-text box that stays inside
                # the bubble (with a margin). Previously the Hebrew was
                # squeezed into the tight box of the ORIGINAL text, so any
                # translation longer than the source shrank to tiny letters.
                margin = max(3, int((src_letter_h or 10) * 0.45))
                lb = _grow_inside(flood_interior,
                                  (nx1 - cx1, ny1 - cy1, nx2 - cx1, ny2 - cy1),
                                  margin)
                if lb is not None:
                    lx1, ly1, lx2, ly2 = lb
                    new_bubble["layout_bbox"] = (cx1 + lx1, cy1 + ly1,
                                                 cx1 + lx2, cy1 + ly2)
        refined.append(new_bubble)

    refined = _resolve_overlaps(refined)
    _separate_layouts(refined)
    return refined, full_mask, interior_mask


def _grow_inside(mask: np.ndarray, box, margin: int, step: int = 2,
                 max_iter: int = 400):
    """Grow *box* (x1,y1,x2,y2 in mask coords) outward, one side at a time,
    while the added strip stays inside *mask* eroded by *margin*.

    Returns the grown box, or None when even the starting box pokes outside
    the (eroded) interior by a lot. The result keeps roughly the bubble's
    proportions, so text laid out in it stays clear of the outline.
    """
    mh, mw = mask.shape[:2]
    inner = cv2.erode(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    if cv2.countNonZero(inner) == 0:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mw, x2), min(mh, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    # Start box must be (mostly) inside the eroded interior already
    if inner[y1:y2, x1:x2].mean() / 255.0 < 0.85:
        return (x1, y1, x2, y2)

    def ok(a1, b1, a2, b2):
        if a1 < 0 or b1 < 0 or a2 > mw or b2 > mh:
            return False
        return inner[b1:b2, a1:a2].mean() / 255.0 >= 0.985

    grow = [True, True, True, True]
    for _ in range(max_iter):
        if not any(grow):
            break
        if grow[0]:
            grow[0] = ok(x1 - step, y1, x1, y2) and bool(x1 - step >= 0)
            if grow[0]: x1 -= step
        if grow[2]:
            grow[2] = ok(x2, y1, x2 + step, y2)
            if grow[2]: x2 += step
        if grow[1]:
            grow[1] = ok(x1, y1 - step, x2, y1)
            if grow[1]: y1 -= step
        if grow[3]:
            grow[3] = ok(x1, y2, x2, y2 + step)
            if grow[3]: y2 += step
    return (x1, y1, x2, y2)


def _separate_layouts(bubbles: List[Dict]) -> None:
    """Two text blocks inside ONE bubble get the same interior, so both
    layout boxes would grow to fill it and the Hebrew would print on top of
    each other. Split overlapping layout boxes between the two source-text
    boxes (along the axis where the texts are further apart)."""
    for i in range(len(bubbles)):
        for j in range(i + 1, len(bubbles)):
            a, b = bubbles[i], bubbles[j]
            la, lb = a.get("layout_bbox"), b.get("layout_bbox")
            if not la or not lb:
                continue
            if min(la[2], lb[2]) <= max(la[0], lb[0]) or \
               min(la[3], lb[3]) <= max(la[1], lb[1]):
                continue  # no overlap
            ta, tb = a["merged_bbox"], b["merged_bbox"]
            gap_y = max(tb[1] - ta[3], ta[1] - tb[3])
            gap_x = max(tb[0] - ta[2], ta[0] - tb[2])
            la, lb = list(la), list(lb)
            if gap_y >= gap_x:
                top, bot = (la, lb) if ta[1] <= tb[1] else (lb, la)
                ttop = ta if top is la else tb
                tbot = tb if top is la else ta
                cut = (ttop[3] + tbot[1]) // 2
                top[3] = min(top[3], cut - 2)
                bot[1] = max(bot[1], cut + 2)
            else:
                lft, rgt = (la, lb) if ta[0] <= tb[0] else (lb, la)
                tl = ta if lft is la else tb
                tr = tb if lft is la else ta
                cut = (tl[2] + tr[0]) // 2
                lft[2] = min(lft[2], cut - 2)
                rgt[0] = max(rgt[0], cut + 2)
            for bub, lay in ((a, la), (b, lb)):
                if lay[2] - lay[0] < 8 or lay[3] - lay[1] < 8:
                    bub.pop("layout_bbox", None)
                else:
                    bub["layout_bbox"] = tuple(lay)


def flat_fill_text(
    image: np.ndarray,
    text_mask: np.ndarray,
    interior_mask: np.ndarray,
    dark_thresh: int = 110,
    min_area: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Erase text inside FLAT bubble interiors by painting the bubble's own
    paper colour, and return (image, remaining_mask) - the text pixels that
    still need inpainting (non-flat / untrusted areas).

    Inpainting (TELEA) on a flat bubble produces soft grey blotches exactly
    where the letters were - on white paper the eye sees them at once.
    Painting the measured paper colour leaves no trace. Unlike
    :func:`clean_bubble_interiors` this touches ONLY the text pixels, so the
    paper texture of the rest of the bubble is preserved (used when the
    'clean bubble background' option is off).
    """
    out = image.copy()
    remaining = text_mask.copy()
    if cv2.countNonZero(interior_mask) == 0 or cv2.countNonZero(text_mask) == 0:
        return out, remaining
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(interior_mask, connectivity=8)
    for lbl in range(1, num):
        if int(stats[lbl, cv2.CC_STAT_AREA]) < min_area:
            continue
        x, y, w, h = (int(stats[lbl, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                    cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        comp = labels[y:y + h, x:x + w] == lbl
        tm = text_mask[y:y + h, x:x + w] > 0
        paper = comp & ~tm & (gray[y:y + h, x:x + w] > dark_thresh + 20)
        if paper.sum() < 50:
            continue
        sub = out[y:y + h, x:x + w]
        bg = np.median(sub[paper], axis=0)
        dist = np.abs(sub[paper].astype(np.int16) - bg.astype(np.int16)).sum(axis=1)
        if float((dist > 60).mean()) > 0.10:
            continue  # not flat (gradient / texture / art) -> inpaint instead
        # Paint text pixels that belong to this interior (grown by 2px so the
        # anti-aliased fringe next to the interior edge is included).
        pad = 3
        X1, Y1 = max(0, x - pad), max(0, y - pad)
        X2, Y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        compg = np.zeros((Y2 - Y1, X2 - X1), np.uint8)
        compg[y - Y1:y - Y1 + h, x - X1:x - X1 + w][comp] = 255
        compg = cv2.dilate(compg, np.ones((5, 5), np.uint8))
        sel = (compg > 0) & (text_mask[Y1:Y2, X1:X2] > 0)
        out[Y1:Y2, X1:X2][sel] = bg.astype(np.uint8)
        remaining[Y1:Y2, X1:X2][sel] = 0
    return out, remaining


def clean_bubble_interiors(
    image: np.ndarray,
    interior_mask: np.ndarray,
    dark_thresh: int = 110,
    min_area: int = 300,
) -> np.ndarray:
    """Repaint each detected bubble interior with its own uniform background.

    Old / low-quality scans have bubbles full of print noise, dots, and
    stains. For every connected interior region this paints ALL its pixels
    with the MEDIAN of its light pixels -- so white bubbles become clean
    white, yellowed caption boxes stay yellow but uniform, and all the noise
    disappears. Text pixels inside are covered too (they are removed anyway).

    The bubble outline itself is never part of the interior, so it is
    untouched.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {image.shape}")
    if cv2.countNonZero(interior_mask) == 0:
        return image.copy()

    out = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(interior_mask, connectivity=8)
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == lbl
        light = comp & (gray > dark_thresh + 20)
        n_light = int(light.sum())
        if n_light < 50:
            continue

        # --- SAFETY FILTERS: only repaint genuinely FLAT bubble interiors.
        # A flood that swallowed a face / art area must never be repainted.
        # 1. Dark fraction: text occupies a small share of a real bubble;
        #    hair / heavy art inside means this isn't a flat bubble.
        dark_frac = 1.0 - (n_light / area)
        if dark_frac > 0.30:
            continue
        # 2. Flatness: among the light pixels, most must be close in color
        #    to the median background. Skin/shading midtones fail this.
        bg_color = np.median(out[light], axis=0)
        dist = np.abs(out[light].astype(np.int16) - bg_color.astype(np.int16)).sum(axis=1)
        nonflat_frac = float((dist > 90).mean())
        if nonflat_frac > 0.12:
            continue

        out[comp] = bg_color.astype(np.uint8)

    return out


def _overlap_ratio(a, b) -> float:
    """Intersection area divided by the SMALLER box's area (0..1)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


def _resolve_overlaps(bubbles: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """Prevent text-on-text rendering when refined boxes collide.

    Two cases:
    * The ORIGINAL (Gemini) boxes also overlapped heavily -> Gemini returned
      the same text twice; keep only the bubble with the longer Hebrew text.
    * The originals did NOT overlap -> refinement dragged the boxes onto
      each other; revert both to their original boxes so each text renders
      in its own place.
    """
    alive = [True] * len(bubbles)
    for i in range(len(bubbles)):
        if not alive[i]:
            continue
        for j in range(i + 1, len(bubbles)):
            if not alive[j]:
                continue
            bi, bj = bubbles[i], bubbles[j]
            if _overlap_ratio(bi["merged_bbox"], bj["merged_bbox"]) < threshold:
                continue

            oi = bi.get("_orig_bbox", bi["merged_bbox"])
            oj = bj.get("_orig_bbox", bj["merged_bbox"])
            if _overlap_ratio(oi, oj) >= threshold:
                # Duplicate detection from the AI - drop the shorter text
                keep_i = len(str(bi.get("hebrew_text", ""))) >= len(str(bj.get("hebrew_text", "")))
                alive[j if keep_i else i] = False
                if not keep_i:
                    break  # bubble i is gone; stop comparing it
            else:
                # Refinement collision - trust the original positions
                bi["merged_bbox"] = oi
                bj["merged_bbox"] = oj

    return [b for b, ok in zip(bubbles, alive) if ok]


def _find_light_seed(
    crop: np.ndarray,
    cx: int,
    cy: int,
    dark_thresh: int,
    max_radius: int = 40,
) -> "tuple[int, int] | None":
    """Find a light (background) pixel near (cx, cy) to seed the flood fill.

    Two passes: first a strict brightness bar (white-ish bubbles), then an
    adaptive bar so MID-TONE colored captions (purple, olive, saturated
    yellow) also qualify -- their background is clearly lighter than the
    ink, but fails the strict test.
    """
    ch, cw = crop.shape
    cx = int(np.clip(cx, 0, cw - 1))
    cy = int(np.clip(cy, 0, ch - 1))

    bars = [dark_thresh + 40, dark_thresh + 15]

    for light in bars:
        if crop[cy, cx] >= light:
            return (cx, cy)
        for r in range(2, max_radius, 2):
            for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r),
                           (r, r), (-r, r), (r, -r), (-r, -r)):
                px, py = cx + dx, cy + dy
                if 0 <= px < cw and 0 <= py < ch and crop[py, px] >= light:
                    return (px, py)
    return None


def remove_text_by_mask(
    image: np.ndarray,
    text_mask: np.ndarray,
    inpaint_radius: int = 4,
) -> np.ndarray:
    """Inpaint only the given text-pixel mask (from :func:`refine_text_regions`).

    Because the mask covers just the ink strokes (plus a small dilation) and
    never the bubble outlines, the fill comes from the immediately
    surrounding bubble background -- no black smearing.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {image.shape}")
    if cv2.countNonZero(text_mask) == 0:
        return image.copy()
    return cv2.inpaint(image, text_mask, inpaint_radius, cv2.INPAINT_TELEA)


def remove_text_from_bubbles(
    image: np.ndarray,
    bubbles: List[Dict],
) -> np.ndarray:
    """Remove text from comic speech/thought bubbles using OpenCV inpainting.

    A binary mask is built where each bubble region is white (255) and the
    rest is black (0).  The mask is dilated slightly so that text edges are
    fully covered, then ``cv2.inpaint`` is applied.

    Parameters
    ----------
    image:
        Source image as a NumPy array (H x W x 3, uint8).
    bubbles:
        List of bubble dictionaries.  Each dict must contain the key
        ``"merged_bbox"`` with value ``(x1, y1, x2, y2)``.

    Returns
    -------
    np.ndarray
        Inpainted image with original text removed from the bubble regions.

    Raises
    ------
    ValueError
        If *image* does not have 3 channels or if a bubble lacks
        ``merged_bbox``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image, got shape {image.shape}"
        )

    # Build mask
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for bubble in bubbles:
        if "merged_bbox" not in bubble:
            raise ValueError(
                f"Bubble dictionary missing required key 'merged_bbox': {bubble}"
            )

        x1, y1, x2, y2 = map(int, bubble["merged_bbox"])
        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Dilate mask to cover text edges
    kernel = np.ones(
        (_INPAINT_DILATE_KERNEL_SIZE, _INPAINT_DILATE_KERNEL_SIZE),
        dtype=np.uint8,
    )
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpaint (INPAINT_TELEA is generally faster; INPAINT_NS often looks
    # smoother for large regions -- we default to TELEA here.)
    result = cv2.inpaint(image, mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA)

    return result


def render_hebrew_text(
    image: np.ndarray,
    translated_bubbles: List[Dict],
    font_path: str | None = None,
    mirrored_page: bool = False,
) -> np.ndarray:
    """Render Hebrew text into comic speech/thought bubbles.

    Parameters
    ----------
    image:
        Background image (BGR, uint8) that has already been inpainted.
        The image may already be horizontally mirrored (for RTL panel
        order) -- that does not matter here, because the text is drawn
        ON TOP of the final image and is never flipped afterwards.
    translated_bubbles:
        List of bubble dicts, each containing at minimum:

        * ``"merged_bbox"`` -- ``(x1, y1, x2, y2)`` bounding box.
        * ``"hebrew_text"`` -- the translated Hebrew string.
    font_path:
        Path to a TrueType/OpenType font file.  If ``None``, a default
        font is selected from a list of known system paths.
    mirrored_page:
        DEPRECATED and ignored.  Text is always rendered in normal RTL
        (reshape + Bidi).  The old behaviour of reversing the string on
        mirrored pages produced mirror-writing, because the page is
        mirrored BEFORE rendering and never flipped again.

    Returns
    -------
    np.ndarray
        The image with Hebrew text drawn into each bubble.
    """
    # Resolve font path
    resolved_font = font_path or _find_default_font()
    if resolved_font is None or not os.path.isfile(resolved_font):
        raise ValueError(
            f"No suitable font found. Provide an explicit font_path. "
            f"Searched: {_DEFAULT_FONT_CANDIDATES}"
        )

    # Convert OpenCV BGR -> RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    img_h, img_w = image.shape[:2]

    for bubble in translated_bubbles:
        bbox = bubble.get("merged_bbox")
        text = bubble.get("hebrew_text", "")

        if bbox is None or not text:
            continue

        # Per-bubble font override (e.g. style-matched Hebrew font),
        # falling back to the global font if missing/invalid.
        bubble_font = bubble.get("font_path")
        if not bubble_font or not os.path.isfile(str(bubble_font)):
            bubble_font = resolved_font

        layout = bubble.get("layout_bbox")
        if layout:
            # Room measured INSIDE the bubble (already clear of the outline)
            x1, y1, x2, y2 = map(int, layout)
            pad_ratio = 0.03
        else:
            x1, y1, x2, y2 = map(int, bbox)
            # Tight box of the ORIGINAL text: give the Hebrew a little
            # breathing room (~8% per side), then pad inside.
            ex = int((x2 - x1) * 0.08)
            ey = int((y2 - y1) * 0.08)
            x1 = max(0, x1 - ex); y1 = max(0, y1 - ey)
            x2 = min(img_w, x2 + ex); y2 = min(img_h, y2 + ey)
            pad_ratio = 0.06

        bubble_w = x2 - x1
        bubble_h = y2 - y1
        if bubble_w <= 0 or bubble_h <= 0:
            continue

        pad_x = int(bubble_w * pad_ratio)
        pad_y = int(bubble_h * pad_ratio)
        max_text_w = bubble_w - 2 * pad_x
        max_text_h = bubble_h - 2 * pad_y
        if max_text_w <= 0 or max_text_h <= 0:
            continue

        # SIZE MATCHING: the Hebrew letters should be as tall as the ORIGINAL
        # lettering. That is the ceiling; the fitter only goes smaller when
        # the translation is too long for the room available.
        max_font = None
        slh = bubble.get("src_letter_h")
        if slh:
            max_font = _font_size_for_letter_height(bubble_font, slh * 1.05)

        font, wrapped_lines = _fit_font_and_wrap(
            text, bubble_font, max_text_w, max_text_h, max_font=max_font)
        if font is None or not wrapped_lines:
            continue

        display_lines = [_prepare_rtl_text(line) for line in wrapped_lines]

        # Uniform line pitch from the FONT (not from each line's own bbox):
        # per-line bboxes made spacing jump whenever a line had ל or a final
        # letter, and drawing at the bbox top shifted lines unevenly.
        top_off, line_h = _line_metrics(font)
        pitch = int(round(line_h * _LINE_SPACING))
        block_h = pitch * (len(display_lines) - 1) + line_h

        cx = (x1 + x2) / 2.0  # CENTRED - how Hebrew comics are lettered
        start_y = y1 + pad_y + (max_text_h - block_h) / 2.0

        ink = tuple(bubble.get("ink_rgb") or (0, 0, 0))
        # Light halo only where the text sits on NON-flat background (text
        # box poking over art / a rough inpaint): keeps it readable.
        region = image[max(0, y1):y2, max(0, x1):x2]
        stroke_w, stroke_fill = 0, None
        # (std on GRAYSCALE - on colour the channel spread of a flat yellow
        # caption alone exceeded the bar)
        if region.size and float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).std()) > 28:
            stroke_w = max(1, int(font.size / 12))
            lum = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
            stroke_fill = (255, 255, 255) if lum < 128 else (0, 0, 0)

        for i, line in enumerate(display_lines):
            baseline_y = start_y + i * pitch + top_off
            draw.text((cx, baseline_y), line, font=font, fill=ink, anchor="ms",
                      stroke_width=stroke_w, stroke_fill=stroke_fill)

    # Convert RGB back to BGR for OpenCV compatibility
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType/OpenType font at the requested *size* (points).

    Parameters
    ----------
    font_path:
        Absolute path to a ``.ttf`` or ``.otf`` font file.
    size:
        Font size in points.

    Returns
    -------
    ImageFont.FreeTypeFont
        A Pillow FreeType font object ready for drawing.

    Raises
    ------
    FileNotFoundError
        If *font_path* does not exist.
    OSError
        If Pillow cannot load the font file.
    """
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")

    return _cached_font(font_path, int(size))


@functools.lru_cache(maxsize=512)
def _cached_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    # The fitter tries ~15 sizes per bubble; re-reading the font file from
    # disk every time was measurable on long albums.
    return ImageFont.truetype(font_path, size)


def _line_metrics(font) -> "tuple[int, int]":
    """(offset from line top to baseline, line height) for Hebrew text."""
    bb = font.getbbox(_HEB_METRIC_REF, anchor="ls")
    top_off = -bb[1]
    return top_off, max(1, bb[3] - bb[1])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_default_font() -> str | None:
    """Return the first existing font path from *_DEFAULT_FONT_CANDIDATES*.

    Returns ``None`` if none of the candidates exist on disk.
    """
    for path in _DEFAULT_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _prepare_rtl_text(text: str, mirrored_page: bool = False) -> str:
    """Prepare Hebrew text for rendering.

    IMPORTANT: In this pipeline the page is mirrored *before* the Hebrew
    text is drawn.  The text itself is therefore never flipped afterwards,
    so it must ALWAYS be rendered in normal RTL mode — regardless of
    whether the underlying page was mirrored.

    (The old behaviour of reversing the raw string when *mirrored_page*
    was True produced mirror-writing, because no second flip ever
    happened.  The parameter is kept only for API compatibility and is
    intentionally ignored.)

    Steps:
        1. If Pillow has the Raqm layout engine, return the text UNCHANGED --
           Pillow will shape and reorder RTL text natively while drawing.
           Applying python-bidi on top would reverse the text twice.
        2. Otherwise: ``arabic_reshaper.reshape`` fixes character forms, then
           ``bidi.algorithm.get_display`` reorders the string for RTL display.

    If the fallback libraries are missing the text is returned unchanged
    (a warning is emitted at import time).
    """
    if _HAS_RAQM:
        return text
    if _HAS_ARABIC_RESHAPER:
        text = arabic_reshaper.reshape(text)
    if _HAS_BIDI:
        text = get_display(text)
    return text


def _font_size_for_letter_height(font_path: str, target_h: float) -> int:
    """Font size at which THIS font's Hebrew letters are *target_h* px tall.

    'Font size' is not glyph height: a Hebrew letter at size N is typically
    ~0.7N tall, and the ratio varies per font. Measure the actual glyph
    height of representative Hebrew letters at a reference size and scale.
    """
    try:
        ref = 100
        f = load_font(font_path, ref)
        bbox = f.getbbox("אבהחשם")  # x-height Hebrew letters, no descenders
        glyph_h = max(1, bbox[3] - bbox[1])
        return max(_MIN_FONT_SIZE, int(round(ref * float(target_h) / glyph_h)))
    except Exception:
        return max(_MIN_FONT_SIZE, int(target_h * 1.4))


def _fit_font_and_wrap(
    text: str,
    font_path: str,
    max_width: int,
    max_height: int,
    mirrored_page: bool = False,
    max_font: int | None = None,
) -> Tuple[ImageFont.FreeTypeFont | None, List[str]]:
    """Largest font size at which *text* fits in *max_width* x *max_height*.

    Words are never broken mid-word (textwrap used to split a long Hebrew
    word in two with no hyphen); a size at which one word alone is too wide
    simply doesn't fit. Only at the minimum size is breaking allowed.
    """
    lo = _MIN_FONT_SIZE
    hi = max(_MIN_FONT_SIZE, min(_MAX_FONT_SIZE, max_height))
    if max_font is not None:
        hi = min(hi, max(max_font, _MIN_FONT_SIZE))
    best_font: ImageFont.FreeTypeFont | None = None
    best_lines: List[str] = []

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = load_font(font_path, mid)
        except OSError:
            hi = mid - 1
            continue
        lines = _wrap_text_for_font(text, font, max_width)
        if lines and _measure_text_block_height(lines, font) <= max_height:
            best_font, best_lines = font, lines
            lo = mid + 1
        else:
            hi = mid - 1

    if best_font is None:
        try:
            best_font = load_font(font_path, _MIN_FONT_SIZE)
            best_lines = (_wrap_text_for_font(text, best_font, max_width)
                          or _wrap_text_for_font(text, best_font, max_width,
                                                 allow_break=True))
        except OSError:
            pass

    return best_font, best_lines


def _text_w(font, s: str) -> int:
    bb = font.getbbox(_prepare_rtl_text(s))
    return (bb[2] - bb[0]) if bb else 0


def _greedy(words: List[str], font, width: int) -> "List[str] | None":
    lines, cur = [], ""
    for w in words:
        t = w if not cur else cur + " " + w
        if _text_w(font, t) <= width:
            cur = t
        else:
            if not cur:
                return None  # a single word wider than the line
            lines.append(cur)
            cur = w
            if _text_w(font, cur) > width:
                return None
    if cur:
        lines.append(cur)
    return lines


def _wrap_text_for_font(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    mirrored_page: bool = False,
    allow_break: bool = False,
) -> List[str]:
    """Pixel-accurate, BALANCED word wrap.

    Greedy wrapping leaves ragged blocks ("long line / long line / one
    word"), which look wrong in a round bubble. After finding the greedy
    line count, the narrowest width giving the SAME count is searched, so
    lines come out even - the classic diamond/oval comic shape.
    Returns [] when the text cannot be wrapped without breaking a word.
    """
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    lines = _greedy(words, font, max_width)
    if lines is None:
        if not allow_break:
            return []
        return textwrap.wrap(text, width=max(1, int(max_width / max(
            1, _text_w(font, "א")))))
    n = len(lines)
    if n > 1:
        lo_w = max(_text_w(font, w) for w in words)
        hi_w = max_width
        while lo_w < hi_w:
            mid = (lo_w + hi_w) // 2
            cand = _greedy(words, font, mid)
            if cand is not None and len(cand) <= n:
                hi_w = mid
            else:
                lo_w = mid + 1
        cand = _greedy(words, font, hi_w)
        if cand:
            lines = cand
    return lines


def _measure_text_block_height(
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float = _LINE_SPACING,
    mirrored_page: bool = False,
) -> int:
    """Total pixel height of *lines* - same uniform pitch the renderer uses."""
    if not lines:
        return 0
    _top, line_h = _line_metrics(font)
    return int(round(line_h * line_spacing)) * (len(lines) - 1) + line_h
