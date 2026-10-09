#!/usr/bin/env python3
"""
main.py - Comic Translator to Hebrew v4

Three modes:
  Local  (EasyOCR + Google Translate) - free, offline after first run
  Gemini (Google AI vision)           - free API key, great OCR
  Torii  (Professional)               - paid, best inpainting/typesetting

Pipeline: OCR -> Translate -> Refine text regions -> Pixel-mask inpaint 
          -> Mirror page -> Mirror coords -> Render Hebrew RTL
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import comic_loader
import image_processor
import ocr_engine
import translator

# Optional backends
_HAS_GEMINI = False
try:
    import gemini_api
    _HAS_GEMINI = True
except ImportError:
    pass

_HAS_TORII = False
try:
    import torii_api
    _HAS_TORII = True
except ImportError:
    pass


def _bubble_to_dict(bubble) -> Dict[str, Any]:
    return {"text": bubble.text, "merged_bbox": bubble.merged_bbox}


def _remove_text_refined(img_bgr, bubbles, clean_bubbles=True):
    """Smart refine + optional bubble-background cleaning + pixel-mask inpaint."""
    refined, text_mask, interiors = image_processor.refine_text_regions(img_bgr, bubbles)
    if clean_bubbles:
        img_bgr = image_processor.clean_bubble_interiors(img_bgr, interiors)
    img_bgr, rest = image_processor.flat_fill_text(img_bgr, text_mask, interiors)
    cleaned = image_processor.remove_text_by_mask(img_bgr, rest)
    return cleaned, refined


def process_local(image: Image.Image) -> Tuple[Image.Image, str]:
    logs = []
    def log(msg): logs.append(msg); print(msg)

    try:
        log("[Local] Step 1: OCR with EasyOCR...")
        img_arr = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        regions = ocr_engine.detect_text_regions(img_bgr)
        if not regions:
            log("  No text found."); return image, "\n".join(logs)
        bubbles = ocr_engine.group_text_regions(regions)
        log(f"  {len(bubbles)} bubble(s)")

        log("[Local] Step 2: Translate to Hebrew...")
        bdicts = [_bubble_to_dict(b) for b in bubbles]
        bdicts = translator.translate_bubbles(bdicts)
        for b in bdicts:
            log(f"  '{b['text'][:30]}' -> '{b.get('hebrew_text','')[:30]}'")

        log("[Local] Step 3: Smart inpaint (refine + pixel mask)...")
        cleaned, refined = _remove_text_refined(img_bgr, bdicts)

        log("[Local] Step 4: Mirror page + coords for RTL...")
        mirrored = image_processor.mirror_image(cleaned)
        mirrored_bubbles = image_processor.mirror_bubbles(refined, w)

        log("[Local] Step 5: Render Hebrew RTL...")
        result_bgr = image_processor.render_hebrew_text(mirrored, mirrored_bubbles)
        result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        log("  DONE!"); return result, "\n".join(logs)
    except Exception as exc:
        traceback.print_exc(); log(f"ERROR: {exc}"); return image, "\n".join(logs)


def process_gemini(image: Image.Image, memory: str = "") -> Tuple[Image.Image, str, str]:
    logs = []
    def log(msg): logs.append(msg); print(msg)

    try:
        log("[Gemini] Step 1: Gemini OCR + Hebrew translation...")
        bubbles, memory = gemini_api.ocr_and_translate(
            image, memory=memory or None, return_memory=True)
        if not bubbles:
            log("  No text found."); return image.transpose(Image.FLIP_LEFT_RIGHT), "\n".join(logs), memory
        log(f"  {len(bubbles)} region(s)")
        for b in bubbles:
            log(f"  '{b['text'][:35]}' -> '{b['hebrew_text'][:35]}'")

        img_arr = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        log("[Gemini] Step 2: Smart inpaint (refine + pixel mask)...")
        cleaned, refined = _remove_text_refined(img_bgr, bubbles)

        log("[Gemini] Step 3: Mirror page + coords for RTL...")
        mirrored = image_processor.mirror_image(cleaned)
        mirrored_bubbles = image_processor.mirror_bubbles(refined, w)

        log("[Gemini] Step 4: Render Hebrew RTL...")
        result_bgr = image_processor.render_hebrew_text(mirrored, mirrored_bubbles)
        result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        log("  DONE!"); return result, "\n".join(logs), memory
    except Exception as exc:
        traceback.print_exc(); log(f"ERROR: {exc}"); return image, "\n".join(logs), memory


def process_torii(image: Image.Image) -> Tuple[Image.Image, str]:
    logs = []
    def log(msg): logs.append(msg); print(msg)

    # NOTE: never mirror AFTER text is rendered -- that produces mirror-writing.
    # Correct flow: get the INPAINTED image + translated boxes from /v2/upload,
    # mirror the CLEAN image, mirror the box coords, then typeset the Hebrew.
    MIN_FONT = 5
    BOX_SCALE = 0.88  # inner margin so text stays inside the bubble

    try:
        log("[Torii] Step 1: Torii OCR + translate (returns clean image + boxes)...")
        result = torii_api.translate_full(
            image, target_lang="he", translator="gemini-3.1-flash-lite",
            font="NotoSans", text_align="right", min_font_size=MIN_FONT,
        )
        boxes = result["text_boxes"]
        inpainted = result["inpainted"]

        if inpainted is None:
            log("  WARNING: no inpainted image returned; NOT mirroring "
                "(mirroring the rendered image would flip the Hebrew).")
            return result["image"] or image, "\n".join(logs)
        if not boxes:
            log("  No text found - mirroring clean page only.")
            return inpainted.transpose(Image.FLIP_LEFT_RIGHT), "\n".join(logs)

        log(f"  {len(boxes)} text box(es).")
        log("[Torii] Step 2: Mirror CLEAN page + box coords...")
        mirrored = inpainted.transpose(Image.FLIP_LEFT_RIGHT)
        page_w = inpainted.width
        text_boxes = []
        for b in boxes:
            cx, cy = float(b["x"]), float(b["y"])       # upload returns CENTER
            w0, h0 = float(b["width"]) * BOX_SCALE, float(b["height"]) * BOX_SCALE
            text_boxes.append({
                "x": int(round((page_w - cx) - w0 / 2)),  # typeset wants TOP-LEFT
                "y": int(round(cy - h0 / 2)),
                "width": int(round(w0)), "height": int(round(h0)),
                "text": b.get("text", ""), "alignment": "right",
                "text_color": b.get("fillColor", "#000000"),
                "stroke_color": b.get("strokeColor", "#ffffff"),
            })

        log("[Torii] Step 3: Typeset Hebrew on mirrored page...")
        final = torii_api.typeset_text(mirrored, text_boxes,
                                       font="NotoSans", min_font_size=MIN_FONT)
        log("  DONE!"); return final, "\n".join(logs)
    except Exception as exc:
        traceback.print_exc(); log(f"ERROR: {exc}"); return image, "\n".join(logs)


def process_single_page(image: Image.Image, mode: str = "gemini",
                        memory: str = "") -> Tuple[Image.Image, str, str]:
    """Returns (image, log, updated_album_memory)."""
    if mode == "gemini" and _HAS_GEMINI:
        return process_gemini(image, memory=memory)
    if mode == "torii" and _HAS_TORII:
        img, lg = process_torii(image); return img, lg, memory
    img, lg = process_local(image); return img, lg, memory


def process_comic_file(file_path: str, mode: str = "gemini") -> Tuple[List[Image.Image], str]:
    try:
        pages = comic_loader.load_comic_pages(Path(file_path))
    except Exception as exc:
        traceback.print_exc()
        return [], f"FAILED TO LOAD FILE:\n{exc}"
    if not pages:
        return [], f"No pages found in {Path(file_path).name}"
    results, all_logs = [], [f"Loaded {len(pages)} page(s) from {Path(file_path).name}"]
    memory = ""
    for i, page in enumerate(pages):
        all_logs.append(f"\n=== Page {i+1}/{len(pages)} ===")
        proc, log, memory = process_single_page(page, mode=mode, memory=memory)
        results.append(proc); all_logs.append(log)
    return results, "\n".join(all_logs)


def process_and_export(file_path: str, out_fmt: str, mode: str = "gemini") -> Tuple[List, str, str, List[str]]:
    if not file_path or not os.path.exists(file_path):
        return [], "Please upload a file.", "", []
    images, msg = process_comic_file(file_path, mode=mode)
    if not images: return [], msg, msg, []
    stem = Path(file_path).stem
    # Save next to the project, in ./output/<name>_hebrew/ (easy to find!)
    out = Path(__file__).parent / "output" / f"{stem}_hebrew"
    out.mkdir(parents=True, exist_ok=True)
    if out_fmt == "CBZ":
        paths = comic_loader.save_comic_pages(images, out, "page", "png")
        cbz = comic_loader.create_cbz(paths, out / f"{stem}_hebrew.cbz")
        return images, f"Saved CBZ: {cbz}", msg, [str(cbz)]
    elif out_fmt == "PDF":
        pdf = comic_loader.create_pdf(images, out / f"{stem}_hebrew.pdf")
        return images, f"Saved PDF: {pdf}", msg, [str(pdf)]
    else:
        paths = comic_loader.save_comic_pages(images, out, "page", "png")
        return images, f"Saved {len(paths)} images to {out}", msg, [str(p) for p in paths]


def _build_interface() -> gr.Blocks:
    with gr.Blocks(title="Comic Translator to Hebrew") as app:
        gr.Markdown(
            "# Comic Translator to Hebrew\n\n"
            "**Gemini mode** (default): Free, great OCR, uses Google's vision AI.\n"
            "Get a free API key at [aistudio.google.com](https://aistudio.google.com)\n\n"
            "---"
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Upload Comic File",
                    file_types=[".pdf",".cbz",".cbr",".jpg",".jpeg",".png",".webp",".gif"],
                )
                mode_in = gr.Radio(
                    choices=["Gemini (Free - Recommended)", "Local (Free - EasyOCR)", "Torii API (Paid)"],
                    value="Gemini (Free - Recommended)", label="Mode",
                )
                fmt_in = gr.Radio(choices=["Images","CBZ","PDF"], value="Images", label="Output")
                go_btn = gr.Button("Translate to Hebrew!", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=2)
                downloads = gr.Files(label="Download Results", interactive=False)
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Translated Pages", columns=2, rows=2, height="600px")
        with gr.Row():
            logs = gr.Textbox(label="Detailed Logs", interactive=False, lines=15, max_lines=30)

        def on_translate(file_obj, mode_choice, out_fmt):
            """Streaming handler: yields UI updates after EVERY page.

            Long albums take many minutes; a single blocking return lets the
            browser connection time out silently. Streaming keeps it alive,
            shows real progress, and each page is saved to disk immediately.
            """
            if file_obj is None:
                yield [], "Upload a file.", "", []
                return
            mode = "gemini" if "Gemini" in mode_choice else ("torii" if "Torii" in mode_choice else "local")

            src = file_obj.name
            try:
                pages = comic_loader.load_comic_pages(Path(src))
            except Exception as exc:
                traceback.print_exc()
                yield [], f"FAILED TO LOAD FILE:\n{exc}", str(exc), []
                return
            total = len(pages)
            if not total:
                yield [], "No pages found in file.", "", []
                return

            stem = Path(src).stem
            out = Path(__file__).parent / "output" / f"{stem}_hebrew"
            out.mkdir(parents=True, exist_ok=True)

            gallery_imgs, logs, files = [], [f"Loaded {total} page(s) from {Path(src).name}"], []
            album_memory = ""
            yield gallery_imgs, f"Loaded {total} page(s). Starting...", "\n".join(logs), files

            for i, page in enumerate(pages):
                logs.append(f"\n=== Page {i+1}/{total} ===")
                try:
                    proc, plog, album_memory = process_single_page(page, mode=mode, memory=album_memory)
                except Exception as exc:
                    traceback.print_exc()
                    proc, plog = page, f"ERROR on page {i+1}: {exc}"
                logs.append(plog)

                # Save IMMEDIATELY - nothing is lost if the browser/app dies
                fp = out / f"page_{i+1:03d}.png"
                proc.save(fp)
                files.append(str(fp))
                gallery_imgs.append(proc)

                pct = (i + 1) / total * 100
                yield (gallery_imgs,
                       f"Page {i+1}/{total} done ({pct:.0f}%) - saving to: {out}",
                       "\n".join(logs), files)

            # Final packaging
            if out_fmt == "CBZ":
                cbz = comic_loader.create_cbz(files, out / f"{stem}_hebrew.cbz")
                files = [str(cbz)]
                status = f"DONE! {len(gallery_imgs)}/{total} pages. CBZ: {cbz}"
            elif out_fmt == "PDF":
                pdf = comic_loader.create_pdf(gallery_imgs, out / f"{stem}_hebrew.pdf")
                files = [str(pdf)]
                status = f"DONE! {len(gallery_imgs)}/{total} pages. PDF: {pdf}"
            else:
                status = f"DONE! {len(gallery_imgs)}/{total} pages saved in: {out}"
            logs.append(f"\n{status}")
            yield gallery_imgs, status, "\n".join(logs), files

        go_btn.click(fn=on_translate, inputs=[file_in, mode_in, fmt_in],
                     outputs=[gallery, status, logs, downloads])
    return app


if __name__ == "__main__":
    print("=" * 60)
    print("  Comic Translator to Hebrew v4")
    print(f"  Gemini: {'OK' if _HAS_GEMINI else 'N/A'} | Torii: {'OK' if _HAS_TORII else 'N/A'}")
    print("=" * 60)
    _build_interface().launch(share=True)
