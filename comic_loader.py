"""
Comic Book Loader — Extract pages from PDF, CBZ, CBR files.

Supports:
    * PDF  — extracts each page as an image
    * CBZ  — Comic Book Zip (zip archive of images)
    * CBR  — Comic Book RAR (rar archive of images)
    * JPG/PNG/WEBP — single image files (passed through as-is)

Usage::

    pages = load_comic_pages("manga.pdf")
    for i, page_image in enumerate(pages):
        ...
"""

from __future__ import annotations

import io
import os
import warnings
import zipfile
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Supported image extensions (tried in this order)
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


import re


def _natural_key(s: str):
    """Sort key that orders page_2 before page_10 (numeric-aware).

    Critical for scanned albums where lexicographic sort scrambles pages.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _sniff_format(path: Path) -> str:
    """Detect the REAL container format from magic bytes.

    Many scanned comic files are mislabeled: a '.cbz' that is actually a RAR
    or 7z archive (or vice versa) is extremely common in the wild. Returns
    one of: 'zip', 'rar', '7z', 'pdf', or '' (unknown).
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return ""
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") \
            or head.startswith(b"PK\x07\x08"):
        return "zip"
    if head.startswith(b"Rar!"):
        return "rar"
    if head.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if head.startswith(b"%PDF"):
        return "pdf"
    return ""


def _setup_rarfile_tool(rarfile_mod) -> None:
    """Help rarfile find an extraction backend on Windows.

    rarfile needs an external tool (unrar / unar / bsdtar). On Windows,
    users rarely have 'unrar' on PATH, but very often have WinRAR installed
    -- point rarfile at its UnRAR.exe. As a last resort, Windows 10+ ships
    bsdtar as C:\\Windows\\System32\\tar.exe, which rarfile can also use.
    """
    import shutil

    if shutil.which(getattr(rarfile_mod, "UNRAR_TOOL", "unrar")):
        return  # unrar already on PATH

    candidates = [
        r"C:\Program Files\WinRAR\UnRAR.exe",
        r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
        r"C:\Program Files\WinRAR\unrar.exe",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            rarfile_mod.UNRAR_TOOL = cand
            return
    # else: leave rarfile to auto-detect unar/bsdtar (tar.exe on Win10+)


def load_comic_pages(source: Union[str, Path]) -> List[Image.Image]:
    """Load comic pages from a file (PDF, CBZ, CBR, or single image).

    Parameters
    ----------
    source:
        Path to a ``.pdf``, ``.cbz``, ``.cbr``, or image file.

    Returns
    -------
    List[Image.Image]
        A list of PIL RGB images, one per page.

    Raises
    ------
    FileNotFoundError
        If *source* does not exist.
    ValueError
        If the file format is unsupported.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    ext = source.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(source)
    elif ext in (".cbz", ".cbr", ".cb7"):
        # Trust the file's MAGIC BYTES over its extension: '.cbz' files that
        # are really RAR/7z archives (and vice versa) are extremely common
        # in scanned comic collections.
        real = _sniff_format(source)
        if real == "zip":
            return _load_cbz(source)
        if real == "rar":
            return _load_cbr(source)
        if real == "pdf":
            return _load_pdf(source)
        if real == "7z":
            pages = _extract_with_bsdtar(source)
            if pages:
                return pages
            raise RuntimeError(
                f"'{source.name}' is actually a 7-Zip archive. Windows tar.exe "
                "could not extract it. Fix: open it with 7-Zip, extract the "
                "images, and re-zip them as <name>.cbz."
            )
        # Unknown magic: try everything, most likely a corrupt download
        try:
            return _load_cbz(source)
        except Exception:
            pass
        try:
            return _load_cbr(source)
        except Exception:
            pass
        pages = _extract_with_bsdtar(source)
        if pages:
            return pages
        with open(source, "rb") as f:
            head = f.read(8)
        raise RuntimeError(
            f"Cannot open '{source.name}': it is not a ZIP, RAR, 7z or PDF "
            f"(file starts with bytes {head!r}). The download is most likely "
            "CORRUPT or INCOMPLETE - try re-downloading the file and check "
            "that its size matches the source."
        )
    elif ext in _IMAGE_EXTS:
        img = Image.open(source).convert("RGB")
        return [img]
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: .pdf, .cbz, .cbr, {', '.join(_IMAGE_EXTS)}"
        )


def _load_pdf(path: Path) -> List[Image.Image]:
    """Extract pages from a PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. "
            "Install:  pip install PyMuPDF"
        )

    pages: List[Image.Image] = []
    doc = fitz.open(str(path))

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Render at 2x resolution for better OCR quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        pages.append(img)

    doc.close()
    return pages


def _load_cbz(path: Path) -> List[Image.Image]:
    """Extract images from a CBZ (zip) archive."""
    pages: List[Image.Image] = []

    with zipfile.ZipFile(path, "r") as zf:
        # Natural sort (page_2 before page_10) to preserve true page order
        names = sorted(
            [n for n in zf.namelist()
             if Path(n).suffix.lower() in _IMAGE_EXTS
             and not Path(n).name.startswith(".")        # macOS junk
             and "__MACOSX" not in n],
            key=_natural_key,
        )
        for name in names:
            data = zf.read(name)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            pages.append(img)

    if not pages:
        warnings.warn(f"No image files found in CBZ: {path}")
    return pages


def _extract_with_bsdtar(path: Path) -> List[Image.Image]:
    """Fallback CBR extraction using bsdtar.

    Windows 10/11 ship libarchive's bsdtar as C:\\Windows\\System32\\tar.exe,
    which can read many RAR archives -- no installation needed. Returns []
    if tar is unavailable or extraction failed.
    """
    import shutil
    import subprocess
    import tempfile

    tar = shutil.which("tar")
    if not tar:
        return []

    pages: List[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [tar, "-xf", str(path), "-C", tmp],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            return []
        found = sorted(
            (p for p in Path(tmp).rglob("*")
             if p.suffix.lower() in _IMAGE_EXTS
             and not p.name.startswith(".")
             and "__MACOSX" not in str(p)),
            key=lambda p: _natural_key(str(p)),
        )
        for p in found:
            try:
                pages.append(Image.open(p).convert("RGB"))
            except Exception:
                continue
    return pages


def _load_cbr(path: Path) -> List[Image.Image]:
    """Extract images from a CBR (rar) archive.

    Tries, in order:
      1. rarfile + unrar / WinRAR's UnRAR.exe (auto-detected on Windows)
      2. Windows built-in tar.exe (bsdtar) -- no installation needed
    """
    pages: List[Image.Image] = []
    rar_error = None

    # --- Attempt 1: rarfile ------------------------------------------------
    try:
        import rarfile
        _setup_rarfile_tool(rarfile)
        with rarfile.RarFile(str(path), "r") as rf:
            names = sorted(
                [n for n in rf.namelist()
                 if Path(n).suffix.lower() in _IMAGE_EXTS
                 and not Path(n).name.startswith(".")
                 and "__MACOSX" not in n],
                key=_natural_key,
            )
            for name in names:
                data = rf.read(name)
                pages.append(Image.open(io.BytesIO(data)).convert("RGB"))
        if pages:
            return pages
    except ImportError:
        rar_error = "rarfile package not installed (pip install rarfile)"
    except Exception as exc:  # RarCannotExec, corrupt archive, etc.
        rar_error = str(exc)

    # --- Attempt 2: Windows built-in bsdtar --------------------------------
    pages = _extract_with_bsdtar(path)
    if pages:
        return pages

    raise RuntimeError(
        "Could not open this CBR file.\n"
        f"rarfile said: {rar_error}\n"
        "Fixes (any ONE of these):\n"
        "  1. Install WinRAR (https://www.win-rar.com) and restart the app "
        "- it will be auto-detected.\n"
        "  2. Convert once with 7-Zip: right-click the .cbr -> Extract, then "
        "zip the images and rename to <name>.cbz (CBZ needs no extra tools).\n"
        "  3. If the file might be corrupt, try re-downloading it."
    )


def save_comic_pages(
    pages: List[Image.Image],
    output_dir: Union[str, Path],
    prefix: str = "page",
    format: str = "png",
) -> List[Path]:
    """Save a list of comic pages to a directory.

    Parameters
    ----------
    pages:
        List of PIL images to save.
    output_dir:
        Destination directory (created if it doesn't exist).
    prefix:
        Filename prefix (e.g. ``'page'`` produces ``page_001.png``).
    format:
        Output image format (``'png'``, ``'jpg'``, ``'webp'``).

    Returns
    -------
    List[Path]
        Paths of the saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for i, img in enumerate(pages):
        filename = f"{prefix}_{i+1:03d}.{format.lower()}"
        filepath = output_dir / filename
        img.save(filepath, format=format.upper())
        saved.append(filepath)

    return saved


def create_cbz(
    image_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
) -> Path:
    """Create a CBZ (Comic Book Zip) from a list of image files.

    Parameters
    ----------
    image_paths:
        List of image file paths to include.
    output_path:
        Destination ``.cbz`` file path.

    Returns
    -------
    Path
        The created CBZ file path.
    """
    output_path = Path(output_path)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in image_paths:
            img_path = Path(img_path)
            zf.write(img_path, arcname=img_path.name)
    return output_path


def create_pdf(
    images: List[Image.Image],
    output_path: Union[str, Path],
) -> Path:
    """Create a PDF from a list of PIL images.

    Parameters
    ----------
    images:
        List of PIL RGB images.
    output_path:
        Destination ``.pdf`` file path.

    Returns
    -------
    Path
        The created PDF file path.
    """
    output_path = Path(output_path)
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=150.0,
        )
    return output_path
