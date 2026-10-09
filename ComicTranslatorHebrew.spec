# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for ComicTranslatorHebrew.exe
#
# Why a spec file: the default analysis hits Python's recursion limit on
# this dependency tree (deep import chains in opencv/PyMuPDF). Raising the
# limit here is the officially recommended fix.

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 10)

import os
from PyInstaller.utils.hooks import collect_all

# ============================================================
#  שנה כאן את שם קובץ ה-EXE שייווצר.
#  אייקון: שים קובץ app_icon.ico ליד קובץ זה - הוא ייארז אוטומטית
#  (אייקון הקובץ ב-Explorer + אייקון החלון ושורת המשימות).
# ============================================================
EXE_NAME = "ComicTranslatorHebrew"
ICON_FILE = "app_icon.ico" if os.path.exists("app_icon.ico") else None

datas = []

# Bundle the name glossary so the frozen exe behaves like the dev run.
if os.path.exists("glossary.json"):
    datas.append(("glossary.json", "."))

# The page-editor template must ship with the exe, or the editor export
# silently degrades on user machines (the mixed-environment lesson).
if os.path.exists("editor_template.html"):
    datas.append(("editor_template.html", "."))
binaries = []
hiddenimports = ["fitz", "rarfile"]

if ICON_FILE:
    datas.append((ICON_FILE, "."))  # bundle so the window icon works too

# arabic_reshaper ships a config data file; bidi is small - bundle fully.
for pkg in ("arabic_reshaper", "bidi"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # These are installed on the build machine (from the old requirements)
    # but must NOT be dragged into the exe - they are huge and unused here.
    excludes=[
        "torch", "torchvision", "easyocr", "gradio", "deep_translator",
        "matplotlib", "pandas", "scipy", "IPython", "jedi", "PyQt5",
        "PySide6", "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=EXE_NAME,
    icon=ICON_FILE,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # set True temporarily to debug startup crashes
    disable_windowed_traceback=False,
)
