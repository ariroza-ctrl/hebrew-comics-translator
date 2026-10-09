@echo off
chcp 65001 > nul
echo ============================================================
echo   Building ComicTranslatorHebrew.exe  (spec-based build)
echo ============================================================
echo.

REM Install only what the exe needs (NO easyocr/torch - keeps it small)
python -m pip install pyinstaller opencv-python numpy Pillow requests python-bidi arabic-reshaper PyMuPDF rarfile

echo.
echo Building exe from spec (this takes a few minutes)...
python -m PyInstaller --noconfirm ComicTranslatorHebrew.spec

echo.
if exist dist\ComicTranslatorHebrew.exe (
    echo ============================================================
    echo   SUCCESS!  The exe is at:  dist\ComicTranslatorHebrew.exe
    echo   You can copy it anywhere - no Python needed to run it.
    echo ============================================================
) else (
    echo   BUILD FAILED - scroll up to see the error.
)
pause
