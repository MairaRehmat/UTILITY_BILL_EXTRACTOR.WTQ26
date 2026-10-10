"""
OCR module for Utility Bill Decoder.
Provides local OCR extraction using pytesseract as auxiliary context.
Catches errors gracefully if Tesseract binary is not installed or configured.
"""

import os
import logging
from PIL import Image

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

logger = logging.getLogger(__name__)

# Common Tesseract Windows Installation Paths
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Administrator\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]


def _configure_tesseract():
    if not HAS_PYTESSERACT:
        return False
    # Check if tesseract is already executable in PATH or configured
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass

    # Try common Windows paths
    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                pass
    return False


def extract_ocr_text(pil_img: Image.Image) -> str:
    """
    Extract text from image using Tesseract OCR.
    Returns extracted text string, or empty string if Tesseract is unavailable or fails.
    """
    if not HAS_PYTESSERACT:
        logger.info("pytesseract package not imported. Skipping local OCR.")
        return ""

    if not _configure_tesseract():
        logger.info("Tesseract executable not found or configured. Skipping local OCR.")
        return ""

    try:
        # Perform OCR with Urdu + English if available, or default eng
        text = pytesseract.image_to_string(pil_img, config="--psm 6")
        return text.strip()
    except Exception as e:
        logger.warning(f"Local OCR extraction encountered an error: {e}")
        return ""
