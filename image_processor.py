"""
Image Processor module for Utility Bill Decoder.
Handles loading, preprocessing, deskewing, and enhancing bill images for OCR and Gemini Vision.
"""

import os
import io
from typing import Tuple, Optional
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_image(image_path: str) -> Image.Image:
    """Load an image from file path and convert to RGB PIL Image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)  # Correct orientation based on EXIF tag
    return image.convert("RGB")


def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Preprocess image to enhance readability for OCR and Vision models.
    Adjusts contrast, sharpness, and normalizes dimensions safely.
    """
    if pil_img is None:
        raise ValueError("Input PIL Image cannot be None.")

    # Convert to RGB if not already
    enhanced = pil_img.convert("RGB")

    # Resize if extremely large or small for consistent quality
    max_dim = 2400
    w, h = enhanced.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)
    elif max(w, h) < 800:
        scale = 1200 / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)

    # Enhance Contrast & Sharpness safely
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.2)
    sharpener = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpener.enhance(1.3)

    if HAS_CV2:
        try:
            cv_img = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) > 100:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if abs(angle) > 0.5 and abs(angle) < 45.0:
                    (h_cv, w_cv) = cv_img.shape[:2]
                    center = (w_cv // 2, h_cv // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(
                        cv_img, M, (w_cv, h_cv), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )
                    enhanced = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        except Exception:
            pass  # Fall back gracefully if deskew fails

    return enhanced


def pil_to_jpeg_bytes(pil_img: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes for API payload transmission."""
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def save_temp_image(pil_img: Image.Image, output_path: str) -> str:
    """Save PIL Image to temporary path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_img.save(output_path, quality=95)
    return output_path
