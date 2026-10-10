"""
Extraction engine for Utility Bill Decoder using Gemini 3.1 Flash-Lite.
Handles Level 1 JSON vision extraction and Level 2 Q&A answering with full diagnostics.
"""

import os
import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite")

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from source.image_processor import load_image, preprocess_image_for_ocr, pil_to_jpeg_bytes
from source.ocr import extract_ocr_text
from source.prompts import (
    LEVEL1_EXTRACTION_SYSTEM_PROMPT,
    LEVEL2_QA_SYSTEM_PROMPT,
    build_level1_user_prompt,
    build_level2_user_prompt,
)
from source.validation import sanitize_and_validate_level1
from source.calculations import compute_bill_metrics

logger = logging.getLogger(__name__)

# Valid Google AI Studio API keys always start with "AIza"
_VALID_KEY_PREFIX = ""


def get_api_key() -> str:
    """Retrieve Google Gemini API key from environment variables."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    return (key or "").strip()


def validate_api_key_format(key: str) -> Tuple[bool, str]:
    """
    Validate that the API key has the correct Google AI Studio format.
    Returns (is_valid, diagnostic_message).
    """
    if not key:
        return False, (
            "No API key configured. Add `GEMINI_API_KEY=AIzaSy...` to `WTQ/.env`."
        )
    if len(key) < 20:
        prefix = key[:8] if len(key) >= 8 else key
        return False, (
            f"Invalid API key format (starts with '{prefix}...'). "
            "Google AI Studio API keys always start with 'AIzaSy...'. "
            "The current key appears to be an OAuth access token or GCP service credential, "
            "which is NOT supported for direct API key authentication. "
            "Get a valid key from https://aistudio.google.com/apikey"
        )
    return True, "ok"


def get_genai_client() -> Optional[Any]:
    """
    Initialize Google GenAI client using official SDK and API key authentication.
    Returns None with a logged error if key is missing or malformed.
    """
    api_key = get_api_key()
    is_valid, diag = validate_api_key_format(api_key)
    if not is_valid:
        logger.error(f"API Key validation failed: {diag}")
        return None
    if not HAS_GENAI:
        logger.error("google-genai Python library is not installed. Run: pip install google-genai")
        return None
    return genai.Client(api_key=api_key)


def extract_level1_json(image_path: str) -> Tuple[Dict[str, Any], str, str]:
    """
    Extract Level 1 structured bill JSON from image using Gemini 3.1 Flash-Lite.
    Returns (validated_json_dict, raw_response_str, error_diagnostic_message).
    If extraction succeeds, error_diagnostic_message is "".
    """
    # --- Step 1: Validate API key format before attempting any network call ---
    api_key = get_api_key()
    is_valid, key_diag = validate_api_key_format(api_key)
    if not is_valid:
        return sanitize_and_validate_level1({}), "", key_diag

    # --- Step 2: Load and preprocess image ---
    try:
        pil_img = load_image(image_path)
        processed_img = preprocess_image_for_ocr(pil_img)
    except Exception as img_err:
        err_msg = f"Image Processing Error: {img_err}"
        logger.error(err_msg)
        return sanitize_and_validate_level1({}), "", err_msg

    # --- Step 3: Local OCR (supporting evidence only; never blocks Gemini) ---
    ocr_text = ""
    try:
        ocr_text = extract_ocr_text(processed_img)
        if ocr_text:
            logger.info(f"OCR extracted {len(ocr_text)} chars of supporting text.")
    except Exception as ocr_err:
        logger.warning(f"OCR skipped: {ocr_err}")

    # --- Step 4: Initialize Gemini client ---
    client = get_genai_client()
    if not client:
        return sanitize_and_validate_level1({}), "", key_diag

    # --- Step 5: Convert image to JPEG bytes for reliable API transmission ---
    try:
        img_bytes = pil_to_jpeg_bytes(processed_img)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
    except Exception as conv_err:
        err_msg = f"Image Conversion Error: {conv_err}"
        logger.error(err_msg)
        return sanitize_and_validate_level1({}), "", err_msg

    user_prompt = build_level1_user_prompt(ocr_text)

    # --- Step 6: Call Gemini vision with 1 retry on network errors ---
    raw_text = ""
    last_error = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[image_part, user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=LEVEL1_EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text or ""
            last_error = None
            break
        except Exception as e:
            last_error = e
            err_str = str(e)
            logger.warning(f"Gemini attempt {attempt + 1} failed: {err_str[:200]}")
            if "401" in err_str or "403" in err_str:
                break  # Auth errors will not be resolved by retry
            if attempt < 1:
                time.sleep(2)

    if last_error is not None:
        diagnostic_msg = diagnose_api_error(str(last_error))
        logger.error(f"Gemini Vision Extraction Error: {last_error}")
        return sanitize_and_validate_level1({}), str(last_error), diagnostic_msg

    if not raw_text.strip():
        msg = "API returned an empty response body. The model may have declined to process this image."
        return sanitize_and_validate_level1({}), "", msg

    # --- Step 7: Parse JSON from response ---
    json_dict = parse_json_response(raw_text)
    if not json_dict:
        msg = f"Response parsing failed: model output could not be parsed as valid JSON.\nRaw response: {raw_text[:300]}"
        logger.error(msg)
        return sanitize_and_validate_level1({}), raw_text, msg

    # --- Step 8: Pydantic validation ---
    validated_dict = sanitize_and_validate_level1(json_dict)
    logger.info(f"Extraction succeeded. Provider: {validated_dict.get('provider')}, Month: {validated_dict.get('bill_month')}")
    return validated_dict, raw_text, ""


def diagnose_api_error(error_str: str) -> str:
    """Categorize API errors into clear, actionable diagnostic messages without exposing secrets."""
    err_lower = error_str.lower()
    key = get_api_key()
    prefix_hint = f"Key prefix: '{key[:5]}...'" if key else "No key set"

    if "401" in error_str or "unauthenticated" in err_lower or "access_token_type_unsupported" in err_lower:
        return (
            f"Authentication Error (401) — {prefix_hint}. "
            "Your GEMINI_API_KEY is invalid or the wrong type. "
            "Get a valid key starting with 'AIzaSy...' from https://aistudio.google.com/apikey "
            "and update GEMINI_API_KEY in WTQ/.env."
        )
    if "403" in error_str or "permission_denied" in err_lower:
        return f"Permission Error (403) — {prefix_hint}. The key lacks permission for model '{MODEL_NAME}'."
    if "404" in error_str or "not_found" in err_lower:
        return f"Model Not Found (404): Model '{MODEL_NAME}' is not available. Check MODEL_NAME in WTQ/.env."
    if "429" in error_str or "resource_exhausted" in err_lower or "quota" in err_lower:
        return "Rate Limit (429): API quota exceeded. Wait a moment and try again, or check your Google AI Studio plan."
    if "500" in error_str or "503" in error_str or "internal" in err_lower or "unavailable" in err_lower:
        return "Server Error (500/503): Google Gemini API is temporarily unavailable. Please try again."
    if "timeout" in err_lower or "connection" in err_lower or "winerror" in err_lower or "network" in err_lower:
        return "Network Error: Could not reach Google Gemini API. Check your internet connection."
    return f"Gemini API Error: {error_str[:300]}"


def answer_level2_question(image_path: str, question: str, bill_json: Dict[str, Any]) -> str:
    """
    Answer a customer question about the bill using Gemini 3.1 Flash-Lite
    grounded on extracted JSON and Python-calculated metrics.
    """
    if not question or not question.strip():
        return "Please ask a specific question about your bill."

    q_lower = question.strip().lower()
    math_context = compute_bill_metrics(bill_json)

    # Quick deterministic handlers strictly from extracted JSON
    if any(x in q_lower for x in ["how much do i have to pay", "how much is my bill", "need to pay", "amount to pay"]):
        payable = bill_json.get("payable_within_due_date") or bill_json.get("current_bill")
        if payable is not None:
            return f"Your payable bill amount within due date is PKR {payable:,.2f}."
        return "The bill does not provide enough information to determine this."

    if any(x in q_lower for x in ["when is my due date", "due date"]):
        due = bill_json.get("due_date")
        return f"The due date for this bill is {due}." if due else "The bill does not provide enough information to determine this."

    if any(x in q_lower for x in ["how many units", "units did i consume", "units consumed"]):
        units = bill_json.get("units_consumed")
        return f"You consumed {int(units):,} units of electricity this month." if units is not None else "The bill does not provide enough information to determine this."

    if any(x in q_lower for x in ["what taxes", "taxes are included", "tax"]):
        tot_tax = bill_json.get("total_taxes")
        tax_pct = math_context.get("tax_percentage_of_bill", 0)
        return f"Total government taxes amount to PKR {tot_tax:,.2f} ({tax_pct}% of total bill)." if tot_tax is not None else "The bill does not provide enough information to determine this."

    if any(x in q_lower for x in ["additional charges", "other charges"]):
        tot_chg = bill_json.get("total_charges")
        return f"Total electricity charges before taxes amount to PKR {tot_chg:,.2f}." if tot_chg is not None else "The bill does not provide enough information to determine this."

    if any(x in q_lower for x in ["difference between", "payable amount"]):
        pw = bill_json.get("payable_within_due_date") or bill_json.get("current_bill")
        pa = bill_json.get("payable_after_due_date")
        diff = math_context.get("late_payment_surcharge_pkr", 0.0)
        if pa is not None and pw is not None:
            return f"Payable within due date: PKR {pw:,.2f}. After due date: PKR {pa:,.2f} (late payment difference: PKR {diff:,.2f})."
        return "The bill does not provide enough information to determine this."

    # Fall back to Gemini for complex questions
    client = get_genai_client()
    if not client:
        return _fallback_answer_question(question, bill_json, math_context)

    user_prompt = build_level2_user_prompt(question, bill_json, math_context)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=LEVEL2_QA_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        answer = (response.text or "").strip()
        return answer if answer else "The bill does not provide enough information to determine this."
    except Exception as e:
        logger.error(f"Gemini Level 2 Q&A failed: {e}")
        return _fallback_answer_question(question, bill_json, math_context)


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Parse JSON from model response, handling Markdown fences and partial wrapping."""
    if not raw_text:
        return {}

    cleaned = raw_text.strip()

    # Strip ```json ... ``` fences
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
        else:
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

    # Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return {}


def _fallback_answer_question(question: str, bill_json: dict, math_context: dict) -> str:
    """Offline heuristic answering grounded strictly in extracted JSON."""
    q_lower = question.lower()
    payable_within = bill_json.get("payable_within_due_date") or bill_json.get("current_bill")
    payable_after = bill_json.get("payable_after_due_date")
    due_date = bill_json.get("due_date")
    units = bill_json.get("units_consumed")
    total_taxes = bill_json.get("total_taxes")
    total_charges = bill_json.get("total_charges")

    if any(x in q_lower for x in ["bill", "pay", "amount"]):
        if payable_within:
            return f"Your total bill payable within due date is PKR {payable_within:,.2f}."
    if "tax" in q_lower:
        pct = math_context.get("tax_percentage_of_bill", 0)
        if total_taxes:
            return f"Total government taxes amount to PKR {total_taxes:,.2f} ({pct}% of total bill)."
    if any(x in q_lower for x in ["unit", "consumed"]):
        if units:
            return f"You consumed {int(units)} units this billing month."
    if any(x in q_lower for x in ["due date", "when to pay"]):
        if due_date:
            return f"The due date for payment is {due_date}."
    if "charge" in q_lower and total_charges:
        return f"Total electricity charges (excluding taxes) are PKR {total_charges:,.2f}."
    if any(x in q_lower for x in ["after due", "late"]) and payable_after:
        return f"Payable after the due date: PKR {payable_after:,.2f}."
    return "The bill does not provide enough information to determine this."
