"""
Validation module for Utility Bill Decoder.
Defines Pydantic schemas, Python sanitization rules, and extraction confidence metrics for Level 1 JSON.
"""

import re
import logging
from typing import List, Optional, Any, Dict, Tuple
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Valid ENUM types per competition guidelines
VALID_PROVIDERS = {"KE", "LESCO", "IESCO"}

VALID_CHARGE_TYPES = {
    "energy",
    "fixed",
    "fpa",
    "quarterly_adjustment",
    "surcharge",
    "meter_rent",
    "subsidy",
    "other",
}

VALID_TAX_TYPES = {
    "gst",
    "electricity_duty",
    "income_tax",
    "municipal_tax",
    "other_tax",
}


class ChargeItem(BaseModel):
    type: str = Field(default="other")
    amount: Optional[float] = None

    @field_validator("type", mode="before")
    def validate_charge_type(cls, v):
        if not v or not isinstance(v, str):
            return "other"
        v_clean = v.strip().lower()
        if v_clean in VALID_CHARGE_TYPES:
            return v_clean
        # Map common variations
        if "energy" in v_clean or "variable" in v_clean:
            return "energy"
        if "fixed" in v_clean:
            return "fixed"
        if "fuel" in v_clean or "fpa" in v_clean or "fca" in v_clean:
            return "fpa"
        if "quarter" in v_clean or "qtr" in v_clean:
            return "quarterly_adjustment"
        if "surcharge" in v_clean or "fc" in v_clean or "debt" in v_clean:
            return "surcharge"
        if "rent" in v_clean:
            return "meter_rent"
        if "subsidy" in v_clean or "credit" in v_clean or "relief" in v_clean:
            return "subsidy"
        return "other"

    @field_validator("amount", mode="before")
    def parse_float(cls, v):
        return parse_numeric(v)


class TaxItem(BaseModel):
    type: str = Field(default="other_tax")
    amount: Optional[float] = None

    @field_validator("type", mode="before")
    def validate_tax_type(cls, v):
        if not v or not isinstance(v, str):
            return "other_tax"
        v_clean = v.strip().lower()
        if v_clean in VALID_TAX_TYPES:
            return v_clean
        if "gst" in v_clean or "sales" in v_clean:
            return "gst"
        if "duty" in v_clean or "ed" in v_clean:
            return "electricity_duty"
        if "income" in v_clean or "wht" in v_clean or "235" in v_clean:
            return "income_tax"
        if "municipal" in v_clean or "neelum" in v_clean or "local" in v_clean or "tv" in v_clean:
            return "municipal_tax"
        return "other_tax"

    @field_validator("amount", mode="before")
    def parse_float(cls, v):
        return parse_numeric(v)


class BillLevel1Schema(BaseModel):
    provider: Optional[str] = None
    tariff: Optional[str] = None
    sanctioned_load_kw: Optional[float] = None
    bill_month: Optional[str] = None
    reading_date: Optional[str] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    previous_reading: Optional[float] = None
    current_reading: Optional[float] = None
    units_consumed: Optional[float] = None
    charges: List[ChargeItem] = Field(default_factory=list)
    total_charges: Optional[float] = None
    taxes: List[TaxItem] = Field(default_factory=list)
    total_taxes: Optional[float] = None
    current_bill: Optional[float] = None
    arrears: Optional[float] = None
    payable_within_due_date: Optional[float] = None
    payable_after_due_date: Optional[float] = None

    @field_validator("provider", mode="before")
    def normalize_provider(cls, v):
        if not v or not isinstance(v, str):
            return None
        v_upper = v.strip().upper()
        if "KE" in v_upper or "K-ELECTRIC" in v_upper or "KARACHI" in v_upper:
            return "KE"
        if "LESCO" in v_upper or "LAHORE" in v_upper:
            return "LESCO"
        if "IESCO" in v_upper or "ISLAMABAD" in v_upper:
            return "IESCO"
        if v_upper in VALID_PROVIDERS:
            return v_upper
        return None

    @field_validator("bill_month", mode="before")
    def normalize_bill_month(cls, v):
        if not v or not isinstance(v, str):
            return None
        v_clean = v.strip()
        # Format YYYY-MM
        match = re.search(r"(\d{4})[-/](\d{1,2})", v_clean)
        if match:
            year, month = match.groups()
            return f"{year}-{int(month):02d}"
        return None

    @field_validator("reading_date", "issue_date", "due_date", mode="before")
    def normalize_date(cls, v):
        if not v or not isinstance(v, str):
            return None
        v_clean = v.strip()
        # YYYY-MM-DD
        match1 = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", v_clean)
        if match1:
            y, m, d = match1.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"
        # DD-MM-YYYY or DD/MM/YYYY
        match2 = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", v_clean)
        if match2:
            d, m, y = match2.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"
        return None

    @field_validator(
        "sanctioned_load_kw",
        "previous_reading",
        "current_reading",
        "units_consumed",
        "total_charges",
        "total_taxes",
        "current_bill",
        "arrears",
        "payable_within_due_date",
        "payable_after_due_date",
        mode="before",
    )
    def validate_numeric(cls, v):
        return parse_numeric(v)


def parse_numeric(v: Any) -> Optional[float]:
    """Safely convert strings, ints, or floats with commas to float or None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v_clean = v.replace(",", "").strip()
        if not v_clean or v_clean.lower() in ["null", "none", "n/a", "-", ""]:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", v_clean)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def sanitize_and_validate_level1(data: dict) -> dict:
    """
    Validates and cleans extracted dictionary into Level 1 schema compliance.
    Strips PII and enforces default structures.
    """
    if not isinstance(data, dict):
        data = {}

    try:
        model = BillLevel1Schema(**data)
        validated_dict = model.model_dump()
    except Exception as e:
        logger.warning(f"Validation error encountered, fixing structure: {e}")
        validated_dict = BillLevel1Schema().model_dump()

    # Enforce PII stripping on all text fields
    for k, v in list(validated_dict.items()):
        if isinstance(v, str):
            validated_dict[k] = strip_pii_from_string(v)

    return validated_dict


def strip_pii_from_string(text: str) -> str:
    """Remove potential CNICs, reference numbers, or personal IDs from text strings."""
    if not text:
        return ""
    # Strip 13-digit CNIC pattern (e.g. 42101-1234567-1)
    text = re.sub(r"\b\d{5}[-]\d{7}[-]\d{1}\b", "[REDACTED]", text)
    # Strip long numeric IDs (e.g. 14+ digit reference numbers)
    text = re.sub(r"\b\d{12,20}\b", "[REDACTED]", text)
    return text


def calculate_extraction_confidence(bill_data: Dict[str, Any]) -> Tuple[int, str, List[Tuple[bool, str]]]:
    """
    Calculates extraction confidence score (0-100), status label, and checklist of detected fields.
    """
    checks = [
        ("provider", "Provider detected", 15),
        ("bill_month", "Billing month detected", 15),
        ("units_consumed", "Units consumed detected", 15),
        ("current_bill", "Current bill amount detected", 20),
        ("due_date", "Due date detected", 15),
        ("payable_within_due_date", "Payable within due date detected", 10),
        ("total_charges", "Total charges detected", 5),
        ("total_taxes", "Total taxes detected", 5),
    ]

    total_score = 0
    max_score = sum(w for _, _, w in checks)
    checklist = []

    for key, label, weight in checks:
        val = bill_data.get(key)
        is_present = val is not None and val != "" and val != "Unknown" and val != "N/A"
        if is_present:
            total_score += weight
            checklist.append((True, f"✓ {label}"))
        else:
            checklist.append((False, f"⚠ {label.replace('detected', 'unclear')}"))

    score_pct = int(round((total_score / max_score) * 100)) if max_score > 0 else 0

    if score_pct >= 90:
        status_label = "Excellent"
    elif score_pct >= 70:
        status_label = "Good"
    elif score_pct >= 50:
        status_label = "Needs Review"
    else:
        status_label = "Low Confidence"

    return score_pct, status_label, checklist
