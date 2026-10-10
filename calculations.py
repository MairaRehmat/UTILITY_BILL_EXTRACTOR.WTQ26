"""
Calculations module for Utility Bill Decoder.
Performs deterministic mathematical computations on Level 1 JSON for Level 2 Q&A and UI insights.
All arithmetic calculations are strictly performed in Python.
"""

from typing import Dict, Any, Optional

CHARGE_EXPLANATIONS = {
    "energy": "Variable cost for electricity units consumed during the billing month based on applicable tariff slab.",
    "fixed": "Fixed monthly charge based on consumer sanctioned load capacity.",
    "fpa": "Fuel Price Adjustment (FPA/FCA) reflecting variations in fuel cost for electricity generation.",
    "quarterly_adjustment": "Quarterly Tariff Adjustment (QTR) determined periodically by NEPRA.",
    "surcharge": "Government imposed surcharge (such as FC Surcharge or Debt Servicing Surcharge).",
    "meter_rent": "Monthly rental fee for electricity meter equipment and service line.",
    "subsidy": "Government tariff differential subsidy or consumer relief credit (reduces bill amount).",
    "other": "Other miscellaneous non-tax billing charges.",
}

TAX_EXPLANATIONS = {
    "gst": "General Sales Tax (GST) levied on electricity supply as per government tax regulations.",
    "electricity_duty": "State/Provincial duty levied on electricity consumption.",
    "income_tax": "Advance Income Tax / Withholding Tax (WHT) collected under Section 235 of Income Tax Ordinance.",
    "municipal_tax": "Local Municipal Tax, TV Fee, or Neelum Jhelum Surcharge.",
    "other_tax": "Other statutory government taxes or local surcharges.",
}


def compute_bill_metrics(bill_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes mathematical metrics, percentages, breakdowns, and verifications
    from extracted Level 1 bill data strictly using Python arithmetic.
    """
    current_bill = bill_data.get("current_bill") or 0.0
    units = bill_data.get("units_consumed") or 0.0
    total_charges = bill_data.get("total_charges") or 0.0
    total_taxes = bill_data.get("total_taxes") or 0.0
    payable_within = bill_data.get("payable_within_due_date") or current_bill
    payable_after = bill_data.get("payable_after_due_date") or 0.0
    arrears = bill_data.get("arrears") or 0.0

    # Calculate itemized charge sum
    charges_list = bill_data.get("charges") or []
    itemized_charges_sum = sum(
        c.get("amount") for c in charges_list if isinstance(c, dict) and c.get("amount") is not None
    )

    # Calculate itemized tax sum
    taxes_list = bill_data.get("taxes") or []
    itemized_taxes_sum = sum(
        t.get("amount") for t in taxes_list if isinstance(t, dict) and t.get("amount") is not None
    )

    # Tax & Charge Percentages (Safe division against 0)
    effective_bill = current_bill if current_bill > 0 else (total_charges + total_taxes)
    tax_percentage = (total_taxes / effective_bill * 100.0) if effective_bill > 0 else 0.0
    charge_percentage = (total_charges / effective_bill * 100.0) if effective_bill > 0 else 0.0

    # Per-unit effective rate
    per_unit_cost = (current_bill / units) if units > 0 else 0.0

    # Late Payment Surcharge / Penalty
    late_payment_surcharge = (payable_after - payable_within) if (payable_after and payable_within and payable_after > payable_within) else 0.0

    # Largest Charge Item
    largest_charge = None
    if charges_list:
        valid_charges = [c for c in charges_list if isinstance(c, dict) and c.get("amount")]
        if valid_charges:
            largest_charge = max(valid_charges, key=lambda x: x.get("amount", 0))

    # Largest Tax Item
    largest_tax = None
    if taxes_list:
        valid_taxes = [t for t in taxes_list if isinstance(t, dict) and t.get("amount")]
        if valid_taxes:
            largest_tax = max(valid_taxes, key=lambda x: x.get("amount", 0))

    metrics = {
        "current_bill": current_bill,
        "units_consumed": units,
        "total_charges": total_charges,
        "total_taxes": total_taxes,
        "itemized_charges_sum": round(itemized_charges_sum, 2),
        "itemized_taxes_sum": round(itemized_taxes_sum, 2),
        "tax_percentage_of_bill": round(tax_percentage, 1),
        "charge_percentage_of_bill": round(charge_percentage, 1),
        "effective_per_unit_rate_pkr": round(per_unit_cost, 2),
        "late_payment_surcharge_pkr": round(late_payment_surcharge, 2),
        "arrears_pkr": round(arrears, 2),
        "largest_charge_item": largest_charge,
        "largest_tax_item": largest_tax,
    }

    return metrics


def generate_simple_bill_explanation(bill_data: Dict[str, Any]) -> str:
    """
    Generates a short, beginner-friendly explanation based ONLY on extracted bill data.
    """
    provider = bill_data.get("provider") or "Electricity"
    units = bill_data.get("units_consumed")
    total_charges = bill_data.get("total_charges")
    total_taxes = bill_data.get("total_taxes")
    current_bill = bill_data.get("current_bill") or bill_data.get("payable_within_due_date")
    due_date = bill_data.get("due_date")

    sentences = []

    if units is not None and current_bill is not None:
        sentences.append(f"Your {provider} bill shows {int(units):,} units consumed this billing period for a total current bill of PKR {current_bill:,.2f}.")
    elif units is not None:
        sentences.append(f"Your {provider} bill shows {int(units):,} units consumed this billing period.")
    elif current_bill is not None:
        sentences.append(f"Your {provider} bill shows a total current bill of PKR {current_bill:,.2f}.")
    else:
        sentences.append(f"Your {provider} bill records your electricity consumption and billing details.")

    if total_charges is not None:
        sentences.append(f"Electricity charges before taxes are PKR {total_charges:,.2f}.")

    if total_taxes is not None:
        sentences.append(f"Taxes and government charges are PKR {total_taxes:,.2f}.")

    if due_date:
        sentences.append(f"The due date for payment is {due_date}.")

    return " ".join(sentences)


def get_charge_explanation(charge_type: str) -> str:
    """Get concise explanation for a charge type."""
    return CHARGE_EXPLANATIONS.get(charge_type.lower(), CHARGE_EXPLANATIONS["other"])


def get_tax_explanation(tax_type: str) -> str:
    """Get concise explanation for a tax type."""
    return TAX_EXPLANATIONS.get(tax_type.lower(), TAX_EXPLANATIONS["other_tax"])
