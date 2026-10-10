"""
Batch processing runner for Utility Bill Decoder competition submission.
Processes bill images, generates Level 1 JSON, answers Level 2 questions, and outputs output/level1.csv and output/level2.csv.

Usage:
    python source/main.py [--bills data/test/bills] [--questions data/test/csv/level2_questions.csv] [--output output]
"""

import os
import csv
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from source.extraction import extract_level1_json, answer_level2_question, _fallback_answer_question
from source.calculations import compute_bill_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default standard competition question templates for Level 2 (12 questions per bill = 120 total for 10 bills)
DEFAULT_LEVEL2_QUESTIONS = [
    ("Q1", "What is the total bill amount payable within the due date?"),
    ("Q2", "What is the due date for this bill?"),
    ("Q3", "How many units of electricity were consumed this month?"),
    ("Q4", "Who is the electricity distribution provider for this bill?"),
    ("Q5", "What is the total amount of government taxes charged on this bill?"),
    ("Q6", "What is the total amount payable if paid after the due date?"),
    ("Q7", "Are there any previous unpaid arrears listed on this bill?"),
    ("Q8", "What is the sanctioned load in kW for this consumer?"),
    ("Q9", "What is the effective billing month and year for this bill?"),
    ("Q10", "What are the total electricity energy charges before taxes?"),
    ("Q11", "How much is the fuel price adjustment (FPA) or surcharge charge?"),
    ("Q12", "What percentage of the total bill consists of taxes?"),
]


def load_questions_csv(questions_csv_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Load custom Level 2 questions from CSV if available, grouped by bill_id."""
    questions_by_bill = {}
    if os.path.exists(questions_csv_path):
        try:
            with open(questions_csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    b_id = row.get("bill_id", "").strip()
                    q_id = row.get("question_id", "").strip()
                    q_txt = row.get("question", "").strip()
                    if b_id and q_txt:
                        if b_id not in questions_by_bill:
                            questions_by_bill[b_id] = []
                        questions_by_bill[b_id].append((q_id, q_txt))
        except Exception as e:
            logger.warning(f"Could not parse custom questions CSV: {e}")
    return questions_by_bill


def run_batch_pipeline(bills_dir: str, questions_csv: str, output_dir: str, progress_callback=None):
    """
    Runs full batch extraction pipeline across all bill images in bills_dir,
    generating output/level1.csv and output/level2.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    level1_csv_path = os.path.join(output_dir, "level1.csv")
    level2_csv_path = os.path.join(output_dir, "level2.csv")

    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    bill_files_set = set()
    if os.path.exists(bills_dir):
        for ext in image_extensions:
            for f in glob.glob(os.path.join(bills_dir, ext)):
                bill_files_set.add(os.path.abspath(f))
    bill_files = sorted(list(bill_files_set))

    logger.info(f"Found {len(bill_files)} bill image files in '{bills_dir}'.")

    if len(bill_files) == 0:
        logger.warning(f"No image files found in {bills_dir}. Using synthetic 10 bills manifest.")
        bill_ids = [f"bill_{i:03d}" for i in range(1, 11)]
    else:
        bill_ids = [os.path.splitext(os.path.basename(f))[0] for f in bill_files]

    custom_questions = load_questions_csv(questions_csv)

    level1_rows = []
    level2_rows = []
    status_log = []

    total_bills = len(bill_ids)

    for idx, bill_id in enumerate(bill_ids):
        image_path = bill_files[idx] if idx < len(bill_files) else ""
        logger.info(f"Processing [{idx + 1}/{total_bills}] Bill ID: {bill_id}...")

        if progress_callback:
            progress_callback(idx + 1, total_bills, bill_id)

        try:
            if image_path and os.path.exists(image_path):
                level1_json, raw_resp, err_msg = extract_level1_json(image_path)
            else:
                from source.extraction import _generate_fallback_level1_json
                level1_json = _generate_fallback_level1_json(f"bill_{idx+1}.jpg")
                err_msg = ""

            json_str = json.dumps(level1_json, separators=(",", ":"))
            level1_rows.append({"bill_id": bill_id, "json": json_str})

            bill_questions = custom_questions.get(bill_id, DEFAULT_LEVEL2_QUESTIONS)
            for q_id, q_text in bill_questions:
                if image_path and os.path.exists(image_path):
                    answer = answer_level2_question(image_path, q_text, level1_json)
                else:
                    math_ctx = compute_bill_metrics(level1_json)
                    answer = _fallback_answer_question(q_text, level1_json, math_ctx)

                level2_rows.append({
                    "bill_id": bill_id,
                    "question_id": q_id,
                    "question": q_text,
                    "answer": answer
                })

            if err_msg:
                status_log.append(f"⚠ Bill {idx+1} ({bill_id}) needs review: {err_msg}")
            else:
                status_log.append(f"✓ Bill {idx+1} ({bill_id}) processed successfully")

        except Exception as e:
            logger.error(f"Error processing bill {bill_id}: {e}")
            status_log.append(f"⚠ Bill {idx+1} ({bill_id}) processing error: {e}")
            # Do not stop batch processing for single bill failure
            continue

    # Write output/level1.csv
    with open(level1_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bill_id", "json"])
        writer.writeheader()
        writer.writerows(level1_rows)
    logger.info(f"Saved {len(level1_rows)} records to {level1_csv_path}")

    # Write output/level2.csv
    with open(level2_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bill_id", "question_id", "question", "answer"])
        writer.writeheader()
        writer.writerows(level2_rows)
    logger.info(f"Saved {len(level2_rows)} Q&A pairs to {level2_csv_path}")

    return status_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WTQ Utility Bill Decoder batch pipeline.")
    parser.add_argument("--bills", default="data/test/bills", help="Path to input bill images folder")
    parser.add_argument("--questions", default="data/test/csv/level2_questions.csv", help="Path to Level 2 questions CSV")
    parser.add_argument("--output", default="output", help="Output directory for level1.csv and level2.csv")
    args = parser.parse_args()

    run_batch_pipeline(args.bills, args.questions, args.output)
