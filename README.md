# ⚡ Utility Bill Decoder — Technical Source Documentation

## Project Overview
**Utility Bill Decoder** is an AI-powered competition application that reads electricity bills from Pakistani distribution companies (**K-Electric / KE**, **LESCO**, and **IESCO**) and explains them in simple English. It extracts structured, non-identifying billing information (Level 1) and answers natural language questions (Level 2).

---

## Technical Specifications
- **Python Version:** Python 3.10+
- **Primary AI Model:** Google Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`)
- **Model Provider:** Google AI Studio / Gemini API (`google-genai` SDK)
- **Validation Engine:** Pydantic v2
- **OCR Engine:** Tesseract OCR / PIL & OpenCV Image Preprocessing
- **Web UI Framework:** Streamlit

---

## Installation & Setup

### 1. Environment Setup
Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required packages from `source/requirements.txt`:

```bash
pip install -r source/requirements.txt
```

### 3. Environment Variables Configuration
Ensure `.env` exists at the project root (`WTQ/.env`):

```env
MODEL_PROVIDER=google
MODEL_NAME=gemini-3.1-flash-lite
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

---

## Execution Modes

### 1. Interactive Streamlit Web Interface
Launch the consumer web interface:

```bash
streamlit run source/app.py
```

### 2. Competition Batch Processing Mode
Run automated extraction across all test bill images to generate submission CSVs:

```bash
python source/main.py --bills data/test/bills --output output
```

---

## Output CSV Formats

### 1. `output/level1.csv`
Contains extracted Level 1 structured JSON per bill (`bill_id,json`). Single-line minified JSON.

### 2. `output/level2.csv`
Contains grounded answers to Level 2 questions per bill (`bill_id,question_id,question,answer`).

---

## AI Usage Disclosure
Per competition transparency requirements, the following AI tools and services were used in this project:

- **LLM / Vision Model:** Google Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) via Google AI Studio API
- **AI Pair Programming Assistant:** Antigravity (Google DeepMind)
- **OCR:** Tesseract OCR (`pytesseract`)
- **API Client:** `google-genai` Python SDK
