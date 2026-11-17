from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
from datetime import datetime
import pandas as pd
from eda_engine import run_eda
from charts import generate_charts
import sys

# Support both new google-genai and legacy google-generativeai packages
try:
    from google import genai
    from google.genai import types
    _SDK_MODE = "google-genai"
except ImportError:
    try:
        import google.generativeai as legacy_genai
        _SDK_MODE = "legacy-genai"
    except ImportError:
        _SDK_MODE = "none"


def log(msg):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


# ── Configuration ─────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT", "120"))
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2


SYSTEM_PROMPT = """
You are a senior data scientist writing a professional EDA report for a business audience.

You will receive a JSON summary of a dataset's statistics. Your job is to:
1. Interpret the numbers — don't just repeat them, explain what they MEAN
2. Flag potential data quality issues a junior analyst might miss
3. Identify patterns that would affect machine learning model choices
4. Write actionable recommendations, not generic advice

Structure your report with exactly these sections:
## 1. Dataset Overview
## 2. Data Quality Assessment
## 3. Key Statistical Insights
## 4. Correlation Analysis
## 5. Categorical Variable Analysis
## 6. ML Readiness Assessment
## 7. Recommended Next Steps

Rules:
- Reference actual column names and specific numbers from the data
- Use markdown tables where comparisons are clearer in tabular form
- Flag any column with >5% missing values as HIGH PRIORITY
- Flag any correlation above 0.7 as potentially multicollinear
- Be specific — "column X has 23% missing values which will require imputation"
  not "some columns have missing values"
- Keep the tone professional but accessible
- Be concise — focus on the most important findings
"""


def _compact_stats(eda_stats: dict) -> str:
    """
    Produce a compact JSON representation of EDA stats to minimize
    prompt token count. Uses compact separators and trims whitespace.
    """
    return json.dumps(eda_stats, separators=(",", ":"))


def _is_retryable(error: Exception) -> bool:
    """Check if an API error is transient and worth retrying."""
    err_str = str(error).lower()
    retryable_indicators = ["503", "unavailable", "overloaded",
                            "resource exhausted", "429", "rate limit",
                            "deadline", "timeout", "internal"]
    return any(indicator in err_str for indicator in retryable_indicators)


def generate_report(eda_stats: dict, chart_paths: list, filename: str) -> str:
    """
    Sends compact EDA stats to Gemini and gets back a markdown report.
    Uses a single verified model with retries and exponential backoff.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in your .env file.")

    # Build chart references
    chart_refs = ""
    if chart_paths:
        chart_refs = "\n\nCharts generated:\n"
        for path in chart_paths:
            chart_refs += f"- {path}\n"
        chart_refs += "Reference these charts in the appropriate report sections using: ![Title](path)"

    # Build compact prompt — no indent=2, no redundant whitespace
    compact_json = _compact_stats(eda_stats)
    user_message = (
        f"Generate a full EDA report for '{filename}'.\n\n"
        f"Dataset statistics:\n{compact_json}"
        f"{chart_refs}\n\n"
        f"Generate the complete report now."
    )

    prompt_size_kb = len(user_message.encode("utf-8")) / 1024
    log(f"🤖 Sending stats to Gemini ({GEMINI_MODEL})  prompt={prompt_size_kb:.1f} KB")

    if _SDK_MODE == "google-genai":
        return _call_google_genai(api_key, user_message)
    elif _SDK_MODE == "legacy-genai":
        return _call_legacy_genai(api_key, user_message)
    else:
        raise RuntimeError(
            "No Google GenAI SDK found. Install one with:\n"
            "  pip install google-genai"
        )


def _extract_text_safely(response) -> str:
    """Extract text from Gemini response, handling both .text and candidates structure."""
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        parts = []
        for candidate in response.candidates:
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)
        if parts:
            return "".join(parts)
    return ""


def _call_google_genai(api_key: str, user_message: str) -> str:
    """Call the google-genai SDK with retries and backoff."""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=4096,
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"   Attempt {attempt}/{MAX_RETRIES}...")
            start_time = time.time()

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_message,
                config=config,
            )

            elapsed = time.time() - start_time
            log(f"   ✅ Gemini responded in {elapsed:.1f}s")

            text = _extract_text_safely(response)
            if not text:
                raise RuntimeError("Gemini returned an empty text response.")
            return text

        except Exception as e:
            elapsed = time.time() - start_time
            last_error = e
            log(f"   ❌ Attempt {attempt} failed after {elapsed:.1f}s: {e}")

            if not _is_retryable(e) or attempt == MAX_RETRIES:
                break

            backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            log(f"   ⏳ Retrying in {backoff}s...")
            time.sleep(backoff)

    raise RuntimeError(f"Gemini API call failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def _call_legacy_genai(api_key: str, user_message: str) -> str:
    """Call the legacy google-generativeai SDK with retries and backoff."""
    legacy_genai.configure(api_key=api_key)
    model = legacy_genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"   Attempt {attempt}/{MAX_RETRIES}...")
            start_time = time.time()

            response = model.generate_content(user_message)

            elapsed = time.time() - start_time
            log(f"   ✅ Gemini responded in {elapsed:.1f}s")

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return response.text

        except Exception as e:
            elapsed = time.time() - start_time
            last_error = e
            log(f"   ❌ Attempt {attempt} failed after {elapsed:.1f}s: {e}")

            if not _is_retryable(e) or attempt == MAX_RETRIES:
                break

            backoff = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            log(f"   ⏳ Retrying in {backoff}s...")
            time.sleep(backoff)

    raise RuntimeError(f"Gemini API call failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def save_report(report: str, filename: str) -> str:
    """Saves the report as a markdown file with a timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eda_report_{timestamp}.md"

    header = (
        f"# EDA Report: {filename}\n"
        f"*Generated automatically by EDA Agent on {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n\n"
        f"---\n\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + report)

    log(f"✅ Report saved: {output_path}")
    return output_path


def run_full_pipeline(csv_path: str):
    """
    Orchestrates the full pipeline:
    1. Run EDA (eda_engine)
    2. Generate charts (charts)
    3. Send to Gemini, get report
    4. Save report to file
    """
    log(f"\n📂 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    log("📊 Running EDA engine...")
    eda_stats = run_eda(df)
    log(f"   → {eda_stats['shape']['rows']} rows, {eda_stats['shape']['columns']} columns")

    log("🎨 Generating charts...")
    chart_paths = generate_charts(df)

    log("🤖 Generating AI report...")
    report = generate_report(eda_stats, chart_paths, csv_path)

    log("💾 Saving report...")
    output_path = save_report(report, csv_path)

    return output_path


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Car_Purchasing_Data.csv"
    run_full_pipeline(csv_path)
