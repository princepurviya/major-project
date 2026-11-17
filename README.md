# MCP-Powered Conversational Data Analytics and Intelligence Platform

An intelligent, autonomous enterprise data intelligence platform powered by **Google Gemini**, **DuckDB in-memory SQL**, **Pandas**, and the **Model Context Protocol (MCP)**.

Upload any **CSV** or **Excel (.xlsx, .xls)** dataset to conduct real-time natural-language conversational analytics, audit data quality and anomalies, execute high-speed SQL queries, and generate comprehensive Exploratory Data Analysis (EDA) reports with interactive visualizations.

---

## 🌟 Key Features Across All Phases

### 1. 💬 Conversational Data Analyst with Conversation Memory (Phases 3 & 4)
- **Natural Language Querying:** Ask complex analytical questions in plain English (*"Which department has the highest average salary?"*, *"Find top 5 products by revenue"*).
- **Follow-up Question Awareness:** Understands conversational follow-ups (*"Show them"*, *"Which department do they belong to?"*, *"Filter by high severity"*, *"Create a chart for them"*) without needing to repeat previous context.
- **Fact-Grounded Responses:** Gemini inspects schema, writes standard SQL, executes queries in DuckDB, and delivers factual answers grounded strictly in actual data computations.
- **Interactive Chart Generation:** Dynamically creates Plotly bar, line, scatter, box, and histogram charts inline.
- **Export Conversation Transcripts:** Download full conversation histories including prompts, answers, and generated SQL as Markdown files.

### 2. 🛡️ Data Quality Analysis & Remediation (Phase 1)
- **Missing Value Detection:** Audits `NaN`, `None`, empty strings `""`, and whitespace-only cells (`"   "`). Identifies total missing cells, affected rows, and highlights columns with the highest missingness.
- **Duplicate Detection:** Identifies full-row duplicates as well as key-subset duplicates (e.g. repeated `employee_id` or `email`) with frequency tables.
- **Configurable Data Validation:** Flags invalid email formats, unparseable dates, out-of-range numeric boundaries, and inconsistent categorical casing.
- **Documented Composite Quality Score (0–100):**
  $$\text{Quality Score} = (\text{Completeness} \times 0.40) + (\text{Uniqueness} \times 0.30) + (\text{Validity} \times 0.30)$$
- **Non-Destructive Data Cleaning:** Safely clean data on an isolated copy without mutating the original upload. Supports mean/median numerical imputation, mode categorical imputation, row dropping, and deduplication with before-and-after metrics and export to **CSV** or **Excel (.xlsx)**.

### 3. 🔍 Explainable Anomaly Detection (Phase 2)
- **IQR-Based Outlier Detection:** Non-parametric detection ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$) with customizable multiplier sliders.
- **Z-Score-Based Outlier Detection:** Identifies extreme deviations ($|Z| > \text{threshold}$, default 3.0) with automatic skewness profiling and Gaussian normality warnings for $|skewness| > 1.0$.
- **Human-Readable Explanations & Severity Scoring:** Every detected anomaly provides its exact value, expected normal range, detection method, plain-English reason, and severity badge (**Low**, **Medium**, or **High**).
- **Method Comparison:** Compare IQR vs Z-Score consensus, identifying overlapping anomalies and discrepancies.
- **Interactive Visualizations:** High-fidelity Plotly charts including **Box Plots** with highlighted anomalies, **Row Index vs Value Scatter Plots** with threshold lines, and **Distribution Histograms**.
- **Comprehensive Data Export:** Export anomalies list as **CSV** or **Excel (.xlsx)**, or download an enriched copy of the dataset containing `is_anomaly`, `anomaly_severity`, and `anomaly_reasons` columns.

### 4. ⚡ High-Speed DuckDB In-Memory SQL Workspace (Phase 5 Optimized)
- High-speed zero-configuration analytical SQL execution on table `dataset`.
- Read-only security layer strictly blocks destructive queries (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, semicolon chaining).
- Includes pre-built SQL query templates, execution timers, and CSV export.
- **Zero-Leakage Dataset Switching:** Instantly switches between datasets and Excel sheets without memory leaks or schema contamination.
- **Concurrency & Concurrency Locking:** Thread-safe connection pooling protected by `threading.RLock()`.
- **ResourceInUse Safe:** All query executions and cursor results are finalized via `con.complete()`.

### 5. 📁 CSV & Multi-Sheet Excel Support
- Ingest `.csv`, `.xlsx`, and `.xls` files.
- Automatically inspects multi-sheet Excel workbooks and provides a dynamic sheet switcher.

### 6. 📊 Automated Full EDA Pipeline
- Generates a structured 7-section statistical markdown report covering distributions, correlation heatmaps, missing value analyses, and ML readiness assessments.

---

## 🏗️ Architecture Overview

```
User Upload (CSV / Multi-Sheet Excel)
                  │
                  ▼
         data_loader.py (Auto-format & sheet detection)
                  │
                  ▼
          DuckDB In-Memory Engine (sql_engine.py)
          ├── In-memory isolated table: 'dataset'
          ├── Read-only query security filter
          └── Thread-safe connection & concurrency lock
                  │
      ┌───────────┴───────────────────────────────┐
      ▼                                           ▼
Streamlit Web UI                            Gemini AI Agent
├── 💬 Natural Language Analyst                   │ (Selects MCP Tools)
├── 🛡️ Data Quality Analysis (data_quality.py)     ▼
├── 🔍 Anomaly Detection (anomaly_detection.py)   mcp_server.py (FastMCP)
├── ⚡ DuckDB SQL Workspace                 ├── SQL & Aggregations (8 tools)
├── 📊 Automated Full EDA Report           ├── Quality Diagnostics & Scoring (8 tools)
└── 📥 Multi-Format Reports & Downloads     ├── Explainable Anomaly Detection (6 tools)
                                           └── Plotly Chart Creation
```

---

## 🛠️ Model Context Protocol (MCP) Tools Catalog

The MCP server (`mcp_server.py`) exposes 22 specialized analytical tools via the Model Context Protocol:

| Category | Tool Name | Description |
|---|---|---|
| **SQL & Exploration** | `get_dataset_schema` | Retrieves column names, data types, nullability, and preview rows |
| | `get_dataset_preview` | Inspects the first N rows of the active dataset |
| | `execute_sql_query` | Executes read-only SQL queries against DuckDB table `dataset` |
| | `get_table_statistics` | Computes column-level summary statistics |
| | `filter_dataset` | Filters dataset rows based on a SQL WHERE clause |
| | `aggregate_dataset` | Aggregates columns with GROUP BY and ORDER BY |
| | `create_chart` | Generates Plotly bar, line, scatter, histogram, or box charts |
| | `run_full_eda_pipeline`| Executes complete automated EDA pipeline |
| **Data Quality** | `analyze_data_quality` | Full audit computing composite quality score (0-100) and breakdown |
| | `detect_missing_values`| Audits missing, null, and blank cells across all columns |
| | `detect_duplicates` | Finds duplicate rows and repeated entity keys |
| | `validate_dataset` | Validates email patterns, date parsing, and numeric ranges |
| | `get_column_quality` | Deep dive into nulls, uniques, and anomalies for a column |
| | `preview_problematic_rows` | Previews rows containing quality defects |
| | `clean_dataset` | Performs selective imputation and deduplication on a copy |
| | `get_quality_report` | Generates structured text summary report of dataset quality |
| **Anomaly Detection** | `detect_anomalies` | Detects outliers using IQR, Z-Score, or comparison |
| | `detect_iqr_outliers` | Non-parametric IQR outlier detection with severity ratings |
| | `detect_zscore_outliers` | Z-score detection with Gaussian normality & skewness warnings |
| | `get_anomaly_summary` | Summary of anomaly counts, percentages, and normal bounds |
| | `get_anomaly_details` | Retrieves specific anomalous records with explainable reasons |
| | `compare_anomaly_methods`| Compares IQR vs Z-Score consensus and discrepancies |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **AI Model** | Google Gemini (`gemini-3.5-flash-lite`) via `google-genai` SDK |
| **Agent Protocol** | Model Context Protocol (MCP v2) |
| **SQL Engine** | DuckDB in-memory v2.0.0 |
| **Data Processing** | Pandas, NumPy, OpenPyXL |
| **Visualizations** | Plotly Express & Graph Objects |
| **Web Interface** | Streamlit |
| **Language** | Python 3.10+ |

---

## 📂 Project Structure

```
eda-agent-mcp/
├── app.py                           # Streamlit UI (Chat, Quality, Anomaly, SQL Workspace, EDA)
├── chat_analyst.py                  # Conversational loop, memory, retry backoff & tool dispatcher
├── data_quality.py                  # Phase 1: Missingness, duplicates, validation, scoring & cleaning
├── anomaly_detection.py             # Phase 2: IQR, Z-Score, severity scoring & explainability
├── sql_engine.py                    # Thread-safe DuckDB in-memory query engine with guardrails
├── data_loader.py                   # CSV & multi-sheet Excel (.xlsx, .xls) ingestion
├── query_engine.py                  # Analytical layer linking DuckDB, Pandas, and Plotly
├── mcp_server.py                    # FastMCP server exposing 22 analytical tools
├── mcp_client.py                    # Client helper for MCP stdio protocol
├── agent.py                         # Automated 7-section EDA report generator
├── eda_engine.py                    # Statistical profiling, correlations & distributions
├── charts.py                        # Standard visualization suites for EDA reports
├── test_e2e_finalization.py         # Phase 5 master end-to-end verification suite (30 tests)
├── test_duckdb_dataset_switching.py # Verification of dataset switching & ResourceInUse fix
├── test_data_quality.py             # Automated test suite for Data Quality & Gemini tools
├── test_anomaly_detection.py        # Automated test suite for Anomaly Detection & Gemini tools
├── test_excel_duckdb.py             # Test suite for Excel, DuckDB SQL & security
├── verify_mcp.py                    # Real MCP stdio client-server JSON-RPC verification
├── requirements.txt                 # Python package dependencies
├── run.bat                          # Windows one-click launcher
├── run.sh                           # macOS/Linux one-click launcher
├── .env                             # Environment variables (API Key, Model name)
└── README.md                        # Complete project documentation
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
- **Python 3.10 or higher**
- A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).

### 2. Setup Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create or verify [`.env`](file:///.env) in the project root:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_TIMEOUT=60
```

---

## 🖥️ Running the Application

### Option A: Web Application (Streamlit) — Recommended
```bash
streamlit run app.py
```
*(On Windows, you can also double-click `run.bat`)*

Once loaded in your browser at `http://localhost:8501`:
1. **Upload Dataset:** Choose any `.csv`, `.xlsx`, or `.xls` file in the sidebar. Select your sheet if using a multi-sheet Excel file.
2. **Conversational Analyst:** Head to the **`💬 Natural Language Analyst`** tab to chat with your data (*"What is my dataset quality score?"*, *"Find salary outliers"*, *"Which department has highest revenue?"*, *"Visualize department salaries"*).
3. **Data Quality Analysis:** Check the **`🛡️ Data Quality Analysis`** tab to inspect dataset health, audit missing values, review validation anomalies, clean data, and download the cleaned CSV/Excel.
4. **Anomaly Detection:** Explore the **`🔍 Anomaly Detection`** tab to audit IQR and Z-Score outliers, inspect explainability reasons and severity grades, view interactive Box and Scatter charts, and export enriched datasets.
5. **SQL Workspace:** Write and run custom DuckDB queries directly in the **`⚡ DuckDB SQL Workspace`** tab.
6. **Automated EDA Report:** Generate a full 7-section statistical report in the **`📊 Automated Full EDA Report`** tab.

---

## 🧪 Automated Testing & Verification

Run any of the automated test suites to verify system health:

```bash
# 1. Master Phase 5 End-to-End Verification Suite (30/30 tests covering all phases)
python test_e2e_finalization.py

# 2. DuckDB Dataset-Switching & ResourceInUse Fix Verification (7 test suites)
python test_duckdb_dataset_switching.py

# 3. Real MCP stdio Protocol Verification (JSON-RPC handshake & tool execution)
python verify_mcp.py

# 4. Phase 1 Data Quality Verification Suite
python test_data_quality.py

# 5. Phase 2 Explainable Anomaly Detection Verification Suite
python test_anomaly_detection.py

# 6. Multi-Sheet Excel, SQL Security & Conversational Gemini Suite
python test_excel_duckdb.py
```

---

## 🔒 Security & Safety Model

1. **API Key Protection:** The Gemini API key is strictly read from `.env` and excluded from version control via `.gitignore`. Error messages sanitize credentials.
2. **Read-Only SQL Sandbox:** All SQL queries are executed against DuckDB with an AST security filter blocking destructive statements (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `PRAGMA`, `ATTACH`, `DETACH`) and query chaining.
3. **In-Memory Isolation:** Datasets are registered in DuckDB in-memory (`:memory:`). When switching datasets, previous connection states and tables are fully discarded, preventing cross-dataset data leakage.
4. **Non-Destructive Operations:** Cleaning and anomaly detection operations never modify uploaded files on disk; transformations are performed on isolated in-memory copies.

---

## ⚠️ Known Limitations & Best Practices

1. **Large Excel Files:** While CSV files are processed in-memory at high speed, extremely large multi-sheet Excel workbooks (>100MB) may take a few seconds to parse via OpenPyXL. For maximum throughput on massive datasets, CSV format is recommended.
2. **Gemini API Rate Limits:** Free-tier Gemini API keys may occasionally encounter `429 Too Many Requests` or `503 Unavailable` under peak Google traffic. The application includes automatic exponential backoff retries (up to 3 attempts) to handle transient spikes smoothly.
3. **Z-Score Skewness Assumption:** Z-Score outlier detection assumes an approximately normal distribution. For highly skewed data ($|skewness| > 1.0$), the application issues a statistical warning and recommends using IQR instead.

---

## 📄 License

MIT License. Free for academic, personal, and commercial use.
