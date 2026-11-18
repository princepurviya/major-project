# 🦆 MCP-Powered Conversational Data Analytics and Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory%20OLAP-FFF000.svg?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash%20Lite-8E75C2.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![MCP](https://img.shields.io/badge/Protocol-Model%20Context%20Protocol%20(FastMCP)-0055FF.svg)](https://modelcontextprotocol.io/)
[![Plotly](https://img.shields.io/badge/Charts-Plotly%20Interactive-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous, enterprise-grade conversational data intelligence platform combining **Google Gemini**, **DuckDB in-memory SQL**, **Pandas**, and the **Model Context Protocol (MCP)**. 

Upload any **CSV** or multi-sheet **Excel (.xlsx, .xls)** dataset to perform real-time natural language analytics, audit data quality and anomalies, execute high-speed analytical SQL queries, and generate automated Exploratory Data Analysis (EDA) reports with rich interactive visualizations.

---

## 📑 Table of Contents

- [🌟 Executive Summary & Motivation](#-executive-summary--motivation)
- [🏗️ System Architecture](#️-system-architecture)
- [🚀 Key Features by Phase](#-key-features-by-phase)
  - [Phase 1: Data Quality Analysis & Remediation](#phase-1-data-quality-analysis--remediation)
  - [Phase 2: Explainable Anomaly Detection](#phase-2-explainable-anomaly-detection)
  - [Phase 3: Conversational AI Analyst with Memory](#phase-3-conversational-ai-analyst-with-memory)
  - [Phase 4: SQL Workspace, Reporting & Multi-Format Exports](#phase-4-sql-workspace-reporting--multi-format-exports)
  - [Phase 5: High-Performance Engine, Resource Safety & Concurrency](#phase-5-high-performance-engine-resource-safety--concurrency)
- [🛠️ MCP Tools Catalog (22 Tools)](#️-mcp-tools-catalog-22-tools)
- [💻 Technology Stack](#-technology-stack)
- [📂 Project Directory Structure](#-project-directory-structure)
- [⚙️ Installation & Setup Guide](#️-installation--setup-guide)
- [🖥️ How to Run the Application](#️-how-to-run-the-application)
- [❓ Troubleshooting & Common Errors (FAQ)](#-troubleshooting--common-errors-faq)
  - [Fixing `ModuleNotFoundError: No module named 'duckdb'`](#fixing-modulenotfounderror-no-module-named-duckdb)
  - [Handling Gemini API Rate Limits (`429 Too Many Requests`)](#handling-gemini-api-rate-limits-429-too-many-requests)
  - [PowerShell Script Execution Policy Warning](#powershell-script-execution-policy-warning)
- [🧪 Comprehensive Testing & Verification Suite](#-comprehensive-testing--verification-suite)
- [🔒 Security & Privacy Model](#-security--privacy-model)
- [🎓 College Demonstration & Evaluation Guide](#-college-demonstration--evaluation-guide)
- [📄 License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Executive Summary & Motivation

Traditional data analytics presents severe bottlenecks for modern organizations and researchers:
1. **The SQL & Code Barrier:** Non-technical domain experts must wait for data engineers to write complex SQL and aggregation scripts.
2. **LLM Hallucinations:** Direct LLM chat interfaces (e.g., standard ChatGPT) regularly invent fictional numbers and statistical summaries when answering questions about large datasets.
3. **Black-Box Anomaly Detection:** Most statistical anomaly algorithms output binary labels without human-readable reasons or normality evaluations.
4. **Data Quality Neglect:** Analysts frequently query uncleaned datasets containing hidden whitespace, inconsistent casing, and duplicate keys.

**Our Solution:** The **MCP-Powered Conversational Data Analytics and Intelligence Platform** solves these challenges by anchoring Google Gemini to a deterministic, zero-copy, in-memory **DuckDB SQL engine** through Anthropic’s **Model Context Protocol (FastMCP)**. Every numerical result and chart generated in conversation is computed from actual data, ensuring 100% factual accuracy and zero hallucinations.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   User Upload Interface   │
                                  │   (CSV / Multi-Sheet XLS) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │       data_loader.py      │
                                  │  - Auto-format detection  │
                                  │  - Multi-sheet inspection │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │    DuckDB In-Memory OLAP  │
                                  │       (sql_engine.py)     │
                                  │  - Thread-Safe RLock      │
                                  │  - Table: 'dataset'       │
                                  │  - Read-Only AST Guard    │
                                  └─────────────┬─────────────┘
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                                                       ▼
      ┌───────────────────────────┐                           ┌───────────────────────────┐
      │     Streamlit Web UI      │                           │      Google Gemini AI     │
      │  (Interactive Workspace)  │                           │  (gemini-3.5-flash-lite)  │
      │                           │                           └─────────────┬─────────────┘
      │ 💬 Conversational Analyst │                                         │ Function Calling
      │ 🛡️ Data Quality Studio    │                                         ▼
      │ 🔍 Anomaly Detection      │                           ┌───────────────────────────┐
      │ ⚡ DuckDB SQL Console     │                           │   mcp_server.py (FastMCP) │
      │ 📊 Automated EDA Pipeline │◄──────────────────────────┤   22 Analytical Tools:    │
      │ 📥 Multi-Format Exporter  │     Structured Results    │   - SQL & Exploratory (8) │
      └───────────────────────────┘     & Plotly Visuals      │   - Data Quality (8)      │
                                                              │   - Anomaly Detection (6) │
                                                              └───────────────────────────┘
```

---

## 🚀 Key Features by Phase

### Phase 1: Data Quality Analysis & Remediation
- **Missing Value Detection:** Scans for `NaN`, `None`, empty strings `""`, and whitespace-only strings (`"   "`). Identifies total missing cells, percentage of affected rows, and ranks columns by missingness.
- **Duplicate Record Detection:** Detects full-row duplicates as well as key-subset duplicates (e.g., repeated `employee_id` or `email`) with frequency counts.
- **Configurable Data Validation:** Flags format anomalies:
  - Email pattern violations via RFC regex matching.
  - Unparseable date strings across multiple timestamp formats.
  - Out-of-bounds numeric anomalies (e.g., negative ages, impossible percentages).
  - Categorical casing inconsistencies (e.g., `"Sales"`, `"sales"`, `"SALES"`).
- **Documented Composite Quality Score (0–100):**
  $$\text{Quality Score} = (\text{Completeness} \times 0.40) + (\text{Uniqueness} \times 0.30) + (\text{Validity} \times 0.30)$$
  - **Completeness (40%):** Measures absence of missing/blank cells ($\max(0, 100 - \text{Missing Cell \%})$).
  - **Uniqueness (30%):** Measures absence of duplicate records ($\max(0, 100 - \text{Duplicate Row \%})$).
  - **Validity (30%):** Measures adherence to validation rules ($\max(0, 100 - (\text{Affected Rows \%} \times 2))$).
- **Non-Destructive Data Cleaning Studio:** Perform remediation on an isolated copy without mutating the uploaded source:
  - Numerical imputation (Mean, Median).
  - Categorical imputation (Mode).
  - Row dropping (null thresholds).
  - Deduplication.
  - Download cleaned datasets as **CSV** or **Excel (.xlsx)**.

---

### Phase 2: Explainable Anomaly Detection
- **Dual Outlier Detection Engines:**
  - **IQR (Interquartile Range) Method:** Non-parametric statistical bounds ($[Q1 - k \times IQR, Q3 + k \times IQR]$) where $k$ is adjustable via slider (default 1.5).
  - **Z-Score Method:** Parametric standard deviation thresholding ($|Z| = \frac{|x - \mu|}{\sigma} > \text{threshold}$, default 3.0).
- **Normality & Skewness Profiling:** Automatically calculates sample skewness. Emits statistical warnings if $|skewness| > 1.0$, cautioning that Gaussian assumptions for Z-Score may yield biased outliers and recommending IQR.
- **Explainability & Plain-English Rationales:** Every anomaly is explained with its exact numerical value, expected normal interval, calculation method, distance beyond threshold, and plain-English narrative.
- **Granular Severity Ratings:**
  - 🔴 **High:** Extreme outliers ($> 3.0 \times IQR$ beyond bounds, or $|Z| > 5.0$).
  - 🟡 **Medium:** Significant outliers ($1.5 \times IQR$ to $3.0 \times IQR$, or $4.0 < |Z| \le 5.0$).
  - 🟢 **Low:** Borderline outliers ($0 < \text{distance} \le 1.5 \times IQR$, or $3.0 < |Z| \le 4.0$).
- **Method Consensus Analysis:** Side-by-side comparison identifying consensus anomalies (flagged by both methods) vs discrepancies (IQR-only or Z-Score-only).
- **Interactive Visualizations:**
  - **Box Plots:** Displays distributions with jittered anomalous points highlighted.
  - **Scatter Plots:** Plots row index against column values with interactive upper/lower threshold boundaries.
  - **Distribution Histograms:** Visualizes bin distributions with color-coded outlier tails.
- **Enriched Dataset Export:** Download the dataset enriched with audit columns: `is_anomaly`, `anomaly_severity`, and `anomaly_reasons` as **CSV** or **Excel (.xlsx)**.

---

### Phase 3: Conversational AI Analyst with Memory
- **Natural Language to SQL:** Ask freeform business questions (*"Which department has the highest average salary?"*, *"Top 5 customers by revenue"*, *"Show salary distribution"*).
- **Multi-Turn Context & Follow-Up Awareness:** Remembers previous questions and results. Seamlessly resolves follow-up questions such as:
  - *"Show them"*
  - *"Which department do they belong to?"*
  - *"Filter only high severity records"*
  - *"Create a bar chart for that"*
- **Fact-Grounded Execution:** Gemini inspects the DuckDB schema and issues read-only SQL queries or MCP tools. Numerical values in chat are strictly derived from actual query results.
- **Dynamic Plotly Visualizations:** Generates interactive bar, line, scatter, box, and histogram charts directly inside the conversation stream.
- **Resilient API Architecture:** Incorporates bounded exponential backoff retries (handling `429 Too Many Requests`, `503 Service Unavailable`, and network timeouts).
- **Thread-Isolated Asynchronous Processing:** Executes tool calls in a background thread to prevent Windows `asyncio` event-loop drops.

---

### Phase 4: SQL Workspace, Reporting & Multi-Format Exports
- **Interactive DuckDB SQL Console:**
  - Direct SQL query execution on the `dataset` table.
  - Sub-millisecond execution times displayed in real time.
  - Pre-built quick-query templates (Filter, Group By, Top 10, Summary Stats).
  - One-click CSV export of query result sets.
- **Automated 7-Section EDA Markdown Report:**
  - Overview & Shape
  - Data Types & Memory
  - Missingness & Health
  - Numerical Statistics
  - Categorical Distributions
  - Correlation Matrix & Insights
  - Machine Learning Readiness Assessment
- **Export Capabilities:**
  - Download conversational transcripts as formatted **Markdown (.md)** files.
  - Export cleaned data as **CSV** or **Excel (.xlsx)**.
  - Export anomaly audit reports as **CSV** or **Excel (.xlsx)**.

---

### Phase 5: High-Performance Engine, Resource Safety & Concurrency
- **DuckDB Connection Lifecycle Management:**
  - Uses `con.complete()` and explicit unregistration to eliminate DuckDB `ResourceInUseException`.
  - Complete connection isolation: Switching datasets automatically flushes active cursors and re-initializes memory without process restarts.
- **Thread-Safe Concurrency:** Guarded by `threading.RLock()` to prevent race conditions during simultaneous UI operations and background agent requests.
- **Strict Read-Only Security Guardrails:** An AST-level regex validator strictly allows `SELECT` and `WITH` (CTE) statements while blocking 16 destructive keywords:
  `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `TRUNCATE`, `GRANT`, `REVOKE`, `REPLACE`, `ATTACH`, `DETACH`, `PRAGMA`, `CALL`, `EXPORT`, `COPY`, `EXECUTE`.

---

## 🛠️ MCP Tools Catalog (22 Tools)

The platform exposes 22 analytical tools via the Model Context Protocol in `mcp_server.py`:

| Category | Tool Name | Key Parameters | Functional Output |
|---|---|---|---|
| **SQL & Exploration** | `get_dataset_schema` | `filepath`, `sheet_name` | Column names, data types, nullability, and preview rows |
| | `get_dataset_preview` | `filepath`, `n`, `sheet_name` | Top $N$ rows of the dataset |
| | `execute_sql_query` | `sql_query`, `filepath`, `max_rows` | Executes read-only SQL query against DuckDB table `dataset` |
| | `get_table_statistics`| `filepath`, `sheet_name` | Summary statistics (count, mean, std, quartiles) |
| | `filter_dataset` | `where_clause`, `filepath`, `limit` | Filters dataset rows based on a SQL WHERE clause |
| | `aggregate_dataset` | `group_by_columns`, `aggregations` | Group-by aggregations (`AVG`, `SUM`, `COUNT`, `MIN`, `MAX`) |
| | `create_chart` | `chart_type`, `x_col`, `y_col`, `title`| Generates Plotly bar, line, scatter, box, or histogram |
| | `run_full_eda_pipeline`| `filepath`, `sheet_name` | Executes complete automated 7-section statistical EDA |
| **Data Quality** | `analyze_data_quality` | `filepath`, `sheet_name` | Full quality audit, composite score (0-100), and breakdown |
| | `detect_missing_values`| `filepath`, `sheet_name` | Missing/blank cell counts, percentages, and affected rows |
| | `detect_duplicates` | `filepath`, `subset_columns` | Exact row duplicates and repeated key-column identifiers |
| | `validate_dataset` | `filepath`, `rules` | Checks email formatting, dates, ranges, and casing |
| | `get_column_quality` | `column_name`, `filepath` | Column-specific nulls, distinct counts, and format validity |
| | `preview_problematic_rows` | `filepath`, `issue_type`, `n` | Previews rows with missing, duplicate, or invalid data |
| | `clean_dataset` | `filepath`, `operations` | Imputes values, removes duplicates, or drops rows on a copy |
| | `get_quality_report` | `filepath`, `sheet_name` | Structured text report of quality metrics and recommendations |
| **Anomaly Detection**| `detect_anomalies` | `filepath`, `method`, `multiplier` | Complete outlier detection via IQR, Z-Score, or consensus |
| | `detect_iqr_outliers` | `filepath`, `columns`, `multiplier` | IQR outlier detection with severity and boundary distances |
| | `detect_zscore_outliers`| `filepath`, `columns`, `threshold` | Z-Score detection with skewness check and normality alerts |
| | `get_anomaly_summary` | `filepath`, `sheet_name` | Summary of outlier counts, percentages, and normal bounds |
| | `get_anomaly_details` | `column`, `filepath`, `method` | Individual anomalous records with plain-English reasons |
| | `compare_anomaly_methods`| `filepath`, `columns` | Compares IQR vs Z-Score consensus, overlaps, and discrepancies |

---

## 💻 Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Large Language Model** | Google Gemini (`gemini-3.5-flash-lite`) | Natural language reasoning, SQL synthesis, and tool selection |
| **Agent Protocol** | FastMCP / Model Context Protocol | Standardized client-server tool execution |
| **Analytical Database** | DuckDB in-memory OLAP | High-speed vectorized SQL execution directly on DataFrames |
| **Data Processing** | Pandas, NumPy, OpenPyXL | Dataset parsing, statistical computations, and cleaning |
| **Visualization** | Plotly Express & Graph Objects | Dynamic, responsive HTML5 interactive charts |
| **User Interface** | Streamlit | 5-tab multi-page responsive web application |
| **Environment** | Python 3.10+ | Core language environment |

---

## 📂 Project Directory Structure

```
major-project/
├── app.py                           # Main Streamlit web application & 5-tab dashboard
├── chat_analyst.py                  # Conversational Gemini agent, memory, retries & tool dispatcher
├── data_quality.py                  # Phase 1: Missingness, duplicate, validation, scoring & cleaning
├── anomaly_detection.py             # Phase 2: IQR, Z-Score, skewness, severity & explainability
├── sql_engine.py                    # DuckDB in-memory engine, connection lifecycle & security guardrails
├── data_loader.py                   # CSV & multi-sheet Excel (.xlsx, .xls) data ingestion
├── query_engine.py                  # Analytical bridging layer linking DuckDB, Pandas, and Plotly
├── mcp_server.py                    # FastMCP server exposing 22 analytical tools
├── mcp_client.py                    # Stdio JSON-RPC client interface for MCP
├── agent.py                         # Automated 7-section statistical EDA report builder
├── eda_engine.py                    # Statistical profiling, correlations & distribution analysis
├── charts.py                        # Standard chart generators for automated reports
├── cleanup.py                       # Project housekeeping script for cleaning temporary files
├── requirements.txt                 # Pinned project dependencies
├── run.bat                          # One-click Windows launch script
├── run.sh                           # One-click macOS/Linux launch script
├── .env.example                     # Environment template for API keys
├── .gitignore                       # Git ignore rules (protects .env and .venv)
├── test_e2e_finalization.py         # Phase 5 master verification suite (30/30 tests)
├── test_duckdb_dataset_switching.py # Verification of dataset switching & ResourceInUse fix
├── test_data_quality.py             # Phase 1 Data Quality & Gemini tools test suite
├── test_anomaly_detection.py        # Phase 2 Explainable Anomaly Detection test suite
├── test_excel_duckdb.py             # Multi-sheet Excel, SQL security & Gemini query test suite
├── test_chat_analyst.py             # Conversational agent & follow-up memory test suite
└── verify_mcp.py                    # Stdio MCP protocol JSON-RPC verification script
```

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- **Python 3.10 or higher** installed. Verify via:
  ```bash
  python --version
  ```
- A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).

### 2. Clone the Repository
```bash
git clone https://github.com/princepurviya/major-project.git
cd major-project
```

### 3. Create and Activate Virtual Environment

> [!IMPORTANT]
> Always use a dedicated virtual environment (`.venv`) so all packages (including `duckdb`) are properly recognized and do not conflict with global or Anaconda Python installations.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure API Key
Create a `.env` file in the project root directory (you can copy `.env.example`):
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_TIMEOUT=60
```

---

## 🖥️ How to Run the Application

### Option A: One-Click Launcher (Windows — Recommended)
Double-click `run.bat` or run:
```cmd
run.bat
```
*`run.bat` automatically detects your `.venv` and executes Streamlit using the dedicated virtual environment.*

### Option B: Streamlit CLI (Inside Activated `.venv`)
```bash
streamlit run app.py
```

### Option C: Explicit Virtual Environment Python Command
If you encounter environment path conflicts (e.g. from Anaconda), run directly with the virtual environment binary:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Option D: macOS / Linux Shell Launcher
```bash
chmod +x run.sh
./run.sh
```

Once launched, open your web browser at **`http://localhost:8501`**.

---

## ❓ Troubleshooting & Common Errors (FAQ)

### Fixing `ModuleNotFoundError: No module named 'duckdb'`

> **Symptom:** When running `streamlit run app.py`, the application crashes with:
> `ModuleNotFoundError: No module named 'duckdb'` in `sql_engine.py`.

- **Root Cause:** The command was executed in the system's global or Anaconda base Python environment (`C:\Users\HP\anaconda3\...`), where `duckdb` is not installed, instead of the project's `.venv`.
- **Solution:** Always run via the virtual environment:
  ```powershell
  # Method 1: Activate .venv first
  .\.venv\Scripts\Activate.ps1
  streamlit run app.py

  # Method 2: Direct call without activation
  .\.venv\Scripts\python.exe -m streamlit run app.py

  # Method 3: Simply double-click
  run.bat
  ```

---

### Handling Gemini API Rate Limits (`429 Too Many Requests`)

- **Root Cause:** Free-tier Google Gemini API keys are subject to requests-per-minute (RPM) limits during peak hours.
- **Solution:** 
  - The platform includes built-in **bounded exponential backoff** in `chat_analyst.py`. When a `429` or `503` occurs, the application automatically retries up to 3 times with progressive delays.
  - If you consistently hit rate limits, obtain a standard tier API key from [Google AI Studio](https://aistudio.google.com/) or wait 30 seconds before submitting subsequent queries.

---

### PowerShell Script Execution Policy Warning

- **Symptom:** Running `.\.venv\Scripts\Activate.ps1` produces `cannot be loaded because running scripts is disabled on this system`.
- **Solution:** Allow local scripts for the current PowerShell session:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

---

## 🧪 Comprehensive Testing & Verification Suite

The repository contains an exhaustive automated test suite to ensure stability across all features and platforms:

```bash
# 1. Master Phase 5 End-to-End Verification Suite (30/30 tests covering all 5 phases)
python test_e2e_finalization.py

# 2. DuckDB Dataset-Switching & ResourceInUse Fix Verification (7 test suites)
python test_duckdb_dataset_switching.py

# 3. Conversational AI Agent, Memory & Follow-Up Understanding Suite
python test_chat_analyst.py

# 4. Phase 1 Data Quality Engine & MCP Tools Suite
python test_data_quality.py

# 5. Phase 2 Explainable Anomaly Detection & Comparison Suite
python test_anomaly_detection.py

# 6. Multi-Sheet Excel, SQL Security Guardrails & Gemini Query Suite
python test_excel_duckdb.py

# 7. Real Stdio Model Context Protocol (FastMCP) Handshake Suite
python verify_mcp.py
```

---

## 🔒 Security & Privacy Model

1. **Zero Secret Leakage:** The Gemini API key is loaded strictly through `.env` and kept out of Git via `.gitignore`.
2. **In-Memory SQL Sandbox:** Queries execute against an ephemeral, in-memory DuckDB instance (`:memory:`). The application strictly permits `SELECT` and `WITH` statements, rejecting all modifications and system PRAGMAs.
3. **No File Mutation:** Uploaded datasets remain read-only on disk. All cleaning, imputation, and outlier flag operations occur strictly on in-memory DataFrame copies.
4. **Data Isolation:** When a user uploads a new dataset or switches Excel sheets, the DuckDB connection is reset, unregistering previous tables to prevent cross-dataset contamination.

---

## 🎓 College Demonstration & Evaluation Guide

For university evaluations, thesis defenses, or technical presentations, follow this suggested live demo workflow:

1. **Dataset Loading:** In the sidebar, upload a multi-sheet Excel file or CSV (e.g. `Car_Prices_Poland.csv`). Demonstrate dynamic sheet switching and schema inspection.
2. **Conversational Intelligence:** Go to **`💬 Natural Language Analyst`**:
   - Ask: *"What is the average price by brand?"* $\to$ Watch Gemini execute SQL and format results.
   - Follow-up: *"Which one is the most expensive?"* $\to$ Observe conversation memory in action.
   - Follow-up: *"Generate a bar chart for the top 5"* $\to$ See dynamic Plotly visualization rendered inline.
3. **Data Quality Audit:** Open **`🛡️ Data Quality Analysis`**:
   - Review the **Composite Quality Score (0–100)** with Completeness, Uniqueness, and Validity breakdown.
   - Demonstrate the **Cleaning Studio**: Apply median imputation and export the cleaned dataset.
4. **Explainable Anomaly Detection:** Open **`🔍 Anomaly Detection`**:
   - Run IQR and Z-Score outlier detection.
   - Highlight the **Normality/Skewness warning** and the **plain-English explanation table**.
   - Show the interactive Box Plot with highlighted outliers.
5. **SQL Console:** Open **`⚡ DuckDB SQL Workspace`**:
   - Run a complex aggregation query and demonstrate sub-millisecond execution.
   - Attempt a destructive statement (e.g. `DROP TABLE dataset;`) $\to$ Show security guardrail blocking it.
6. **Export:** Export conversation history as a **Markdown transcript** and download the enriched anomaly dataset.

---

## 📄 License & Acknowledgments

This project is licensed under the **MIT License** — free for academic, personal, and enterprise use.

### Acknowledgments
- **Google DeepMind & Google AI Studio** for the Gemini API.
- **DuckDB Labs** for the lightning-fast in-memory analytical SQL database.
- **Anthropic & MCP Community** for the Model Context Protocol specifications.
- **Streamlit & Plotly** for the rapid interactive data visualization frameworks.
