import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import time
import threading
from typing import Optional
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Direct tool dispatcher import for in-process fallback & performance
import mcp_server

try:
    from google import genai
    from google.genai import types
    _SDK_MODE = "google-genai"
except ImportError:
    _SDK_MODE = "none"

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

def log(msg: str):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

CHAT_ANALYST_SYSTEM_PROMPT = """
You are the AI engine of the MCP-Powered Conversational Data Analytics and Intelligence Platform, powered by Google Gemini and connected via DuckDB and analytical MCP tools for deep dataset intelligence.

A user has uploaded a CSV or Excel dataset. The active dataset is loaded into DuckDB as a table named 'dataset'.

CORE CAPABILITIES & TOOLS:
1. SQL & Exploratory Analysis:
   - 'get_dataset_schema': Inspect schema, columns, data types, and row counts.
   - 'get_dataset_preview': Inspect the first N rows of the dataset.
   - 'execute_sql_query': Run standard SQL queries against table 'dataset' (e.g. SELECT department, AVG(salary) FROM dataset GROUP BY 1).
   - 'create_chart': Generate interactive Plotly charts (bar, line, scatter, histogram, box).

2. Data Quality Analysis:
   - 'analyze_data_quality': Run overall data quality audit and get composite quality score (0-100), category breakdown, and suggestions.
   - 'detect_missing_values': Audit missing, null, and blank cells across all columns or identify the column with most missing values.
   - 'detect_duplicates': Find duplicate rows or identify duplicate IDs with frequency counts.
   - 'validate_dataset': Check for email format violations, invalid date strings, or out-of-range numeric values.
   - 'get_column_quality': Deep dive into nulls, uniques, and anomalies for a specific column.
   - 'preview_problematic_rows': Inspect rows with missing values, duplicate entries, or validation anomalies.
   - 'clean_dataset': Perform user-requested data cleaning (mean/median/mode imputation, dropping nulls, removing duplicates).
   - 'get_quality_report': Generate complete structured text data quality report.

3. Explainable Anomaly Detection:
   - 'detect_anomalies': Run IQR or Z-Score anomaly detection across numeric columns.
   - 'detect_iqr_outliers': Find outliers using IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR) with severity (Low, Medium, High).
   - 'detect_zscore_outliers': Find outliers using Z-score (|Z| > 3.0), evaluating Gaussian normality and skewness.
   - 'get_anomaly_summary': Get high-level anomaly counts, percentages, and bounds across numeric columns.
   - 'get_anomaly_details': Retrieve specific anomalous records and reasons for a given column.
   - 'compare_anomaly_methods': Compare IQR vs Z-Score consensus and discrepancies.

SECURITY & ACCURACY RULES:
1. NEVER guess or fabricate numerical counts, percentages, or analysis results.
2. Use the exact numbers returned by the MCP tools.
3. For SQL, only read-only operations (SELECT, WITH) are permitted. Destructive statements (DROP, DELETE, UPDATE, INSERT, ALTER) are strictly blocked.
4. If a user asks about a column that does not exist in the dataset:
   - Identify the available columns and ask the user to clarify.
5. Present numbers cleanly and format tabular answers in clean Markdown tables.
6. Provide concise, friendly, and factual explanations of findings like ChatGPT.
7. Anomaly Explanation: Always communicate the statistical reason, severity (Low/Medium/High), and expected normal range. If data is skewed (|skewness| > 1.0), explain that Z-score may be biased and recommend IQR. Clarify that anomalies are statistical outliers, not necessarily data entry mistakes.
8. Follow-up Question Awareness:
   - When the user asks follow-up questions referencing previous answers (e.g., "show them", "filter these results", "which department do they belong to?", "sort them by salary", "explain the first record", "compare these results", "create a chart for them"), seamlessly maintain context from the previous tool results and conversation history.
   - Formulate specific SQL queries or tool calls that build directly upon the previous results.
"""

def mcp_to_gemini_tools(mcp_tools):
    """Convert MCP tool declarations into Gemini function declarations."""
    gemini_func_declarations = []
    for t in mcp_tools:
        schema = t.get("input_schema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        clean_props = {}
        for p_name, p_val in properties.items():
            prop_type = p_val.get("type", "STRING").upper()
            prop_dict = {
                "type": prop_type,
                "description": p_val.get("description", "")
            }
            if prop_type == "ARRAY":
                prop_dict["items"] = p_val.get("items", {"type": "STRING"})
            if "enum" in p_val:
                prop_dict["enum"] = p_val["enum"]
            clean_props[p_name] = prop_dict

        func_decl = {
            "name": t["name"],
            "description": t["description"].strip(),
            "parameters": {
                "type": "OBJECT",
                "properties": clean_props,
                "required": required
            }
        }
        gemini_func_declarations.append(func_decl)

    return [{"function_declarations": gemini_func_declarations}]

def execute_tool_direct(name: str, args: dict) -> str:
    """Direct in-process execution fallback of MCP tools for 100% stability."""
    try:
        if name == "get_dataset_schema":
            return mcp_server.get_dataset_schema(args.get("filepath", ""), args.get("sheet_name"))
        elif name == "get_dataset_preview":
            return mcp_server.get_dataset_preview(args.get("filepath", ""), args.get("n_rows", 10), args.get("sheet_name"))
        elif name == "execute_sql_query":
            return mcp_server.execute_sql_query(args.get("filepath", ""), args.get("query", ""), args.get("sheet_name"))
        elif name == "get_table_statistics":
            return mcp_server.get_table_statistics(args.get("filepath", ""), args.get("sheet_name"))
        elif name == "filter_dataset":
            return mcp_server.filter_dataset(args.get("filepath", ""), args.get("where_clause", ""), args.get("limit", 50), args.get("sheet_name"))
        elif name == "aggregate_dataset":
            return mcp_server.aggregate_dataset(
                args.get("filepath", ""),
                args.get("aggregations", []),
                args.get("group_by"),
                args.get("order_by"),
                args.get("limit", 50),
                args.get("sheet_name")
            )
        elif name == "create_chart":
            return mcp_server.create_chart(
                args.get("filepath", ""),
                args.get("chart_type", "bar"),
                args.get("x_col", ""),
                args.get("y_col"),
                args.get("title", ""),
                args.get("top_n", 20),
                args.get("sheet_name")
            )
        elif name == "run_full_eda_pipeline":
            return mcp_server.run_full_eda_pipeline_tool(args.get("filepath", ""))
        # Phase 1 Data Quality Tools
        elif name == "analyze_data_quality":
            return mcp_server.analyze_data_quality(args.get("filepath", ""), args.get("sheet_name"))
        elif name == "detect_missing_values":
            return mcp_server.detect_missing_values_tool(args.get("filepath", ""), args.get("sheet_name"))
        elif name == "detect_duplicates":
            return mcp_server.detect_duplicates_tool(args.get("filepath", ""), args.get("subset"), args.get("sheet_name"))
        elif name == "validate_dataset":
            return mcp_server.validate_dataset_tool(args.get("filepath", ""), args.get("rules"), args.get("sheet_name"))
        elif name == "get_column_quality":
            return mcp_server.get_column_quality(args.get("filepath", ""), args.get("column", ""), args.get("sheet_name"))
        elif name == "preview_problematic_rows":
            return mcp_server.preview_problematic_rows(args.get("filepath", ""), args.get("issue_type", "all"), args.get("limit", 15), args.get("sheet_name"))
        elif name == "clean_dataset":
            return mcp_server.clean_dataset_tool(args.get("filepath", ""), args.get("operations", {}), args.get("sheet_name"))
        elif name == "get_quality_report":
            return mcp_server.get_quality_report(args.get("filepath", ""), args.get("sheet_name"))
        # Phase 2 Explainable Anomaly Detection Tools
        elif name == "detect_anomalies":
            return mcp_server.detect_anomalies_tool(
                args.get("filepath", ""),
                args.get("columns"),
                args.get("method", "iqr"),
                args.get("threshold", 3.0),
                args.get("multiplier", 1.5),
                args.get("sheet_name")
            )
        elif name == "detect_iqr_outliers":
            return mcp_server.detect_iqr_outliers_tool(
                args.get("filepath", ""),
                args.get("columns"),
                args.get("multiplier", 1.5),
                args.get("sheet_name")
            )
        elif name == "detect_zscore_outliers":
            return mcp_server.detect_zscore_outliers_tool(
                args.get("filepath", ""),
                args.get("columns"),
                args.get("threshold", 3.0),
                args.get("sheet_name")
            )
        elif name == "get_anomaly_summary":
            return mcp_server.get_anomaly_summary_tool(
                args.get("filepath", ""),
                args.get("columns"),
                args.get("method", "iqr"),
                args.get("sheet_name")
            )
        elif name == "get_anomaly_details":
            return mcp_server.get_anomaly_details_tool(
                args.get("filepath", ""),
                args.get("column", ""),
                args.get("method", "iqr"),
                args.get("limit", 20),
                args.get("sheet_name")
            )
        elif name == "compare_anomaly_methods":
            return mcp_server.compare_anomaly_methods_tool(
                args.get("filepath", ""),
                args.get("columns"),
                args.get("sheet_name")
            )
        else:
            return json.dumps({"status": "error", "error": f"Unknown tool: {name}"})
    except Exception as exc:
        log(f"❌ Error in execute_tool_direct('{name}'): {exc}")
        return json.dumps({
            "status": "error",
            "error_type": "tool_execution_error",
            "tool": name,
            "error": str(exc)
        })

def get_static_gemini_tools():
    """Static definitions of tools for immediate zero-latency initialization."""
    return [
        {
            "function_declarations": [
                {
                    "name": "get_dataset_schema",
                    "description": "Get column names, data types, nullability, and preview sample of the active dataset.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to CSV or Excel file"},
                            "sheet_name": {"type": "STRING", "description": "Optional Excel sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "get_dataset_preview",
                    "description": "Retrieve a preview of the first N rows (default 10) of the dataset.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to CSV or Excel file"},
                            "n_rows": {"type": "INTEGER", "description": "Number of rows (default 10)"},
                            "sheet_name": {"type": "STRING", "description": "Optional Excel sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "execute_sql_query",
                    "description": "Execute read-only SQL query against DuckDB table 'dataset' (e.g. SELECT department, AVG(salary) FROM dataset GROUP BY department ORDER BY 2 DESC).",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "query": {"type": "STRING", "description": "SQL query where table is 'dataset'"},
                            "sheet_name": {"type": "STRING", "description": "Optional Excel sheet name"}
                        },
                        "required": ["filepath", "query"]
                    }
                },
                {
                    "name": "create_chart",
                    "description": "Generate an interactive Plotly visualization ('bar', 'line', 'scatter', 'histogram', 'box').",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "chart_type": {"type": "STRING", "description": "bar, line, scatter, histogram, box"},
                            "x_col": {"type": "STRING", "description": "X axis column"},
                            "y_col": {"type": "STRING", "description": "Y axis column"},
                            "title": {"type": "STRING", "description": "Chart title"},
                            "top_n": {"type": "INTEGER", "description": "Top N records to plot"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath", "chart_type", "x_col"]
                    }
                },
                # Data Quality Tools
                {
                    "name": "analyze_data_quality",
                    "description": "Comprehensive assessment of dataset quality: completeness, uniqueness, validity, quality score (0-100), and improvement recommendations.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "detect_missing_values",
                    "description": "Detect missing, NaN, None, and whitespace-only cells across all columns, returning counts, percentages, and highest missing column.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "detect_duplicates",
                    "description": "Identify duplicate rows in the dataset, optionally focusing on specific subset columns like 'employee_id' or 'customer_id'.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "subset": {
                                "type": "ARRAY",
                                "description": "Optional list of columns to check for duplicates",
                                "items": {"type": "STRING"}
                            },
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "validate_dataset",
                    "description": "Check for invalid emails, date parsing failures, out-of-range numeric values, and inconsistent categorical casing.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "get_column_quality",
                    "description": "Get specific quality diagnostics for a single column: null count, unique count, data type, and validation status.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "column": {"type": "STRING", "description": "Target column name"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath", "column"]
                    }
                },
                {
                    "name": "preview_problematic_rows",
                    "description": "Retrieve rows that contain quality defects (missing cells, duplicates, or validation anomalies) with issue explanations.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "issue_type": {"type": "STRING", "description": "'all', 'missing', 'duplicate', or 'validation'"},
                            "limit": {"type": "INTEGER", "description": "Max rows to return"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "clean_dataset",
                    "description": "Perform selective data cleaning (impute numerical mean/median, impute categorical mode, drop null rows, remove duplicates) on a copy of dataset.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "operations": {
                                "type": "OBJECT",
                                "description": "Cleaning operations dict"
                            },
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath", "operations"]
                    }
                },
                {
                    "name": "get_quality_report",
                    "description": "Generate complete structured text summary of data quality metrics, score, category breakdown, and suggestions.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "detect_anomalies",
                    "description": "Detect anomalies in numeric columns using IQR, Z-Score, or comparison. Returns summary, severity levels, and explainable reasons.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "columns": {
                                "type": "ARRAY",
                                "description": "Optional list of numeric columns to inspect",
                                "items": {"type": "STRING"}
                            },
                            "method": {"type": "STRING", "description": "'iqr', 'zscore', or 'both' (default 'iqr')"},
                            "multiplier": {"type": "NUMBER", "description": "IQR multiplier (default 1.5)"},
                            "threshold": {"type": "NUMBER", "description": "Z-Score threshold (default 3.0)"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "detect_iqr_outliers",
                    "description": "Find outliers using IQR method (Q1 - multiplier*IQR, Q3 + multiplier*IQR). Returns severity (Low/Medium/High) and plain-English reasons.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "columns": {
                                "type": "ARRAY",
                                "description": "Optional list of numeric columns to inspect",
                                "items": {"type": "STRING"}
                            },
                            "multiplier": {"type": "NUMBER", "description": "IQR multiplier (default 1.5)"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "detect_zscore_outliers",
                    "description": "Find outliers using Z-score (|Z| > threshold, default 3.0). Checks skewness and warns if Gaussian normality assumption is compromised.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "columns": {
                                "type": "ARRAY",
                                "description": "Optional list of numeric columns to inspect",
                                "items": {"type": "STRING"}
                            },
                            "threshold": {"type": "NUMBER", "description": "Z-score threshold (default 3.0)"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "get_anomaly_summary",
                    "description": "Get high-level summary of anomalies across numeric columns (counts, percentages, normal bounds, skewness, warnings).",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "columns": {
                                "type": "ARRAY",
                                "description": "Optional list of numeric columns",
                                "items": {"type": "STRING"}
                            },
                            "method": {"type": "STRING", "description": "'iqr' or 'zscore' (default 'iqr')"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                },
                {
                    "name": "get_anomaly_details",
                    "description": "Retrieve specific anomalous records for a single column with actual values, expected range, severity, and plain-English reasons.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "column": {"type": "STRING", "description": "Column name to inspect"},
                            "method": {"type": "STRING", "description": "'iqr' or 'zscore' (default 'iqr')"},
                            "limit": {"type": "INTEGER", "description": "Max anomalous records to retrieve (default 20)"},
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath", "column"]
                    }
                },
                {
                    "name": "compare_anomaly_methods",
                    "description": "Compare IQR vs Z-Score detection side-by-side to identify consensus anomalies, IQR-only, and Z-score-only records.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "filepath": {"type": "STRING", "description": "Path to dataset"},
                            "columns": {
                                "type": "ARRAY",
                                "description": "Optional list of numeric columns",
                                "items": {"type": "STRING"}
                            },
                            "sheet_name": {"type": "STRING", "description": "Optional sheet name"}
                        },
                        "required": ["filepath"]
                    }
                }
            ]
        }
    ]


class ChatAnalystMCPClient:
    """
    Manages conversational interactions between the user, Gemini AI, and the MCP server.
    Handles SQL & Data Quality tool execution, conversation context, and active dataset switching.
    """
    def __init__(self, filepath: str, sheet_name: Optional[str] = None):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.chat_session = None
        self.gemini_tools = get_static_gemini_tools()
        self._server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
        self.last_tool_results: dict = {}

    def update_dataset(self, filepath: str, sheet_name: Optional[str] = None):
        """Update active dataset and sheet without recreating client."""
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.chat_session = None

    def init_chat_session(self, initial_summary: str = ""):
        """Initialize or reset the Gemini chat with system prompt and dataset awareness."""
        if _SDK_MODE != "google-genai":
            raise RuntimeError("google-genai SDK is required.")

        sheet_info = f" (Sheet: '{self.sheet_name}')" if self.sheet_name else ""
        system_instruction = CHAT_ANALYST_SYSTEM_PROMPT
        system_instruction += f"\n\nCURRENT ACTIVE DATASET: '{self.filepath}'{sheet_info}"
        if initial_summary:
            system_instruction += f"\nDATASET METADATA:\n{initial_summary}"

        self.chat_session = self.client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=self.gemini_tools,
                temperature=0.2,
                max_output_tokens=4096
            )
        )

    def _send_message_with_retry(self, message_content, max_retries: int = 3):
        """Send message to Gemini chat with bounded exponential backoff for transient errors."""
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return self.chat_session.send_message(message_content)
            except Exception as exc:
                last_exc = exc
                err_msg = str(exc).lower()
                is_transient = any(k in err_msg for k in ["429", "503", "resource_exhausted", "quota", "timeout", "unavailable", "deadline"])
                if is_transient and attempt < max_retries:
                    backoff = attempt * 2
                    log(f"⚠️ Gemini transient error: {exc}. Retrying in {backoff}s (attempt {attempt}/{max_retries})...")
                    time.sleep(backoff)
                else:
                    break
        raise last_exc

    def execute_turn(self, user_message: str) -> dict:
        """
        Execute one chat turn with Gemini and handle tool calls.
        Uses in-process direct tool dispatcher for 100% stability on Windows without TaskGroup pipe drops.
        """
        if self.chat_session is None:
            self.init_chat_session()

        charts_produced = []
        tools_executed = []
        sql_queries = []

        log(f"💬 User question: {user_message}")
        turn_start = time.time()

        try:
            gemini_resp = self._send_message_with_retry(user_message)
        except Exception as exc:
            err_str = str(exc)
            if any(k in err_str.lower() for k in ["429", "quota", "resource_exhausted"]):
                user_facing = "⚠️ **Rate Limit Reached:** Gemini API rate limit / quota exceeded. Please wait a moment and try again."
            elif any(k in err_str.lower() for k in ["api_key", "unauthenticated", "invalid api key"]):
                user_facing = "❌ **Authentication Error:** Invalid or missing Gemini API key. Please verify your GEMINI_API_KEY in `.env`."
            elif any(k in err_str.lower() for k in ["timeout", "deadline"]):
                user_facing = "⏱️ **Request Timeout:** The query took too long to complete. Please try a simpler question."
            else:
                user_facing = f"❌ **Error processing query:** {err_str}"
            return {
                "response_text": user_facing,
                "charts": [],
                "tools_executed": [],
                "sql_queries": [],
                "error": err_str
            }

        turn = 0
        while True:
            turn += 1
            function_calls = gemini_resp.function_calls

            if not function_calls:
                final_text = ""
                if hasattr(gemini_resp, "text") and gemini_resp.text:
                    final_text = gemini_resp.text
                elif hasattr(gemini_resp, "candidates") and gemini_resp.candidates:
                    for cand in gemini_resp.candidates:
                        if hasattr(cand, "content") and cand.content:
                            for part in cand.content.parts:
                                if hasattr(part, "text") and part.text:
                                    final_text += part.text

                log(f"✅ Gemini turn finished in {time.time() - turn_start:.1f}s")
                return {
                    "response_text": final_text,
                    "charts": charts_produced,
                    "tools_executed": tools_executed,
                    "sql_queries": sql_queries
                }

            # Execute tool calls
            function_responses = []
            for call in function_calls:
                name = call.name
                args = dict(call.args) if call.args else {}
                
                # Guarantee active filepath and sheet are passed
                if "filepath" not in args or not args["filepath"]:
                    args["filepath"] = self.filepath
                if self.sheet_name and "sheet_name" not in args:
                    args["sheet_name"] = self.sheet_name

                if name == "execute_sql_query" and "query" in args:
                    sql_queries.append(args["query"])

                log(f"🔧 Tool called: {name} with args: {args}")
                tool_start = time.time()
                
                # Execute tool directly in-process
                raw_result = execute_tool_direct(name, args)
                tool_elapsed = time.time() - tool_start

                # Cache in last_tool_results for conversational follow-ups
                self.last_tool_results[name] = raw_result

                tools_executed.append({
                    "tool": name,
                    "args": args,
                    "time_s": round(tool_elapsed, 2)
                })

                if name == "create_chart":
                    try:
                        chart_data = json.loads(raw_result)
                        if chart_data.get("status") == "success":
                            charts_produced.append(chart_data)
                    except Exception:
                        pass

                function_responses.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": raw_result}
                    )
                )

            try:
                gemini_resp = self._send_message_with_retry(function_responses)
            except Exception as exc:
                err_str = str(exc)
                log(f"⚠️ Error returning tool responses to Gemini: {err_str}")
                return {
                    "response_text": f"⚠️ Analysis completed partially. Tool execution succeeded but encountered an error generating the final summary: {err_str}",
                    "charts": charts_produced,
                    "tools_executed": tools_executed,
                    "sql_queries": sql_queries,
                    "error": err_str
                }

            if turn > 8:
                log("⚠️ Max tool turns reached.")
                return {
                    "response_text": gemini_resp.text or "Analysis completed with available data.",
                    "charts": charts_produced,
                    "tools_executed": tools_executed,
                    "sql_queries": sql_queries
                }


def run_chat_turn_sync(client: ChatAnalystMCPClient, user_message: str) -> dict:
    """
    Synchronous helper for running the turn within Streamlit or scripts.
    Executes in a dedicated background thread with its own event loop to avoid
    TaskGroup / asyncio conflicts with Streamlit's internal event loop on Windows.
    """
    result_holder = {}
    error_holder = {}

    def _thread_target():
        # Each thread gets its own brand-new event loop — no sharing with Streamlit's loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder["result"] = client.execute_turn(user_message)
        except Exception as exc:
            error_holder["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join(timeout=180)  # 3-minute hard timeout for complex multi-tool analytical workflows

    if "error" in error_holder:
        raise error_holder["error"]
    if "result" not in result_holder:
        raise TimeoutError("Chat turn timed out after 180 seconds.")
    return result_holder["result"]
