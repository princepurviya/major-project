import asyncio
import json
import sys

# Windows encoding fix — must be at the very top
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def log(msg: str):
    """
    Safe logging for MCP servers — always use stderr, never stdout.
    stdout is reserved exclusively for MCP JSON communication.
    Any print() to stdout will corrupt the MCP message stream.
    """
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

from mcp.server.mcpserver import MCPServer
from agent import run_full_pipeline
from eda_engine import run_eda
from charts import generate_charts
from query_engine import (
    get_dataset_metadata,
    run_duckdb_query,
    compute_pandas_metric,
    generate_custom_chart,
    get_or_load_dataset
)
from sql_engine import (
    get_table_schema,
    get_table_info,
    execute_readonly_sql
)
from data_quality import (
    detect_missing_values as dq_detect_missing,
    detect_duplicates as dq_detect_duplicates,
    validate_dataset as dq_validate_dataset,
    calculate_quality_score,
    clean_dataset as dq_clean_dataset
)
from anomaly_detection import (
    detect_iqr_outliers as ad_detect_iqr,
    detect_zscore_outliers as ad_detect_zscore,
    compare_anomaly_methods as ad_compare_methods,
    create_anomaly_report_dataframe
)

server = MCPServer("eda-agent")

# ── Existing Core Tools ──────────────────────────────────────────

@server.tool(
    name="get_dataset_schema",
    description="Get column names, data types, nullability, and preview sample of the active dataset."
)
def get_dataset_schema(filepath: str, sheet_name: str = None) -> str:
    """Get schema, column names, data types, missing values, and a sample of the dataset."""
    log(f"MCP: get_dataset_schema for {filepath} (sheet: {sheet_name})")
    meta = get_dataset_metadata(filepath, sheet_name=sheet_name)
    return json.dumps(meta, indent=2)

@server.tool(
    name="get_dataset_preview",
    description="Retrieve a preview of the first N rows (default 10) of the dataset."
)
def get_dataset_preview(filepath: str, n_rows: int = 10, sheet_name: str = None) -> str:
    """Retrieve preview rows from dataset."""
    log(f"MCP: get_dataset_preview {n_rows} rows for {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    preview = df.head(n_rows).fillna("").to_dict(orient="records")
    return json.dumps({"rows": preview, "total_rows": len(df)}, indent=2)

@server.tool(
    name="execute_sql_query",
    description="Execute read-only SQL query against DuckDB table 'dataset' (e.g., SELECT department, AVG(salary) FROM dataset GROUP BY department ORDER BY 2 DESC)."
)
def execute_sql_query(filepath: str, query: str, sheet_name: str = None) -> str:
    """Execute SQL query using DuckDB against table 'dataset'."""
    log(f"MCP: execute_sql_query on {filepath}: {query}")
    res = run_duckdb_query(filepath, query, sheet_name=sheet_name)
    return json.dumps(res, indent=2)

@server.tool(
    name="get_table_statistics",
    description="Get row count, column count, and statistical summary of table columns."
)
def get_table_statistics(filepath: str, sheet_name: str = None) -> str:
    """Get high-level table statistics."""
    log(f"MCP: get_table_statistics for {filepath}")
    get_or_load_dataset(filepath, sheet_name=sheet_name)
    info = get_table_info()
    return json.dumps(info, indent=2)

@server.tool(
    name="filter_dataset",
    description="Filter dataset records using a SQL WHERE condition (e.g. 'salary > 50000 AND department = Engineering'). Table name is 'dataset'."
)
def filter_dataset(filepath: str, where_clause: str, limit: int = 50, sheet_name: str = None) -> str:
    """Filter records by WHERE condition."""
    log(f"MCP: filter_dataset condition: {where_clause}")
    sql = f"SELECT * FROM dataset WHERE {where_clause} LIMIT {limit}"
    res = run_duckdb_query(filepath, sql, sheet_name=sheet_name)
    return json.dumps(res, indent=2)

@server.tool(
    name="aggregate_dataset",
    description="Aggregate columns by group_by dimensions (e.g. aggregations=['AVG(salary)', 'COUNT(*)'], group_by=['department'])."
)
def aggregate_dataset(filepath: str, aggregations: list, group_by: list = None, order_by: str = None, limit: int = 50, sheet_name: str = None) -> str:
    """Run aggregation query with GROUP BY and ORDER BY."""
    log(f"MCP: aggregate_dataset aggregations: {aggregations}, group_by: {group_by}")
    agg_str = ", ".join(aggregations)
    if group_by:
        grp_str = ", ".join(group_by)
        sql = f"SELECT {grp_str}, {agg_str} FROM dataset GROUP BY {grp_str}"
    else:
        sql = f"SELECT {agg_str} FROM dataset"
    
    if order_by:
        sql += f" ORDER BY {order_by}"
    sql += f" LIMIT {limit}"

    res = run_duckdb_query(filepath, sql, sheet_name=sheet_name)
    return json.dumps(res, indent=2)

@server.tool(
    name="create_chart",
    description="Generate an interactive Plotly visualization ('bar', 'line', 'scatter', 'histogram', 'box')."
)
def create_chart(filepath: str, chart_type: str, x_col: str, y_col: str = None, title: str = "", top_n: int = 20, sheet_name: str = None) -> str:
    """Generate interactive Plotly visualization for dataset."""
    log(f"MCP: create_chart {chart_type} for {x_col} vs {y_col}")
    res = generate_custom_chart(filepath, chart_type, x_col, y_col, title, top_n, sheet_name=sheet_name)
    return json.dumps(res, indent=2)

@server.tool(
    name="run_full_eda_pipeline",
    description="Run automated 7-section statistical EDA pipeline on CSV or Excel file."
)
def run_full_eda_pipeline_tool(filepath: str) -> str:
    """Run complete automated EDA pipeline."""
    log(f"MCP: Running full pipeline on {filepath}")
    output_path = run_full_pipeline(filepath)
    return f"EDA pipeline complete. Report saved to: {output_path}"

# ── Phase 1: Data Quality Analysis MCP Tools ─────────────────────

@server.tool(
    name="analyze_data_quality",
    description="Comprehensive assessment of dataset quality: completeness, uniqueness, validity, quality score (0-100), and improvement recommendations."
)
def analyze_data_quality(filepath: str, sheet_name: str = None) -> str:
    """Run full data quality audit and calculate composite quality score."""
    log(f"MCP: analyze_data_quality for {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    missing = dq_detect_missing(df)
    dupes = dq_detect_duplicates(df)
    val = dq_validate_dataset(df)
    score = calculate_quality_score(missing, dupes, val)
    return json.dumps({
        "quality_score": score,
        "missing_summary": {
            "total_missing_cells": missing["total_missing_cells"],
            "overall_missing_pct": missing["overall_missing_percentage"],
            "worst_column": missing["worst_column"]
        },
        "duplicate_summary": {
            "duplicate_records": dupes["total_duplicate_records"],
            "duplicate_pct": dupes["duplicate_percentage"]
        },
        "validation_summary": {
            "total_anomalies": val["total_anomalies"],
            "affected_rows": val["affected_rows_count"]
        }
    }, indent=2)

@server.tool(
    name="detect_missing_values",
    description="Detect missing, NaN, None, and whitespace-only cells across all columns, returning counts, percentages, and highest missing column."
)
def detect_missing_values_tool(filepath: str, sheet_name: str = None) -> str:
    """Audit missing values per column and overall."""
    log(f"MCP: detect_missing_values for {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    res = dq_detect_missing(df)
    # Return without the large preview dict to optimize tokens
    summary = {
        "total_rows": res["total_rows"],
        "total_missing_cells": res["total_missing_cells"],
        "overall_missing_percentage": res["overall_missing_percentage"],
        "affected_row_count": res["affected_row_count"],
        "columns_summary": res["columns_summary"],
        "worst_column": res["worst_column"]
    }
    return json.dumps(summary, indent=2)

@server.tool(
    name="detect_duplicates",
    description="Identify duplicate rows in the dataset, optionally focusing on specific subset columns like 'employee_id' or 'customer_id'."
)
def detect_duplicates_tool(filepath: str, subset: list = None, sheet_name: str = None) -> str:
    """Detect duplicate records across all columns or specific ID column."""
    log(f"MCP: detect_duplicates for {filepath} (subset: {subset})")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    res = dq_detect_duplicates(df, subset=subset)
    return json.dumps({
        "total_rows": res["total_rows"],
        "subset_used": res["subset_used"],
        "total_duplicate_records": res["total_duplicate_records"],
        "duplicate_percentage": res["duplicate_percentage"],
        "frequency_table": res["frequency_table"]
    }, indent=2)

@server.tool(
    name="validate_dataset",
    description="Check for invalid emails, date parsing failures, out-of-range numeric values, and inconsistent categorical casing."
)
def validate_dataset_tool(filepath: str, rules: dict = None, sheet_name: str = None) -> str:
    """Audit data validity rules and format compliance."""
    log(f"MCP: validate_dataset for {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    res = dq_validate_dataset(df, rules=rules)
    return json.dumps(res, indent=2)

@server.tool(
    name="get_column_quality",
    description="Get specific quality diagnostics for a single column: null count, unique count, data type, and validation status."
)
def get_column_quality(filepath: str, column: str, sheet_name: str = None) -> str:
    """Inspect quality profile of a specific column."""
    log(f"MCP: get_column_quality for column '{column}' in {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found. Available: {list(df.columns)}"})

    series = df[column]
    null_cnt = int(series.isna().sum())
    total = len(df)
    return json.dumps({
        "column": column,
        "dtype": str(series.dtype),
        "total_values": total,
        "missing_count": null_cnt,
        "missing_pct": round(null_cnt / total * 100, 2) if total > 0 else 0,
        "unique_count": int(series.nunique()),
        "sample_values": [str(x) for x in series.dropna().head(5).tolist()]
    }, indent=2)

@server.tool(
    name="preview_problematic_rows",
    description="Retrieve rows that contain quality defects (missing cells, duplicates, or validation anomalies) with issue explanations."
)
def preview_problematic_rows(filepath: str, issue_type: str = "all", limit: int = 15, sheet_name: str = None) -> str:
    """Preview problematic rows for user inspection."""
    log(f"MCP: preview_problematic_rows for {filepath} (issue: {issue_type})")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    
    problematic = []
    if issue_type in ["all", "missing"]:
        missing_res = dq_detect_missing(df)
        for r in missing_res.get("affected_preview", [])[:limit]:
            problematic.append({"defect_category": "missing_value", "record": r})

    if issue_type in ["all", "duplicate"]:
        dupe_res = dq_detect_duplicates(df)
        for r in dupe_res.get("duplicate_preview", [])[:limit]:
            problematic.append({"defect_category": "duplicate_row", "record": r})

    if issue_type in ["all", "validation"]:
        val_res = dq_validate_dataset(df)
        for a in val_res.get("anomalies_preview", [])[:limit]:
            problematic.append({"defect_category": "validation_anomaly", "details": a})

    return json.dumps({"problematic_records": problematic[:limit]}, indent=2)

@server.tool(
    name="clean_dataset",
    description="Perform selective data cleaning (impute numerical mean/median, impute categorical mode, drop null rows, remove duplicates) on a copy of dataset."
)
def clean_dataset_tool(filepath: str, operations: dict, sheet_name: str = None) -> str:
    """Run specified cleaning operations and return summary of changes."""
    log(f"MCP: clean_dataset operations: {operations}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    _, summary = dq_clean_dataset(df, operations)
    return json.dumps(summary, indent=2)

@server.tool(
    name="get_quality_report",
    description="Generate complete structured text summary of data quality metrics, score, category breakdown, and suggestions."
)
def get_quality_report(filepath: str, sheet_name: str = None) -> str:
    """Generate markdown-formatted quality report."""
    log(f"MCP: get_quality_report for {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    missing = dq_detect_missing(df)
    dupes = dq_detect_duplicates(df)
    val = dq_validate_dataset(df)
    score = calculate_quality_score(missing, dupes, val)

    report = (
        f"# Data Quality Report\n\n"
        f"**Overall Score:** {score['overall_score']} / 100 ({score['grade']})\n"
        f"{score['summary']}\n\n"
        f"### Category Scores\n"
        f"- **Completeness:** {score['category_scores']['completeness']} / 100\n"
        f"- **Uniqueness:** {score['category_scores']['uniqueness']} / 100\n"
        f"- **Validity:** {score['category_scores']['validity']} / 100\n\n"
        f"### Key Metrics\n"
        f"- Total Missing Cells: {missing['total_missing_cells']} ({missing['overall_missing_percentage']}%)\n"
        f"- Duplicate Rows: {dupes['total_duplicate_records']} ({dupes['duplicate_percentage']}%)\n"
        f"- Validation Anomalies: {val['total_anomalies']}\n\n"
        f"### Recommendations\n"
    )
    for s in score["suggestions"]:
        report += f"- {s}\n"

    return report

# ── Phase 2: Explainable Anomaly Detection Tools ────────────────

@server.tool(
    name="detect_anomalies",
    description="Detect anomalies in dataset columns using IQR or Z-score method. Returns summary and explainable anomalous records."
)
def detect_anomalies_tool(
    filepath: str,
    columns: list = None,
    method: str = "iqr",
    threshold: float = 3.0,
    multiplier: float = 1.5,
    sheet_name: str = None
) -> str:
    """Detect anomalies in specified columns using IQR or Z-Score."""
    log(f"MCP: detect_anomalies on {filepath}, method={method}, cols={columns}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    m = method.lower()
    if m == "zscore" or m == "z-score":
        res = ad_detect_zscore(df, columns=columns, threshold=threshold)
    elif m == "both" or m == "compare":
        res = ad_compare_methods(df, columns=columns, iqr_multiplier=multiplier, zscore_threshold=threshold)
    else:
        res = ad_detect_iqr(df, columns=columns, multiplier=multiplier)
    return json.dumps(res, indent=2)

@server.tool(
    name="detect_iqr_outliers",
    description="Detect outliers using the Interquartile Range (IQR) method (Q1 - 1.5*IQR, Q3 + 1.5*IQR). Returns severity ratings and reasons."
)
def detect_iqr_outliers_tool(
    filepath: str,
    columns: list = None,
    multiplier: float = 1.5,
    sheet_name: str = None
) -> str:
    """Run IQR-based outlier detection with configurable multiplier."""
    log(f"MCP: detect_iqr_outliers on {filepath}, mult={multiplier}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    res = ad_detect_iqr(df, columns=columns, multiplier=multiplier)
    return json.dumps(res, indent=2)

@server.tool(
    name="detect_zscore_outliers",
    description="Detect outliers using Z-Score (|Z| > threshold, default 3.0). Checks for skewness and evaluates Gaussian normality."
)
def detect_zscore_outliers_tool(
    filepath: str,
    columns: list = None,
    threshold: float = 3.0,
    sheet_name: str = None
) -> str:
    """Run Z-score outlier detection with configurable threshold."""
    log(f"MCP: detect_zscore_outliers on {filepath}, threshold={threshold}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    res = ad_detect_zscore(df, columns=columns, threshold=threshold)
    return json.dumps(res, indent=2)

@server.tool(
    name="get_anomaly_summary",
    description="Get aggregate summary of anomalies across all numeric columns (counts, percentages, bounds, skewness, and warnings)."
)
def get_anomaly_summary_tool(
    filepath: str,
    columns: list = None,
    method: str = "iqr",
    sheet_name: str = None
) -> str:
    """Get high-level anomaly stats per column without row-level dumping."""
    log(f"MCP: get_anomaly_summary on {filepath}, method={method}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    m = method.lower()
    if m in ("zscore", "z-score"):
        full = ad_detect_zscore(df, columns=columns)
    else:
        full = ad_detect_iqr(df, columns=columns)
    
    # Return concise summary
    summary_data = {
        "method": full["method"],
        "total_rows": full["total_rows"],
        "analyzed_columns": full["analyzed_columns"],
        "total_anomalies": full["total_anomalies"],
        "affected_rows_count": full["affected_rows_count"],
        "affected_rows_percentage": full["affected_rows_percentage"],
        "columns": [
            {
                "column": c["column"],
                "anomaly_count": c["anomaly_count"],
                "anomaly_percentage": c["anomaly_percentage"],
                "skewness": c.get("skewness"),
                "warning": c.get("warning") or c.get("reason"),
                "bounds": f"[{c.get('lower_bound')}, {c.get('upper_bound')}]"
            }
            for c in full["column_summaries"]
        ]
    }
    return json.dumps(summary_data, indent=2)

@server.tool(
    name="get_anomaly_details",
    description="Get specific anomalous records for a single column with actual values, expected range, severity, and plain-English reasons."
)
def get_anomaly_details_tool(
    filepath: str,
    column: str,
    method: str = "iqr",
    limit: int = 20,
    sheet_name: str = None
) -> str:
    """Retrieve detailed anomaly explanations for a single column."""
    log(f"MCP: get_anomaly_details for {column} on {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    m = method.lower()
    if m in ("zscore", "z-score"):
        res = ad_detect_zscore(df, columns=[column])
    else:
        res = ad_detect_iqr(df, columns=[column])
    
    col_anomalies = [a for a in res["anomalies"] if a["column"] == column]
    col_summary = next((c for c in res["column_summaries"] if c["column"] == column), {})

    details = {
        "column": column,
        "method": res["method"],
        "total_anomalies": len(col_anomalies),
        "bounds": f"[{col_summary.get('lower_bound')}, {col_summary.get('upper_bound')}]",
        "skewness": col_summary.get("skewness"),
        "warning": col_summary.get("warning"),
        "records": col_anomalies[:limit]
    }
    return json.dumps(details, indent=2)

@server.tool(
    name="compare_anomaly_methods",
    description="Compare IQR vs Z-Score anomaly detection results side-by-side. Identifies consensus anomalies, IQR-only, and Z-score-only records."
)
def compare_anomaly_methods_tool(
    filepath: str,
    columns: list = None,
    sheet_name: str = None
) -> str:
    """Compare both detection techniques on dataset."""
    log(f"MCP: compare_anomaly_methods on {filepath}")
    df, _ = get_or_load_dataset(filepath, sheet_name=sheet_name)
    res = ad_compare_methods(df, columns=columns)
    return json.dumps(res, indent=2)

if __name__ == "__main__":
    log("EDA Agent MCP Server (Data Quality & DuckDB) starting...")
    server.run(transport="stdio")
