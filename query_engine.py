import os
import json
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from data_loader import load_dataset_file, inspect_excel_sheets
from sql_engine import (
    initialize_database,
    load_dataframe,
    get_table_schema,
    get_table_info,
    execute_readonly_sql
)
from charts import THEME

# In-memory session tracking for active dataset & sheet
_CURRENT_ACTIVE_PATH: Optional[str] = None
_CURRENT_ACTIVE_SHEET: Optional[str] = None
_CURRENT_DATAFRAME: Optional[pd.DataFrame] = None
_CURRENT_METADATA: Optional[Dict[str, Any]] = None

def get_or_load_dataset(filepath: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Retrieve the cached active dataset or load and register it with DuckDB."""
    global _CURRENT_ACTIVE_PATH, _CURRENT_ACTIVE_SHEET, _CURRENT_DATAFRAME, _CURRENT_METADATA

    if (
        _CURRENT_DATAFRAME is not None
        and _CURRENT_ACTIVE_PATH == filepath
        and _CURRENT_ACTIVE_SHEET == sheet_name
    ):
        return _CURRENT_DATAFRAME, _CURRENT_METADATA

    # Load via data_loader
    df, meta = load_dataset_file(filepath, sheet_name=sheet_name)
    
    # Register directly with DuckDB as table 'dataset'
    load_dataframe(df, dataset_id=f"{filepath}:{sheet_name}")

    _CURRENT_ACTIVE_PATH = filepath
    _CURRENT_ACTIVE_SHEET = sheet_name
    _CURRENT_DATAFRAME = df
    _CURRENT_METADATA = meta

    return df, meta

def get_dataset_metadata(filepath: str, sheet_name: Optional[str] = None) -> dict:
    """Return dataset metadata and schema for Gemini context and UI."""
    _, meta = get_or_load_dataset(filepath, sheet_name)
    return meta

def run_duckdb_query(filepath: str, sql_query: str, sheet_name: Optional[str] = None) -> dict:
    """Execute read-only SQL query via DuckDB on table 'dataset'."""
    get_or_load_dataset(filepath, sheet_name)
    _, meta = execute_readonly_sql(sql_query, max_rows=50)
    return meta

def compute_pandas_metric(filepath: str, operation: str, column: str, group_by: Optional[str] = None, sheet_name: Optional[str] = None) -> dict:
    """Compute specific summary metrics using Pandas (mean, sum, median, min, max, count, std, nunique)."""
    df, _ = get_or_load_dataset(filepath, sheet_name)
    if column not in df.columns:
        return {
            "status": "error",
            "error": f"Column '{column}' does not exist. Available columns: {list(df.columns)}"
        }

    try:
        if group_by:
            if group_by not in df.columns:
                return {
                    "status": "error",
                    "error": f"Group by column '{group_by}' does not exist. Available: {list(df.columns)}"
                }
            grouped = df.groupby(group_by)[column]
            if operation == "mean":
                res = grouped.mean()
            elif operation == "median":
                res = grouped.median()
            elif operation == "sum":
                res = grouped.sum()
            elif operation == "min":
                res = grouped.min()
            elif operation == "max":
                res = grouped.max()
            elif operation == "count":
                res = grouped.count()
            elif operation == "std":
                res = grouped.std()
            else:
                return {"status": "error", "error": f"Unsupported operation: {operation}"}
            
            res_df = res.reset_index()
            return {
                "status": "success",
                "operation": operation,
                "column": column,
                "group_by": group_by,
                "results": res_df.to_dict(orient="records")
            }
        else:
            series = df[column]
            if operation == "mean":
                val = float(series.mean())
            elif operation == "median":
                val = float(series.median())
            elif operation == "sum":
                val = float(series.sum())
            elif operation == "min":
                val = float(series.min()) if pd.api.types.is_numeric_dtype(series) else str(series.min())
            elif operation == "max":
                val = float(series.max()) if pd.api.types.is_numeric_dtype(series) else str(series.max())
            elif operation == "count":
                val = int(series.count())
            elif operation == "std":
                val = float(series.std())
            elif operation == "nunique":
                val = int(series.nunique())
            else:
                return {"status": "error", "error": f"Unsupported operation: {operation}"}

            return {
                "status": "success",
                "operation": operation,
                "column": column,
                "value": val
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def generate_custom_chart(filepath: str, chart_type: str, x_col: str, y_col: Optional[str] = None, title: str = "", top_n: int = 20, sheet_name: Optional[str] = None) -> dict:
    """Generate interactive Plotly chart from the dataset."""
    import plotly.express as px
    import plotly.io as pio

    df, _ = get_or_load_dataset(filepath, sheet_name)
    if x_col not in df.columns:
        return {"status": "error", "error": f"Column '{x_col}' not found in dataset."}
    if y_col and y_col not in df.columns:
        return {"status": "error", "error": f"Column '{y_col}' not found in dataset."}

    os.makedirs("charts", exist_ok=True)
    import time
    timestamp = int(time.time() * 1000)
    chart_filename = f"chat_chart_{timestamp}.html"
    chart_path = os.path.join("charts", chart_filename)

    plot_df = df
    if top_n and top_n > 0 and len(df) > top_n:
        if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
            plot_df = df.nlargest(top_n, y_col)
        else:
            plot_df = df.head(top_n)

    title = title or f"{chart_type.title()} Chart: {x_col}" + (f" vs {y_col}" if y_col else "")

    try:
        if chart_type == "bar":
            if y_col:
                fig = px.bar(plot_df, x=x_col, y=y_col, title=title, template=THEME)
            else:
                val_counts = df[x_col].value_counts().head(top_n).reset_index()
                val_counts.columns = [x_col, "count"]
                fig = px.bar(val_counts, x=x_col, y="count", title=title, template=THEME)
        elif chart_type == "line":
            fig = px.line(plot_df, x=x_col, y=y_col, title=title, template=THEME)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, title=title, template=THEME)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col, title=title, template=THEME)
        elif chart_type == "box":
            fig = px.box(plot_df, x=x_col, y=y_col, title=title, template=THEME)
        else:
            return {"status": "error", "error": f"Unsupported chart_type '{chart_type}'."}

        fig.write_html(chart_path)
        fig_json = pio.to_json(fig)

        return {
            "status": "success",
            "chart_type": chart_type,
            "chart_path": chart_path,
            "title": title,
            "fig_json": fig_json
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
