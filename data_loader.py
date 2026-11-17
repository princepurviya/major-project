import os
import io
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

def inspect_excel_sheets(filepath: str) -> List[str]:
    """
    Get the list of sheet names from an Excel file without loading entire data.
    Supports .xlsx and .xls.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".xlsx", ".xls"]:
        return []
    
    excel_file = pd.ExcelFile(filepath)
    return excel_file.sheet_names

def load_dataset_file(filepath: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load CSV (.csv), Excel (.xlsx), or Excel (.xls) dataset into a Pandas DataFrame.
    Returns (DataFrame, metadata_dict).
    Handles empty files, corrupted files, and format detection gracefully.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)

    try:
        if ext == ".csv":
            # Attempt to read CSV with common encodings
            try:
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding="latin1")
            active_sheet = None
            available_sheets = []

        elif ext in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(filepath)
            available_sheets = excel_file.sheet_names
            if not available_sheets:
                raise ValueError("Excel workbook contains no sheets.")
            
            # Select specified sheet or default to first sheet
            if sheet_name and sheet_name in available_sheets:
                active_sheet = sheet_name
            else:
                active_sheet = available_sheets[0]
            
            df = pd.read_excel(filepath, sheet_name=active_sheet)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .csv, .xlsx, .xls")

    except Exception as e:
        raise RuntimeError(f"Error reading file '{filename}': {str(e)}")

    if df.empty:
        raise ValueError(f"The dataset in '{filename}' is empty (0 rows).")

    # Clean string columns if needed, replace infinities
    columns_info = []
    for col in df.columns:
        non_null = int(df[col].count())
        missing = int(df[col].isnull().sum())
        missing_pct = round((missing / len(df)) * 100, 2) if len(df) > 0 else 0.0
        dtype_str = str(df[col].dtype)
        unique_cnt = int(df[col].nunique())
        sample_vals = [str(v) for v in df[col].dropna().head(3).tolist()]

        columns_info.append({
            "name": str(col),
            "type": dtype_str,
            "non_null_count": non_null,
            "missing_count": missing,
            "missing_pct": f"{missing_pct}%",
            "unique_count": unique_cnt,
            "sample_values": sample_vals
        })

    meta = {
        "filepath": filepath,
        "filename": filename,
        "format": ext.replace(".", "").upper(),
        "active_sheet": active_sheet,
        "available_sheets": available_sheets if ext in [".xlsx", ".xls"] else [],
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "columns_detail": columns_info,
        "sample_preview_10": df.head(10).fillna("").to_dict(orient="records")
    }

    return df, meta
