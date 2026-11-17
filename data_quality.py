import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

EMAIL_REGEX = re.compile(r"^[\w\.-]+@([\w-]+\.)+[\w-]{2,4}$")

def is_blank_or_whitespace(val) -> bool:
    """Check if value is NaN, None, empty string, or whitespace-only."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    if isinstance(val, str):
        return len(val.strip()) == 0
    return False

def detect_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify missing values in DataFrame.
    Detects NaN, None, empty strings, and whitespace-only values.
    Returns counts, percentages, affected row indices, and columns summary.
    """
    total_rows = len(df)
    total_cells = df.size
    
    # Create a boolean mask of all missing values including whitespace-only
    missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        # Check standard nulls
        null_mask = df[col].isna()
        # Check string whitespace-only
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            str_mask = df[col].astype(str).str.strip().eq("") | df[col].astype(str).str.strip().eq("None") | df[col].astype(str).str.strip().eq("nan") | df[col].astype(str).str.strip().eq("NaN")
            missing_mask[col] = null_mask | str_mask
        else:
            missing_mask[col] = null_mask

    column_stats = []
    total_missing_cells = int(missing_mask.sum().sum())
    
    for col in df.columns:
        cnt = int(missing_mask[col].sum())
        pct = round((cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
        column_stats.append({
            "column": str(col),
            "missing_count": cnt,
            "missing_percentage": pct,
            "has_missing": cnt > 0
        })

    # Sort columns by missing percentage descending
    column_stats = sorted(column_stats, key=lambda x: x["missing_percentage"], reverse=True)
    
    # Identify rows containing at least one missing value
    rows_with_missing_mask = missing_mask.any(axis=1)
    affected_row_count = int(rows_with_missing_mask.sum())
    affected_row_pct = round((affected_row_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    
    # Preview up to 20 affected records
    preview_df = df[rows_with_missing_mask].head(20).fillna("<MISSING>")
    affected_preview = preview_df.to_dict(orient="records")

    return {
        "total_rows": total_rows,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "overall_missing_percentage": round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0,
        "affected_row_count": affected_row_count,
        "affected_row_percentage": affected_row_pct,
        "columns_summary": column_stats,
        "worst_column": column_stats[0] if column_stats and column_stats[0]["missing_count"] > 0 else None,
        "affected_preview": affected_preview
    }

def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect duplicate records in DataFrame.
    If subset is provided, detects duplicates based on selected columns (e.g. Employee ID).
    """
    total_rows = len(df)
    
    # Validate subset columns exist
    if subset:
        valid_cols = [c for c in subset if c in df.columns]
        if not valid_cols:
            valid_cols = None
    else:
        valid_cols = None

    duplicate_mask = df.duplicated(subset=valid_cols, keep=False)
    total_duplicate_rows = int(df.duplicated(subset=valid_cols, keep="first").sum())
    total_duplicate_instances = int(duplicate_mask.sum())

    duplicate_preview = []
    frequency_table = []

    if total_duplicate_rows > 0:
        dupes_df = df[duplicate_mask].sort_values(by=valid_cols if valid_cols else list(df.columns))
        duplicate_preview = dupes_df.head(25).fillna("").to_dict(orient="records")

        # Frequency calculation if subset column provided (e.g. ID frequency)
        if valid_cols and len(valid_cols) == 1:
            val_col = valid_cols[0]
            counts = df[val_col].value_counts()
            repeated = counts[counts > 1].head(20)
            frequency_table = [
                {"value": str(k), "frequency": int(v)}
                for k, v in repeated.items()
            ]

    return {
        "total_rows": total_rows,
        "subset_used": valid_cols,
        "is_subset_mode": valid_cols is not None,
        "total_duplicate_records": total_duplicate_rows,
        "total_affected_instances": total_duplicate_instances,
        "duplicate_percentage": round((total_duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "has_duplicates": total_duplicate_rows > 0,
        "duplicate_preview": duplicate_preview,
        "frequency_table": frequency_table
    }

def validate_dataset(df: pd.DataFrame, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform configurable data validation across columns:
    - Inferred or rule-based data type mismatches
    - Invalid date formats
    - Inconsistent categorical values / casing
    - Email format validation
    - Numerical range validation (min/max boundaries)
    Returns detailed validation anomalies and row indices.
    """
    rules = rules or {}
    total_rows = len(df)
    anomalies = []
    affected_indices = set()

    # Rule checks per column
    for col in df.columns:
        col_rules = rules.get(col, {})
        series = df[col].dropna()
        
        # 1. Email format validation
        if col_rules.get("is_email") or "email" in str(col).lower():
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                for idx, val in series.items():
                    s_val = str(val).strip()
                    if s_val and not EMAIL_REGEX.match(s_val):
                        anomalies.append({
                            "row_index": int(idx),
                            "column": str(col),
                            "value": s_val,
                            "issue_type": "invalid_email_format",
                            "explanation": f"Value '{s_val}' does not match standard email pattern."
                        })
                        affected_indices.add(idx)

        # 2. Date format validation
        if col_rules.get("is_date") or any(k in str(col).lower() for k in ["date", "time", "dob", "joined"]):
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                for idx, val in series.items():
                    s_val = str(val).strip()
                    if s_val:
                        try:
                            pd.to_datetime(s_val)
                        except Exception:
                            anomalies.append({
                                "row_index": int(idx),
                                "column": str(col),
                                "value": s_val,
                                "issue_type": "invalid_date_format",
                                "explanation": f"Value '{s_val}' cannot be parsed as a valid datetime."
                            })
                            affected_indices.add(idx)

        # 3. Numeric range validation
        min_val = col_rules.get("min_val")
        max_val = col_rules.get("max_val")
        if pd.api.types.is_numeric_dtype(df[col]):
            if min_val is not None:
                viol_min = df[df[col] < min_val]
                for idx, row in viol_min.head(10).iterrows():
                    anomalies.append({
                        "row_index": int(idx),
                        "column": str(col),
                        "value": float(row[col]),
                        "issue_type": "range_underflow",
                        "explanation": f"Value {row[col]} is less than allowed minimum ({min_val})."
                    })
                    affected_indices.add(idx)
            if max_val is not None:
                viol_max = df[df[col] > max_val]
                for idx, row in viol_max.head(10).iterrows():
                    anomalies.append({
                        "row_index": int(idx),
                        "column": str(col),
                        "value": float(row[col]),
                        "issue_type": "range_overflow",
                        "explanation": f"Value {row[col]} exceeds allowed maximum ({max_val})."
                    })
                    affected_indices.add(idx)

        # 4. Inconsistent categorical casing detection
        if (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])) and df[col].nunique() < 50:
            str_series = series.astype(str).str.strip()
            lower_to_originals = {}
            for val in str_series.unique():
                low = val.lower()
                lower_to_originals.setdefault(low, set()).add(val)
            
            inconsistent_groups = [list(origs) for origs in lower_to_originals.values() if len(origs) > 1]
            if inconsistent_groups:
                for group in inconsistent_groups:
                    anomalies.append({
                        "row_index": -1,
                        "column": str(col),
                        "value": ", ".join(group),
                        "issue_type": "inconsistent_casing",
                        "explanation": f"Inconsistent capitalization detected: {group}"
                    })

    # Summary
    return {
        "total_rows": total_rows,
        "total_anomalies": len(anomalies),
        "affected_rows_count": len(affected_indices),
        "has_validation_errors": len(anomalies) > 0,
        "anomalies_preview": anomalies[:25]
    }

def calculate_quality_score(
    missing_summary: Dict[str, Any],
    duplicate_summary: Dict[str, Any],
    validation_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate an application-defined Data Quality Score between 0 and 100.
    
    SCORING FORMULA DOCUMENTATION:
    Overall Score = (Completeness Score * 0.40) + (Uniqueness Score * 0.30) + (Validity Score * 0.30)
    
    1. Completeness Score (Weight: 40%):
       - Measures absence of missing/blank values across all cells.
       - Formula: max(0, 100 - (total_missing_cells / total_cells * 100))
    
    2. Uniqueness Score (Weight: 30%):
       - Measures absence of duplicate records.
       - Formula: max(0, 100 - (total_duplicate_records / total_rows * 100))
    
    3. Validity Score (Weight: 30%):
       - Measures absence of format/rule validation anomalies.
       - Formula: max(0, 100 - (affected_rows_count / total_rows * 100 * 2))
    """
    # 1. Completeness
    missing_pct = missing_summary.get("overall_missing_percentage", 0.0)
    completeness_score = round(max(0.0, 100.0 - float(missing_pct)), 1)

    # 2. Uniqueness
    dupe_pct = duplicate_summary.get("duplicate_percentage", 0.0)
    uniqueness_score = round(max(0.0, 100.0 - float(dupe_pct)), 1)

    # 3. Validity
    total_rows = validation_summary.get("total_rows", 1) or 1
    val_affected = validation_summary.get("affected_rows_count", 0)
    val_penalty = min(100.0, (val_affected / total_rows) * 100.0 * 2.0)
    validity_score = round(max(0.0, 100.0 - val_penalty), 1)

    # Composite weighted score
    overall_score = round(
        (completeness_score * 0.40) +
        (uniqueness_score * 0.30) +
        (validity_score * 0.30),
        1
    )

    # Quality Grade & Interpretation
    if overall_score >= 90:
        grade = "Excellent"
        summary_text = "Dataset demonstrates exceptional quality with negligible defects."
    elif overall_score >= 75:
        grade = "Good"
        summary_text = "Dataset is in good analytical shape with minor missingness or duplicate records."
    elif overall_score >= 50:
        grade = "Fair"
        summary_text = "Noticeable data defects found. Data cleaning is recommended before training ML models."
    else:
        grade = "Poor"
        summary_text = "Significant data quality issues detected across multiple dimensions. Requires cleanup."

    # Improvement Suggestions
    suggestions = []
    if completeness_score < 95:
        worst = missing_summary.get("worst_column")
        worst_name = worst["column"] if worst else "several columns"
        suggestions.append(f"Impute missing values in '{worst_name}' using mean/median or remove empty records.")
    if uniqueness_score < 98:
        suggestions.append(f"Remove or review the {duplicate_summary.get('total_duplicate_records')} duplicate rows.")
    if validity_score < 95:
        suggestions.append(f"Address the {validation_summary.get('total_anomalies')} validation errors in dates, emails, or casing.")
    if not suggestions:
        suggestions.append("Data is clean and ready for immediate statistical modeling and dashboarding!")

    return {
        "overall_score": overall_score,
        "grade": grade,
        "summary": summary_text,
        "category_scores": {
            "completeness": completeness_score,
            "uniqueness": uniqueness_score,
            "validity": validity_score
        },
        "scoring_formula": "Overall Score = (Completeness * 0.40) + (Uniqueness * 0.30) + (Validity * 0.30)",
        "suggestions": suggestions
    }

def clean_dataset(
    df: pd.DataFrame,
    operations: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply selective, user-confirmed data cleaning operations.
    NEVER modifies original DataFrame in-place; creates and returns a clean copy.
    
    Supported operations dict:
    - remove_duplicates: bool
    - remove_duplicate_subset: list[str] (optional)
    - drop_missing_rows: bool
    - drop_missing_cols: list[str] (optional)
    - impute_numerical: dict[col_name, 'mean' | 'median' | value]
    - impute_categorical: dict[col_name, 'mode' | value]
    - convert_types: dict[col_name, 'numeric' | 'datetime' | 'string']
    """
    cleaned_df = df.copy()
    initial_rows = len(df)
    initial_cells = df.size
    changes_log = []

    # 1. Remove duplicates
    if operations.get("remove_duplicates"):
        subset = operations.get("remove_duplicate_subset")
        valid_subset = [c for c in subset if c in cleaned_df.columns] if subset else None
        dupes_count = int(cleaned_df.duplicated(subset=valid_subset).sum())
        if dupes_count > 0:
            cleaned_df = cleaned_df.drop_duplicates(subset=valid_subset, keep="first").reset_index(drop=True)
            changes_log.append({
                "operation": "remove_duplicates",
                "details": f"Removed {dupes_count} duplicate rows (subset: {valid_subset or 'All Columns'})."
            })

    # 2. Remove rows with missing values
    if operations.get("drop_missing_rows"):
        cols_to_check = operations.get("drop_missing_cols")
        valid_cols = [c for c in cols_to_check if c in cleaned_df.columns] if cols_to_check else None
        before_drop = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=valid_cols).reset_index(drop=True)
        dropped_cnt = before_drop - len(cleaned_df)
        if dropped_cnt > 0:
            changes_log.append({
                "operation": "drop_missing_rows",
                "details": f"Dropped {dropped_cnt} rows with missing values."
            })

    # 3. Impute Numerical Columns
    impute_num = operations.get("impute_numerical", {})
    for col, method in impute_num.items():
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            missing_cnt = int(cleaned_df[col].isna().sum())
            if missing_cnt > 0:
                if method == "mean":
                    val = float(cleaned_df[col].mean())
                elif method == "median":
                    val = float(cleaned_df[col].median())
                elif isinstance(method, (int, float)):
                    val = method
                else:
                    val = 0.0
                cleaned_df[col] = cleaned_df[col].fillna(val)
                changes_log.append({
                    "operation": "impute_numerical",
                    "column": col,
                    "details": f"Imputed {missing_cnt} missing values in '{col}' using {method} ({round(val, 2)})."
                })

    # 4. Impute Categorical Columns
    impute_cat = operations.get("impute_categorical", {})
    for col, method in impute_cat.items():
        if col in cleaned_df.columns:
            # Replace whitespace-only strings with NaN
            if cleaned_df[col].dtype == object:
                cleaned_df[col] = cleaned_df[col].replace(r"^\s*$", np.nan, regex=True)
            missing_cnt = int(cleaned_df[col].isna().sum())
            if missing_cnt > 0:
                if method == "mode":
                    mode_vals = cleaned_df[col].mode()
                    val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                elif isinstance(method, str):
                    val = method
                else:
                    val = "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(val)
                changes_log.append({
                    "operation": "impute_categorical",
                    "column": col,
                    "details": f"Imputed {missing_cnt} missing values in '{col}' with '{val}'."
                })

    # 5. Type Conversions
    type_conv = operations.get("convert_types", {})
    for col, target_type in type_conv.items():
        if col in cleaned_df.columns:
            try:
                if target_type == "numeric":
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
                elif target_type == "datetime":
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce")
                elif target_type == "string":
                    cleaned_df[col] = cleaned_df[col].astype(str)
                changes_log.append({
                    "operation": "convert_type",
                    "column": col,
                    "details": f"Converted '{col}' to {target_type}."
                })
            except Exception as e:
                pass

    summary = {
        "initial_rows": initial_rows,
        "final_rows": len(cleaned_df),
        "rows_removed": initial_rows - len(cleaned_df),
        "total_operations_applied": len(changes_log),
        "changes_log": changes_log
    }

    return cleaned_df, summary
