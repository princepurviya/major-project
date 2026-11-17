import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple, Union

THEME = "plotly_dark"


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Retrieve numeric column names excluding all-null columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_cols = []
    for c in num_cols:
        if df[c].dropna().nunique() > 0:
            valid_cols.append(str(c))
    return valid_cols


def calculate_column_skewness(series: pd.Series) -> Optional[float]:
    """Calculate sample skewness safely."""
    cleaned = series.dropna()
    if len(cleaned) < 3:
        return None
    try:
        val = float(cleaned.skew())
        return round(val, 3) if not np.isnan(val) else None
    except Exception:
        return None


def detect_iqr_outliers(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Formulas:
        IQR = Q3 - Q1
        Lower Bound = Q1 - multiplier * IQR
        Upper Bound = Q3 + multiplier * IQR

    Severity scoring based on distance beyond bounds:
        Low:    0 < distance <= 1.5 * IQR
        Medium: 1.5 * IQR < distance <= 3.0 * IQR
        High:   distance > 3.0 * IQR (Extreme Outlier)
    """
    if columns is None:
        target_cols = get_numeric_columns(df)
    elif isinstance(columns, str):
        target_cols = [columns]
    else:
        target_cols = [c for c in columns if c in df.columns]

    total_rows = len(df)
    column_summaries = []
    all_anomalies = []
    affected_row_indices = set()

    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid_data = series.dropna()
        n_valid = len(valid_data)

        if n_valid < 4:
            column_summaries.append({
                "column": col,
                "status": "skipped",
                "reason": "Insufficient numeric records (< 4) to calculate quartiles",
                "total_valid": n_valid,
                "anomaly_count": 0,
                "anomaly_percentage": 0.0,
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_bound": None,
                "upper_bound": None
            })
            continue

        q1 = float(valid_data.quantile(0.25))
        q3 = float(valid_data.quantile(0.75))
        iqr = float(q3 - q1)
        lower_bound = float(q1 - multiplier * iqr)
        upper_bound = float(q3 + multiplier * iqr)
        skew = calculate_column_skewness(valid_data)

        col_anomalies = []

        if iqr == 0.0:
            # Constant or near-constant column
            column_summaries.append({
                "column": col,
                "status": "constant_or_zero_iqr",
                "reason": "IQR is 0.0 (identical 25th and 75th percentiles; data has zero variance in interquartile range)",
                "total_valid": n_valid,
                "anomaly_count": 0,
                "anomaly_percentage": 0.0,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": 0.0,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "skewness": skew
            })
            continue

        for idx, val in series.items():
            if pd.isna(val):
                continue

            val_float = float(val)
            is_low = val_float < lower_bound
            is_high = val_float > upper_bound

            if is_low or is_high:
                if is_high:
                    distance = val_float - upper_bound
                    direction = "above upper bound"
                    ratio = distance / iqr
                    bound_ref = upper_bound
                    reason = (
                        f"Value {val_float:,.2f} exceeds upper bound {upper_bound:,.2f} by "
                        f"{distance:,.2f} ({ratio:.2f}× IQR beyond upper limit)"
                    )
                else:
                    distance = lower_bound - val_float
                    direction = "below lower bound"
                    ratio = distance / iqr
                    bound_ref = lower_bound
                    reason = (
                        f"Value {val_float:,.2f} falls below lower bound {lower_bound:,.2f} by "
                        f"{distance:,.2f} ({ratio:.2f}× IQR below lower limit)"
                    )

                # Determine severity
                if ratio <= 1.5:
                    severity = "Low"
                elif ratio <= 3.0:
                    severity = "Medium"
                else:
                    severity = "High"

                anomaly_item = {
                    "row_index": int(idx),
                    "column": col,
                    "value": round(val_float, 4),
                    "expected_range": f"[{lower_bound:,.2f}, {upper_bound:,.2f}]",
                    "method": "IQR",
                    "severity": severity,
                    "reason": reason,
                    "distance_from_bound": round(distance, 4),
                    "iqr_ratio": round(ratio, 2),
                    "direction": direction,
                    "bound_violated": round(bound_ref, 4)
                }
                col_anomalies.append(anomaly_item)
                all_anomalies.append(anomaly_item)
                affected_row_indices.add(int(idx))

        anomaly_pct = round((len(col_anomalies) / total_rows * 100), 2) if total_rows > 0 else 0.0

        column_summaries.append({
            "column": col,
            "status": "analyzed",
            "total_valid": n_valid,
            "anomaly_count": len(col_anomalies),
            "anomaly_percentage": anomaly_pct,
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "multiplier": multiplier,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "skewness": skew,
            "anomalies_sample": col_anomalies[:10]
        })

    affected_rows_pct = round((len(affected_row_indices) / total_rows * 100), 2) if total_rows > 0 else 0.0

    return {
        "method": "IQR",
        "multiplier": multiplier,
        "total_rows": total_rows,
        "analyzed_columns": target_cols,
        "total_anomalies": len(all_anomalies),
        "affected_rows_count": len(affected_row_indices),
        "affected_rows_percentage": affected_rows_pct,
        "column_summaries": column_summaries,
        "anomalies": all_anomalies
    }


def detect_zscore_outliers(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect outliers using the Z-Score method.

    Formulas:
        Z = (X - Mean) / Std
        Expected Range: [Mean - threshold * Std, Mean + threshold * Std]

    Severity scoring based on absolute Z-Score:
        Low:    threshold < |Z| <= threshold + 1.0  (e.g., 3.0 to 4.0)
        Medium: threshold + 1.0 < |Z| <= threshold + 2.0 (e.g., 4.0 to 5.0)
        High:   |Z| > threshold + 2.0 (e.g., > 5.0, extreme sigma deviation)

    Note:
        Evaluates skewness. If |skewness| > 1.0, data is significantly skewed and
        Z-Score assumptions (Gaussian normality) are compromised; warnings are attached.
    """
    if columns is None:
        target_cols = get_numeric_columns(df)
    elif isinstance(columns, str):
        target_cols = [columns]
    else:
        target_cols = [c for c in columns if c in df.columns]

    total_rows = len(df)
    column_summaries = []
    all_anomalies = []
    affected_row_indices = set()

    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid_data = series.dropna()
        n_valid = len(valid_data)

        if n_valid < 3:
            column_summaries.append({
                "column": col,
                "status": "skipped",
                "reason": "Insufficient numeric records (< 3) to calculate mean and standard deviation",
                "total_valid": n_valid,
                "anomaly_count": 0,
                "anomaly_percentage": 0.0,
                "mean": None,
                "std": None
            })
            continue

        mean = float(valid_data.mean())
        std = float(valid_data.std(ddof=1))
        skew = calculate_column_skewness(valid_data)

        skew_warning = None
        if skew is not None and abs(skew) > 1.0:
            skew_warning = (
                f"Column has high skewness ({skew:+.2f}). Z-score assumes a normal bell curve. "
                "Consider using IQR for more robust non-parametric anomaly detection."
            )

        if std == 0.0 or np.isnan(std):
            column_summaries.append({
                "column": col,
                "status": "zero_variance",
                "reason": "Standard deviation is 0.0 (all values in this column are identical)",
                "total_valid": n_valid,
                "anomaly_count": 0,
                "anomaly_percentage": 0.0,
                "mean": round(mean, 4),
                "std": 0.0,
                "skewness": skew,
                "warning": skew_warning
            })
            continue

        lower_bound = float(mean - threshold * std)
        upper_bound = float(mean + threshold * std)
        col_anomalies = []

        for idx, val in series.items():
            if pd.isna(val):
                continue

            val_float = float(val)
            z_score = (val_float - mean) / std
            abs_z = abs(z_score)

            if abs_z > threshold:
                diff_from_mean = val_float - mean
                direction = "above mean" if z_score > 0 else "below mean"

                # Severity calculation
                if abs_z <= threshold + 1.0:
                    severity = "Low"
                elif abs_z <= threshold + 2.0:
                    severity = "Medium"
                else:
                    severity = "High"

                reason = (
                    f"Z-score of {z_score:+.2f} exceeds threshold ±{threshold:.1f} "
                    f"({abs(diff_from_mean):,.2f} units / {abs_z:.2f} standard deviations {direction})"
                )

                anomaly_item = {
                    "row_index": int(idx),
                    "column": col,
                    "value": round(val_float, 4),
                    "expected_range": f"[{lower_bound:,.2f}, {upper_bound:,.2f}]",
                    "method": "Z-Score",
                    "severity": severity,
                    "reason": reason,
                    "z_score": round(float(z_score), 2),
                    "distance_from_mean": round(float(diff_from_mean), 4),
                    "mean": round(mean, 4),
                    "std": round(std, 4)
                }
                col_anomalies.append(anomaly_item)
                all_anomalies.append(anomaly_item)
                affected_row_indices.add(int(idx))

        anomaly_pct = round((len(col_anomalies) / total_rows * 100), 2) if total_rows > 0 else 0.0

        column_summaries.append({
            "column": col,
            "status": "analyzed",
            "total_valid": n_valid,
            "anomaly_count": len(col_anomalies),
            "anomaly_percentage": anomaly_pct,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "threshold": threshold,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "skewness": skew,
            "warning": skew_warning,
            "anomalies_sample": col_anomalies[:10]
        })

    affected_rows_pct = round((len(affected_row_indices) / total_rows * 100), 2) if total_rows > 0 else 0.0

    return {
        "method": "Z-Score",
        "threshold": threshold,
        "total_rows": total_rows,
        "analyzed_columns": target_cols,
        "total_anomalies": len(all_anomalies),
        "affected_rows_count": len(affected_row_indices),
        "affected_rows_percentage": affected_rows_pct,
        "column_summaries": column_summaries,
        "anomalies": all_anomalies
    }


def compare_anomaly_methods(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Run both IQR and Z-Score methods and compare results side-by-side.
    Identifies records flagged by both methods, IQR only, or Z-score only.
    """
    iqr_res = detect_iqr_outliers(df, columns=columns, multiplier=iqr_multiplier)
    z_res = detect_zscore_outliers(df, columns=columns, threshold=zscore_threshold)

    # Build unique anomaly keys (row_index, column)
    iqr_map = {(a["row_index"], a["column"]): a for a in iqr_res["anomalies"]}
    z_map = {(a["row_index"], a["column"]): a for a in z_res["anomalies"]}

    iqr_keys = set(iqr_map.keys())
    z_keys = set(z_map.keys())

    both_keys = iqr_keys.intersection(z_keys)
    iqr_only_keys = iqr_keys - z_keys
    z_only_keys = z_keys - iqr_keys
    union_keys = iqr_keys.union(z_keys)

    agreement_pct = round((len(both_keys) / len(union_keys) * 100), 2) if union_keys else 100.0

    comparison_records = []
    for key in sorted(union_keys):
        row_idx, col = key
        val = iqr_map.get(key, z_map.get(key))["value"]
        flagged_by = []
        if key in iqr_keys:
            flagged_by.append("IQR")
        if key in z_keys:
            flagged_by.append("Z-Score")

        iqr_info = iqr_map.get(key)
        z_info = z_map.get(key)

        comparison_records.append({
            "row_index": row_idx,
            "column": col,
            "value": val,
            "flagged_by": " + ".join(flagged_by),
            "agreement": "Both" if len(flagged_by) == 2 else ("IQR Only" if "IQR" in flagged_by else "Z-Score Only"),
            "iqr_severity": iqr_info["severity"] if iqr_info else "Normal",
            "iqr_reason": iqr_info["reason"] if iqr_info else "Within IQR range",
            "zscore_severity": z_info["severity"] if z_info else "Normal",
            "z_score": z_info["z_score"] if z_info else None,
            "zscore_reason": z_info["reason"] if z_info else "Within Z-score limit"
        })

    # Per-column comparison table
    col_comparison = []
    iqr_col_map = {c["column"]: c for c in iqr_res["column_summaries"]}
    z_col_map = {c["column"]: c for c in z_res["column_summaries"]}

    for col in iqr_res["analyzed_columns"]:
        iqr_col = iqr_col_map.get(col, {})
        z_col = z_col_map.get(col, {})
        col_comparison.append({
            "column": col,
            "iqr_anomalies": iqr_col.get("anomaly_count", 0),
            "zscore_anomalies": z_col.get("anomaly_count", 0),
            "iqr_bounds": f"[{iqr_col.get('lower_bound')}, {iqr_col.get('upper_bound')}]" if iqr_col.get("lower_bound") is not None else "N/A",
            "zscore_bounds": f"[{z_col.get('lower_bound')}, {z_col.get('upper_bound')}]" if z_col.get("lower_bound") is not None else "N/A",
            "skewness": iqr_col.get("skewness"),
            "skew_warning": z_col.get("warning") is not None
        })

    return {
        "iqr_multiplier": iqr_multiplier,
        "zscore_threshold": zscore_threshold,
        "total_iqr_anomalies": len(iqr_keys),
        "total_zscore_anomalies": len(z_keys),
        "both_flagged_count": len(both_keys),
        "iqr_only_count": len(iqr_only_keys),
        "zscore_only_count": len(z_only_keys),
        "total_unique_anomalies": len(union_keys),
        "method_agreement_percentage": agreement_pct,
        "column_comparison": col_comparison,
        "comparison_records": comparison_records
    }


def create_anomaly_report_dataframe(df: pd.DataFrame, anomalies: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Attach anomaly flags, reasons, and severities to a copy of the dataset
    so users can download the enriched data.
    """
    enriched_df = df.copy()
    enriched_df["is_anomaly"] = False
    enriched_df["anomaly_severity"] = "None"
    enriched_df["anomaly_columns"] = ""
    enriched_df["anomaly_reasons"] = ""

    severity_order = {"None": 0, "Low": 1, "Medium": 2, "High": 3}

    # Group anomalies by row_index
    row_anomalies: Dict[int, List[Dict[str, Any]]] = {}
    for a in anomalies:
        idx = a["row_index"]
        if idx not in row_anomalies:
            row_anomalies[idx] = []
        row_anomalies[idx].append(a)

    for idx, items in row_anomalies.items():
        if idx in enriched_df.index:
            enriched_df.at[idx, "is_anomaly"] = True
            cols = list({item["column"] for item in items})
            enriched_df.at[idx, "anomaly_columns"] = ", ".join(cols)
            reasons = [f"[{item['column']}] {item['reason']}" for item in items]
            enriched_df.at[idx, "anomaly_reasons"] = "; ".join(reasons)

            # Assign maximum severity
            max_sev = "None"
            max_rank = 0
            for item in items:
                sev = item.get("severity", "Low")
                rank = severity_order.get(sev, 1)
                if rank > max_rank:
                    max_rank = rank
                    max_sev = sev
            enriched_df.at[idx, "anomaly_severity"] = max_sev

    return enriched_df


def generate_anomaly_charts(
    df: pd.DataFrame,
    column: str,
    method_result: Dict[str, Any],
    chart_type: str = "box"
) -> go.Figure:
    """
    Generate interactive Plotly visualizations with highlighted anomaly points.
    Supported types: 'box', 'scatter', 'histogram'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    series = pd.to_numeric(df[column], errors="coerce")
    clean_series = series.dropna()

    # Find anomalies for this column
    col_anomalies = [a for a in method_result.get("anomalies", []) if a["column"] == column]
    anomaly_indices = {a["row_index"] for a in col_anomalies}
    anomaly_map = {a["row_index"]: a for a in col_anomalies}

    # Look up summary for bounds
    col_summary = next((c for c in method_result.get("column_summaries", []) if c["column"] == column), {})
    lower_bound = col_summary.get("lower_bound")
    upper_bound = col_summary.get("upper_bound")

    method_name = method_result.get("method", "Anomaly Detection")

    if chart_type == "box":
        fig = go.Figure()

        # Box plot for all data
        fig.add_trace(go.Box(
            y=clean_series,
            name=column,
            boxpoints=False,
            marker_color="#4A90E2",
            line_color="#4A90E2",
            fillcolor="rgba(74, 144, 226, 0.3)"
        ))

        # Overlay anomalies
        if col_anomalies:
            anom_y = [a["value"] for a in col_anomalies]
            anom_text = [
                f"Row {a['row_index']}<br>Value: {a['value']:,.2f}<br>Severity: {a['severity']}<br>{a['reason']}"
                for a in col_anomalies
            ]
            fig.add_trace(go.Scatter(
                x=[column] * len(col_anomalies),
                y=anom_y,
                mode="markers",
                name=f"Anomalies ({len(col_anomalies)})",
                marker=dict(
                    color="#FF4B4B",
                    size=10,
                    symbol="x",
                    line=dict(width=2, color="#FFFFFF")
                ),
                text=anom_text,
                hoverinfo="text"
            ))

        fig.update_layout(
            title=f"Box Plot: {column} ({method_name} Outliers Highlighted)",
            template=THEME,
            height=420,
            showlegend=True
        )
        return fig

    elif chart_type == "scatter":
        # Scatter: Row Index vs Column Value
        fig = go.Figure()

        normal_indices = [idx for idx in clean_series.index if idx not in anomaly_indices]
        normal_values = clean_series.loc[normal_indices]

        fig.add_trace(go.Scatter(
            x=normal_indices,
            y=normal_values,
            mode="markers",
            name="Normal Points",
            marker=dict(color="#4A90E2", size=6, opacity=0.65),
            hoverinfo="x+y"
        ))

        if col_anomalies:
            anom_x = [a["row_index"] for a in col_anomalies]
            anom_y = [a["value"] for a in col_anomalies]
            sev_colors = {"Low": "#FFB020", "Medium": "#FF6B4A", "High": "#D32F2F"}
            colors = [sev_colors.get(a["severity"], "#FF4B4B") for a in col_anomalies]
            hover_texts = [
                f"Row {a['row_index']}<br>Value: {a['value']:,.2f}<br>Severity: {a['severity']}<br>{a['reason']}"
                for a in col_anomalies
            ]

            fig.add_trace(go.Scatter(
                x=anom_x,
                y=anom_y,
                mode="markers",
                name=f"Anomalies ({len(col_anomalies)})",
                marker=dict(color=colors, size=11, symbol="diamond", line=dict(width=1.5, color="#FFFFFF")),
                text=hover_texts,
                hoverinfo="text"
            ))

        # Add horizontal threshold lines if available
        if lower_bound is not None:
            fig.add_hline(
                y=lower_bound,
                line_dash="dash",
                line_color="#FFA726",
                annotation_text=f"Lower Bound: {lower_bound:,.2f}",
                annotation_position="bottom right"
            )
        if upper_bound is not None:
            fig.add_hline(
                y=upper_bound,
                line_dash="dash",
                line_color="#FFA726",
                annotation_text=f"Upper Bound: {upper_bound:,.2f}",
                annotation_position="top right"
            )

        fig.update_layout(
            title=f"Scatter Plot: {column} across Rows ({method_name})",
            xaxis_title="Row Index",
            yaxis_title=column,
            template=THEME,
            height=420,
            showlegend=True
        )
        return fig

    elif chart_type == "histogram":
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=clean_series,
            name="Distribution",
            nbinsx=35,
            marker_color="#5C6BC0",
            opacity=0.75
        ))

        if lower_bound is not None:
            fig.add_vline(
                x=lower_bound,
                line_dash="dash",
                line_color="#FF7043",
                annotation_text=f"Lower: {lower_bound:,.2f}",
                annotation_position="top left"
            )
        if upper_bound is not None:
            fig.add_vline(
                x=upper_bound,
                line_dash="dash",
                line_color="#FF7043",
                annotation_text=f"Upper: {upper_bound:,.2f}",
                annotation_position="top right"
            )

        fig.update_layout(
            title=f"Histogram & Detection Thresholds: {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            template=THEME,
            height=420,
            showlegend=False
        )
        return fig

    else:
        raise ValueError(f"Unsupported chart_type '{chart_type}'. Choose 'box', 'scatter', or 'histogram'.")
