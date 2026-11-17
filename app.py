import sys
import os
import time
import json
import io
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.io as pio
from datetime import datetime
from dotenv import load_dotenv

from data_loader import load_dataset_file, inspect_excel_sheets
from sql_engine import (
    initialize_database,
    load_dataframe,
    get_table_schema,
    get_table_info,
    execute_readonly_sql
)
from chat_analyst import ChatAnalystMCPClient, run_chat_turn_sync
from data_quality import (
    detect_missing_values,
    detect_duplicates,
    validate_dataset,
    calculate_quality_score,
    clean_dataset
)
from anomaly_detection import (
    detect_iqr_outliers,
    detect_zscore_outliers,
    compare_anomaly_methods,
    create_anomaly_report_dataframe,
    generate_anomaly_charts,
    get_numeric_columns
)

load_dotenv()

# Page config
st.set_page_config(
    page_title="MCP-Powered Conversational Data Analytics and Intelligence Platform",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%);
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 24px;
        color: #FFFFFF;
    }
    .main-header p {
        margin: 4px 0 0;
        font-size: 14px;
        color: #D6E4FF;
    }
    .score-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    </style>
    <div class="main-header">
        <h1>🦆 MCP-Powered Conversational Data Analytics and Intelligence Platform</h1>
        <p>Enterprise conversational analytics, explainable anomaly detection, automated data quality diagnostics, and high-speed DuckDB SQL engine via Model Context Protocol</p>
    </div>
""", unsafe_allow_html=True)

# ── SIDEBAR: Dataset Upload & Sheet Selector ─────────────────
with st.sidebar:
    st.markdown("### 📁 Dataset Management")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
        help="Supports .csv and multi-sheet .xlsx / .xls workbooks"
    )

    selected_sheet = None

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # If Excel, inspect sheet names
        if ext in [".xlsx", ".xls"]:
            try:
                available_sheets = inspect_excel_sheets(file_path)
                if len(available_sheets) > 1:
                    selected_sheet = st.selectbox(
                        "📑 Select Excel Sheet",
                        options=available_sheets,
                        index=0,
                        help="Choose which sheet to query"
                    )
                elif available_sheets:
                    selected_sheet = available_sheets[0]
                    st.caption(f"Sheet: `{selected_sheet}`")
            except Exception as e:
                st.error(f"Error inspecting sheets: {e}")

        # Check if active dataset or sheet changed
        current_active = st.session_state.get("active_filepath")
        current_sheet = st.session_state.get("active_sheet")

        if current_active != file_path or current_sheet != selected_sheet:
            try:
                df, meta = load_dataset_file(file_path, sheet_name=selected_sheet)
                load_dataframe(df, dataset_id=f"{file_path}:{selected_sheet}")

                st.session_state["active_filepath"] = file_path
                st.session_state["active_filename"] = uploaded_file.name
                st.session_state["active_sheet"] = selected_sheet
                st.session_state["df"] = df
                st.session_state["metadata"] = meta

                # Reset cleaned df session state on new upload
                st.session_state["cleaned_df"] = None
                st.session_state["cleaning_summary"] = None

                # Reset conversation and recreate Chat client
                st.session_state["chat_messages"] = []
                st.session_state["mcp_client"] = ChatAnalystMCPClient(file_path, sheet_name=selected_sheet)

                summary_snippet = json.dumps({
                    "total_rows": meta["total_rows"],
                    "total_columns": meta["total_columns"],
                    "columns": [c["name"] for c in meta["columns_detail"]],
                    "sample": meta["sample_preview_10"][:3]
                })
                st.session_state["mcp_client"].init_chat_session(summary_snippet)

                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": f"👋 **Hello!** Loaded dataset **`{uploaded_file.name}`**"
                               + (f" (Sheet: `{selected_sheet}`)" if selected_sheet else "")
                               + f" with **{meta['total_rows']:,}** rows and **{meta['total_columns']}** columns into DuckDB.\n\n"
                                 "You can ask natural-language questions, explore the **Data Quality Analysis** tab, or run SQL queries!",
                    "charts": [],
                    "sql_queries": []
                })
                st.toast(f"✅ Loaded {uploaded_file.name} successfully into DuckDB!", icon="🦆")
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")

    # Fallback to default CSV if none uploaded
    if "active_filepath" not in st.session_state:
        default_file = os.path.join("data", "Car_Prices_Poland.csv")
        if os.path.exists(default_file):
            try:
                df, meta = load_dataset_file(default_file)
                load_dataframe(df, dataset_id=default_file)
                st.session_state["active_filepath"] = default_file
                st.session_state["active_filename"] = "Car_Prices_Poland.csv"
                st.session_state["active_sheet"] = None
                st.session_state["df"] = df
                st.session_state["metadata"] = meta
                st.session_state["cleaned_df"] = None
                st.session_state["cleaning_summary"] = None
                st.session_state["chat_messages"] = []
                st.session_state["mcp_client"] = ChatAnalystMCPClient(default_file)
                summary_snippet = json.dumps({
                    "total_rows": meta["total_rows"],
                    "total_columns": meta["total_columns"],
                    "columns": [c["name"] for c in meta["columns_detail"]],
                    "sample": meta["sample_preview_10"][:3]
                })
                st.session_state["mcp_client"].init_chat_session(summary_snippet)
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": f"👋 Loaded default dataset **`Car_Prices_Poland.csv`** ({meta['total_rows']:,} rows). Ask questions in chat or check out **Data Quality Analysis**!",
                    "charts": [],
                    "sql_queries": []
                })
            except Exception as e:
                st.error(f"Error loading default dataset: {e}")

    # Display active dataset metadata
    if "metadata" in st.session_state:
        meta = st.session_state["metadata"]
        st.markdown("---")
        sheet_text = f" | Sheet: `{meta['active_sheet']}`" if meta.get("active_sheet") else ""
        st.markdown(f"**Active:** `{meta['filename']}`{sheet_text}")
        st.caption(f"📊 Format: **{meta['format']}** | **{meta['total_rows']:,}** rows × **{meta['total_columns']}** columns")
        
        with st.expander("📋 Columns & Schema Details", expanded=False):
            cols_df = pd.DataFrame([
                {"Column": c["name"], "Type": c["type"], "Missing": c["missing_pct"]}
                for c in meta["columns_detail"]
            ])
            st.dataframe(cols_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧭 Navigation Mode")
    app_mode = st.radio(
        "Mode",
        [
            "💬 Natural Language Analyst",
            "🛡️ Data Quality Analysis",
            "🔍 Anomaly Detection",
            "⚡ DuckDB SQL Workspace",
            "📊 Automated Full EDA Report"
        ],
        label_visibility="collapsed"
    )

    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        if st.button("➕ New Chat", use_container_width=True, help="Start fresh conversation with active dataset"):
            if "active_filepath" in st.session_state:
                st.session_state["chat_messages"] = []
                st.session_state["mcp_client"] = ChatAnalystMCPClient(
                    st.session_state["active_filepath"],
                    sheet_name=st.session_state.get("active_sheet")
                )
                st.session_state["mcp_client"].init_chat_session()
                st.toast("Started new conversation session!", icon="💬")
                st.rerun()
    with sb_col2:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clear chat message history"):
            st.session_state["chat_messages"] = []
            st.rerun()

# ── MODE 1: Conversational AI Analyst (Natural Language to SQL & DQ) ──
if app_mode == "💬 Natural Language Analyst":
    if "active_filepath" not in st.session_state:
        st.info("👈 Upload a CSV or Excel dataset in the sidebar to get started.")
        st.stop()

    meta = st.session_state["metadata"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{meta['total_rows']:,}")
    c2.metric("Columns", meta['total_columns'])
    c3.metric("SQL & DQ Engine", "DuckDB + MCP")
    c4.metric("AI Agent", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))

    # Preview expander
    with st.expander("🔍 Dataset Preview (First 10 Rows)", expanded=False):
        st.dataframe(st.session_state["df"].head(10), use_container_width=True)

    st.markdown("---")

    # Render Chat History
    chat_history = st.session_state.get("chat_messages", [])
    if len(chat_history) > 1:
        with st.expander("📥 Export Conversation Transcript", expanded=False):
            transcript_lines = [
                f"# MCP-Powered Conversational Data Analytics and Intelligence Platform",
                f"**Dataset:** `{meta['filename']}`" + (f" (Sheet: `{meta.get('active_sheet')}`)" if meta.get('active_sheet') else ""),
                f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                "---"
            ]
            for m in chat_history:
                sender = "🧑‍💻 **User**" if m["role"] == "user" else "🤖 **AI Analyst**"
                transcript_lines.append(f"\n{sender}:\n{m['content']}")
                if m.get("sql_queries"):
                    for q in m["sql_queries"]:
                        transcript_lines.append(f"\n```sql\n{q}\n```")
            transcript_text = "\n".join(transcript_lines)
            st.download_button(
                "📥 Download Chat Transcript (.md)",
                data=transcript_text,
                file_name=f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

    for msg in chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            if msg.get("sql_queries"):
                with st.expander("🔍 Generated SQL Query", expanded=False):
                    for q in msg["sql_queries"]:
                        st.code(q, language="sql")

            if msg.get("charts"):
                for chart_info in msg["charts"]:
                    if "fig_json" in chart_info:
                        try:
                            fig = pio.from_json(chart_info["fig_json"])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            if os.path.exists(chart_info.get("chart_path", "")):
                                with open(chart_info["chart_path"], "r", encoding="utf-8") as f:
                                    components.html(f.read(), height=420)

    # Chat Input
    user_prompt = st.chat_input("Ask about your data (e.g., 'Analyze quality', 'How many missing values?', 'Find duplicates', 'Top 5 by sales')...")
    if user_prompt:
        st.session_state["chat_messages"].append({
            "role": "user",
            "content": user_prompt,
            "charts": [],
            "sql_queries": []
        })
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            status_container = st.status("Analyzing with Gemini & MCP...", expanded=True)
            try:
                client = st.session_state["mcp_client"]
                status_container.write("🧠 Querying dataset & computing quality diagnostics...")

                result = run_chat_turn_sync(client, user_prompt)

                tools_used = result.get("tools_executed", [])
                if tools_used:
                    tool_names = [f"`{t['tool']}` ({t['time_s']}s)" for t in tools_used]
                    status_container.update(label=f"Done! Executed MCP tools: {', '.join(tool_names)}", state="complete")
                else:
                    status_container.update(label="Analysis complete!", state="complete")

                response_text = result["response_text"]
                charts = result.get("charts", [])
                sql_queries = result.get("sql_queries", [])

                st.markdown(response_text)

                if sql_queries:
                    with st.expander("🔍 Generated SQL Query", expanded=False):
                        for q in sql_queries:
                            st.code(q, language="sql")

                for chart_info in charts:
                    if "fig_json" in chart_info:
                        try:
                            fig = pio.from_json(chart_info["fig_json"])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            if os.path.exists(chart_info.get("chart_path", "")):
                                with open(chart_info["chart_path"], "r", encoding="utf-8") as f:
                                    components.html(f.read(), height=420)

                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": response_text,
                    "charts": charts,
                    "sql_queries": sql_queries
                })

            except Exception as e:
                status_container.update(label="Error processing query", state="error")
                err_msg = f"❌ **Error:** {str(e)}"
                st.error(err_msg)
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": err_msg,
                    "charts": [],
                    "sql_queries": []
                })

# ── MODE 2: Data Quality Analysis (Phase 1 Feature) ───────────
elif app_mode == "🛡️ Data Quality Analysis":
    st.markdown("### 🛡️ Data Quality Analysis & Remediation")
    st.caption("Deep diagnostics on missing values, duplicate records, validation errors, and selective cleaning.")

    if "df" not in st.session_state:
        st.info("👈 Please upload a CSV or Excel dataset in the sidebar to begin.")
        st.stop()

    df = st.session_state["df"]

    # Compute Quality Metrics
    missing_res = detect_missing_values(df)
    dupe_res = detect_duplicates(df)
    val_res = validate_dataset(df)
    score_res = calculate_quality_score(missing_res, dupe_res, val_res)

    # 1. Overall Score & Category Breakdown
    st.markdown("#### 1. Overall Data Quality Score")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.metric("Overall Score", f"{score_res['overall_score']} / 100", f"Grade: {score_res['grade']}")
    with s_col2:
        st.metric("Completeness", f"{score_res['category_scores']['completeness']}%", f"{missing_res['total_missing_cells']} null cells")
    with s_col3:
        st.metric("Uniqueness", f"{score_res['category_scores']['uniqueness']}%", f"{dupe_res['total_duplicate_records']} dupes")
    with s_col4:
        st.metric("Validity", f"{score_res['category_scores']['validity']}%", f"{val_res['total_anomalies']} anomalies")

    with st.expander("ℹ️ How is the Data Quality Score calculated?", expanded=False):
        st.markdown(f"""
        **Scoring Formula:**
        `{score_res['scoring_formula']}`
        - **Completeness (40%):** Measures non-null, non-blank values across all cells.
        - **Uniqueness (30%):** Measures absence of duplicate rows.
        - **Validity (30%):** Penalizes formatting anomalies (invalid dates, invalid emails, out-of-range bounds).
        """)
        st.markdown("**Improvement Recommendations:**")
        for sugg in score_res["suggestions"]:
            st.markdown(f"- 💡 {sugg}")

    st.markdown("---")

    # 2. Missing Value Detection
    st.markdown("#### 2. Missing Value Analysis")
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.metric("Total Missing Cells", f"{missing_res['total_missing_cells']:,}")
        st.metric("Affected Rows", f"{missing_res['affected_row_count']:,} ({missing_res['affected_row_percentage']}%)")
        if missing_res["worst_column"]:
            st.warning(f"Highest Missingness: **`{missing_res['worst_column']['column']}`** ({missing_res['worst_column']['missing_percentage']}%)")
        else:
            st.success("🎉 No missing values detected in any column!")

    with m_col2:
        missing_df = pd.DataFrame(missing_res["columns_summary"])
        st.dataframe(
            missing_df.style.bar(subset=["missing_percentage"], color="#FFA07A"),
            use_container_width=True,
            hide_index=True
        )

    if missing_res["affected_preview"]:
        with st.expander("🔍 Preview Rows with Missing Values", expanded=False):
            st.dataframe(pd.DataFrame(missing_res["affected_preview"]), use_container_width=True)

    st.markdown("---")

    # 3. Duplicate Record Detection
    st.markdown("#### 3. Duplicate Record Analysis")
    d_col1, d_col2 = st.columns([1, 2])
    with d_col1:
        st.metric("Total Duplicate Rows", f"{dupe_res['total_duplicate_records']:,}")
        st.metric("Duplicate Percentage", f"{dupe_res['duplicate_percentage']}%")
        
        subset_cols = st.multiselect(
            "Check duplicates on specific columns (e.g. ID):",
            options=list(df.columns),
            help="Select key columns to identify repeated entities"
        )
        if subset_cols:
            custom_dupes = detect_duplicates(df, subset=subset_cols)
            st.info(f"Duplicates based on {subset_cols}: **{custom_dupes['total_duplicate_records']}** records")

    with d_col2:
        if dupe_res["duplicate_preview"]:
            st.warning(f"Found {dupe_res['total_duplicate_records']} duplicate rows:")
            st.dataframe(pd.DataFrame(dupe_res["duplicate_preview"]), use_container_width=True)
        else:
            st.success("🎉 No duplicate rows found in dataset!")

    st.markdown("---")

    # 4. Data Validation Anomalies
    st.markdown("#### 4. Data Validation Anomalies")
    if val_res["has_validation_errors"]:
        st.warning(f"Identified {val_res['total_anomalies']} potential data anomalies:")
        anom_df = pd.DataFrame(val_res["anomalies_preview"])
        st.dataframe(anom_df, use_container_width=True, hide_index=True)
    else:
        st.success("🎉 All validated dates, emails, and numeric types conform to standard patterns!")

    st.markdown("---")

    # 5. Selective Data Cleaning
    st.markdown("#### 5. Selective Data Cleaning & Remediation")
    st.caption("Apply optional data cleaning operations to a copy of your dataset. The original dataset is never modified.")

    with st.form("cleaning_form"):
        cl_c1, cl_c2 = st.columns(2)
        with cl_c1:
            st.markdown("**Row Filtering & Deduplication**")
            rem_dupes = st.checkbox("Remove Duplicate Rows", value=False)
            drop_nulls = st.checkbox("Drop rows with any missing values", value=False)
            
        with cl_c2:
            st.markdown("**Imputation Options**")
            num_cols = df.select_dtypes(include="number").columns.tolist()
            num_impute_col = st.selectbox("Select numeric column to impute (optional):", ["None"] + num_cols)
            num_impute_strategy = st.radio("Strategy:", ["mean", "median"], horizontal=True)

            cat_cols = df.select_dtypes(include="object").columns.tolist()
            cat_impute_col = st.selectbox("Select categorical column to impute (optional):", ["None"] + cat_cols)

        submit_clean = st.form_submit_button("🧹 Clean Dataset & Compare", type="primary", use_container_width=True)

    if submit_clean:
        cleaning_ops = {
            "remove_duplicates": rem_dupes,
            "drop_missing_rows": drop_nulls,
            "impute_numerical": {num_impute_col: num_impute_strategy} if num_impute_col != "None" else {},
            "impute_categorical": {cat_impute_col: "mode"} if cat_impute_col != "None" else {}
        }
        cleaned_data, summary = clean_dataset(df, cleaning_ops)
        st.session_state["cleaned_df"] = cleaned_data
        st.session_state["cleaning_summary"] = summary

    # Display Before-and-After Comparison if cleaned dataset exists
    if st.session_state.get("cleaned_df") is not None:
        cleaned_df = st.session_state["cleaned_df"]
        summary = st.session_state["cleaning_summary"]

        st.success(f"✅ Cleaning completed! Applied {summary['total_operations_applied']} operations.")
        
        comp1, comp2, comp3 = st.columns(3)
        comp1.metric("Original Rows", f"{summary['initial_rows']:,}")
        comp2.metric("Cleaned Rows", f"{summary['final_rows']:,}")
        comp3.metric("Rows Removed", f"{summary['rows_removed']:,}")

        if summary["changes_log"]:
            st.markdown("**Operations Applied:**")
            for log_entry in summary["changes_log"]:
                st.markdown(f"- 🛠️ {log_entry.get('details', '')}")

        st.markdown("**Cleaned Dataset Preview:**")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

        # Download cleaned dataset
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            csv_buf = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Cleaned CSV", data=csv_buf, file_name="cleaned_dataset.csv", mime="text/csv", use_container_width=True)
        with dl_c2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                cleaned_df.to_excel(writer, index=False, sheet_name="CleanedData")
            st.download_button("📥 Download Cleaned Excel (.xlsx)", data=excel_buf.getvalue(), file_name="cleaned_dataset.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ── MODE 3: Explainable Anomaly Detection (Phase 2 Feature) ───
elif app_mode == "🔍 Anomaly Detection":
    st.markdown("### 🔍 Explainable Anomaly Detection Dashboard")
    st.caption("Robust statistical anomaly detection with IQR, Z-Score, multi-column profiling, severity scoring, and human-readable explanations.")

    if "df" not in st.session_state:
        st.info("👈 Please upload a CSV or Excel dataset in the sidebar to begin.")
        st.stop()

    df = st.session_state["df"]
    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        st.warning("⚠️ No numerical columns found in the active dataset for anomaly detection.")
        st.stop()

    # Controls Section
    st.markdown("#### ⚙️ Configuration & Target Columns")
    cfg_c1, cfg_c2, cfg_c3 = st.columns([2, 1, 1])

    with cfg_c1:
        selected_cols = st.multiselect(
            "Select Numerical Columns to Inspect",
            options=numeric_cols,
            default=numeric_cols[:min(8, len(numeric_cols))],
            help="Choose one or multiple numerical columns for anomaly detection."
        )

    with cfg_c2:
        method_choice = st.selectbox(
            "Detection Method",
            ["IQR (Interquartile Range)", "Z-Score (Standard Deviations)", "Both / Method Comparison"],
            help="IQR is non-parametric and robust to skewed data. Z-Score assumes approximate Gaussian normality."
        )

    with cfg_c3:
        if "IQR" in method_choice:
            iqr_mult = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1, help="Default is 1.5 (Tukey's fence). 3.0 represents extreme outliers.")
            z_thresh = 3.0
        elif "Z-Score" in method_choice:
            z_thresh = st.slider("Z-Score Threshold (σ)", min_value=1.5, max_value=5.0, value=3.0, step=0.1, help="Default is 3.0 standard deviations (|Z| > 3.0).")
            iqr_mult = 1.5
        else:
            iqr_mult = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            z_thresh = st.slider("Z-Score Threshold (σ)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)

    if not selected_cols:
        st.info("Please select at least one numerical column to analyze.")
        st.stop()

    st.markdown("---")

    # Run Detection
    if method_choice == "Both / Method Comparison":
        comp_res = compare_anomaly_methods(df, columns=selected_cols, iqr_multiplier=iqr_mult, zscore_threshold=z_thresh)

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IQR Anomalies", f"{comp_res['total_iqr_anomalies']:,}")
        m2.metric("Z-Score Anomalies", f"{comp_res['total_zscore_anomalies']:,}")
        m3.metric("Consensus (Both Flagged)", f"{comp_res['both_flagged_count']:,}")
        m4.metric("Method Agreement", f"{comp_res['method_agreement_percentage']}%")

        # Column comparison table
        st.markdown("#### 📊 Side-by-Side Method Comparison by Column")
        col_comp_df = pd.DataFrame(comp_res["column_comparison"])
        st.dataframe(col_comp_df, use_container_width=True, hide_index=True)

        # Detailed Comparison Records
        st.markdown("#### 📋 Detailed Records Comparison")
        if comp_res["comparison_records"]:
            comp_df = pd.DataFrame(comp_res["comparison_records"])
            
            # Filter controls
            filt_c1, filt_c2 = st.columns(2)
            with filt_c1:
                col_filt = st.multiselect("Filter by Column", options=selected_cols, default=[])
            with filt_c2:
                agree_filt = st.selectbox("Filter by Agreement", ["All Records", "Both", "IQR Only", "Z-Score Only"])

            filtered_comp = comp_df
            if col_filt:
                filtered_comp = filtered_comp[filtered_comp["column"].isin(col_filt)]
            if agree_filt != "All Records":
                filtered_comp = filtered_comp[filtered_comp["agreement"] == agree_filt]

            st.dataframe(filtered_comp, use_container_width=True, hide_index=True)

            # Downloads for Comparison
            st.markdown("##### 📥 Export Comparison Results")
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                csv_data = filtered_comp.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Comparison (CSV)", data=csv_data, file_name="anomaly_comparison.csv", mime="text/csv", use_container_width=True)
            with exp_c2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    filtered_comp.to_excel(writer, index=False, sheet_name="Comparison")
                st.download_button("📥 Download Comparison (Excel)", data=buf.getvalue(), file_name="anomaly_comparison.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.success("🎉 Zero anomalies flagged by either IQR or Z-Score across selected columns!")

    else:
        # Single Method (IQR or Z-Score)
        is_iqr = "IQR" in method_choice
        if is_iqr:
            detect_res = detect_iqr_outliers(df, columns=selected_cols, multiplier=iqr_mult)
        else:
            detect_res = detect_zscore_outliers(df, columns=selected_cols, threshold=z_thresh)

        # 1. Metric Cards
        st.markdown(f"#### 📊 {detect_res['method']} Detection Overview")
        met_c1, met_c2, met_c3, met_c4 = st.columns(4)
        met_c1.metric("Columns Analyzed", len(detect_res["analyzed_columns"]))
        met_c2.metric("Total Anomalies", f"{detect_res['total_anomalies']:,}")
        met_c3.metric("Affected Rows", f"{detect_res['affected_rows_count']:,} ({detect_res['affected_rows_percentage']}%)")

        # Find most anomalous column
        worst_col = "None"
        max_anoms = 0
        for cs in detect_res["column_summaries"]:
            if cs.get("anomaly_count", 0) > max_anoms:
                max_anoms = cs["anomaly_count"]
                worst_col = f"{cs['column']} ({max_anoms})"
        met_c4.metric("Highest Outliers", worst_col)

        # 2. Column Summaries Table
        st.markdown("#### 📋 Column Profiles & Normal Bounds")
        cols_summary_list = []
        for cs in detect_res["column_summaries"]:
            row_dict = {
                "Column": cs["column"],
                "Status": cs.get("status", "analyzed"),
                "Valid Records": cs.get("total_valid", 0),
                "Anomalies": cs.get("anomaly_count", 0),
                "Anomaly %": f"{cs.get('anomaly_percentage', 0.0)}%",
                "Lower Bound": cs.get("lower_bound"),
                "Upper Bound": cs.get("upper_bound"),
                "Skewness": cs.get("skewness"),
            }
            if cs.get("warning"):
                row_dict["Warning / Note"] = cs["warning"]
            elif cs.get("reason"):
                row_dict["Warning / Note"] = cs["reason"]
            else:
                row_dict["Warning / Note"] = "Normal"
            cols_summary_list.append(row_dict)

        st.dataframe(pd.DataFrame(cols_summary_list), use_container_width=True, hide_index=True)

        # 3. Explainable Anomalous Records Table
        st.markdown("#### 🔎 Explainable Anomalous Records")
        anomalies_list = detect_res.get("anomalies", [])

        if anomalies_list:
            anom_df = pd.DataFrame(anomalies_list)

            # Filtering options
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                filt_target_col = st.selectbox("Filter by Column", ["All Columns"] + selected_cols)
            with f_col2:
                filt_severity = st.selectbox("Filter by Severity", ["All Severities", "High", "Medium", "Low"])
            with f_col3:
                sort_by = st.selectbox("Sort By", ["Severity (High to Low)", "Row Index", "Value (Desc)"])

            display_df = anom_df.copy()
            if filt_target_col != "All Columns":
                display_df = display_df[display_df["column"] == filt_target_col]
            if filt_severity != "All Severities":
                display_df = display_df[display_df["severity"] == filt_severity]

            if sort_by == "Severity (High to Low)":
                sev_map = {"High": 3, "Medium": 2, "Low": 1}
                display_df["_rank"] = display_df["severity"].map(sev_map)
                display_df = display_df.sort_values(by="_rank", ascending=False).drop(columns=["_rank"])
            elif sort_by == "Row Index":
                display_df = display_df.sort_values(by="row_index", ascending=True)
            elif sort_by == "Value (Desc)":
                display_df = display_df.sort_values(by="value", ascending=False)

            # Reorder columns for optimal readability
            preferred_cols = ["row_index", "column", "value", "expected_range", "method", "severity", "reason"]
            show_cols = [c for c in preferred_cols if c in display_df.columns]
            st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

            # 4. Interactive Visualizations
            st.markdown("#### 📈 Interactive Visualizations")
            viz_col = st.selectbox("Select Column to Visualize", selected_cols)
            
            tab_box, tab_scatter, tab_hist = st.tabs(["📦 Box Plot (Outliers Highlighted)", "📈 Row Scatter & Thresholds", "📊 Distribution & Bounds"])

            with tab_box:
                try:
                    fig_b = generate_anomaly_charts(df, viz_col, detect_res, chart_type="box")
                    st.plotly_chart(fig_b, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating box plot: {e}")

            with tab_scatter:
                try:
                    fig_s = generate_anomaly_charts(df, viz_col, detect_res, chart_type="scatter")
                    st.plotly_chart(fig_s, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating scatter plot: {e}")

            with tab_hist:
                try:
                    fig_h = generate_anomaly_charts(df, viz_col, detect_res, chart_type="histogram")
                    st.plotly_chart(fig_h, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating histogram: {e}")

            # 5. Export / Download Section
            st.markdown("---")
            st.markdown("#### 📥 Export Reports & Enriched Datasets")
            dl_r1, dl_r2 = st.columns(2)

            with dl_r1:
                st.markdown("**1. Download Anomalies Table Only:**")
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    anom_csv = display_df[show_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Anomalies (CSV)",
                        data=anom_csv,
                        file_name=f"anomalies_{detect_res['method'].lower()}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with c_dl2:
                    anom_excel_buf = io.BytesIO()
                    with pd.ExcelWriter(anom_excel_buf, engine="openpyxl") as writer:
                        display_df[show_cols].to_excel(writer, index=False, sheet_name="Anomalies")
                    st.download_button(
                        "📥 Anomalies (Excel)",
                        data=anom_excel_buf.getvalue(),
                        file_name=f"anomalies_{detect_res['method'].lower()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            with dl_r2:
                st.markdown("**2. Download Enriched Dataset (with anomaly flags & reasons):**")
                enriched = create_anomaly_report_dataframe(df, anomalies_list)
                e_dl1, e_dl2 = st.columns(2)
                with e_dl1:
                    enr_csv = enriched.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Enriched Dataset (CSV)",
                        data=enr_csv,
                        file_name=f"enriched_dataset_with_anomalies.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with e_dl2:
                    enr_excel_buf = io.BytesIO()
                    with pd.ExcelWriter(enr_excel_buf, engine="openpyxl") as writer:
                        enriched.to_excel(writer, index=False, sheet_name="EnrichedData")
                    st.download_button(
                        "📥 Enriched Dataset (Excel)",
                        data=enr_excel_buf.getvalue(),
                        file_name=f"enriched_dataset_with_anomalies.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        else:
            st.success(f"🎉 No anomalies detected in selected columns using {detect_res['method']} method!")

# ── MODE 4: DuckDB SQL Workspace ─────────────────────────────
elif app_mode == "⚡ DuckDB SQL Workspace":
    st.markdown("### ⚡ DuckDB Interactive SQL Workspace")
    st.caption("Write and execute standard SQL queries directly against your active dataset (table name: `dataset`).")

    if "df" not in st.session_state:
        st.info("👈 Upload a dataset in the sidebar to run SQL queries.")
        st.stop()

    meta = st.session_state["metadata"]
    cols_list = meta["column_names"]
    
    col_exp, template_exp = st.columns([2, 1])
    with col_exp:
        st.info(f"💡 Active table: **`dataset`** ({meta['total_rows']:,} rows). Available columns:\n`{', '.join(cols_list[:12])}`" + ("..." if len(cols_list) > 12 else ""))
    with template_exp:
        template = st.selectbox(
            "Example Templates",
            [
                "Custom Query",
                "SELECT * FROM dataset LIMIT 10",
                f"SELECT COUNT(*) as total_records FROM dataset",
                f"SELECT {cols_list[0]}, COUNT(*) FROM dataset GROUP BY {cols_list[0]} LIMIT 10"
            ]
        )

    initial_sql = "SELECT * FROM dataset LIMIT 10"
    if template != "Custom Query":
        initial_sql = template

    query_input = st.text_area("SQL Query", value=initial_sql, height=120)

    c_run, c_limit = st.columns([1, 4])
    with c_run:
        run_btn = st.button("🚀 Run SQL", type="primary", use_container_width=True)

    if run_btn and query_input:
        t0 = time.time()
        result_df, query_meta = execute_readonly_sql(query_input, max_rows=500)
        elapsed = time.time() - t0

        if query_meta.get("status") == "success":
            st.success(f"✅ Query executed in **{elapsed*1000:.1f} ms** | Returned **{query_meta['total_rows_returned']:,}** rows")
            if query_meta.get("is_truncated"):
                st.caption(f"Displaying first {query_meta['rows_displayed']} rows.")
            st.dataframe(result_df, use_container_width=True)
            
            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results (CSV)", data=csv_data, file_name="sql_results.csv", mime="text/csv")
        else:
            st.error(f"❌ **{query_meta.get('error_type', 'SQL Error').replace('_', ' ').title()}:** {query_meta.get('error')}")

# ── MODE 5: Automated Full EDA Report ────────────────────────
elif app_mode == "📊 Automated Full EDA Report":
    st.markdown("### 📊 Automated Exploratory Data Analysis Report")
    st.caption("Generate a structured 7-section statistical report with visualizations powered by Gemini.")

    if "df" not in st.session_state:
        st.info("👈 Upload a dataset in the sidebar to begin.")
        st.stop()

    df = st.session_state["df"]
    active_path = st.session_state["active_filepath"]

    if st.button("🚀 Run Full Automated EDA Pipeline", type="primary", use_container_width=True):
        with st.status("Generating comprehensive report...", expanded=True) as status:
            t0 = time.time()
            st.write("📊 Computing descriptive statistics...")
            from eda_engine import run_eda
            stats = run_eda(df)

            st.write("🎨 Generating standard chart suite...")
            from charts import generate_charts
            chart_paths = generate_charts(df)

            st.write(f"🤖 Sending dataset statistics to Gemini ({os.getenv('GEMINI_MODEL')})...")
            from agent import generate_report, save_report
            report = generate_report(stats, chart_paths, os.path.basename(active_path))
            out_path = save_report(report, os.path.basename(active_path))

            status.update(label=f"EDA Complete in {time.time()-t0:.1f}s!", state="complete")

            st.session_state["eda_report"] = report
            st.session_state["eda_out_path"] = out_path
            st.success(f"Report saved to `{out_path}`")

    if "eda_report" in st.session_state:
        st.markdown("---")
        st.download_button(
            "📥 Download Full Report (.md)",
            data=st.session_state["eda_report"],
            file_name=st.session_state.get("eda_out_path", "eda_report.md"),
            mime="text/markdown",
            use_container_width=True
        )
        st.markdown(st.session_state["eda_report"])
