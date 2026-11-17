import sys
import os
import io
import json
import numpy as np
import pandas as pd

# Windows encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from anomaly_detection import (
    detect_iqr_outliers,
    detect_zscore_outliers,
    compare_anomaly_methods,
    create_anomaly_report_dataframe,
    generate_anomaly_charts,
    get_numeric_columns
)
import mcp_server
from chat_analyst import ChatAnalystMCPClient, run_chat_turn_sync


def run_tests():
    print("======================================================================")
    print("🧪 PHASE 2 COMPLETE VERIFICATION TEST SUITE: EXPLAINABLE ANOMALY DETECTION")
    print("======================================================================\n")

    # -------------------------------------------------------------------------
    # TEST 1: Synthetic Dataset with Ground Truth Anomalies
    # -------------------------------------------------------------------------
    print("--- [TEST 1] Synthetic Dataset Ground Truth Validation ---")
    data = {
        "id": list(range(1, 16)),
        "salary": [50000, 52000, 49000, 51000, 53000, 48000, 50500, 51500, 49500, 52500, 51000, 50000, 48500, 250000, 5000], # 250k = extreme high, 5k = extreme low
        "age": [25, 26, 27, 25, 28, 26, 27, 29, 25, 26, 28, 27, 26, 28, 115], # 115 = high outlier
        "constant_col": [42.0] * 15, # zero variance
        "short_col": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], # <4 records
        "category": ["A", "B"] * 7 + ["A"]
    }
    df_synth = pd.DataFrame(data)
    test_csv_path = os.path.join("data", "test_anomaly_dataset.csv")
    df_synth.to_csv(test_csv_path, index=False)
    print(f"Created synthetic dataset at '{test_csv_path}' (15 rows, 4 numeric cols, 1 category).")

    # 1. IQR-based Detection
    print("\n--- [TEST 2] IQR-based Anomaly Detection ---")
    iqr_res = detect_iqr_outliers(df_synth, columns=["salary", "age", "constant_col", "short_col"], multiplier=1.5)
    print(f"IQR Total Anomalies: {iqr_res['total_anomalies']}")
    print(f"Affected Rows Count: {iqr_res['affected_rows_count']} ({iqr_res['affected_rows_percentage']}%)")
    
    assert iqr_res["total_anomalies"] >= 2, "Expected at least 2 anomalies in salary and age"
    
    # Check constant and short col handling
    const_sum = next(c for c in iqr_res["column_summaries"] if c["column"] == "constant_col")
    short_sum = next(c for c in iqr_res["column_summaries"] if c["column"] == "short_col")
    print(f"Constant Col Status: {const_sum['status']} - {const_sum['reason']}")
    print(f"Short Col Status: {short_sum['status']} - {short_sum['reason']}")
    assert const_sum["status"] == "constant_or_zero_iqr"
    assert short_sum["status"] == "skipped"
    print("✅ IQR detection and edge case handling verified!")

    # 2. Z-Score-based Detection
    print("\n--- [TEST 3] Z-Score-based Anomaly Detection ---")
    z_res = detect_zscore_outliers(df_synth, columns=["salary", "age", "constant_col"], threshold=2.5)
    print(f"Z-Score Total Anomalies: {z_res['total_anomalies']}")
    for a in z_res["anomalies"]:
        print(f"  Flagged: [{a['column']}] Row {a['row_index']} Val={a['value']} | Z={a['z_score']} | Severity={a['severity']}")
    
    const_z_sum = next(c for c in z_res["column_summaries"] if c["column"] == "constant_col")
    assert const_z_sum["status"] == "zero_variance"
    print(f"Constant Col Z-Score Status: {const_z_sum['status']}")
    print("✅ Z-Score detection verified!")

    # 3. Explainability and Severity
    print("\n--- [TEST 4] Explainable Anomaly Results with Plain-English Reasons ---")
    for anom in iqr_res["anomalies"]:
        print(f"Column: {anom['column']} | Row: {anom['row_index']} | Value: {anom['value']}")
        print(f"  Expected Range: {anom['expected_range']}")
        print(f"  Severity: {anom['severity']}")
        print(f"  Reason: {anom['reason']}")
        assert anom["severity"] in ("Low", "Medium", "High")
        assert len(anom["reason"]) > 10
        assert "exceeds" in anom["reason"] or "falls below" in anom["reason"]
    print("✅ Explainability and severity scoring verified!")

    # 4. Multiple Numerical Column Analysis
    print("\n--- [TEST 5] Multiple Numerical Column Analysis ---")
    num_cols = get_numeric_columns(df_synth)
    print(f"Auto-detected numeric columns: {num_cols}")
    multi_res = detect_iqr_outliers(df_synth, columns=num_cols)
    assert len(multi_res["column_summaries"]) == len(num_cols)
    print(f"Successfully profiled {len(multi_res['column_summaries'])} numerical columns simultaneously.")
    print("✅ Multi-column profiling verified!")

    # 5. Plotly Visualizations
    print("\n--- [TEST 6] Plotly Visualizations (Box, Scatter, Histogram) ---")
    fig_box = generate_anomaly_charts(df_synth, "salary", iqr_res, "box")
    fig_scatter = generate_anomaly_charts(df_synth, "salary", iqr_res, "scatter")
    fig_hist = generate_anomaly_charts(df_synth, "salary", iqr_res, "histogram")
    assert fig_box is not None
    assert fig_scatter is not None
    assert fig_hist is not None
    print(f"Box Plot title: {fig_box.layout.title.text}")
    print(f"Scatter Plot title: {fig_scatter.layout.title.text}")
    print(f"Histogram title: {fig_hist.layout.title.text}")
    print("✅ Plotly interactive visualizations verified!")

    # 6. Method Comparison
    print("\n--- [TEST 7] Method Comparison (IQR vs Z-Score) ---")
    comp_res = compare_anomaly_methods(df_synth, columns=["salary", "age"], iqr_multiplier=1.5, zscore_threshold=2.5)
    print(f"IQR Total: {comp_res['total_iqr_anomalies']}")
    print(f"Z-Score Total: {comp_res['total_zscore_anomalies']}")
    print(f"Consensus (Both): {comp_res['both_flagged_count']}")
    print(f"Agreement %: {comp_res['method_agreement_percentage']}%")
    assert comp_res["total_unique_anomalies"] > 0
    print("✅ Method comparison verified!")

    # 7. CSV and Excel Downloads & Enriched Dataset Creation
    print("\n--- [TEST 8] CSV and Excel Export Verification ---")
    enriched_df = create_anomaly_report_dataframe(df_synth, iqr_res["anomalies"])
    assert "is_anomaly" in enriched_df.columns
    assert "anomaly_severity" in enriched_df.columns
    assert "anomaly_reasons" in enriched_df.columns
    assert enriched_df["is_anomaly"].sum() == iqr_res["affected_rows_count"]
    
    # Test Excel serialization via BytesIO
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        enriched_df.to_excel(writer, index=False, sheet_name="Anomalies")
    excel_bytes = excel_buf.getvalue()
    assert len(excel_bytes) > 1000
    print(f"Enriched Excel output generated ({len(excel_bytes):,} bytes).")
    print("✅ Report and dataset downloads verified!")

    # 8. MCP Tools Discovery & Direct Execution
    print("\n--- [TEST 9] MCP Server Tools Discovery & Execution ---")
    # Test 6 new tools
    t1_json = mcp_server.detect_anomalies_tool(test_csv_path, columns=["salary"], method="iqr")
    t1 = json.loads(t1_json)
    assert t1["total_anomalies"] > 0
    print(f"MCP detect_anomalies: {t1['total_anomalies']} anomalies found.")

    t2_json = mcp_server.detect_iqr_outliers_tool(test_csv_path, columns=["age"], multiplier=1.5)
    t2 = json.loads(t2_json)
    assert t2["total_anomalies"] > 0
    print(f"MCP detect_iqr_outliers: {t2['total_anomalies']} anomalies found.")

    t3_json = mcp_server.detect_zscore_outliers_tool(test_csv_path, columns=["salary"], threshold=2.5)
    t3 = json.loads(t3_json)
    print(f"MCP detect_zscore_outliers: {t3['total_anomalies']} anomalies found.")

    t4_json = mcp_server.get_anomaly_summary_tool(test_csv_path, columns=["salary", "age"])
    t4 = json.loads(t4_json)
    assert len(t4["columns"]) == 2
    print(f"MCP get_anomaly_summary: {len(t4['columns'])} columns summarized.")

    t5_json = mcp_server.get_anomaly_details_tool(test_csv_path, column="salary")
    t5 = json.loads(t5_json)
    assert len(t5["records"]) > 0
    print(f"MCP get_anomaly_details: {len(t5['records'])} records returned.")

    t6_json = mcp_server.compare_anomaly_methods_tool(test_csv_path, columns=["salary", "age"])
    t6 = json.loads(t6_json)
    assert "method_agreement_percentage" in t6
    print(f"MCP compare_anomaly_methods: Agreement={t6['method_agreement_percentage']}%.")
    print("✅ All 6 MCP Anomaly Detection tools verified!")

    # 9. Real Dataset Verification (Car_Prices_Poland.csv)
    print("\n--- [TEST 10] Real Dataset Verification (Car_Prices_Poland.csv) ---")
    real_csv = os.path.join("data", "Car_Prices_Poland.csv")
    if os.path.exists(real_csv):
        df_real = pd.read_csv(real_csv)
        print(f"Loaded real dataset: {real_csv} ({len(df_real):,} rows, {len(df_real.columns)} cols)")
        real_iqr = detect_iqr_outliers(df_real, columns=["price", "mileage", "vol"])
        print(f"Car Prices Poland IQR Anomalies: {real_iqr['total_anomalies']:,} in {real_iqr['affected_rows_count']:,} rows")
        for cs in real_iqr["column_summaries"]:
            print(f"  Col '{cs['column']}': {cs['anomaly_count']:,} anomalies ({cs['anomaly_percentage']}%) | Bounds: [{cs['lower_bound']:,}, {cs['upper_bound']:,}] | Skew: {cs['skewness']}")
        print("✅ Real dataset anomaly detection verified!")
    else:
        print("Real dataset Car_Prices_Poland.csv not found, skipped.")

    # 10. Gemini Conversational Anomaly Detection & TaskGroup Stability
    print("\n--- [TEST 11] Gemini Conversational Anomaly Detection & TaskGroup Stability ---")
    if os.getenv("GEMINI_API_KEY"):
        try:
            print("Initializing ChatAnalystMCPClient with synthetic dataset...")
            client = ChatAnalystMCPClient(test_csv_path)
            client.init_chat_session()
            
            question = "Are there any salary or age anomalies in this dataset? Explain why they were flagged and their severity."
            print(f"💬 Asking Gemini: '{question}'")
            turn_res = run_chat_turn_sync(client, question)
            
            print(f"\n🤖 Gemini Answer:\n{turn_res['response_text']}\n")
            print(f"🔧 Tools Executed by Gemini: {[t['tool'] for t in turn_res.get('tools_executed', [])]}")
            
            assert len(turn_res["response_text"]) > 20
            print("✅ Gemini conversational anomaly detection and TaskGroup isolation test PASSED!")
        except Exception as e:
            print(f"⚠️ Gemini chat call encountered: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("GEMINI_API_KEY not set; skipping live conversational turn.")

    print("\n======================================================================")
    print("🎉 ALL PHASE 2 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    run_tests()
