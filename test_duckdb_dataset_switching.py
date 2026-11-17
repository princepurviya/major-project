"""
Comprehensive Verification Suite for DuckDB Dataset-Switching and ResourceInUseException Fix.
Validates:
1. DuckDB version and result-cleanup API (complete(), close()).
2. Proper release of unfinished query results and cursors.
3. Switching from Dataset A to Dataset B creates a clean dataset state without data leakage.
4. Absence of private connection attributes (e.g. conn._active_dataset_id).
5. Concurrency safety across threads and MCP tool executions.
6. CSV-to-CSV, CSV-to-Excel, and Excel-to-CSV switching.
7. Switching datasets after executing SELECT queries without fetching all rows.
8. Repeated dataset switching and consecutive MCP requests.
9. Phase 1 Data Quality and Phase 2 Anomaly Detection tool executions on switched datasets.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import threading
import duckdb
import pandas as pd

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import sql_engine
from sql_engine import (
    initialize_database,
    close_connection,
    load_dataframe,
    get_table_schema,
    get_table_info,
    execute_readonly_sql
)
from data_loader import load_dataset_file
import mcp_server

def test_1_version_and_api():
    print("\n--- Test 1: Verify DuckDB Version and Cleanup APIs ---")
    version = duckdb.__version__
    print(f"DuckDB Version: {version}")
    assert version.startswith("2.") or version.startswith("1.") or version.startswith("0."), "Unexpected DuckDB version"
    
    con = duckdb.connect(":memory:")
    assert hasattr(con, "complete"), "Connection must have complete() method"
    assert hasattr(con, "close"), "Connection must have close() method"
    assert hasattr(con, "register"), "Connection must have register() method"
    assert hasattr(con, "unregister"), "Connection must have unregister() method"
    con.close()
    print("✅ Test 1 Passed: DuckDB version and required cleanup APIs verified.")

def test_2_no_private_connection_attributes():
    print("\n--- Test 2: Check No Private Connection Attributes ---")
    close_connection()
    df = pd.DataFrame({"x": [1, 2, 3]})
    load_dataframe(df, dataset_id="test_priv")
    
    con = initialize_database()
    assert not hasattr(con, "_active_dataset_id"), "CRITICAL: Private attribute '_active_dataset_id' must NOT be set on DuckDBPyConnection!"
    assert sql_engine._ACTIVE_DATASET_ID == "test_priv", "Active dataset ID must be tracked via module-level state"
    print("✅ Test 2 Passed: No private attributes on DuckDB connection; tracked cleanly in module state.")

def test_3_unfetched_select_before_switch():
    print("\n--- Test 3: Switching Datasets After SELECT Without Fetching All Rows ---")
    close_connection()
    df_a = pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "customer": ["Alice", "Bob", "Charlie", "David", "Emma"],
        "amount": [150.0, 200.0, 50.0, 320.0, 80.0]
    })
    load_dataframe(df_a, dataset_id="DatasetA_Orders")
    
    # Intentionally execute query and do NOT fetch all rows (leave cursor open)
    con = initialize_database()
    unfinished_result = con.execute("SELECT * FROM dataset")
    # Only fetch one row, leaving 4 rows pending
    first_row = unfinished_result.fetchone()
    assert first_row is not None
    print(f"Executed SELECT on Dataset A, fetched only 1 row: {first_row}")
    
    # Now switch to Dataset B without manual complete()
    df_b = pd.DataFrame({
        "emp_id": [1, 2, 3],
        "employee_name": ["John", "Jane", "Jack"],
        "department": ["Engineering", "Product", "Design"]
    })
    
    # This must NOT raise ResourceInUseException
    try:
        load_dataframe(df_b, dataset_id="DatasetB_Employees")
        print("Successfully loaded Dataset B after unfinished query on Dataset A!")
    except duckdb.ResourceInUseException as e:
        assert False, f"Failed with ResourceInUseException: {e}"
        
    # Verify Dataset B is active and Dataset A columns do not leak
    info_b = get_table_info()
    assert "employee_name" in info_b["columns"]
    assert "department" in info_b["columns"]
    assert "customer" not in info_b["columns"]
    assert "amount" not in info_b["columns"]
    assert info_b["row_count"] == 3
    
    res_b, meta_b = execute_readonly_sql("SELECT employee_name, department FROM dataset")
    assert meta_b["status"] == "success"
    assert len(res_b) == 3
    print("✅ Test 3 Passed: Switching datasets after unfetched SELECT succeeds with zero leakage.")

def test_4_switch_formats():
    print("\n--- Test 4: CSV-to-CSV, CSV-to-Excel, and Excel-to-CSV Switching ---")
    os.makedirs("data", exist_ok=True)
    
    # Create test datasets
    csv_1 = os.path.join("data", "test_cars.csv")
    pd.DataFrame({"brand": ["Toyota", "Ford", "BMW"], "hp": [150, 200, 250]}).to_csv(csv_1, index=False)
    
    csv_2 = os.path.join("data", "test_books.csv")
    pd.DataFrame({"title": ["Book 1", "Book 2"], "pages": [300, 450]}).to_csv(csv_2, index=False)
    
    xlsx_file = os.path.join("data", "test_finance.xlsx")
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as w:
        pd.DataFrame({"revenue": [1000, 2000], "quarter": ["Q1", "Q2"]}).to_excel(w, sheet_name="QReport", index=False)
        pd.DataFrame({"expense": [500, 600], "category": ["Travel", "Software"]}).to_excel(w, sheet_name="Expenses", index=False)

    # 4a: CSV 1 to CSV 2
    df1, _ = load_dataset_file(csv_1)
    load_dataframe(df1, dataset_id="CarsCSV")
    info = get_table_info()
    assert "brand" in info["columns"] and "title" not in info["columns"]
    print("  4a. Loaded Cars CSV: brand, hp")
    
    df2, _ = load_dataset_file(csv_2)
    load_dataframe(df2, dataset_id="BooksCSV")
    info = get_table_info()
    assert "title" in info["columns"] and "brand" not in info["columns"]
    print("  4a -> 4b. Switched CSV to CSV (Books): title, pages (Cars columns removed)")

    # 4b: CSV to Excel Sheet 1
    df_xl1, _ = load_dataset_file(xlsx_file, sheet_name="QReport")
    load_dataframe(df_xl1, dataset_id="QReport")
    info = get_table_info()
    assert "revenue" in info["columns"] and "title" not in info["columns"]
    print("  4b -> 4c. Switched CSV to Excel Sheet 'QReport': revenue, quarter")

    # 4c: Excel Sheet 1 to Excel Sheet 2
    df_xl2, _ = load_dataset_file(xlsx_file, sheet_name="Expenses")
    load_dataframe(df_xl2, dataset_id="Expenses")
    info = get_table_info()
    assert "expense" in info["columns"] and "revenue" not in info["columns"]
    print("  4c -> 4d. Switched Excel Sheet to Excel Sheet 'Expenses': expense, category")

    # 4d: Excel Sheet to CSV 1
    df1_again, _ = load_dataset_file(csv_1)
    load_dataframe(df1_again, dataset_id="CarsCSV")
    info = get_table_info()
    assert "brand" in info["columns"] and "expense" not in info["columns"]
    print("  4d -> 4e. Switched Excel Sheet back to CSV: brand, hp")

    print("✅ Test 4 Passed: All switching combinations (CSV<->CSV, CSV<->Excel, Sheet<->Sheet) successful.")

def test_5_repeated_rapid_switching():
    print("\n--- Test 5: Rapid Repeated Dataset Switching (15 Cycles) ---")
    df_alpha = pd.DataFrame({"alpha_col": range(50), "num": range(50)})
    df_beta = pd.DataFrame({"beta_col": [f"item_{i}" for i in range(30)], "val": range(30)})
    
    t0 = time.time()
    for cycle in range(15):
        # Switch to Alpha
        load_dataframe(df_alpha, dataset_id=f"Alpha_{cycle}")
        info_a = get_table_info()
        assert "alpha_col" in info_a["columns"] and "beta_col" not in info_a["columns"]
        
        # Execute query without fetchdf (leave cursor hanging)
        con = initialize_database()
        con.execute("SELECT * FROM dataset WHERE num > 10")
        
        # Switch to Beta
        load_dataframe(df_beta, dataset_id=f"Beta_{cycle}")
        info_b = get_table_info()
        assert "beta_col" in info_b["columns"] and "alpha_col" not in info_b["columns"]
        
        # Query Beta
        res, meta = execute_readonly_sql("SELECT COUNT(*) as c FROM dataset")
        assert meta["status"] == "success"
        assert res.iloc[0]["c"] == 30
        
    t1 = time.time()
    print(f"✅ Test 5 Passed: 15 rapid cycles (30 switches) completed in {t1 - t0:.2f}s with 0 errors.")

def test_6_concurrency_and_thread_safety():
    print("\n--- Test 6: Concurrency and Thread Safety ---")
    df = pd.DataFrame({"id": range(100), "score": range(100, 200)})
    load_dataframe(df, dataset_id="Concurrent_Test")
    
    errors = []
    
    def worker(worker_id):
        for _ in range(5):
            res_df, meta = execute_readonly_sql(f"SELECT AVG(score) FROM dataset WHERE id >= {worker_id * 5}")
            if meta["status"] != "success":
                errors.append(f"Worker {worker_id} query failed: {meta.get('error')}")
            schema = get_table_schema()
            if schema["status"] != "success":
                errors.append(f"Worker {worker_id} schema failed: {schema.get('error')}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent workers reported errors: {errors}"
    print("✅ Test 6 Passed: 10 concurrent worker threads executed queries with zero conflicts.")

def test_7_mcp_tools_and_phases_compatibility():
    print("\n--- Test 7: Integration with MCP Tools, Phase 1 (DQ) & Phase 2 (Anomaly Detection) ---")
    # Test file
    csv_file = os.path.join("data", "test_cars.csv")
    
    # 1. MCP SQL tool
    sql_res_str = mcp_server.execute_sql_query(csv_file, "SELECT brand, hp FROM dataset WHERE hp > 160")
    import json
    parsed_sql = json.loads(sql_res_str)
    assert parsed_sql["status"] == "success"
    print("  7a. MCP execute_sql_query: passed")

    # 2. Phase 1 Data Quality tools
    dq_res_str = mcp_server.detect_missing_values_tool(csv_file)
    parsed_dq = json.loads(dq_res_str)
    assert "total_missing_cells" in parsed_dq
    assert parsed_dq["total_rows"] == 3
    print("  7b. Phase 1 detect_missing_values: passed")

    # 3. Phase 2 Anomaly Detection tools
    anom_res_str = mcp_server.detect_iqr_outliers_tool(csv_file, columns=["hp"])
    parsed_anom = json.loads(anom_res_str)
    assert parsed_anom["method"] == "IQR"
    assert "column_summaries" in parsed_anom
    print("  7c. Phase 2 detect_iqr_outliers: passed")

    print("✅ Test 7 Passed: MCP tools, Data Quality, and Anomaly Detection work seamlessly on active dataset.")

if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING DUCKDB DATASET-SWITCHING & RESOURCEINUSE FIX VERIFICATION")
    print("======================================================================")
    test_1_version_and_api()
    test_2_no_private_connection_attributes()
    test_3_unfetched_select_before_switch()
    test_4_switch_formats()
    test_5_repeated_rapid_switching()
    test_6_concurrency_and_thread_safety()
    test_7_mcp_tools_and_phases_compatibility()
    print("\n======================================================================")
    print("🎉 ALL 7 TEST SUITES PASSED WITH 100% SUCCESS!")
    print("======================================================================")
