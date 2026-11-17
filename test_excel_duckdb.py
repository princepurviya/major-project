import sys
sys.stdout.reconfigure(encoding="utf-8")

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import pandas as pd
import openpyxl

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from data_loader import load_dataset_file, inspect_excel_sheets
from sql_engine import (
    initialize_database,
    load_dataframe,
    get_table_schema,
    get_table_info,
    execute_readonly_sql,
    is_safe_query
)
from chat_analyst import ChatAnalystMCPClient, run_chat_turn_sync

def create_sample_files():
    """Create test CSV and multi-sheet Excel files."""
    os.makedirs("data", exist_ok=True)
    
    # 1. CSV dataset
    csv_path = os.path.join("data", "test_sales.csv")
    df_sales = pd.DataFrame({
        "product_id": [1, 2, 3, 4, 5],
        "product_name": ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"],
        "category": ["Electronics", "Accessories", "Accessories", "Electronics", "Accessories"],
        "price": [1200, 25, 75, 300, 80],
        "quantity_sold": [10, 150, 80, 25, 60]
    })
    df_sales.to_csv(csv_path, index=False)

    # 2. Multi-sheet Excel workbook (.xlsx)
    xlsx_path = os.path.join("data", "test_multisheet.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_sales.to_excel(writer, sheet_name="Sales", index=False)
        df_emp = pd.DataFrame({
            "emp_id": [101, 102, 103],
            "name": ["Alice", "Bob", "Charlie"],
            "dept": ["IT", "Sales", "HR"],
            "salary": [85000, 62000, 58000]
        })
        df_emp.to_excel(writer, sheet_name="Employees", index=False)

    return csv_path, xlsx_path

def test_excel_and_csv_loading(csv_path, xlsx_path):
    print("\n--- 1. Testing CSV & Multi-Sheet Excel Loading ---")
    
    # CSV
    df_csv, meta_csv = load_dataset_file(csv_path)
    assert meta_csv["format"] == "CSV"
    assert meta_csv["total_rows"] == 5
    print(f"✅ CSV Loading: {meta_csv['filename']} ({meta_csv['total_rows']} rows)")

    # Excel Sheet inspection
    sheets = inspect_excel_sheets(xlsx_path)
    assert "Sales" in sheets and "Employees" in sheets
    print(f"✅ Excel Sheet Detection: Found sheets {sheets}")

    # Load Sheet 1
    df_s1, meta_s1 = load_dataset_file(xlsx_path, sheet_name="Sales")
    assert meta_s1["active_sheet"] == "Sales"
    assert "quantity_sold" in meta_s1["column_names"]
    print(f"✅ Excel Sheet 'Sales' Loaded: {meta_s1['total_rows']} rows, columns: {meta_s1['column_names']}")

    # Load Sheet 2
    df_s2, meta_s2 = load_dataset_file(xlsx_path, sheet_name="Employees")
    assert meta_s2["active_sheet"] == "Employees"
    assert "salary" in meta_s2["column_names"]
    print(f"✅ Excel Sheet 'Employees' Loaded: {meta_s2['total_rows']} rows, columns: {meta_s2['column_names']}")

def test_duckdb_queries(xlsx_path):
    print("\n--- 2. Testing DuckDB SQL Engine ---")
    
    # Load sheet 'Sales'
    df_sales, _ = load_dataset_file(xlsx_path, sheet_name="Sales")
    load_dataframe(df_sales, dataset_id="Sales")

    # Basic SELECT
    res_df, meta = execute_readonly_sql("SELECT product_name, price FROM dataset LIMIT 3")
    assert meta["status"] == "success"
    assert len(res_df) == 3
    print("✅ Basic SELECT passed:")
    print(res_df)

    # GROUP BY & Aggregation
    agg_sql = "SELECT category, SUM(price * quantity_sold) as total_revenue, COUNT(*) as count FROM dataset GROUP BY category ORDER BY total_revenue DESC"
    res_agg, meta_agg = execute_readonly_sql(agg_sql)
    assert meta_agg["status"] == "success"
    print("✅ GROUP BY & Aggregations passed:")
    print(res_agg)

    # SQL Injection / Destructive Query Protection
    print("\n--- 3. Testing SQL Injection & Security Constraints ---")
    destructive_queries = [
        "DROP TABLE dataset",
        "DELETE FROM dataset WHERE price > 100",
        "INSERT INTO dataset VALUES (6, 'Phone', 'Electronics', 800, 5)",
        "ALTER TABLE dataset ADD COLUMN tax INT",
        "SELECT * FROM dataset; DROP TABLE dataset;"
    ]

    for d_query in destructive_queries:
        res, meta = execute_readonly_sql(d_query)
        assert meta["status"] == "error"
        assert meta["error_type"] == "security_violation"
        print(f"✅ Security blocked destructive query: '{d_query}' -> {meta['error']}")

def test_dataset_switch_safety(csv_path, xlsx_path):
    print("\n--- 4. Testing Dataset Switching & Isolation ---")
    # Load Employees sheet
    df_emp, _ = load_dataset_file(xlsx_path, sheet_name="Employees")
    load_dataframe(df_emp, dataset_id="Employees")

    # Verify 'salary' exists and 'quantity_sold' does not exist
    info = get_table_info()
    assert "salary" in info["columns"]
    assert "quantity_sold" not in info["columns"]
    print(f"✅ Dataset switched to Employees. Active columns: {info['columns']}")

    # Load Sales CSV
    df_csv, _ = load_dataset_file(csv_path)
    load_dataframe(df_csv, dataset_id="SalesCSV")
    info2 = get_table_info()
    assert "quantity_sold" in info2["columns"]
    assert "salary" not in info2["columns"]
    print(f"✅ Dataset switched to Sales CSV. Active columns: {info2['columns']}")

def test_natural_language_to_sql(xlsx_path):
    print("\n--- 5. Testing Gemini Natural Language to DuckDB SQL via MCP ---")
    client = ChatAnalystMCPClient(xlsx_path, sheet_name="Sales")
    client.init_chat_session()

    questions = [
        "Which product has the highest price?",
        "What is the total revenue by category?"
    ]

    for q in questions:
        print(f"\nUser: {q}")
        res = run_chat_turn_sync(client, q)
        print(f"AI: {res['response_text']}")
        if res.get("sql_queries"):
            print(f"   Generated SQL: {res['sql_queries']}")
        if res.get("tools_executed"):
            print(f"   Tools called: {[t['tool'] for t in res['tools_executed']]}")
        assert len(res["response_text"].strip()) > 0

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    csv_file, xlsx_file = create_sample_files()
    test_excel_and_csv_loading(csv_file, xlsx_file)
    test_duckdb_queries(xlsx_file)
    test_dataset_switch_safety(csv_file, xlsx_file)
    test_natural_language_to_sql(xlsx_file)
