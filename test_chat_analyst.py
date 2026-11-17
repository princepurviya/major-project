import sys
sys.stdout.reconfigure(encoding="utf-8")

# Windows fix: must be BEFORE any asyncio import
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import json
import pandas as pd

# Load environment
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from query_engine import (
    get_or_load_dataset,
    get_dataset_metadata,
    run_duckdb_query,
    compute_pandas_metric,
    generate_custom_chart
)
from chat_analyst import ChatAnalystMCPClient, run_chat_turn_sync

def create_sample_dataset():
    """Create a sample employee dataset for testing."""
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "test_employees.csv")
    df = pd.DataFrame({
        "employee_id": [101, 102, 103, 104, 105, 106],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona"],
        "department": ["Engineering", "Sales", "Engineering", "HR", "Sales", "Engineering"],
        "salary": [95000, 72000, 110000, 65000, 84000, 105000],
        "age": [29, 34, 41, 31, 38, 35],
        "years_experience": [4, 7, 15, 6, 11, 9]
    })
    df.to_csv(csv_path, index=False)
    return csv_path

def test_query_engine(csv_path):
    print("\n--- 1. Testing Query Engine (DuckDB & Pandas) ---")
    
    # Metadata
    meta = get_dataset_metadata(csv_path)
    assert meta["total_rows"] == 6
    assert meta["total_columns"] == 6
    assert "salary" in meta["column_names"]
    print(f"✅ Metadata extraction passed: {meta['total_rows']} rows, columns: {meta['column_names']}")

    # DuckDB SQL
    sql_res = run_duckdb_query(csv_path, "SELECT department, AVG(salary) as avg_sal FROM dataset GROUP BY department ORDER BY avg_sal DESC")
    assert sql_res["status"] == "success"
    assert len(sql_res["records"]) == 3
    print(f"✅ DuckDB SQL Aggregation passed: {sql_res['records']}")

    # Pandas Metric
    mean_sal = compute_pandas_metric(csv_path, "mean", "salary")
    assert mean_sal["status"] == "success"
    assert round(mean_sal["value"], 2) == 88500.0
    print(f"✅ Pandas Metric passed: Mean salary = {mean_sal['value']}")

    # Chart Generation
    chart_res = generate_custom_chart(csv_path, "bar", "department", "salary", "Salary by Department")
    assert chart_res["status"] == "success"
    assert os.path.exists(chart_res["chart_path"])
    print(f"✅ Plotly Chart generation passed: {chart_res['chart_path']}")

def test_chat_analyst_turn(csv_path):
    print("\n--- 2. Testing Gemini + MCP Conversational Analyst ---")
    client = ChatAnalystMCPClient(csv_path)

    questions = [
        "What is the average salary in this dataset?",
        "Which department has the highest average salary?",
        "Can you visualize the salaries by department?"
    ]

    for q in questions:
        print(f"\nUser: {q}")
        res = run_chat_turn_sync(client, q)
        print(f"AI: {res['response_text']}")
        if res.get("tools_executed"):
            print(f"   Tools called: {[t['tool'] for t in res['tools_executed']]}")
        if res.get("charts"):
            print(f"   Charts produced: {len(res['charts'])}")
        assert len(res["response_text"].strip()) > 0

    print("\n✅ All Conversational AI Analyst tests passed!")

if __name__ == "__main__":
    path = create_sample_dataset()
    test_query_engine(path)
    test_chat_analyst_turn(path)
