import sys
sys.stdout.reconfigure(encoding="utf-8")

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from data_quality import (
    detect_missing_values,
    detect_duplicates,
    validate_dataset,
    calculate_quality_score,
    clean_dataset
)
from chat_analyst import ChatAnalystMCPClient, run_chat_turn_sync

def test_data_quality_suite():
    print("--- 1. Testing Synthetic Dirty Dataset ---")
    dirty_data = {
        "id": [1, 2, 2, 3, 4, 5, 6],
        "name": ["Alice", "Bob", "Bob", "Charlie", "   ", None, "David"],
        "email": ["alice@work.com", "bob@work.com", "bob@work.com", "not-an-email", "charlie@corp.org", "eva@co.uk", "bad@@domain"],
        "age": [25, 30, 30, 150, 22, np.nan, 29],
        "join_date": ["2022-01-15", "2021-06-20", "2021-06-20", "invalid-date", "2023-03-10", "2020-11-05", "2022-08-30"],
        "status": ["Active", "active", "active", "Pending", "ACTIVE", "Active", "Pending"]
    }
    df = pd.DataFrame(dirty_data)
    csv_path = os.path.join("data", "test_quality_dataset.csv")
    df.to_csv(csv_path, index=False)

    # 1. Missing Values
    print("\n--- Missing Value Detection ---")
    missing_res = detect_missing_values(df)
    print(f"Total Missing Cells: {missing_res['total_missing_cells']}")
    print(f"Affected Rows: {missing_res['affected_row_count']}")
    print(f"Worst Column: {missing_res['worst_column']}")
    assert missing_res["total_missing_cells"] > 0
    print("✅ Missing value detection passed!")

    # 2. Duplicate Detection
    print("\n--- Duplicate Detection ---")
    dupe_res = detect_duplicates(df)
    print(f"Total Duplicates (all cols): {dupe_res['total_duplicate_records']}")
    assert dupe_res["total_duplicate_records"] == 1
    
    dupe_id_res = detect_duplicates(df, subset=["id"])
    print(f"Total Duplicate IDs: {dupe_id_res['total_duplicate_records']}")
    print(f"Frequency table: {dupe_id_res['frequency_table']}")
    assert dupe_id_res["total_duplicate_records"] >= 1
    print("✅ Duplicate detection passed!")

    # 3. Validation
    print("\n--- Data Validation ---")
    val_rules = {
        "age": {"min_val": 18, "max_val": 100}
    }
    val_res = validate_dataset(df, rules=val_rules)
    print(f"Total Anomalies: {val_res['total_anomalies']}")
    for a in val_res["anomalies_preview"]:
        print(f"  * {a['column']} ({a['issue_type']}): {a['explanation']}")
    assert val_res["total_anomalies"] >= 3
    print("✅ Validation anomalies detection passed!")

    # 4. Quality Score
    print("\n--- Quality Score ---")
    score_res = calculate_quality_score(missing_res, dupe_res, val_res)
    print(f"Quality Score: {score_res['overall_score']} / 100 (Grade: {score_res['grade']})")
    print(f"Category breakdown: {score_res['category_scores']}")
    print(f"Formula: {score_res['scoring_formula']}")
    print(f"Suggestions: {score_res['suggestions']}")
    assert 0 <= score_res["overall_score"] <= 100
    print("✅ Quality score calculation passed!")

    # 5. Data Cleaning
    print("\n--- Data Cleaning ---")
    cleaning_ops = {
        "remove_duplicates": True,
        "impute_numerical": {"age": "median"},
        "impute_categorical": {"name": "mode"}
    }
    cleaned_df, summary = clean_dataset(df, cleaning_ops)
    print(f"Initial rows: {summary['initial_rows']}, Final rows: {summary['final_rows']}")
    print(f"Operations log: {summary['changes_log']}")
    assert len(cleaned_df) == len(df) - 1 # 1 duplicate removed
    assert cleaned_df["age"].isna().sum() == 0 # Imputed
    print("✅ Data cleaning passed!")

    # 6. Conversational Gemini Integration
    print("\n--- 6. Testing Conversational Gemini Quality Questions ---")
    client = ChatAnalystMCPClient(csv_path)
    questions = [
        "What is the data quality score of this dataset?",
        "How many missing values are present in this dataset?"
    ]
    for q in questions:
        print(f"\nUser: {q}")
        res = run_chat_turn_sync(client, q)
        print(f"AI: {res['response_text'][:250]}...")
        print(f"Tools called: {[t['tool'] for t in res.get('tools_executed', [])]}")
        assert len(res["response_text"].strip()) > 0

    print("\n🎉 ALL PHASE 1 DATA QUALITY TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_data_quality_suite()
