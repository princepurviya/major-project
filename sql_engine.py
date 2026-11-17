import re
import threading
import duckdb
import pandas as pd
from typing import Optional, Dict, Any, Tuple

# Global in-memory connection, concurrency lock, and active dataset state tracking
_DB_LOCK = threading.RLock()
_DB_CONN: Optional[duckdb.DuckDBPyConnection] = None
_ACTIVE_DATASET_ID: Optional[str] = None
TABLE_NAME = "dataset"

# Regex patterns for destructive SQL statements
DESTRUCTIVE_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|REPLACE|ATTACH|DETACH|PRAGMA|CALL|EXPORT|COPY|EXECUTE)\b",
    re.IGNORECASE
)

def initialize_database() -> duckdb.DuckDBPyConnection:
    """Initialize or return the existing in-memory DuckDB connection."""
    global _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is None:
            _DB_CONN = duckdb.connect(database=":memory:", read_only=False)
        return _DB_CONN

def close_connection():
    """Safely close and clean up the database connection."""
    global _DB_CONN, _ACTIVE_DATASET_ID
    with _DB_LOCK:
        if _DB_CONN is not None:
            try:
                _DB_CONN.complete()
            except Exception:
                pass
            try:
                _DB_CONN.close()
            except Exception:
                pass
            _DB_CONN = None
            _ACTIVE_DATASET_ID = None

def load_dataframe(df: pd.DataFrame, dataset_id: str = "active") -> str:
    """
    Register or replace the active Pandas DataFrame in DuckDB under table name 'dataset'.
    Does not copy or duplicate memory; DuckDB views the DataFrame in-process.
    Provides complete isolation when switching datasets and eliminates ResourceInUseException.
    """
    global _ACTIVE_DATASET_ID
    with _DB_LOCK:
        # If active dataset changes, reset connection for complete isolation
        if _ACTIVE_DATASET_ID is not None and _ACTIVE_DATASET_ID != dataset_id:
            close_connection()

        con = initialize_database()

        # Guarantee any unfinished query on this connection is cleared
        try:
            con.complete()
        except Exception:
            pass

        # Safely unregister any existing table mapping
        try:
            con.unregister(TABLE_NAME)
        except Exception:
            pass

        # Register DataFrame into DuckDB table registry
        con.register(TABLE_NAME, df)
        _ACTIVE_DATASET_ID = dataset_id
        return TABLE_NAME

def get_table_schema(table_name: str = TABLE_NAME) -> Dict[str, Any]:
    """Retrieve schema information (columns, types, nullability) for the active table."""
    with _DB_LOCK:
        con = initialize_database()
        try:
            schema_df = con.execute(f"DESCRIBE {table_name}").fetchdf()
            return {
                "status": "success",
                "table": table_name,
                "columns": schema_df.to_dict(orient="records")
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            try:
                con.complete()
            except Exception:
                pass

def get_row_count(table_name: str = TABLE_NAME) -> int:
    """Get the row count of the active table."""
    with _DB_LOCK:
        con = initialize_database()
        try:
            row = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0
        finally:
            try:
                con.complete()
            except Exception:
                pass

def get_table_info(table_name: str = TABLE_NAME) -> Dict[str, Any]:
    """Get high-level table statistics: row count, column count, and column names."""
    with _DB_LOCK:
        con = initialize_database()
        try:
            cols_df = con.execute(f"DESCRIBE {table_name}").fetchdf()
            row_cnt = get_row_count(table_name)
            return {
                "status": "success",
                "table_name": table_name,
                "row_count": row_cnt,
                "column_count": len(cols_df),
                "columns": cols_df["column_name"].tolist(),
                "types": dict(zip(cols_df["column_name"], cols_df["column_type"]))
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            try:
                con.complete()
            except Exception:
                pass

def is_safe_query(sql_query: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that the SQL query is read-only and analytical.
    Allows SELECT, WITH (CTEs). Blocks any destructive statements.
    """
    cleaned = sql_query.strip()
    
    # Strip comments
    cleaned = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL).strip()
    
    if not cleaned:
        return False, "Query is empty."

    # Must start with SELECT or WITH
    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return False, "Security Error: Only read-only queries starting with 'SELECT' or 'WITH' are permitted."

    # Check for destructive commands anywhere in the query
    match = DESTRUCTIVE_PATTERN.search(cleaned)
    if match:
        return False, f"Security Violation: Destructive statement '{match.group(1).upper()}' is forbidden."

    # Disallow multiple semicolons which could chain statements
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple SQL statements in a single query are not allowed."

    return True, None

def execute_readonly_sql(sql_query: str, max_rows: int = 100) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Execute read-only SQL query against DuckDB table 'dataset'.
    Returns (result_df, metadata_dict).
    Handles SQL syntax errors, invalid columns, and safety constraints.
    """
    con = initialize_database()

    # Safety check
    is_safe, error_msg = is_safe_query(sql_query)
    if not is_safe:
        return None, {
            "status": "error",
            "error_type": "security_violation",
            "error": error_msg,
            "sql": sql_query
        }

    with _DB_LOCK:
        con = initialize_database()
        try:
            # Execute query on the connection where dataset is registered
            result_df = con.execute(sql_query).fetchdf()

            total_rows = len(result_df)
            is_truncated = False
            display_df = result_df

            if total_rows > max_rows:
                display_df = result_df.head(max_rows)
                is_truncated = True

            meta = {
                "status": "success",
                "sql": sql_query,
                "total_rows_returned": total_rows,
                "rows_displayed": len(display_df),
                "is_truncated": is_truncated,
                "columns": list(result_df.columns),
                "records": display_df.fillna("").to_dict(orient="records")
            }
            return display_df, meta

        except Exception as e:
            err_str = str(e)
            return None, {
                "status": "error",
                "error_type": "sql_execution_error",
                "error": err_str,
                "sql": sql_query
            }
        finally:
            try:
                con.complete()
            except Exception:
                pass
