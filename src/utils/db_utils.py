"""SQLite connection helper shared by the ETL script, notebooks, and the dashboard."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "db" / "supply_chain.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def run_query(query: str, db_path: Path = DB_PATH, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
