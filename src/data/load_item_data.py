"""
Loads the Store Item Demand Forecasting Challenge dataset into SQLite.

A second, more granular data source than the Rossmann network used
elsewhere in this project: 10 stores x 50 items, daily unit sales for
2013-2017. Unlike Rossmann it carries no promo/competition/holiday fields --
it exists to demonstrate genuine SKU x location forecasting on the
Product-Level Forecasting dashboard page, not to replace the richer
Rossmann-based explainability story on pages 1-5.
"""
from pathlib import Path

import pandas as pd

from src.utils.db_utils import DB_PATH, get_connection

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "item_level"


def load_item_csv_to_sqlite(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR) -> None:
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["date"])

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        train.to_sql("item_sales", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_item_sales_store_item ON item_sales(store, item)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_item_sales_date ON item_sales(date)")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    load_item_csv_to_sqlite()
    print(f"Loaded table 'item_sales' into {DB_PATH}")
