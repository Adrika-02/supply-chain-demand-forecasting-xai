"""
Loads the Rossmann Store Sales CSVs into SQLite.

Framed as an FMCG manufacturer's distribution network: each `Store` is a
retail/distribution node, `StoreType` a store-format / product-category
segment, and `Assortment` the breadth of SKU assortment carried at that node.
The dataset is store-level, not true SKU-level — `Store` doubles as the
demand-category proxy throughout this project (documented in the README).
"""
from pathlib import Path

import pandas as pd

from src.utils.db_utils import DB_PATH, get_connection

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_csvs_to_sqlite(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR) -> None:
    train = pd.read_csv(
        raw_dir / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
    )
    store = pd.read_csv(raw_dir / "store.csv")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        train.to_sql("sales", conn, if_exists="replace", index=False)
        store.to_sql("stores", conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_store ON sales(Store)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(Date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stores_store ON stores(Store)")

        conn.execute("DROP VIEW IF EXISTS sales_enriched")
        conn.execute(
            """
            CREATE VIEW sales_enriched AS
            SELECT
                s."Store"           AS Store,
                s."Date"            AS Date,
                s."DayOfWeek"       AS DayOfWeek,
                s."Sales"           AS Sales,
                s."Customers"       AS Customers,
                s."Open"            AS Open,
                s."Promo"           AS Promo,
                s."StateHoliday"    AS StateHoliday,
                s."SchoolHoliday"   AS SchoolHoliday,
                st."StoreType"      AS StoreType,
                st."Assortment"     AS Assortment,
                st."CompetitionDistance" AS CompetitionDistance,
                st."CompetitionOpenSinceMonth" AS CompetitionOpenSinceMonth,
                st."CompetitionOpenSinceYear"  AS CompetitionOpenSinceYear,
                st."Promo2"         AS Promo2,
                st."Promo2SinceWeek" AS Promo2SinceWeek,
                st."Promo2SinceYear" AS Promo2SinceYear,
                st."PromoInterval"   AS PromoInterval
            FROM sales s
            LEFT JOIN stores st ON s."Store" = st."Store"
            """
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    load_csvs_to_sqlite()
    print(f"Loaded tables 'sales', 'stores', and view 'sales_enriched' into {DB_PATH}")
