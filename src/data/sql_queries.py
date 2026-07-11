"""
SQL query library used by the EDA notebook and the dashboard's SQL Insights page.

All queries read from the `sales_enriched` view (see load_data.py), which joins
daily store-level demand with store-format metadata. `Store` stands in for a
SKU-category-carrying distribution node (see README dataset section for why).
`StoreType` doubles as a distribution-segment proxy where a query calls for
"region", since Rossmann does not publish store geography.
"""

TOTAL_DEMAND_BY_STORE_TYPE_ASSORTMENT = """
SELECT
    StoreType,
    Assortment,
    SUM(Sales)                                      AS total_demand,
    COUNT(DISTINCT Store)                           AS node_count,
    ROUND(SUM(Sales) * 1.0 / COUNT(DISTINCT Store), 2) AS avg_demand_per_node
FROM sales_enriched
WHERE Open = 1
GROUP BY StoreType, Assortment
ORDER BY total_demand DESC;
"""

TOP_20_DEMAND_NODES = """
SELECT
    Store,
    StoreType,
    Assortment,
    SUM(Sales)          AS total_demand,
    ROUND(AVG(Sales), 2) AS avg_daily_demand
FROM sales_enriched
WHERE Open = 1
GROUP BY Store, StoreType, Assortment
ORDER BY total_demand DESC
LIMIT 20;
"""

DEMAND_PROMO_VS_NON_PROMO = """
SELECT
    CASE WHEN Promo = 1 THEN 'Promotional' ELSE 'Non-Promotional' END AS period_type,
    COUNT(*)              AS days_observed,
    SUM(Sales)            AS total_demand,
    ROUND(AVG(Sales), 2)  AS avg_daily_demand,
    ROUND(AVG(Customers), 2) AS avg_daily_customers
FROM sales_enriched
WHERE Open = 1
GROUP BY period_type;
"""

HOLIDAY_IMPACT_BY_SEGMENT = """
SELECT
    StoreType AS distribution_segment,
    CASE WHEN StateHoliday != '0' THEN 'Holiday' ELSE 'Regular Day' END AS day_type,
    COUNT(*)             AS days_observed,
    ROUND(AVG(Sales), 2) AS avg_daily_demand
FROM sales_enriched
WHERE Open = 1
GROUP BY StoreType, day_type
ORDER BY StoreType, day_type;
"""

WEEKLY_DEMAND_TRENDS = """
SELECT
    strftime('%Y-%W', Date) AS year_week,
    SUM(Sales)               AS total_demand,
    ROUND(AVG(Sales), 2)     AS avg_daily_demand
FROM sales_enriched
WHERE Open = 1
GROUP BY year_week
ORDER BY year_week;
"""

MONTHLY_DEMAND_TRENDS = """
SELECT
    strftime('%Y-%m', Date) AS year_month,
    SUM(Sales)                AS total_demand,
    ROUND(AVG(Sales), 2)      AS avg_daily_demand
FROM sales_enriched
WHERE Open = 1
GROUP BY year_month
ORDER BY year_month;
"""

QUERY_LIBRARY = {
    "Total demand by store type & assortment": TOTAL_DEMAND_BY_STORE_TYPE_ASSORTMENT,
    "Top 20 highest-demand nodes": TOP_20_DEMAND_NODES,
    "Demand: promotional vs non-promotional": DEMAND_PROMO_VS_NON_PROMO,
    "Holiday impact by distribution segment": HOLIDAY_IMPACT_BY_SEGMENT,
    "Weekly demand trend": WEEKLY_DEMAND_TRENDS,
    "Monthly demand trend": MONTHLY_DEMAND_TRENDS,
}


if __name__ == "__main__":
    from src.utils.db_utils import run_query

    for name, query in QUERY_LIBRARY.items():
        print(f"\n=== {name} ===")
        print(run_query(query).head(10).to_string(index=False))
