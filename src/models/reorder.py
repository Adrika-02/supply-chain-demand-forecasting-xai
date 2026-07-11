"""
Reorder-point / safety-stock calculator.

    Reorder Point   = (Average Daily Demand x Lead Time) + Safety Stock
    Safety Stock    = Z(service_level) x Demand Std Dev x sqrt(Lead Time)

Demand and its variability come from each store's own recent rolling
statistics (RollingMean7 / RollingStd7, already in the feature table) --
not from an assumed physical inventory count, since Rossmann has no real
inventory/stock-on-hand data. Because `Sales` is revenue-equivalent, not a
literal unit count, the reorder point is expressed in revenue-equivalent
terms by default; an optional `unit_price` converts it to a physical unit
quantity if the caller supplies one (there is no real per-SKU price in this
dataset, so that conversion is always a user-supplied assumption, never a
derived figure).

This computes a *target stocking level* to cover lead-time demand at a
chosen service level -- not "how many more units to order right now", which
would additionally require real current on-hand inventory (not available
here).
"""
from scipy.stats import norm


def z_score_for_service_level(service_level: float) -> float:
    return float(norm.ppf(service_level))


def compute_reorder_point(
    avg_daily_demand: float,
    demand_std: float,
    lead_time_days: float,
    service_level: float = 0.95,
    unit_price: float | None = None,
) -> dict:
    z = z_score_for_service_level(service_level)
    safety_stock = z * demand_std * (lead_time_days ** 0.5)
    lead_time_demand = avg_daily_demand * lead_time_days
    reorder_point = lead_time_demand + safety_stock

    result = {
        "z_score": z,
        "lead_time_demand": lead_time_demand,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
    }
    if unit_price:
        result["safety_stock_units"] = safety_stock / unit_price
        result["reorder_point_units"] = reorder_point / unit_price
    return result


if __name__ == "__main__":
    example = compute_reorder_point(
        avg_daily_demand=20718.52,  # Store 262's historical average daily demand
        demand_std=4906.33,          # Store 262's RollingStd7
        lead_time_days=7,
        service_level=0.95,
    )
    print(example)
