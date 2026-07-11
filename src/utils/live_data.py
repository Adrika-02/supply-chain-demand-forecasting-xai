"""Live external data connector.

Rossmann's historical sales data (2013-2015) has no live feed, and the
dataset carries no per-store geography. But temperature and precipitation
are well-established exogenous drivers of FMCG footfall and demand (heat
drives beverage/ice-cream demand, heavy rain suppresses foot traffic), so a
live weather read for the retailer's core German market is a legitimate,
honestly-sourced "real-time signal" a demand planner would monitor alongside
the model's forecast -- unlike per-store sales, which cannot be live since
the underlying dataset stops in 2015.

Uses the Open-Meteo current-weather API (free, no key required). Every value
returned here is fetched live at call time; nothing is simulated.
"""
import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

GERMAN_CITIES = {
    "Berlin": (52.520, 13.405),
    "Hamburg": (53.550, 9.993),
    "Munich": (48.137, 11.575),
    "Frankfurt": (50.110, 8.682),
    "Cologne": (50.937, 6.960),
}

WEATHER_CODE_LABELS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm",
}


def fetch_live_weather(cities: dict = GERMAN_CITIES, timeout: float = 5.0) -> pd.DataFrame:
    """Fetch current, live weather for each city from Open-Meteo.

    One row per city: temperature, precipitation, wind speed, a human-readable
    condition, and the observation timestamp the API itself reports (proof
    this is a live read, not a cached or synthetic value).
    """
    rows = []
    for city, (lat, lon) in cities.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
            "timezone": "Europe/Berlin",
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        current = resp.json()["current"]
        rows.append({
            "City": city,
            "Temperature (C)": current["temperature_2m"],
            "Precipitation (mm)": current["precipitation"],
            "Wind Speed (km/h)": current["wind_speed_10m"],
            "Condition": WEATHER_CODE_LABELS.get(current["weather_code"], "Unknown"),
            "Observed At (Europe/Berlin)": current["time"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(fetch_live_weather())
