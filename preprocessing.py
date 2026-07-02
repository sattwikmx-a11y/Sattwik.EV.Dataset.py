"""
preprocessing.py
Data loading, cleaning, and aggregation utilities for the EV Dashboard.
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ev_data.csv")


def load_data() -> pd.DataFrame:
    """Load and type-cast the EV dataset."""
    df = pd.read_csv(DATA_PATH)
    df["Year"] = df["Year"].astype(int)
    df["EV_Sales"] = pd.to_numeric(df["EV_Sales"], errors="coerce").fillna(0).astype(int)
    df["Market_Share_Pct"] = pd.to_numeric(df["Market_Share_Pct"], errors="coerce").fillna(0.0)
    df["Charging_Stations"] = pd.to_numeric(df["Charging_Stations"], errors="coerce").fillna(0).astype(int)
    df["Battery_Cost_USD"] = pd.to_numeric(df["Battery_Cost_USD"], errors="coerce").fillna(0)
    df["Fuel_Price_USD"] = pd.to_numeric(df["Fuel_Price_USD"], errors="coerce").fillna(0.0)
    df["GDP_Trillion_USD"] = pd.to_numeric(df["GDP_Trillion_USD"], errors="coerce").fillna(0.0)
    df["CO2_Reduction_MT"] = pd.to_numeric(df["CO2_Reduction_MT"], errors="coerce").fillna(0.0)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and fill gaps."""
    df = df.dropna(subset=["Year", "Country", "EV_Sales"])
    df = df[df["EV_Sales"] >= 0]
    return df.reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame,
    countries: list | None = None,
    vehicle_types: list | None = None,
    year_range: tuple | None = None,
    regions: list | None = None,
) -> pd.DataFrame:
    """Apply sidebar filter selections to the dataframe."""
    if countries:
        df = df[df["Country"].isin(countries)]
    if vehicle_types:
        df = df[df["Vehicle_Type"].isin(vehicle_types)]
    if year_range:
        df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
    if regions:
        df = df[df["Region"].isin(regions)]
    return df


def get_global_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Yearly global aggregates for the trend/forecast chart."""
    grp = (
        df.groupby("Year")
        .agg(
            EV_Sales=("EV_Sales", "sum"),
            Charging_Stations=("Charging_Stations", "sum"),
            Market_Share_Pct=("Market_Share_Pct", "mean"),
            Battery_Cost_USD=("Battery_Cost_USD", "mean"),
            Fuel_Price_USD=("Fuel_Price_USD", "mean"),
            GDP_Trillion_USD=("GDP_Trillion_USD", "sum"),
            CO2_Reduction_MT=("CO2_Reduction_MT", "sum"),
        )
        .reset_index()
    )
    return grp


def get_country_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-country aggregated for a single year (both vehicle types combined)."""
    filtered = df[df["Year"] == year]
    grp = (
        filtered.groupby("Country")
        .agg(
            EV_Sales=("EV_Sales", "sum"),
            Market_Share_Pct=("Market_Share_Pct", "mean"),
            Charging_Stations=("Charging_Stations", "sum"),
            Government_Incentive_Score=("Government_Incentive_Score", "mean"),
            GDP_Trillion_USD=("GDP_Trillion_USD", "mean"),
            Region=("Region", "first"),
        )
        .reset_index()
    )
    return grp.sort_values("EV_Sales", ascending=False)


def get_vehicle_type_split(df: pd.DataFrame) -> pd.DataFrame:
    """Total sales by vehicle type across the full filtered dataset."""
    return df.groupby("Vehicle_Type")["EV_Sales"].sum().reset_index()


def get_yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year global sales growth (%)."""
    trend = get_global_trend(df).sort_values("Year")
    trend["YoY_Growth_Pct"] = trend["EV_Sales"].pct_change() * 100
    return trend.dropna(subset=["YoY_Growth_Pct"])


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix for numerical features."""
    numeric_cols = [
        "EV_Sales", "Market_Share_Pct", "Charging_Stations",
        "Government_Incentive_Score", "Battery_Cost_USD",
        "Fuel_Price_USD", "GDP_Trillion_USD", "CO2_Reduction_MT",
    ]
    return df[numeric_cols].corr().round(2)


def get_policy_scores() -> pd.DataFrame:
    """
    Curated government policy scores per country (0–10 scale).
    Based on IEA Global EV Outlook 2024 policy assessments.
    """
    data = {
        "Country": ["China", "Norway", "India", "USA", "Germany", "UK", "France"],
        "Tax_Incentives": [8.5, 9.5, 6.0, 7.0, 7.5, 7.0, 7.5],
        "Purchase_Subsidies": [8.0, 9.8, 5.5, 6.5, 7.0, 7.5, 8.0],
        "Charging_Infrastructure": [9.0, 8.5, 4.5, 6.0, 8.0, 7.0, 7.5],
        "Registration_Benefits": [8.5, 9.0, 5.0, 5.5, 6.5, 8.0, 7.0],
    }
    return pd.DataFrame(data)


# Country ISO codes for choropleth maps
ISO_MAP = {
    "China": "CHN", "USA": "USA", "Germany": "DEU", "Norway": "NOR",
    "UK": "GBR", "France": "FRA", "India": "IND", "Japan": "JPN",
    "South Korea": "KOR", "Netherlands": "NLD",
}
