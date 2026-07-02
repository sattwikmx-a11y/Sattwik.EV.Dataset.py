"""
forecast.py
Machine-learning forecast models for EV sales prediction (2026–2035).
Uses scikit-learn: Polynomial Regression + Random Forest Regressor.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FORECAST_YEARS = list(range(2026, 2036))


# ── Polynomial Regression Forecast ───────────────────────────────────────────

def polynomial_forecast(
    trend_df: pd.DataFrame,
    future_years: list = FORECAST_YEARS,
    degree: int = 3,
) -> pd.DataFrame:
    """
    Fit a degree-3 polynomial regression on global yearly EV sales
    and predict for future_years.

    Parameters
    ----------
    trend_df : pd.DataFrame   Output of preprocessing.get_global_trend()
    future_years : list       Years to predict
    degree : int              Polynomial degree

    Returns
    -------
    pd.DataFrame with columns: Year, Predicted_Sales, Lower_CI, Upper_CI, Model
    """
    hist = trend_df[["Year", "EV_Sales"]].dropna().sort_values("Year")
    if len(hist) < 3:
        all_years = sorted(set(hist["Year"].tolist() + future_years)) if not hist.empty else list(range(2010, 2036))
        df_out = pd.DataFrame({
            "Year": all_years,
            "Predicted_Sales": 0,
            "Lower_CI": 0,
            "Upper_CI": 0,
            "Model": "Polynomial (deg-3) [No Data]",
            "Is_Forecast": [yr in future_years for yr in all_years],
        })
        return df_out

    X = hist["Year"].values.reshape(-1, 1)
    y = hist["EV_Sales"].values.astype(float)

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)

    all_years = sorted(set(hist["Year"].tolist() + future_years))
    X_pred = np.array(all_years).reshape(-1, 1)
    y_pred = model.predict(X_pred)

    # Clip to non-negative
    y_pred = np.clip(y_pred, 0, None)

    # Confidence band: ±8% for near-term, ±15% for far future
    bands = []
    for yr, pred in zip(all_years, y_pred):
        spread = 0.08 if yr <= 2028 else 0.15
        bands.append((pred * (1 - spread), pred * (1 + spread)))

    df_out = pd.DataFrame({
        "Year": all_years,
        "Predicted_Sales": y_pred.astype(int),
        "Lower_CI": [int(b[0]) for b in bands],
        "Upper_CI": [int(b[1]) for b in bands],
        "Model": "Polynomial (deg-3)",
        "Is_Forecast": [yr in future_years for yr in all_years],
    })
    return df_out


# ── Random Forest Multi-Feature Forecast ─────────────────────────────────────

def rf_forecast(
    trend_df: pd.DataFrame,
    future_years: list = FORECAST_YEARS,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """
    Train a Random Forest on historical features and predict future sales.

    Features used: Year, Charging_Stations, Battery_Cost_USD, Fuel_Price_USD,
                   GDP_Trillion_USD (global sum), Market_Share_Pct (mean)

    Returns
    -------
    pd.DataFrame with columns: Year, Predicted_Sales, Lower_CI, Upper_CI, Model
    """
    feature_cols = [
        "Year", "Charging_Stations", "Battery_Cost_USD",
        "Fuel_Price_USD", "GDP_Trillion_USD", "Market_Share_Pct",
    ]
    hist = trend_df[feature_cols + ["EV_Sales"]].dropna().sort_values("Year")
    if len(hist) < 3:
        all_years = sorted(hist["Year"].tolist() + future_years) if not hist.empty else list(range(2010, 2036))
        df_out = pd.DataFrame({
            "Year": all_years,
            "Predicted_Sales": 0,
            "Lower_CI": 0,
            "Upper_CI": 0,
            "Model": "Random Forest [No Data]",
            "Is_Forecast": [yr in future_years for yr in all_years],
        })
        return df_out

    X = hist[feature_cols].values
    y = hist["EV_Sales"].values.astype(float)

    rf = make_pipeline(StandardScaler(), RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, n_jobs=-1
    ))
    rf.fit(X, y)

    # ── Project features into the future ─────────────────────────────────────
    last = hist.iloc[-1]
    future_rows = []
    for yr in future_years:
        t = yr - last["Year"]
        future_rows.append({
            "Year": yr,
            # Charging stations grow ~20%/yr (decelerating)
            "Charging_Stations": int(last["Charging_Stations"] * (1.18 ** t)),
            # Battery cost drops ~10%/yr
            "Battery_Cost_USD": max(int(last["Battery_Cost_USD"] * (0.90 ** t)), 60),
            # Fuel price rises ~3%/yr
            "Fuel_Price_USD": round(last["Fuel_Price_USD"] * (1.03 ** t), 2),
            # Global GDP grows ~2.5%/yr
            "GDP_Trillion_USD": round(last["GDP_Trillion_USD"] * (1.025 ** t), 2),
            # Market share increases ~3 pp/yr up to 60%
            "Market_Share_Pct": min(last["Market_Share_Pct"] + 3.0 * t, 60.0),
        })

    all_years = sorted(hist["Year"].tolist() + future_years)
    all_X_rows = hist[feature_cols].values.tolist() + \
                 [[r[c] for c in feature_cols] for r in future_rows]
    X_all = np.array(all_X_rows)
    y_all = rf.predict(X_all)
    y_all = np.clip(y_all, 0, None)

    bands = []
    for yr, pred in zip(all_years, y_all):
        spread = 0.10 if yr <= 2028 else 0.18
        bands.append((pred * (1 - spread), pred * (1 + spread)))

    df_out = pd.DataFrame({
        "Year": all_years,
        "Predicted_Sales": y_all.astype(int),
        "Lower_CI": [int(b[0]) for b in bands],
        "Upper_CI": [int(b[1]) for b in bands],
        "Model": "Random Forest",
        "Is_Forecast": [yr in future_years for yr in all_years],
    })
    return df_out


# ── KPI Forecast: single 2030 number ─────────────────────────────────────────

def forecast_2030(trend_df: pd.DataFrame) -> int:
    """Return the polynomial regression forecast for year 2030."""
    df = polynomial_forecast(trend_df, future_years=[2030])
    row = df[df["Year"] == 2030]
    return int(row["Predicted_Sales"].values[0]) if len(row) else 0


def compute_cagr(trend_df: pd.DataFrame, start_year: int = 2015, end_year: int = 2024) -> float:
    """Compute Compound Annual Growth Rate between two years."""
    try:
        v0 = trend_df.loc[trend_df["Year"] == start_year, "EV_Sales"].values[0]
        v1 = trend_df.loc[trend_df["Year"] == end_year, "EV_Sales"].values[0]
        n = end_year - start_year
        return round(((v1 / v0) ** (1 / n) - 1) * 100, 1)
    except Exception:
        return 0.0
