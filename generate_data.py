"""
generate_data.py
Generates a realistic synthetic EV dataset (2010–2025) based on
IEA Global EV Outlook 2024 publicly known statistics.
Run once to produce: data/ev_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Country profiles ──────────────────────────────────────────────────────────
# Each entry: (region, 2015 sales seed, growth_factor, market_share_2024,
#              charging_2024, gov_score, gdp_trillion, population_M)
COUNTRIES = {
    "China":        ("Asia",         200_000, 1.68, 38.0, 2_800_000, 8.5, 17.7, 1412),
    "USA":          ("Americas",      50_000, 1.42, 9.5,    180_000, 7.0, 27.4,  335),
    "Germany":      ("Europe",        25_000, 1.38, 22.0,   120_000, 7.8,  4.5,   84),
    "Norway":       ("Europe",        20_000, 1.32, 88.0,    25_000, 9.5,  0.55,   5),
    "UK":           ("Europe",        10_000, 1.35, 18.5,    65_000, 7.2,  3.1,   67),
    "France":       ("Europe",         8_000, 1.33, 17.0,    55_000, 7.5,  3.0,   68),
    "India":        ("Asia",           2_000, 1.55,  2.1,    12_000, 6.0,  3.7, 1428),
    "Japan":        ("Asia",          50_000, 1.20,  4.2,    40_000, 6.5,  4.2,  125),
    "South Korea":  ("Asia",          10_000, 1.38,  9.0,    18_000, 7.0,  1.7,   52),
    "Netherlands":  ("Europe",         8_000, 1.40, 30.0,    22_000, 8.0,  1.1,   17),
}

YEARS = list(range(2010, 2026))

rows = []

for country, (region, seed_2015, gf, ms_2024, ch_2024, gov, gdp, pop) in COUNTRIES.items():
    # ── EV Sales: logistic-ish growth seeded from 2015 value ──────────────────
    sales_by_year = {}
    for y in YEARS:
        t = y - 2015
        base = seed_2015 * (gf ** t)
        # saturation dampening after 2020
        if y > 2020:
            damp = 1 - 0.02 * (y - 2020)
            base *= max(damp, 0.7)
        noise = np.random.uniform(0.93, 1.07)
        sales_by_year[y] = max(int(base * noise), 0)

    # ── Charging stations: grows proportionally ──────────────────────────────
    ch_by_year = {}
    ch_base_2024 = ch_2024
    for y in YEARS:
        t = 2025 - y
        ch_by_year[y] = max(int(ch_base_2024 / (1.25 ** t) * np.random.uniform(0.95, 1.05)), 50)

    # ── Battery cost: drops ~15%/yr from $800 in 2010 to ~$120 in 2024 ───────
    battery_costs = {y: max(int(800 * (0.86 ** (y - 2010)) * np.random.uniform(0.97, 1.03)), 100)
                     for y in YEARS}

    # ── Fuel price (USD/litre) ────────────────────────────────────────────────
    fuel_base = {"China": 1.1, "USA": 0.95, "Germany": 1.9, "Norway": 2.1,
                 "UK": 1.8, "France": 1.75, "India": 1.05, "Japan": 1.4,
                 "South Korea": 1.3, "Netherlands": 2.0}
    fuel_p = fuel_base[country]

    # ── GDP: ~2%/yr real growth ───────────────────────────────────────────────
    gdp_by_year = {y: round(gdp * (0.975 ** (2024 - y)) * np.random.uniform(0.98, 1.02), 2)
                   for y in YEARS}

    # ── Market share: interpolate to 2024 target ─────────────────────────────
    ms_2010 = ms_2024 / (gf ** 14) * 0.5
    for y in YEARS:
        t = (y - 2010) / 14.0
        ms = ms_2010 + (ms_2024 - ms_2010) * (t ** 1.5)
        ms = round(min(ms * np.random.uniform(0.95, 1.05), 95.0), 2)

        # ── CO2 reduction (MT): proportional to EV penetration ───────────────
        co2 = round(sales_by_year[y] * 2.1e-6 * ms, 3)   # rough MT equivalent

        for vtype, frac in [("BEV", 0.65), ("PHEV", 0.35)]:
            rows.append({
                "Year":                     y,
                "Country":                  country,
                "Region":                   region,
                "Vehicle_Type":             vtype,
                "EV_Sales":                 int(sales_by_year[y] * frac),
                "Market_Share_Pct":         ms,
                "Charging_Stations":        int(ch_by_year[y] * (0.7 if vtype == "BEV" else 0.3)),
                "Government_Incentive_Score": gov + np.random.uniform(-0.3, 0.3),
                "Battery_Cost_USD":         battery_costs[y],
                "Fuel_Price_USD":           round(fuel_p * np.random.uniform(0.9, 1.15), 2),
                "Population_M":             pop,
                "GDP_Trillion_USD":         gdp_by_year[y],
                "CO2_Reduction_MT":         co2,
            })

df = pd.DataFrame(rows)
df["Government_Incentive_Score"] = df["Government_Incentive_Score"].round(2)

out_path = os.path.join(os.path.dirname(__file__), "ev_data.csv")
df.to_csv(out_path, index=False)
print(f"[OK] Dataset generated: {out_path}")
print(f"   Rows: {len(df):,}  |  Years: {df['Year'].min()}-{df['Year'].max()}")
print(f"   Countries: {df['Country'].nunique()}  |  Columns: {list(df.columns)}")
