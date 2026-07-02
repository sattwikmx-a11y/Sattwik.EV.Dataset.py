# ⚡ EV Market Forecast Dashboard

An interactive web dashboard for analyzing global Electric Vehicle (EV) adoption trends and predicting future market growth using machine learning.

Built with **Python · Pandas · Plotly · Dash · Scikit-learn**

---

## 🚀 Quick Start

### 1. Install Python dependencies

```bash
cd EV_Dashboard
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python data/generate_data.py
```

This creates `data/ev_data.csv` — a realistic synthetic EV dataset based on IEA Global EV Outlook 2024 statistics, covering 10 countries from 2010–2025.

### 3. Run the dashboard

```bash
python app.py
```

Open your browser at **http://127.0.0.1:8050**

---

## 📁 Project Structure

```
EV_Dashboard/
├── data/
│   ├── generate_data.py     # Synthetic dataset generator
│   └── ev_data.csv          # Generated dataset (auto-created)
├── assets/
│   └── custom.css           # Dark theme, glassmorphism, animations
├── app.py                   # Main Dash application & callbacks
├── preprocessing.py         # Data loading, cleaning, aggregation
├── forecast.py              # ML forecast models (Poly + Random Forest)
├── graphs.py                # All 10+ Plotly figure builders
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 📊 Dashboard Features

| Section | Description |
|---|---|
| **KPI Cards** | Total Sales, CAGR, Top Country, Charging Stations, 2030 Forecast |
| **Sales Trend** | Historical line chart + ML forecast curve with confidence band |
| **Vehicle Type Pie** | BEV vs PHEV market split (donut chart) |
| **Market Share** | Country-ranked horizontal bar chart |
| **YoY Growth** | Year-over-year growth rate bar chart |
| **World Map** | Interactive choropleth — EV Sales / Market Share / Charging |
| **ML Forecast** | Polynomial Regression or Random Forest to 2035 |
| **Battery Cost** | Declining battery cost trend area chart |
| **Policy Analysis** | Government policy scores by country (grouped bar) |
| **GDP Scatter** | GDP vs EV Sales bubble chart |
| **Correlation Heatmap** | Feature correlation matrix |
| **Data Table** | Sortable, filterable data explorer + CSV download |

---

## 🎛️ Sidebar Filters

- **Country** — Multi-select any combination of the 10 countries
- **Region** — Filter by Asia / Europe / Americas
- **Vehicle Type** — BEV, PHEV, or both
- **Year Range** — Slider from 2010–2025
- **Map Metric** — Switch choropleth between Sales / Market Share / Charging
- **Forecast Model** — Toggle between Polynomial Regression and Random Forest

---

## 🤖 Machine Learning Models

### Polynomial Regression (degree 3)
- Fits a cubic curve to global yearly EV sales
- Extrapolates growth pattern to 2035
- Includes ±8–15% confidence band

### Random Forest Regressor
- Features: Year, Charging Stations, Battery Cost, Fuel Price, GDP, Market Share
- 200 estimators, trained on historical data
- Future features projected using domain knowledge (IEA trends)

---

## 🌍 Dataset

The synthetic dataset models real-world EV statistics for 10 countries:

| Country | Region |
|---|---|
| China | Asia |
| USA | Americas |
| Germany | Europe |
| Norway | Europe |
| UK | Europe |
| France | Europe |
| India | Asia |
| Japan | Asia |
| South Korea | Asia |
| Netherlands | Europe |

**Columns:**
`Year, Country, Region, Vehicle_Type, EV_Sales, Market_Share_Pct, Charging_Stations, Government_Incentive_Score, Battery_Cost_USD, Fuel_Price_USD, Population_M, GDP_Trillion_USD, CO2_Reduction_MT`

---

## 📦 Dependencies

```
dash>=2.17.0
dash-bootstrap-components>=1.5.0
plotly>=5.20.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

---

## 📸 Screenshots

Open the dashboard at http://127.0.0.1:8050 after running `python app.py`.

---

*Data based on IEA Global EV Outlook 2024 published statistics. Synthetic dataset generated for educational/analytical purposes.*
