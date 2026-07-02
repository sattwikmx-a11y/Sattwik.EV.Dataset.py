"""
app.py
EV Market Forecast Dashboard — Main Dash Application
Run: python app.py   →   http://127.0.0.1:8050
"""

import os
import sys
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc

# ── Ensure local imports resolve correctly ────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import preprocessing as pp
import forecast as fc
import graphs as gr

# ── Bootstrap dark theme ──────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="EV Market Forecast Dashboard",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Interactive EV Market Forecast Dashboard — global EV adoption analysis and prediction using ML"},
    ],
)
server = app.server  # For production deployment

# ══════════════════════════════════════════════════════════════════════════════
# Data bootstrap (loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════
_RAW_DF  = pp.load_data()
_RAW_DF  = pp.clean_data(_RAW_DF)
_TREND   = pp.get_global_trend(_RAW_DF)
_POLY_FC = fc.polynomial_forecast(_TREND)
_RF_FC   = fc.rf_forecast(_TREND)
_POLICY  = pp.get_policy_scores()

ALL_COUNTRIES = sorted(_RAW_DF["Country"].unique().tolist())
ALL_REGIONS   = sorted(_RAW_DF["Region"].unique().tolist())
YEAR_MIN      = int(_RAW_DF["Year"].min())
YEAR_MAX      = int(_RAW_DF["Year"].max())

# ══════════════════════════════════════════════════════════════════════════════
# Helper — KPI card builder
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(icon, label, value, sub, color_class):
    return html.Div([
        html.Span(icon, className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(sub,   className="kpi-sub"),
    ], className=f"kpi-card {color_class}")


def section_header(title: str) -> html.Div:
    return html.Div([
        html.Span(title, className="section-title"),
        html.Div(className="section-divider"),
    ], className="section-header")

# ══════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_sidebar():
    return html.Div([
        html.Div("Filters", className="sidebar-section-title", style={"marginBottom": "18px"}),

        # Country
        html.Div([
            html.Label("🌍  Country", className="filter-label"),
            dcc.Dropdown(
                id="filter-country",
                options=[{"label": c, "value": c} for c in ALL_COUNTRIES],
                value=ALL_COUNTRIES,
                multi=True,
                placeholder="All Countries",
                style={"fontSize": "0.82rem"},
            ),
        ], className="filter-group"),

        # Region
        html.Div([
            html.Label("🗺️  Region", className="filter-label"),
            dcc.Dropdown(
                id="filter-region",
                options=[{"label": r, "value": r} for r in ALL_REGIONS],
                value=None,
                multi=True,
                placeholder="All Regions",
                style={"fontSize": "0.82rem"},
            ),
        ], className="filter-group"),

        # Vehicle Type
        html.Div([
            html.Label("🔋  Vehicle Type", className="filter-label"),
            dcc.Checklist(
                id="filter-vehicle-type",
                options=[
                    {"label": "  Battery EV (BEV)", "value": "BEV"},
                    {"label": "  Plug-in Hybrid (PHEV)", "value": "PHEV"},
                ],
                value=["BEV", "PHEV"],
                inputStyle={"marginRight": "6px", "accentColor": "#39d0d8"},
                labelStyle={"display": "flex", "alignItems": "center",
                            "marginBottom": "8px", "color": "#c9d1d9",
                            "fontSize": "0.82rem", "cursor": "pointer"},
            ),
        ], className="filter-group"),

        # Year Range
        html.Div([
            html.Label("📅  Year Range", className="filter-label"),
            dcc.RangeSlider(
                id="filter-year",
                min=YEAR_MIN, max=YEAR_MAX,
                step=1,
                value=[2010, YEAR_MAX],
                marks={y: {"label": str(y), "style": {"color": "#8b949e", "fontSize": "0.7rem"}}
                       for y in range(YEAR_MIN, YEAR_MAX + 1, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], className="filter-group", style={"paddingBottom": "20px"}),

        # Map metric
        html.Div([
            html.Label("🗺️  Map Metric", className="filter-label"),
            dcc.RadioItems(
                id="filter-map-metric",
                options=[
                    {"label": "  EV Sales", "value": "EV_Sales"},
                    {"label": "  Market Share %", "value": "Market_Share_Pct"},
                    {"label": "  Charging Stations", "value": "Charging_Stations"},
                ],
                value="EV_Sales",
                inputStyle={"marginRight": "6px", "accentColor": "#39d0d8"},
                labelStyle={"display": "flex", "alignItems": "center",
                            "marginBottom": "8px", "color": "#c9d1d9",
                            "fontSize": "0.82rem", "cursor": "pointer"},
            ),
        ], className="filter-group"),

        # Forecast model
        html.Div([
            html.Label("🤖  Forecast Model", className="filter-label"),
            dcc.RadioItems(
                id="filter-forecast-model",
                options=[
                    {"label": "  Polynomial Reg.", "value": "poly"},
                    {"label": "  Random Forest", "value": "rf"},
                ],
                value="poly",
                inputStyle={"marginRight": "6px", "accentColor": "#3fb950"},
                labelStyle={"display": "flex", "alignItems": "center",
                            "marginBottom": "8px", "color": "#c9d1d9",
                            "fontSize": "0.82rem", "cursor": "pointer"},
            ),
        ], className="filter-group"),

        # Reset
        html.Button(
            "↺ Reset Filters",
            id="btn-reset",
            n_clicks=0,
            style={
                "width": "100%", "padding": "9px",
                "background": "rgba(139,148,158,0.10)",
                "border": "1px solid #30363d",
                "borderRadius": "8px",
                "color": "#8b949e",
                "fontSize": "0.8rem",
                "cursor": "pointer",
                "fontFamily": "Inter, sans-serif",
                "transition": "all 0.2s ease",
                "marginTop": "6px",
            },
        ),

    ], className="sidebar")


def build_header():
    return html.Div([
        html.Div([
            html.Div("⚡ EV Market Forecast Dashboard", className="header-title"),
            html.Div("Global Electric Vehicle Adoption Analysis · 2010–2035",
                     className="header-subtitle"),
        ]),
        html.Div([
            html.Div([
                html.Div(className="live-dot"),
                "LIVE ANALYSIS",
            ], className="live-badge"),
        ], className="header-badge"),
    ], className="header-bar")


# ══════════════════════════════════════════════════════════════════════════════
# App Layout
# ══════════════════════════════════════════════════════════════════════════════
app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    build_header(),

    # ── Body (sidebar + main) ────────────────────────────────────────────────
    html.Div([

        # Sidebar (left)
        html.Div(build_sidebar(), style={"width": "260px", "flexShrink": "0"}),

        # Main Content (right)
        html.Div([

            # ── KPI Row ─────────────────────────────────────────────────────
            section_header("📊 Key Performance Indicators"),
            html.Div(id="kpi-row", className="kpi-grid"),

            # ── Section: Trend + Market Share ───────────────────────────────
            section_header("📈 Sales Trends & Market Distribution"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="graph-trend", config={"displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "toImageButtonOptions": {"format": "png", "filename": "ev_sales_trend"},
                }), className="chart-card"), width=8),
                dbc.Col(html.Div(dcc.Graph(id="graph-vehicle-type",
                    config={"displayModeBar": False}), className="chart-card"), width=4),
            ], className="g-3"),

            # ── Section: Country + Growth ────────────────────────────────────
            section_header("🌏 Country Performance"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="graph-market-share",
                    config={"displayModeBar": False}), className="chart-card"), width=7),
                dbc.Col(html.Div(dcc.Graph(id="graph-growth-rate",
                    config={"displayModeBar": False}), className="chart-card"), width=5),
            ], className="g-3"),

            # ── Section: Choropleth Map ──────────────────────────────────────
            section_header("🗺️ Global EV Adoption Map"),
            html.Div([
                html.Div([
                    html.Label("Select Year for Map: ", style={"color": "#8b949e",
                        "fontSize": "0.82rem", "fontWeight": "500", "marginRight": "10px"}),
                    dcc.Slider(
                        id="map-year-slider",
                        min=YEAR_MIN, max=YEAR_MAX, step=1,
                        value=YEAR_MAX,
                        marks={y: {"label": str(y), "style": {"color": "#8b949e", "fontSize": "0.7rem"}}
                               for y in range(YEAR_MIN, YEAR_MAX + 1, 3)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px",
                          "padding": "0 12px"}),
                html.Div(dcc.Graph(id="graph-map", config={
                    "displayModeBar": True,
                    "toImageButtonOptions": {"format": "png", "filename": "ev_world_map"},
                }), className="chart-card"),
            ]),

            # ── Section: Forecasts ───────────────────────────────────────────
            section_header("🔮 Machine Learning Forecast (2026–2035)"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="graph-forecast",
                    config={"displayModeBar": True,
                            "toImageButtonOptions": {"format": "png", "filename": "ev_forecast"},
                    }), className="chart-card"), width=8),
                dbc.Col(html.Div(dcc.Graph(id="graph-battery-cost",
                    config={"displayModeBar": False}), className="chart-card"), width=4),
            ], className="g-3"),

            # ── Section: Policy + GDP scatter ───────────────────────────────
            section_header("🏛️ Policy Analysis & Economic Correlation"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="graph-policy",
                    config={"displayModeBar": False}), className="chart-card"), width=7),
                dbc.Col(html.Div(dcc.Graph(id="graph-gdp-scatter",
                    config={"displayModeBar": False}), className="chart-card"), width=5),
            ], className="g-3"),

            # ── Section: Correlation Heatmap ─────────────────────────────────
            section_header("🔬 Feature Correlation Analysis"),
            html.Div(dcc.Graph(id="graph-heatmap",
                config={"displayModeBar": False}), className="chart-card"),

            # ── Section: Data Table ──────────────────────────────────────────
            section_header("📋 Raw Data Explorer"),
            html.Div([
                html.Div([
                    html.Span("Showing filtered dataset",
                              style={"color": "#8b949e", "fontSize": "0.8rem"}),
                    dbc.Button(
                        "⬇ Download CSV",
                        id="btn-download-csv",
                        n_clicks=0,
                        className="btn-download",
                        style={"marginLeft": "auto"},
                    ),
                    dcc.Download(id="download-csv"),
                ], style={"display": "flex", "alignItems": "center",
                           "marginBottom": "12px", "padding": "0 4px"}),
                html.Div(id="data-table-container"),
            ]),

            # Footer
            html.Div([
                "⚡ EV Market Forecast Dashboard  ·  Built with Python, Plotly & Dash  ·  "
                "Data based on IEA Global EV Outlook 2024 statistics"
            ], className="footer", style={"marginTop": "32px"}),

        ], className="main-content", style={"flex": "1", "overflow": "auto"}),

    ], style={"display": "flex", "height": "calc(100vh - 73px)", "overflow": "hidden"}),

], style={"fontFamily": "Inter, sans-serif", "backgroundColor": "#0d1117"})


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

def _get_filtered(countries, vehicle_types, year_range, regions):
    """Apply all sidebar filters to the raw dataframe."""
    if not countries or not vehicle_types:
        return _RAW_DF.iloc[0:0]

    df = _RAW_DF.copy()
    df = pp.apply_filters(
        df,
        countries=countries,
        vehicle_types=vehicle_types,
        year_range=tuple(year_range) if year_range else None,
        regions=regions if regions else None,
    )
    return df


# ── Reset Callback ────────────────────────────────────────────────────────────
@app.callback(
    Output("filter-country", "value"),
    Output("filter-region", "value"),
    Output("filter-vehicle-type", "value"),
    Output("filter-year", "value"),
    Output("filter-forecast-model", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return ALL_COUNTRIES, None, ["BEV", "PHEV"], [YEAR_MIN, YEAR_MAX], "poly"


# ── Master update callback ─────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row",           "children"),
    Output("graph-trend",       "figure"),
    Output("graph-vehicle-type","figure"),
    Output("graph-market-share","figure"),
    Output("graph-growth-rate", "figure"),
    Output("graph-policy",      "figure"),
    Output("graph-gdp-scatter", "figure"),
    Output("graph-heatmap",     "figure"),
    Output("graph-forecast",    "figure"),
    Output("graph-battery-cost","figure"),
    Output("data-table-container", "children"),

    Input("filter-country",        "value"),
    Input("filter-region",         "value"),
    Input("filter-vehicle-type",   "value"),
    Input("filter-year",           "value"),
    Input("filter-forecast-model", "value"),
)
def update_all(countries, regions, vehicle_types, year_range, forecast_model):
    df      = _get_filtered(countries, vehicle_types, year_range, regions)
    trend   = pp.get_global_trend(df)
    yoy     = pp.get_yoy_growth(df)
    corr    = pp.get_correlation_matrix(df)
    vtype   = pp.get_vehicle_type_split(df)

    # Latest year for country breakdown
    latest_year = int(df["Year"].max()) if not df.empty else YEAR_MAX
    country_yr  = pp.get_country_year(df, latest_year)

    # Forecast (trained on the filtered trend subset — only fits the chosen model)
    if forecast_model == "poly":
        fcast = fc.polynomial_forecast(trend)
        model_label = "Polynomial (deg-3)"
    else:
        fcast = fc.rf_forecast(trend)
        model_label = "Random Forest"

    # ── KPI values ───────────────────────────────────────────
    total_sales    = int(df["EV_Sales"].sum())
    cagr           = fc.compute_cagr(trend) if len(trend) >= 2 else 0.0
    top_country    = country_yr.iloc[0]["Country"] if not country_yr.empty else "N/A"
    total_charging = int(df["Charging_Stations"].sum())
    
    # Extract 2030 forecast directly from fcast instead of running a separate model fit
    row_2030 = fcast[fcast["Year"] == 2030]
    sales_2030 = int(row_2030["Predicted_Sales"].values[0]) if not row_2030.empty else 0

    kpis = [
        kpi_card("🚗", "Total EV Sales", gr.fmt_number(total_sales),
                 f"Filtered view · {len(df):,} records", "cyan"),
        kpi_card("📈", "CAGR (2015–2024)", f"{cagr:.1f}%",
                 "Compound Annual Growth Rate", "green"),
        kpi_card("🏆", "Top EV Market", top_country,
                 f"Highest sales in {latest_year}", "purple"),
        kpi_card("⚡", "Charging Stations", gr.fmt_number(total_charging),
                 "Total public chargers", "orange"),
        kpi_card("🔮", "2030 Forecast", gr.fmt_number(sales_2030),
                 f"{model_label} projection", "gold"),
    ]

    # ── Figures ───────────────────────────────────────────────
    fig_trend    = gr.sales_trend_fig(trend, fcast)
    fig_vtype    = gr.vehicle_type_pie(vtype)
    fig_mshare   = gr.market_share_fig(country_yr)
    fig_growth   = gr.growth_rate_fig(yoy)
    fig_policy   = gr.policy_fig(_POLICY)
    fig_gdp      = gr.gdp_scatter_fig(country_yr)
    fig_heatmap  = gr.correlation_heatmap(corr)
    fig_forecast = gr.forecast_fig(fcast, trend, model_label)
    fig_battery  = gr.battery_cost_fig(trend)

    # ── Data Table ────────────────────────────────────────────
    display_cols = [
        "Year", "Country", "Region", "Vehicle_Type", "EV_Sales",
        "Market_Share_Pct", "Charging_Stations", "Government_Incentive_Score",
        "Battery_Cost_USD", "Fuel_Price_USD", "GDP_Trillion_USD",
    ]
    tbl_df = df[display_cols].sort_values(["Year", "Country"]).reset_index(drop=True)
    tbl_df["Market_Share_Pct"] = tbl_df["Market_Share_Pct"].round(2)
    tbl_df["Government_Incentive_Score"] = tbl_df["Government_Incentive_Score"].round(2)

    table = dash_table.DataTable(
        data=tbl_df.head(200).to_dict("records"),
        columns=[{"name": c.replace("_", " "), "id": c} for c in display_cols],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_cell={
            "backgroundColor": "#161b22",
            "color": "#e6edf3",
            "border": "1px solid #30363d",
            "fontFamily": "Inter, sans-serif",
            "fontSize": "12px",
            "padding": "8px 12px",
            "whiteSpace": "normal",
            "minWidth": "80px",
        },
        style_header={
            "backgroundColor": "#21262d",
            "color": "#8b949e",
            "fontWeight": "600",
            "fontSize": "11px",
            "textTransform": "uppercase",
            "letterSpacing": "0.6px",
            "border": "1px solid #30363d",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#1c2128"},
        ],
        style_filter={"backgroundColor": "#21262d", "color": "#e6edf3",
                      "border": "1px solid #30363d"},
    )

    return (kpis, fig_trend, fig_vtype, fig_mshare, fig_growth,
            fig_policy, fig_gdp, fig_heatmap, fig_forecast, fig_battery, table)


# ── Choropleth Map (separate slider, respects region filter) ──────────────────
@app.callback(
    Output("graph-map", "figure"),
    Input("map-year-slider",   "value"),
    Input("filter-map-metric", "value"),
    Input("filter-country",    "value"),
    Input("filter-vehicle-type","value"),
    Input("filter-region",     "value"),
)
def update_map(year, metric, countries, vehicle_types, regions):
    df = _get_filtered(countries, vehicle_types, [year, year], regions)
    country_yr = pp.get_country_year(df, year)
    return gr.choropleth_fig(country_yr, color_metric=metric or "EV_Sales")


# ── Download CSV ──────────────────────────────────────────────────────────────
@app.callback(
    Output("download-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State("filter-country",      "value"),
    State("filter-region",       "value"),
    State("filter-vehicle-type", "value"),
    State("filter-year",         "value"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, countries, regions, vehicle_types, year_range):
    df = _get_filtered(countries, vehicle_types, year_range, regions)
    return dcc.send_data_frame(df.to_csv, "ev_dashboard_export.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EV Market Forecast Dashboard")
    print("  Open -> http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8050)
