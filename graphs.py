"""
graphs.py
All Plotly figure builders for the EV Market Forecast Dashboard.
Each function returns a plotly.graph_objects.Figure.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from preprocessing import ISO_MAP

# ── Shared theme tokens ───────────────────────────────────────────────────────
DARK_BG     = "#0d1117"
CARD_BG     = "#161b22"
BORDER      = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED  = "#8b949e"
ACCENT_CYAN = "#39d0d8"
ACCENT_GREEN = "#3fb950"
ACCENT_PURPLE = "#bc8cff"
ACCENT_ORANGE = "#ff9f1c"
ACCENT_PINK  = "#f85149"

COUNTRY_COLORS = {
    "China":       "#f85149",
    "USA":         "#39d0d8",
    "Germany":     "#bc8cff",
    "Norway":      "#3fb950",
    "UK":          "#ff9f1c",
    "France":      "#58a6ff",
    "India":       "#ffa657",
    "Japan":       "#ff7b72",
    "South Korea": "#d2a8ff",
    "Netherlands": "#56d364",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
        title=dict(font=dict(size=15, color=TEXT_PRIMARY), x=0.02),
        xaxis=dict(
            gridcolor=BORDER, linecolor=BORDER,
            tickfont=dict(color=TEXT_MUTED),
            zerolinecolor=BORDER,
        ),
        yaxis=dict(
            gridcolor=BORDER, linecolor=BORDER,
            tickfont=dict(color=TEXT_MUTED),
            zerolinecolor=BORDER,
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.8)", bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT_PRIMARY),
        ),
        margin=dict(l=50, r=20, t=45, b=40),
        hoverlabel=dict(
            bgcolor=CARD_BG, bordercolor=BORDER,
            font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
        ),
    )
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


def fmt_number(n: int | float) -> str:
    """Human-readable number: 1.2M, 340K, etc."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(int(n))


# ── 1. Sales Trend Line Chart ─────────────────────────────────────────────────

def sales_trend_fig(trend_df: pd.DataFrame, forecast_df: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=trend_df["Year"], y=trend_df["EV_Sales"],
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color=ACCENT_CYAN, width=3),
        marker=dict(size=6, color=ACCENT_CYAN, line=dict(width=1.5, color=DARK_BG)),
        hovertemplate="<b>%{x}</b><br>Sales: %{y:,.0f}<extra></extra>",
    ))

    if forecast_df is not None:
        fcast = forecast_df[forecast_df["Is_Forecast"]]
        bridge_year = trend_df["Year"].max()
        bridge_row = trend_df[trend_df["Year"] == bridge_year]

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([fcast["Year"], fcast["Year"].iloc[::-1]]),
            y=pd.concat([fcast["Upper_CI"], fcast["Lower_CI"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(57,208,216,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band",
            hoverinfo="skip",
        ))
        # Forecast line
        bridge_x = [bridge_year] + fcast["Year"].tolist()
        bridge_y = [bridge_row["EV_Sales"].values[0]] + fcast["Predicted_Sales"].tolist()
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=ACCENT_GREEN, width=3, dash="dot"),
            marker=dict(size=6, color=ACCENT_GREEN, symbol="diamond"),
            hovertemplate="<b>%{x} (Forecast)</b><br>Sales: %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        title="Global EV Sales — Historical & Forecast",
        xaxis_title="Year",
        yaxis_title="EV Sales (units)",
    )
    return _apply_theme(fig)


# ── 2. Market Share by Country (Horizontal Bar) ───────────────────────────────

def market_share_fig(country_df: pd.DataFrame) -> go.Figure:
    df = country_df.sort_values("EV_Sales", ascending=True)
    colors = [COUNTRY_COLORS.get(c, ACCENT_CYAN) for c in df["Country"]]

    fig = go.Figure(go.Bar(
        x=df["EV_Sales"], y=df["Country"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.9,
        ),
        text=[fmt_number(v) for v in df["EV_Sales"]],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Sales: %{x:,.0f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="EV Sales by Country",
        xaxis_title="EV Sales (units)",
        yaxis_title="",
        bargap=0.25,
    )
    return _apply_theme(fig)


# ── 3. Choropleth World Map ───────────────────────────────────────────────────

def choropleth_fig(
    country_df: pd.DataFrame,
    color_metric: str = "EV_Sales",
) -> go.Figure:
    df = country_df.copy()
    df["ISO"] = df["Country"].map(ISO_MAP)
    df = df.dropna(subset=["ISO"])

    label_map = {
        "EV_Sales": "EV Sales",
        "Market_Share_Pct": "Market Share (%)",
        "Charging_Stations": "Charging Stations",
    }

    fig = px.choropleth(
        df, locations="ISO",
        color=color_metric,
        hover_name="Country",
        hover_data={
            "ISO": False,
            "EV_Sales": ":,.0f",
            "Market_Share_Pct": ":.1f",
            "Government_Incentive_Score": ":.1f",
        },
        color_continuous_scale=[
            [0.0, "#0d2b45"],
            [0.3, "#0e6fa5"],
            [0.6, "#39d0d8"],
            [0.85, "#3fb950"],
            [1.0, "#ffd700"],
        ],
        labels={color_metric: label_map.get(color_metric, color_metric)},
    )
    fig.update_geos(
        bgcolor=DARK_BG,
        landcolor="#1c2733",
        oceancolor=DARK_BG,
        showocean=True,
        lakecolor=DARK_BG,
        framecolor=BORDER,
        showframe=True,
        projection_type="natural earth",
        showcountries=True, countrycolor=BORDER,
    )
    fig.update_layout(
        title=f"Global EV Adoption Map — {label_map.get(color_metric, color_metric)}",
        coloraxis_colorbar=dict(
            title=label_map.get(color_metric, color_metric),
            tickfont=dict(color=TEXT_MUTED),
            title_font=dict(color=TEXT_MUTED),
            outlinewidth=0,
        ),
        geo=dict(bgcolor=DARK_BG),
    )
    return _apply_theme(fig)


# ── 4. Government Policy Analysis (Grouped Bar) ───────────────────────────────

def policy_fig(policy_df: pd.DataFrame) -> go.Figure:
    categories = ["Tax_Incentives", "Purchase_Subsidies", "Charging_Infrastructure", "Registration_Benefits"]
    labels     = ["Tax Incentives", "Purchase Subsidies", "Charging Infrastructure", "Registration Benefits"]
    cat_colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_ORANGE]

    fig = go.Figure()
    for cat, label, color in zip(categories, labels, cat_colors):
        fig.add_trace(go.Bar(
            name=label,
            x=policy_df["Country"],
            y=policy_df[cat],
            marker_color=color,
            opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}/10<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="Government Policy Scores by Country (0–10)",
        xaxis_title="Country",
        yaxis_title="Policy Score (0–10)",
        yaxis=dict(range=[0, 10.5]),
        bargap=0.15,
        bargroupgap=0.05,
    )
    return _apply_theme(fig)


# ── 5. Forecast Graph (Actual + Predicted + CI band) ─────────────────────────

def forecast_fig(forecast_df: pd.DataFrame, trend_df: pd.DataFrame, model_label: str = "Polynomial") -> go.Figure:
    fig = go.Figure()

    hist = trend_df.sort_values("Year")
    pred = forecast_df.sort_values("Year")
    fcast_only = pred[pred["Is_Forecast"]]
    hist_pred  = pred[~pred["Is_Forecast"]]

    # CI band (forecast region only)
    fig.add_trace(go.Scatter(
        x=pd.concat([fcast_only["Year"], fcast_only["Year"].iloc[::-1]]),
        y=pd.concat([fcast_only["Upper_CI"], fcast_only["Lower_CI"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(63,185,80,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band",
        hoverinfo="skip",
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=hist["Year"], y=hist["EV_Sales"],
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color=ACCENT_CYAN, width=3),
        marker=dict(size=7, color=ACCENT_CYAN),
        hovertemplate="<b>%{x}</b><br>Actual: %{y:,.0f}<extra></extra>",
    ))

    # In-sample model fit
    fig.add_trace(go.Scatter(
        x=hist_pred["Year"], y=hist_pred["Predicted_Sales"],
        mode="lines",
        name=f"{model_label} Fit",
        line=dict(color=ACCENT_PURPLE, width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Fit: %{y:,.0f}<extra></extra>",
    ))

    # Forecast
    bridge_x = [hist["Year"].max()] + fcast_only["Year"].tolist()
    bridge_y = [hist[hist["Year"] == hist["Year"].max()]["EV_Sales"].values[0]] + \
               fcast_only["Predicted_Sales"].tolist()
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines+markers",
        name="Forecast",
        line=dict(color=ACCENT_GREEN, width=3),
        marker=dict(size=8, color=ACCENT_GREEN, symbol="diamond"),
        hovertemplate="<b>%{x} (Forecast)</b><br>Sales: %{y:,.0f}<extra></extra>",
    ))

    # Vertical separator
    fig.add_vline(
        x=2025.5, line_dash="dash", line_color=TEXT_MUTED, line_width=1,
        annotation_text="  Forecast Start",
        annotation_font=dict(color=TEXT_MUTED, size=10),
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"EV Sales Forecast 2026–2035 ({model_label} Model)",
        xaxis_title="Year",
        yaxis_title="Global EV Sales (units)",
    )
    return _apply_theme(fig)


# ── 6. Year-over-Year Growth Rate (Area Chart) ────────────────────────────────

def growth_rate_fig(yoy_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    colors = [ACCENT_GREEN if v >= 0 else ACCENT_PINK for v in yoy_df["YoY_Growth_Pct"]]

    fig.add_trace(go.Bar(
        x=yoy_df["Year"], y=yoy_df["YoY_Growth_Pct"],
        marker_color=colors,
        name="YoY Growth (%)",
        hovertemplate="<b>%{x}</b><br>Growth: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=yoy_df["Year"], y=yoy_df["YoY_Growth_Pct"],
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=2, dash="dot"),
        name="Trend Line",
        hoverinfo="skip",
    ))
    fig.add_hline(y=0, line_color=BORDER, line_width=1)

    fig.update_layout(
        title="Year-over-Year EV Sales Growth Rate (%)",
        xaxis_title="Year",
        yaxis_title="YoY Growth (%)",
    )
    return _apply_theme(fig)


# ── 7. Vehicle Type Pie/Donut Chart ──────────────────────────────────────────

def vehicle_type_pie(vtype_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=vtype_df["Vehicle_Type"],
        values=vtype_df["EV_Sales"],
        hole=0.55,
        marker=dict(
            colors=[ACCENT_CYAN, ACCENT_PURPLE],
            line=dict(color=DARK_BG, width=3),
        ),
        textfont=dict(color=TEXT_PRIMARY, size=13),
        hovertemplate="<b>%{label}</b><br>Sales: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="BEV vs PHEV Market Split",
        legend=dict(orientation="h", y=-0.1, x=0.25),
        annotations=[dict(
            text="Vehicle<br>Types",
            x=0.5, y=0.5, font_size=13,
            showarrow=False,
            font=dict(color=TEXT_MUTED),
        )],
    )
    return _apply_theme(fig)


# ── 8. GDP vs EV Sales Scatter ────────────────────────────────────────────────

def gdp_scatter_fig(country_df: pd.DataFrame) -> go.Figure:
    df = country_df.copy()
    df["color"] = df["Country"].map(COUNTRY_COLORS).fillna(ACCENT_CYAN)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["GDP_Trillion_USD"]],
            y=[row["EV_Sales"]],
            mode="markers+text",
            name=row["Country"],
            marker=dict(
                size=max(10, int(row["Market_Share_Pct"] * 1.5)),
                color=COUNTRY_COLORS.get(row["Country"], ACCENT_CYAN),
                line=dict(width=1.5, color=DARK_BG),
                opacity=0.85,
            ),
            text=[row["Country"]],
            textposition="top center",
            textfont=dict(size=10, color=TEXT_MUTED),
            hovertemplate=(
                f"<b>{row['Country']}</b><br>"
                f"GDP: ${row['GDP_Trillion_USD']:.1f}T<br>"
                f"EV Sales: {row['EV_Sales']:,.0f}<br>"
                f"Market Share: {row['Market_Share_Pct']:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title="GDP vs EV Sales by Country<br><sup style='color:#8b949e'>Bubble size = Market Share (%)</sup>",
        xaxis_title="GDP (Trillion USD)",
        yaxis_title="EV Sales (units)",
    )
    return _apply_theme(fig)


# ── 9. Correlation Heatmap ────────────────────────────────────────────────────

def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    labels = [
        "EV Sales", "Mkt Share", "Charging", "Gov Score",
        "Battery $", "Fuel $", "GDP", "CO₂ Red."
    ]
    z = corr_df.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels, y=labels,
        colorscale=[
            [0.0, "#f85149"],
            [0.5, CARD_BG],
            [1.0, "#3fb950"],
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT_PRIMARY),
        hovertemplate="<b>%{x} × %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="r", tickfont=dict(color=TEXT_MUTED),
            title_font=dict(color=TEXT_MUTED), outlinewidth=0,
        ),
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    return _apply_theme(fig)


# ── 10. Battery Cost Decline ──────────────────────────────────────────────────

def battery_cost_fig(trend_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["Year"], y=trend_df["Battery_Cost_USD"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(188,140,255,0.10)",
        line=dict(color=ACCENT_PURPLE, width=3),
        marker=dict(size=6, color=ACCENT_PURPLE),
        name="Avg Battery Cost (USD/kWh)",
        hovertemplate="<b>%{x}</b><br>Battery Cost: $%{y:.0f}/kWh<extra></extra>",
    ))
    fig.update_layout(
        title="Average EV Battery Cost Decline (USD/kWh)",
        xaxis_title="Year",
        yaxis_title="Cost (USD/kWh)",
    )
    return _apply_theme(fig)
