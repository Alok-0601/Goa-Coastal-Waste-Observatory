"""Interactive research dashboard for Goa coastal-waste forecasting.

Run locally with: streamlit run app.py
"""

from __future__ import annotations

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Goa_Beach.csv"
MODEL_PATH = ROOT / "Model" / "goa_beach_waste_model.pkl"
FEATURE_PATH = ROOT / "Model" / "feature_names.json"
TARGET = "beach_waste_daily_mt"

# Streamlit themes can otherwise pass a light text colour through to Plotly.
px.defaults.template = "plotly_white"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


st.set_page_config(
    page_title="Goa Coastal Waste Observatory",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load the daily research dataset and prepare display fields."""
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    data["wave_height_m"] = data["wave_height_m"].fillna(data["wave_height_m"].median())
    data["month_label"] = pd.Categorical(
        data["month_name"], categories=MONTH_NAMES, ordered=True
    )
    data["week_part"] = np.where(data["is_weekend"].eq(1), "Weekend", "Weekday")
    return data


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Deserialize the pre-trained Gradient Boosting model once per session."""
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data(show_spinner=False)
def load_feature_names(feature_path: str) -> list[str]:
    with open(feature_path, "r", encoding="utf-8") as feature_file:
        return json.load(feature_file)


def fmt_mt(value: float) -> str:
    return f"{value:,.2f} MT"


def risk_band(value: float, reference: pd.Series) -> tuple[str, str, str]:
    """Return an interpretable band based on distribution quartiles."""
    q1, q3 = reference.quantile([0.25, 0.75])
    if value >= q3:
        return "High attention", "#ef8354", "Plan enhanced collection and monitoring."
    if value >= q1:
        return "Moderate", "#f3b61f", "Maintain routine collection and field checks."
    return "Lower", "#36a878", "Routine collection conditions are indicated."


def estimate_feature_baseline(data: pd.DataFrame, selected_date: date) -> pd.Series:
    """Use observations from the selected calendar month, falling back to all data."""
    month_sample = data.loc[data["month"].eq(selected_date.month)]
    return month_sample.median(numeric_only=True).combine_first(data.median(numeric_only=True))


def build_feature_row(
    feature_names: list[str],
    selected_date: date,
    rainfall: float,
    rainfall_lag1: float,
    rainfall_lag2: float,
    wave_height: float,
    wind_speed: float,
    population: int,
    tourists: int,
    holiday_today: bool,
    event_today: bool,
    holiday_yesterday: bool,
    event_yesterday: bool,
) -> pd.DataFrame:
    selected = pd.Timestamp(selected_date)
    yesterday = selected - pd.Timedelta(days=1)
    is_weekend = int(selected.weekday() >= 5)
    is_monsoon = int(selected.month in (6, 7, 8, 9))
    row = {
        "day_of_week_num": selected.weekday(),
        "is_weekend": is_weekend,
        "is_monsoon": is_monsoon,
        "is_public_holiday": int(holiday_today),
        "is_major_festival_or_event": int(event_today),
        "rainfall_mm": rainfall,
        "wave_height_m": wave_height,
        "wind_speed_10m_mps": wind_speed,
        "population_thousand": population,
        "total_tourists_annual": tourists,
        "rainfall_lag1": rainfall_lag1,
        "rainfall_3day_sum": rainfall + rainfall_lag1 + rainfall_lag2,
        "is_weekend_lag1": int(yesterday.weekday() >= 5),
        "is_public_holiday_lag1": int(holiday_yesterday),
        "is_major_festival_or_event_lag1": int(event_yesterday),
        "day": selected.day,
        "month": selected.month,
        "year": selected.year,
    }
    return pd.DataFrame([{name: row[name] for name in feature_names}])


def predictor_chart(prediction: float, data: pd.DataFrame, colour: str) -> go.Figure:
    maximum = max(float(data[TARGET].max() * 1.12), prediction * 1.12, 1.0)
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"suffix": " MT", "font": {"size": 42, "color": "#102a43"}},
            title={"text": "Estimated daily beach waste", "font": {"size": 18, "color": "#486581"}},
            gauge={
                "axis": {"range": [0, maximum], "tickcolor": "#627d98", "tickfont": {"color": "#486581"}},
                "bar": {"color": colour, "thickness": 0.55},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, float(data[TARGET].quantile(0.25))], "color": "#e7f5ee"},
                    {"range": [float(data[TARGET].quantile(0.25)), float(data[TARGET].quantile(0.75))], "color": "#fff4d6"},
                    {"range": [float(data[TARGET].quantile(0.75)), maximum], "color": "#fde9df"},
                ],
            },
        )
    ).update_layout(
        height=260,
        margin=dict(l=25, r=25, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#102a43"),
    )


def add_download(data: pd.DataFrame) -> None:
    export_columns = [
        "date", TARGET, "rainfall_mm", "wave_height_m", "wind_speed_10m_mps",
        "is_weekend", "is_monsoon", "is_public_holiday", "is_major_festival_or_event",
    ]
    st.download_button(
        "Download filtered observations (CSV)",
        data[export_columns].to_csv(index=False).encode("utf-8"),
        file_name="goa_coastal_waste_observations.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f5f8fa; color: #102a43; }
        [data-testid="stMain"] { color: #102a43; }
        [data-testid="stSidebar"] { background: #0b3c5d; }
        [data-testid="stSidebar"] * { color: #eef7f8; }
        .hero { background: linear-gradient(120deg, #063b5c 0%, #087e8b 58%, #2fa7a0 100%);
                padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 1.5rem;
                box-shadow: 0 10px 24px rgba(6, 59, 92, .18); }
        .hero h1 { color: white; font-size: 2.35rem; margin: 0 0 .35rem 0; letter-spacing: -.5px; }
        [data-testid="stMain"] .hero p { color: #ffffff !important; font-size: 1.03rem; margin: 0; max-width: 780px; }
        .eyebrow { color: #9ee5dc; font-size: .74rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; }
        .section-title { color: #102a43; font-size: 1.24rem; font-weight: 700; margin: .55rem 0 .1rem; }
        .section-subtitle { color: #627d98; margin-bottom: .8rem; }
        div[data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #486581 !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #102a43 !important; }
        [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color: #16865b !important; }
        /* Explicit label colours keep the interface readable with dark Streamlit themes. */
        [data-testid="stMain"] [data-testid="stWidgetLabel"],
        [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
        [data-testid="stMain"] div[data-testid="stNumberInput"] > label,
        [data-testid="stMain"] div[data-testid="stDateInput"] > label,
        [data-testid="stMain"] div[data-testid="stMultiSelect"] > label,
        [data-testid="stMain"] div[data-testid="stSelectbox"] > label { color: #243b53 !important; opacity: 1 !important; }
        [data-testid="stMain"] [data-baseweb="checkbox"] + div,
        [data-testid="stMain"] [data-baseweb="checkbox"] + div * { color: #243b53 !important; }
        [data-testid="stMain"] .stMarkdown p,
        [data-testid="stMain"] .stMarkdown li { color: #334e68; }
        [data-testid="stMain"] [data-testid="stDataFrame"] { color: #102a43 !important; }
        .prediction-card { background: white; border: 1px solid #d9e2ec; border-radius: 14px; padding: 1.1rem 1.25rem; }
        .prediction-card h3 { margin: 0 0 .35rem; color: #102a43; }
        .prediction-card p { margin: 0; color: #486581; }
        .research-note { background: #e8f4f5; border-left: 4px solid #087e8b; border-radius: 4px;
                         padding: .8rem 1rem; color: #243b53; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.2rem; }
        .stTabs [data-baseweb="tab"] { height: 48px; background: transparent; font-weight: 600; color: #486581 !important; }
        .stTabs [data-baseweb="tab"] * { color: #486581 !important; }
        .stTabs [data-baseweb="tab"][aria-selected="true"],
        .stTabs [data-baseweb="tab"][aria-selected="true"] * { color: #087e8b !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists() or not MODEL_PATH.exists() or not FEATURE_PATH.exists():
        st.error("Required project assets are missing. Keep `Goa_Beach.csv` and the `Model` folder beside app.py.")
        st.stop()

    try:
        data = load_data(str(DATA_PATH))
        feature_names = load_feature_names(str(FEATURE_PATH))
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error("The dashboard could not load its data or trained model.")
        st.exception(exc)
        st.stop()

    latest_date = data["date"].max().date()
    earliest_date = data["date"].min().date()
    dataset_average = float(data[TARGET].mean())

    with st.sidebar:
        st.markdown("## 🌊 Observatory")
        st.caption("Goa coastal waste research interface")
        st.divider()
        st.markdown("**Dataset coverage**")
        st.write(f"{earliest_date:%d %b %Y} — {latest_date:%d %b %Y}")
        st.markdown("**Model**")
        st.write("Gradient Boosting Regressor")
        st.markdown("**Target**")
        st.write("Daily beach waste (metric tonnes)")
        st.divider()
        st.caption("Use the forecast workspace to test operational scenarios. Interpret estimates alongside local monitoring and collection conditions.")

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Applied environmental intelligence</div>
          <h1>Goa Coastal Waste Observatory</h1>
          <p>Explore four years of daily coastal-waste observations and generate transparent, scenario-based estimates for planning and research.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Observations", f"{len(data):,}", "daily records")
    metric_b.metric("Mean daily waste", fmt_mt(dataset_average))
    metric_c.metric("Observed peak", fmt_mt(float(data[TARGET].max())))
    metric_d.metric("Coverage", f"{data['year'].nunique()} years", f"through {latest_date:%Y}")

    forecast_tab, explore_tab, methods_tab = st.tabs(
        ["Forecast workspace", "Evidence explorer", "Data & methodology"]
    )

    with forecast_tab:
        st.markdown('<div class="section-title">Scenario forecast</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Adjust environmental and activity conditions. Calendar-derived features update from the selected date.</div>', unsafe_allow_html=True)

        input_col, result_col = st.columns([1.05, 0.95], gap="large")
        with input_col:
            selected_date = st.date_input(
                "Forecast date", value=latest_date + timedelta(days=1),
                help="Weekend and monsoon features are derived automatically from this date.",
            )
            baseline = estimate_feature_baseline(data, selected_date)
            environment, context = st.columns(2)
            with environment:
                st.markdown("##### Environmental conditions")
                rainfall = st.number_input("Rainfall today (mm)", 0.0, 300.0, float(round(baseline["rainfall_mm"], 1)), 0.5)
                rainfall_lag1 = st.number_input("Rainfall yesterday (mm)", 0.0, 300.0, float(round(baseline["rainfall_lag1"], 1)), 0.5)
                rainfall_lag2 = st.number_input("Rainfall two days ago (mm)", 0.0, 300.0, 0.0, 0.5)
                wave_height = st.number_input("Wave height (m)", 0.0, 8.0, float(round(baseline["wave_height_m"], 2)), 0.05)
                wind_speed = st.number_input("Wind speed at 10 m (m/s)", 0.0, 25.0, float(round(baseline["wind_speed_10m_mps"], 2)), 0.1)
            with context:
                st.markdown("##### Activity & calendar context")
                population = st.number_input("Population (thousand)", 1, 5000, int(round(baseline["population_thousand"])), 1)
                tourists = st.number_input("Annual tourist count", 0, 30000000, int(round(baseline["total_tourists_annual"])), 10000, format="%d")
                holiday_today = st.checkbox("Public holiday today")
                event_today = st.checkbox("Major festival or event today")
                holiday_yesterday = st.checkbox("Public holiday yesterday")
                event_yesterday = st.checkbox("Major festival or event yesterday")
                st.info(
                    f"**{selected_date:%A, %d %B %Y}**  \n"
                    f"{'Weekend' if selected_date.weekday() >= 5 else 'Weekday'} · "
                    f"{'Monsoon season' if selected_date.month in (6, 7, 8, 9) else 'Non-monsoon season'}"
                )

            features = build_feature_row(
                feature_names, selected_date, rainfall, rainfall_lag1, rainfall_lag2,
                wave_height, wind_speed, population, tourists, holiday_today, event_today,
                holiday_yesterday, event_yesterday,
            )
            try:
                prediction = float(model.predict(features)[0])
                prediction = max(0.0, prediction)
            except Exception as exc:
                st.error("The model could not evaluate this scenario.")
                st.exception(exc)
                st.stop()

        with result_col:
            band, band_color, advice = risk_band(prediction, data[TARGET])
            st.plotly_chart(predictor_chart(prediction, data, band_color), use_container_width=True, config={"displayModeBar": False})
            difference = prediction - dataset_average
            st.metric("Difference from study-period mean", fmt_mt(difference), delta=f"{difference:+.2f} MT")
            st.markdown(
                f'<div class="prediction-card"><h3 style="color:{band_color};">{band}</h3><p>{advice}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("View model-ready feature row"):
                display_features = features.T.rename(columns={0: "Value"})
                st.dataframe(display_features, use_container_width=True)

        st.markdown(
            '<div class="research-note"><b>Interpretation note.</b> This is a decision-support estimate, not a direct measurement. It reflects patterns learned from the 2022–2025 study dataset; conditions outside those observed ranges should be interpreted cautiously.</div>',
            unsafe_allow_html=True,
        )

    with explore_tab:
        st.markdown('<div class="section-title">Evidence explorer</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Interrogate the historical record before drawing operational conclusions.</div>', unsafe_allow_html=True)
        left_filter, right_filter, weekend_filter = st.columns([1.3, 1.3, 1])
        with left_filter:
            selected_range = st.date_input("Observation window", value=(earliest_date, latest_date), min_value=earliest_date, max_value=latest_date)
        with right_filter:
            selected_months = st.multiselect("Months", MONTH_NAMES, default=MONTH_NAMES)
        with weekend_filter:
            day_type = st.selectbox("Day type", ["All days", "Weekdays", "Weekends"])

        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date = end_date = selected_range
        filtered = data.loc[
            data["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
            & data["month_name"].isin(selected_months)
        ].copy()
        if day_type == "Weekdays":
            filtered = filtered.loc[filtered["is_weekend"].eq(0)]
        elif day_type == "Weekends":
            filtered = filtered.loc[filtered["is_weekend"].eq(1)]

        if filtered.empty:
            st.warning("No observations match the selected filters. Expand the date range or month selection.")
        else:
            trend_col, seasonal_col = st.columns([1.35, 1])
            with trend_col:
                trend = px.line(filtered, x="date", y=TARGET, title="Daily observed beach waste", labels={TARGET: "Metric tonnes", "date": "Date"})
                trend.update_traces(line_color="#087e8b", line_width=1.5)
                trend.update_layout(height=355, margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(trend, use_container_width=True)
            with seasonal_col:
                monthly = filtered.groupby("month_label", observed=False)[TARGET].mean().reset_index().dropna()
                seasonal = px.bar(monthly, x="month_label", y=TARGET, title="Mean waste by month", labels={"month_label": "", TARGET: "Metric tonnes"}, color=TARGET, color_continuous_scale="Tealgrn")
                seasonal.update_layout(height=355, margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                st.plotly_chart(seasonal, use_container_width=True)

            comparison_col, drivers_col = st.columns(2)
            with comparison_col:
                day_order = DAY_NAMES
                by_day = filtered.groupby("day_of_week", as_index=False)[TARGET].mean()
                by_day["day_of_week"] = pd.Categorical(by_day["day_of_week"], categories=day_order, ordered=True)
                by_day = by_day.sort_values("day_of_week")
                day_fig = px.bar(by_day, x="day_of_week", y=TARGET, color="day_of_week", title="Mean waste by day of week", labels={"day_of_week": "", TARGET: "Metric tonnes"}, color_discrete_sequence=px.colors.sequential.Teal)
                day_fig.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(day_fig, use_container_width=True)
            with drivers_col:
                drivers = ["rainfall_mm", "wave_height_m", "wind_speed_10m_mps", "is_weekend", "is_monsoon", "total_tourists_annual"]
                labels = {"rainfall_mm": "Rainfall", "wave_height_m": "Wave height", "wind_speed_10m_mps": "Wind speed", "is_weekend": "Weekend", "is_monsoon": "Monsoon", "total_tourists_annual": "Tourists"}
                corr = filtered[drivers + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
                corr_frame = corr.rename(index=labels).reset_index(name="Correlation").rename(columns={"index": "Variable"})
                corr_fig = px.bar(corr_frame, x="Correlation", y="Variable", orientation="h", color="Correlation", color_continuous_scale="RdBu", color_continuous_midpoint=0, title="Association with daily waste")
                corr_fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                st.plotly_chart(corr_fig, use_container_width=True)

            st.markdown("##### Filtered observations")
            st.dataframe(
                filtered[["date", TARGET, "rainfall_mm", "wave_height_m", "wind_speed_10m_mps", "week_part", "is_monsoon"]]
                .rename(columns={TARGET: "waste_mt", "is_monsoon": "monsoon"}),
                use_container_width=True, height=260,
            )
            add_download(filtered)

    with methods_tab:
        st.markdown('<div class="section-title">Data & methodology</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">A compact record of the dataset, input structure, and responsible-use limits.</div>', unsafe_allow_html=True)
        method_col, schema_col = st.columns([1.05, 0.95], gap="large")
        with method_col:
            st.markdown("##### Study record")
            st.markdown(
                f"""
                - **Unit of analysis:** one daily coastal observation.
                - **Coverage:** {earliest_date:%d %B %Y} to {latest_date:%d %B %Y} ({len(data):,} records).
                - **Outcome:** observed daily beach waste, measured in metric tonnes.
                - **Estimator:** the supplied Gradient Boosting regression model.
                - **Validation approach in the project notebook:** chronological training through 2024 and holdout testing on 2025.
                """
            )
            st.markdown("##### Responsible interpretation")
            st.write("The dashboard summarizes associations in an observational dataset. It should support field teams and researchers, not replace field measurements, local knowledge, or collection logs.")
        with schema_col:
            st.markdown("##### Forecast input families")
            schema = pd.DataFrame({
                "Input family": ["Calendar", "Weather & sea state", "Activity proxy", "Lagged context"],
                "Included variables": [
                    "day, month, year, weekday/weekend, monsoon, holiday/event flags",
                    "rainfall, wave height, wind speed",
                    "population and annual tourist count",
                    "prior-day rainfall, 3-day rainfall sum, prior-day calendar/event flags",
                ],
            })
            st.dataframe(schema, hide_index=True, use_container_width=True)
            st.markdown("##### Files expected at deployment")
            st.code("app.py\nGoa_Beach.csv\nModel/goa_beach_waste_model.pkl\nModel/feature_names.json", language="text")


if __name__ == "__main__":
    main()
