from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from aqi_core import DatasetError, aqi_category, load_dataset, predict, run_experiment

st.set_page_config(page_title="AQI model workbench", page_icon=":material/air:", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def cached_data(path: str) -> tuple[pd.DataFrame, str]:
    return load_dataset(Path(path))


@st.cache_resource(show_spinner=False)
def cached_experiment(path: str):
    data, source = cached_data(path)
    return run_experiment(data), source


DATA_PATH = str(Path(__file__).with_name("city_day.csv"))

with st.sidebar:
    st.markdown("### AQI model workbench")
    st.caption("Independent applied-ML project by Kushagra Saxena")
    page = st.radio("Navigate", ["Overview", "Predict", "Model comparison", "Explore data", "Method & limitations"], key="page")
    st.caption("Historical CPCB city-day data · 2015–2020")

st.title("AQI model workbench", icon=":material/air:")
st.write("Explore a historical air-quality dataset, compare regression baselines and inspect an educational AQI estimate.")
st.caption("This is not a live forecast or public-health service. Read the methodology and limitations before interpreting results.")

content = st.container()
try:
    with content.skeleton(height=230):
        experiment, data_source = cached_experiment(DATA_PATH)
except DatasetError as exc:
    st.error(str(exc), icon=":material/error:")
    st.caption("The app expects city_day.csv beside app.py and can use the repository snapshot as a network fallback.")
    st.stop()
except Exception as exc:
    st.error("Model preparation failed. The dataset may be incompatible with the pinned runtime.", icon=":material/error:")
    with st.expander("Technical detail"):
        st.code(str(exc))
    st.stop()

data = experiment.data
results = experiment.results
best_name = min(results, key=lambda name: results[name].rmse)

if page == "Overview":
    st.header("Dataset overview", icon=":material/dashboard:")
    with st.container(horizontal=True):
        st.metric("Valid records", f"{len(data):,}", border=True)
        st.metric("Cities", f"{data['City'].nunique()}", border=True)
        st.metric("Date range", f"{data['Date'].dt.year.min()}–{data['Date'].dt.year.max()}", border=True)
        st.metric("Lowest test RMSE", f"{results[best_name].rmse:.1f}", delta=best_name, border=True)
    st.caption(f"Data source: {data_source}. Evaluation uses the latest 20% of dated rows as a chronological holdout.")
    city = data.groupby("City", as_index=False)["AQI"].mean().sort_values("AQI", ascending=False)
    monthly = data.assign(Month=data["Date"].dt.to_period("M").dt.to_timestamp()).groupby("Month", as_index=False)["AQI"].mean()
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Average AQI by city")
        st.bar_chart(city, x="City", y="AQI", horizontal=True)
    with right.container(border=True):
        st.subheader("Monthly mean AQI")
        st.line_chart(monthly, x="Month", y="AQI")

elif page == "Predict":
    st.header("Inspect an estimate", icon=":material/query_stats:")
    st.caption("Inputs are compared with models trained on historical city-day records. This does not predict future conditions.")
    defaults = {column: float(data[column].median()) for column in experiment.pollutants}
    with st.form("prediction_form"):
        city = st.selectbox("City", sorted(data["City"].dropna().unique()))
        model_name = st.selectbox("Model", list(results), index=list(results).index(best_name))
        month = st.slider("Month", 1, 12, 6)
        st.markdown("**Pollutant readings**")
        values: dict[str, float] = {}
        for row_start in range(0, len(experiment.pollutants), 3):
            columns = st.columns(3)
            for column, pollutant in zip(columns, experiment.pollutants[row_start:row_start + 3]):
                values[pollutant] = column.number_input(pollutant, min_value=0.0, value=round(defaults[pollutant], 2), key=f"pollutant_{pollutant}")
        submitted = st.form_submit_button("Estimate AQI", type="primary", icon=":material/play_arrow:")
    if submitted:
        estimate = max(0.0, predict(experiment, model_name, city, month, values))
        category, guidance = aqi_category(estimate)
        with st.container(border=True):
            st.metric("Estimated AQI", f"{estimate:.0f}")
            st.badge(category, color="green" if estimate <= 100 else "orange" if estimate <= 200 else "red")
            st.write(guidance)
            st.caption(f"Model: {model_name}. Educational estimate only; consult official local monitoring for current conditions.")

elif page == "Model comparison":
    st.header("Model comparison", icon=":material/model_training:")
    rows = [{"Model": name, "MAE": result.mae, "RMSE": result.rmse, "R²": result.r2} for name, result in results.items()]
    metrics = pd.DataFrame(rows).sort_values("RMSE")
    st.dataframe(metrics, hide_index=True, column_config={"MAE": st.column_config.NumberColumn(format="%.2f"), "RMSE": st.column_config.NumberColumn(format="%.2f"), "R²": st.column_config.NumberColumn(format="%.3f")})
    chart_data = metrics.melt("Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value")
    chart = alt.Chart(chart_data).mark_bar().encode(x=alt.X("Model:N", sort=metrics["Model"].tolist()), y="Value:Q", color="Metric:N", xOffset="Metric:N", tooltip=["Model", "Metric", alt.Tooltip("Value", format=".2f")])
    st.altair_chart(chart)
    st.caption("Lower MAE/RMSE is better. R² and errors are measured on a chronological holdout; they are not production claims.")

elif page == "Explore data":
    st.header("Explore the historical data", icon=":material/analytics:")
    selected_cities = st.multiselect("Cities", sorted(data["City"].unique()), default=sorted(data["City"].unique())[:4], max_selections=8)
    filtered = data[data["City"].isin(selected_cities)] if selected_cities else data.iloc[0:0]
    if filtered.empty:
        st.info("Select at least one city to render the exploration views.", icon=":material/info:")
    else:
        monthly = filtered.assign(Month=filtered["Date"].dt.to_period("M").dt.to_timestamp()).groupby(["Month", "City"], as_index=False)["AQI"].mean()
        st.line_chart(monthly, x="Month", y="AQI", color="City")
        st.dataframe(filtered[["Date", "City", "AQI", *experiment.pollutants]].tail(500), hide_index=True)
        st.caption("Showing the latest 500 matching records in the table to keep the browser payload bounded.")

else:
    st.header("Method & limitations", icon=":material/science:")
    st.markdown("""
### Method

- Validate dates, target values and required pollutant columns.
- Sort records by date and reserve the latest 20% as a chronological holdout.
- Median-impute numeric inputs, standardize them and one-hot encode city.
- Compare linear regression, decision tree, random forest and XGBoost when available.
- Cache the prepared experiment so widget reruns do not retrain every model.

### Limitations

- The city-day dataset is historical and is not a live sensor feed.
- A row-level chronological split reduces leakage but does not establish geographic or future-year generalization.
- Weather, station topology and live emissions context are not modeled.
- The displayed estimate is educational and must not be used as medical or public-health guidance.
- Results depend on the bundled dataset snapshot and pinned package versions.
""")
    st.warning("No impact, accuracy or deployment-scale claim is made beyond the visible holdout metrics.", icon=":material/warning:")

st.caption("Source and methodology are available in the public repository. Built as an independent project by Kushagra Saxena.")
