# AQI model workbench

An independent Streamlit project for exploring India’s historical city-day air-quality dataset, comparing regression baselines, and inspecting educational AQI estimates.

## What it demonstrates

- Defensive dataset loading and schema validation
- Chronological train/test evaluation
- Shared preprocessing across linear, tree, forest, and XGBoost regressors
- Cached model training for responsive Streamlit reruns
- Clear loading, error, methodology, and limitation states

This app is **not** a live AQI forecast, public-health product, or internship deliverable. It makes no deployment-scale or impact claim.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`.

The checked-in `city_day.csv` file is used first. If unavailable, the loader attempts a time-bounded download from this repository and shows a clear error if that also fails.

## Evaluation notes

Records are sorted by date and the latest 20% are held out for evaluation. This is more realistic than a random split, but it does not prove geographic or future-year generalization. Weather, station topology, and live emissions context are not modeled.

## Project structure

- `app.py` — Streamlit interface
- `aqi_core.py` — loading, validation, training, evaluation, and prediction logic
- `tests/` — focused data and prediction tests
- `.streamlit/config.toml` — native light theme
