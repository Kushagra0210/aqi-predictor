from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA_URL = "https://raw.githubusercontent.com/Kushagra0210/aqi-predictor/main/city_day.csv"
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
REQUIRED_COLUMNS = {"Date", "City", "AQI", "PM2.5", "PM10", "NO2", "CO", "O3"}


class DatasetError(RuntimeError):
    """Raised when the source data cannot support the experiment."""


@dataclass
class ModelResult:
    model: Pipeline
    mae: float
    rmse: float
    r2: float
    actual: pd.Series
    predicted: np.ndarray


@dataclass
class Experiment:
    data: pd.DataFrame
    features: list[str]
    pollutants: list[str]
    results: dict[str, ModelResult]


def _read_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    try:
        return pd.read_csv(source, low_memory=False)
    except Exception as exc:
        raise DatasetError(f"The dataset could not be read: {exc}") from exc


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise DatasetError(f"The dataset is missing required columns: {', '.join(missing)}")
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce")
    data = data.drop_duplicates().dropna(subset=["Date", "City", "AQI"]).sort_values("Date")
    if len(data) < 100:
        raise DatasetError("The dataset has too few valid rows for this experiment.")
    return data


def load_dataset(local_path: Path, allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    if local_path.is_file():
        return validate_dataset(_read_csv(local_path)), "bundled snapshot"
    if not allow_download:
        raise DatasetError(f"Dataset not found at {local_path}")
    try:
        request = Request(DATA_URL, headers={"User-Agent": "aqi-predictor/1.0"})
        with urlopen(request, timeout=15) as response:
            payload = response.read()
        return validate_dataset(_read_csv(BytesIO(payload))), "remote fallback"
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError("The bundled dataset is missing and the fallback download failed. Try again or inspect the repository setup.") from exc


def _estimators() -> dict[str, object]:
    estimators: dict[str, object] = {
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=4, random_state=42),
        "Random forest": RandomForestRegressor(n_estimators=120, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        estimators["XGBoost"] = XGBRegressor(n_estimators=140, max_depth=6, learning_rate=0.08, random_state=42, verbosity=0, n_jobs=2)
    except ImportError:
        pass
    return estimators


def run_experiment(data: pd.DataFrame) -> Experiment:
    pollutants = [column for column in POLLUTANTS if column in data.columns]
    prepared = data.copy()
    for column in pollutants:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["Month"] = prepared["Date"].dt.month
    features = ["City", "Month", *pollutants]
    split_index = int(len(prepared) * 0.8)
    train = prepared.iloc[:split_index]
    test = prepared.iloc[split_index:]
    if train.empty or test.empty:
        raise DatasetError("A chronological train/test split could not be created.")
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["Month", *pollutants]),
        ("city", OneHotEncoder(handle_unknown="ignore"), ["City"]),
    ])
    results: dict[str, ModelResult] = {}
    for name, estimator in _estimators().items():
        model = Pipeline([("prepare", clone(preprocessor)), ("model", estimator)])
        model.fit(train[features], train["AQI"])
        prediction = model.predict(test[features])
        results[name] = ModelResult(model=model, mae=float(mean_absolute_error(test["AQI"], prediction)), rmse=float(np.sqrt(mean_squared_error(test["AQI"], prediction))), r2=float(r2_score(test["AQI"], prediction)), actual=test["AQI"].reset_index(drop=True), predicted=prediction)
    return Experiment(data=prepared, features=features, pollutants=pollutants, results=results)


def predict(experiment: Experiment, model_name: str, city: str, month: int, values: dict[str, float]) -> float:
    row = {feature: values.get(feature, np.nan) for feature in experiment.features}
    row["City"] = city
    row["Month"] = month
    return float(experiment.results[model_name].model.predict(pd.DataFrame([row], columns=experiment.features))[0])


def aqi_category(value: float) -> tuple[str, str]:
    if value <= 50: return "Good", "Air quality is generally satisfactory."
    if value <= 100: return "Satisfactory", "Sensitive people may experience minor discomfort."
    if value <= 200: return "Moderate", "Sensitive groups may experience breathing discomfort."
    if value <= 300: return "Poor", "Most people may experience breathing discomfort."
    if value <= 400: return "Very poor", "Prolonged exposure may increase respiratory risk."
    return "Severe", "Avoid exposure and consult official local health guidance."
