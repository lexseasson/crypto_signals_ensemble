"""Pipeline educativo para entrenamiento offline de modelos de ML."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from data_layer.data_fetcher import fetch_ohlcv_data
from .indicators import calculate_all_indicators


def load_historical_dataset(
    symbol: str,
    timeframe: str,
    limit: int = 2000,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Carga datos históricos desde CSV o desde el exchange."""

    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        df = df.sort_index()
        return df

    return fetch_ohlcv_data(symbol, timeframe=timeframe, limit=limit)


def build_supervised_dataset(
    ohlcv: pd.DataFrame,
    target_window: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Convierte datos OHLCV en un dataset supervisado para clasificación binaria."""

    features = calculate_all_indicators(ohlcv.copy())
    features['Target'] = (features['close'].shift(-target_window) > features['close']).astype(int)
    dataset = features.dropna()

    drop_cols = {
        'open', 'high', 'low', 'close', 'volume',
        'Supertrend_Direction', 'Target'
    }
    feature_cols = [col for col in dataset.columns if col not in drop_cols]

    X = dataset[feature_cols]
    y = dataset['Target']
    return X, y


def train_random_forest_offline(
    symbol: str,
    timeframe: str,
    limit: int = 2000,
    target_window: int = 5,
    model_output: Path = Path("models/random_forest.joblib"),
) -> Tuple[Pipeline, str]:
    """Entrena un modelo Random Forest usando validación temporal."""

    raw_data = load_historical_dataset(symbol, timeframe, limit=limit)
    if raw_data.empty:
        raise ValueError("No se pudieron obtener datos históricos para el entrenamiento.")

    X, y = build_supervised_dataset(raw_data, target_window=target_window)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
        )),
    ])

    splitter = TimeSeriesSplit(n_splits=5)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        scores.append(f"Fold {fold}:\n{report}\n")

    pipeline.fit(X, y)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, model_output)

    summary = "".join(scores)
    return pipeline, summary


__all__ = [
    "load_historical_dataset",
    "build_supervised_dataset",
    "train_random_forest_offline",
]