"""Herramientas para evaluar riesgo y desempeño de estrategias."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

# Aproximación de periodos por año para distintas resoluciones
TIMEFRAME_TO_ANNUAL_FACTOR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


@dataclass
class RiskMetrics:
    """Resumen de métricas de riesgo/retorno."""

    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "cumulative_return": self.cumulative_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
        }


def _infer_periods_per_year(timeframe: str) -> int:
    return TIMEFRAME_TO_ANNUAL_FACTOR.get(timeframe, 365)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Calcula rendimientos logarítmicos."""

    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)


def compute_strategy_returns(returns: pd.Series, signals: pd.Series) -> pd.Series:
    """Calcula los rendimientos de la estrategia aplicando señales desfasadas."""

    aligned_signals = signals.shift(1).fillna(0)
    strategy_returns = returns * aligned_signals
    return strategy_returns.fillna(0)


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """Calcula la curva de *drawdown* a partir de rendimientos simples."""

    cumulative_curve = (1 + returns.fillna(0)).cumprod()
    running_max = cumulative_curve.cummax()
    drawdown = cumulative_curve / running_max - 1
    return drawdown.fillna(0)


def calculate_risk_metrics(
    data: pd.DataFrame,
    signal_column: str = "Signal_Ensemble",
    price_column: str = "close",
    timeframe: str = "1h",
    risk_free_rate: float = 0.02,
) -> RiskMetrics:
    """Genera métricas de riesgo y series auxiliares para visualización.

    La función añade columnas al DataFrame original:
    - ``Log_Return``: rendimientos logarítmicos del activo.
    - ``Strategy_Return``: rendimientos aplicando la señal.
    - ``Cumulative_Strategy_Return``: rendimientos acumulados de la estrategia.
    - ``Drawdown``: drawdown acumulado.

    Args:
        data: DataFrame con los precios y señales calculadas.
        signal_column: Columna con la señal de estrategia a evaluar.
        price_column: Columna con los precios de cierre.
        timeframe: Timeframe de la serie para anualizar métricas.
        risk_free_rate: Tasa libre de riesgo anual usada en el Sharpe.

    Returns:
        RiskMetrics: objeto con las métricas agregadas.
    """

    if price_column not in data or signal_column not in data:
        raise ValueError("El DataFrame debe contener las columnas de precio y señal especificadas.")

    df = data.copy()
    df["Log_Return"] = compute_log_returns(df[price_column])
    df["Strategy_Return"] = compute_strategy_returns(df["Log_Return"], df[signal_column])
    df["Cumulative_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod() - 1
    df["Drawdown"] = compute_drawdown(df["Strategy_Return"]).values

    periods_per_year = _infer_periods_per_year(timeframe)
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    mean_return = df["Strategy_Return"].mean()
    std_return = df["Strategy_Return"].std()
    downside = df.loc[df["Strategy_Return"] < 0, "Strategy_Return"]
    downside_std = downside.std(ddof=0)

    annualized_return = (1 + mean_return) ** periods_per_year - 1 if mean_return != -1 else -1
    annualized_volatility = std_return * np.sqrt(periods_per_year) if std_return != 0 else 0

    excess_return = mean_return - rf_per_period
    sharpe_ratio = (
        excess_return / std_return * np.sqrt(periods_per_year)
        if std_return not in (0, np.nan)
        else 0
    )

    sortino_ratio = (
        excess_return / downside_std * np.sqrt(periods_per_year)
        if downside_std not in (0, np.nan) and not np.isnan(downside_std)
        else 0
    )

    cumulative_curve = (1 + df["Strategy_Return"]).cumprod()
    running_max = cumulative_curve.cummax()
    max_drawdown = ((cumulative_curve / running_max) - 1).min()

    metrics = RiskMetrics(
        cumulative_return=float(cumulative_curve.iloc[-1] - 1 if not cumulative_curve.empty else 0),
        annualized_return=float(annualized_return) if not np.isnan(annualized_return) else 0.0,
        annualized_volatility=float(annualized_volatility) if not np.isnan(annualized_volatility) else 0.0,
        sharpe_ratio=float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
        sortino_ratio=float(sortino_ratio) if not np.isnan(sortino_ratio) else 0.0,
        max_drawdown=float(max_drawdown) if not np.isnan(max_drawdown) else 0.0,
    )

    # Actualizar el DataFrame original con las nuevas columnas
    data["Log_Return"] = df["Log_Return"]
    data["Strategy_Return"] = df["Strategy_Return"]
    data["Cumulative_Strategy_Return"] = df["Cumulative_Strategy_Return"]
    data["Drawdown"] = df["Drawdown"]

    return metrics
