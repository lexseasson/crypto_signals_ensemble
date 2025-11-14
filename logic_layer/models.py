import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_arima_forecast(data: pd.DataFrame, forecast_steps: int = 30) -> pd.Series:
    """
    Realiza un pronóstico de precios utilizando el modelo ARIMA con auto_arima.
    
    Args:
        data (pd.DataFrame): DataFrame con la columna 'close'.
        forecast_steps (int): Número de pasos a pronosticar (e.g., minutos).

    Returns:
        pd.Series: Serie con las predicciones de precios.
    """
    try:
        ts = data['close'].copy()
        ts.index = pd.to_datetime(ts.index)

        # 1. Prueba de Estacionariedad (ADF)
        result = adfuller(ts)
        d = 0
        if result[1] > 0.05:
            ts_diff = ts.diff().dropna()
            d = 1
        else:
            ts_diff = ts

        # 2. Auto-ARIMA para encontrar el mejor orden (p, q)
        model = pm.auto_arima(ts_diff, start_p=1, start_q=1,
                              test='adf',
                              max_p=5, max_q=5,
                              m=1, d=d, # d es el orden de diferenciación
                              seasonal=False,
                              trace=False,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        # 3. Ajustar y Pronosticar
        model_fit = model.fit(ts_diff)
        forecast_diff = model_fit.predict(n_periods=forecast_steps)
        
        # Crear el índice futuro
        last_timestamp = ts.index[-1]
        freq = pd.infer_freq(ts.index) or pd.Timedelta(minutes=1) # Asumir 1 minuto si no se infiere
        forecast_index = pd.date_range(start=last_timestamp + freq, periods=forecast_steps, freq=freq)
        forecast = pd.Series(forecast_diff, index=forecast_index)

        # 4. Invertir la transformación si se diferenció
        if d == 1:
            # Reconstruir la serie original a partir de la diferencia
            last_value = ts.iloc[-1]
            forecast = last_value + forecast.cumsum()

        return forecast

    except Exception as e:
        logging.error(f"Error en model_arima_forecast: {e}")
        return pd.Series()

def model_garch_volatility(data: pd.DataFrame, forecast_steps: int = 30) -> pd.Series:
    """
    Pronostica la volatilidad futura utilizando el modelo GARCH(1,1).

    Args:
        data (pd.DataFrame): DataFrame con la columna 'close'.
        forecast_steps (int): Número de pasos a pronosticar.

    Returns:
        pd.Series: Serie con las predicciones de volatilidad (desviación estándar).
    """
    try:
        # Calcular los rendimientos logarítmicos
        returns = np.log(data['close'] / data['close'].shift(1)).dropna()
        returns_scaled = returns * 1000  # Escalar para mejor ajuste

        # Ajustar el modelo GARCH
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        model_fit = model.fit(disp='off')

        # Predicciones de volatilidad
        forecasts = model_fit.forecast(horizon=forecast_steps)
        # La predicción es la raíz cuadrada de la varianza
        vol_predictions = np.sqrt(forecasts.variance.values[-1, :])
        
        # Crear el índice futuro
        last_timestamp = returns.index[-1]
        freq = pd.infer_freq(returns.index) or pd.Timedelta(minutes=1)
        forecast_index = pd.date_range(start=last_timestamp + freq, periods=forecast_steps, freq=freq)
        
        return pd.Series(vol_predictions, index=forecast_index)

    except Exception as e:
        logging.error(f"Error en model_garch_volatility: {e}")
        return pd.Series()

def model_ml_signal(data: pd.DataFrame, target_window: int = 5) -> pd.Series:
    """
    Genera una señal de trading (Clasificación) usando un modelo de Machine Learning (Random Forest).
    La señal predice si el precio subirá (1) o bajará (0) en la próxima 'target_window' velas.

    Args:
        data (pd.DataFrame): DataFrame con indicadores técnicos ya calculados.
        target_window (int): Ventana de tiempo para definir el objetivo (e.g., 5 minutos).

    Returns:
        pd.Series: Serie con la señal de predicción (1: Compra/Subida, 0: Venta/Bajada).
    """
    try:
        # 1. Ingeniería de Características y Definición del Objetivo
        
        # El objetivo es si el precio de cierre subirá en los próximos 'target_window' periodos
        data['Target'] = (data['close'].shift(-target_window) > data['close']).astype(int)
        
        # Seleccionar características (excluir columnas que no son indicadores o que contienen NaN)
        features = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'Target', 'Supertrend_Direction']]
        
        # Limpiar datos (eliminar filas con NaN generadas por los indicadores y el target)
        data_clean = data.dropna()
        
        if data_clean.empty:
            logging.warning("DataFrame vacío después de limpiar NaNs para ML.")
            return pd.Series()

        X = data_clean[features]
        y = data_clean['Target']
        
        # 2. Escalado de Características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 3. División de Datos (usar el 80% para entrenamiento)
        train_size = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 4. Entrenamiento del Modelo (Random Forest Classifier)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # 5. Predicción (usar todo el conjunto de datos para generar señales históricas)
        X_full_scaled = scaler.transform(data.dropna()[features])
        predictions = model.predict(X_full_scaled)
        
        # 6. Evaluación (Opcional, para fines educativos)
        y_pred_test = model.predict(X_test)
        logging.info(f"Reporte de Clasificación (Test Set):\n{classification_report(y_test, y_pred_test, zero_division=0)}")

        # 7. Devolver las predicciones como una Serie con el índice original
        return pd.Series(predictions, index=data_clean.index)

    except Exception as e:
        logging.error(f"Error en model_ml_signal: {e}")
        return pd.Series()
