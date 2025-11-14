import pandas as pd
from typing import Tuple, Dict, Any
from .indicators import calculate_all_indicators
from .models import model_ml_signal, model_arima_forecast, model_garch_volatility

def generate_ensemble_signals(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Genera un ensamble de señales de trading combinando indicadores técnicos y modelos.

    Args:
        data (pd.DataFrame): DataFrame con datos OHLCV.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: 
            - DataFrame con todos los indicadores y señales añadidas.
            - Diccionario con resultados de modelos (predicciones de ARIMA y GARCH).
    """
    
    # 1. Cálculo de Indicadores Técnicos
    df_signals = calculate_all_indicators(data.copy())
    
    # 2. Generación de Señales Basadas en Indicadores
    
    # Señal 1: Cruce MACD
    df_signals['Signal_MACD'] = 0
    df_signals.loc[(df_signals['MACD'] > df_signals['Signal_Line']) & (df_signals['MACD'].shift(1) <= df_signals['Signal_Line'].shift(1)), 'Signal_MACD'] = 1 # Compra
    df_signals.loc[(df_signals['MACD'] < df_signals['Signal_Line']) & (df_signals['MACD'].shift(1) >= df_signals['Signal_Line'].shift(1)), 'Signal_MACD'] = -1 # Venta

    # Señal 2: RSI sobrevendido/sobrecomprado
    df_signals['Signal_RSI'] = 0
    df_signals.loc[df_signals['RSI'] < 30, 'Signal_RSI'] = 1 # Compra (sobrevendido)
    df_signals.loc[df_signals['RSI'] > 70, 'Signal_RSI'] = -1 # Venta (sobrecomprado)
    
    # Señal 3: Supertrend
    df_signals['Signal_Supertrend'] = df_signals['Supertrend_Direction'].fillna(0).astype(int)
    
    # 3. Generación de Señales Basadas en Machine Learning
    ml_predictions = model_ml_signal(df_signals.copy())
    # Mapear la predicción de ML (0 o 1) a una señal de trading (-1, 0, 1)
    # 1 (Subida) -> 1 (Compra), 0 (Bajada) -> -1 (Venta)
    df_signals['Signal_ML'] = ml_predictions.map({1: 1, 0: -1}).reindex(df_signals.index).fillna(0)
    
    # 4. Ejecución de Modelos de Series de Tiempo (para visualización)
    arima_forecast = model_arima_forecast(data.copy())
    garch_volatility = model_garch_volatility(data.copy())
    
    model_results = {
        'arima_forecast': arima_forecast,
        'garch_volatility': garch_volatility
    }
    
    # 5. Señal de Ensamble (Lógica de Votación Simple)
    # Sumar las señales y aplicar un umbral
    signal_columns = ['Signal_MACD', 'Signal_RSI', 'Signal_Supertrend', 'Signal_ML']
    df_signals['Ensemble_Score'] = df_signals[signal_columns].sum(axis=1)
    
    # Definir la señal final de ensamble
    df_signals['Signal_Ensemble'] = 0
    df_signals.loc[df_signals['Ensemble_Score'] >= 2, 'Signal_Ensemble'] = 1 # Compra fuerte
    df_signals.loc[df_signals['Ensemble_Score'] <= -2, 'Signal_Ensemble'] = -1 # Venta fuerte
    
    return df_signals, model_results
