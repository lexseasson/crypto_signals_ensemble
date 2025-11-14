import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Añadir las capas al path para poder importarlas
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_layer.data_fetcher import fetch_ohlcv_data, CRYPTO_PAIRS
from logic_layer.signal_generator import generate_ensemble_signals

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Crypto Signals Ensemble Dashboard")

st.title("📊 Crypto Signals Ensemble: Dashboard Educacional")
st.markdown("---")

# --- Sidebar para Controles ---
st.sidebar.header("Configuración de Datos")

selected_pair = st.sidebar.selectbox(
    "Seleccione Par de Criptomoneda:",
    options=CRYPTO_PAIRS,
    index=0
)

selected_timeframe = st.sidebar.selectbox(
    "Seleccione Timeframe:",
    options=['1m', '5m', '15m', '1h', '4h', '1d'],
    index=2 # 15m por defecto
)

data_limit = st.sidebar.slider(
    "Límite de Velas (para análisis):",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

st.sidebar.header("Modelos y Señales")
show_arima = st.sidebar.checkbox("Mostrar Pronóstico ARIMA", value=True)
show_garch = st.sidebar.checkbox("Mostrar Volatilidad GARCH", value=True)

# --- Función Principal para Cargar y Procesar Datos ---
@st.cache_data(ttl=60) # Cachear los datos por 60 segundos
def load_and_process_data(symbol, timeframe, limit):
    """Carga los datos y genera las señales."""
    data = fetch_ohlcv_data(symbol, timeframe, limit)
    if data.empty:
        st.error(f"No se pudieron obtener datos para {symbol} en {timeframe}.")
        return None, None
    
    df_signals, model_results = generate_ensemble_signals(data)
    return df_signals, model_results

df_signals, model_results = load_and_process_data(selected_pair, selected_timeframe, data_limit)

if df_signals is not None:
    
    # --- Definición de Pestañas ---
    tab_main, tab_indicators, tab_models, tab_data = st.tabs([
        "📈 Gráfico Principal y Señales", 
        "📊 Indicadores Secundarios", 
        "🧠 Modelos de Alto Nivel", 
        "📋 Datos y Ensamble"
    ])

    # --- PESTAÑA 1: Gráfico Principal y Señales ---
    with tab_main:
        st.header(f"Gráfico de Velas y Señales para {selected_pair} ({selected_timeframe})")
        
        # Crear subplots: 1 para el precio/indicadores, 1 para el volumen
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.75, 0.25],
                            subplot_titles=('Precio, Tendencia y Señales', 'Volumen'))

        # 1.1. Gráfico de Velas (Row 1)
        fig.add_trace(go.Candlestick(x=df_signals.index,
                                     open=df_signals['open'],
                                     high=df_signals['high'],
                                     low=df_signals['low'],
                                     close=df_signals['close'],
                                     name='Precio'), row=1, col=1)

        # 1.2. Indicadores de Tendencia (Row 1)
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['SMA_50'], line=dict(color='blue', width=1), name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['BB_upper'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['BB_lower'], line=dict(color='gray', width=1, dash='dash'), name='BB Lower'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['Supertrend'], line=dict(color='purple', width=2), name='Supertrend'), row=1, col=1)
        
        # 1.3. Proyección ARIMA (si está activada)
        if show_arima and not model_results['arima_forecast'].empty:
            forecast = model_results['arima_forecast']
            # Unir el último punto real con el primer punto de la predicción para continuidad
            last_real_point = df_signals['close'].iloc[-1]
            forecast_start_index = forecast.index[0]
            
            # Crear una serie que conecte el final del precio real con el inicio del pronóstico
            connection_point = pd.Series([last_real_point, forecast.iloc[0]], index=[df_signals.index[-1], forecast_start_index])
            full_forecast = pd.concat([connection_point.iloc[:-1], forecast])
            
            fig.add_trace(go.Scatter(x=full_forecast.index, y=full_forecast.values, mode='lines', 
                                     line=dict(color='yellow', width=2, dash='dot'), 
                                     name='Pronóstico ARIMA'), row=1, col=1)

        # 1.4. Señales de Ensamble (Row 1)
        buy_signals = df_signals[df_signals['Signal_Ensemble'] == 1]
        sell_signals = df_signals[df_signals['Signal_Ensemble'] == -1]
        
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['low'] * 0.99, mode='markers',
                                 marker=dict(symbol='triangle-up', size=10, color='green'),
                                 name='Ensemble BUY'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['high'] * 1.01, mode='markers',
                                 marker=dict(symbol='triangle-down', size=10, color='red'),
                                 name='Ensemble SELL'), row=1, col=1)

        # 2. Volumen (Row 2)
        fig.add_trace(go.Bar(x=df_signals.index, y=df_signals['volume'], name='Volumen', marker_color='rgba(0, 128, 0, 0.5)'), row=2, col=1)

        # Configuración de Layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            template="plotly_dark"
        )
        
        fig.update_yaxes(title_text="Precio", row=1, col=1)
        fig.update_yaxes(title_text="Volumen", row=2, col=1, showticklabels=False)
        
        st.plotly_chart(fig, use_container_width=True)

    # --- PESTAÑA 2: Indicadores Secundarios ---
    with tab_indicators:
        st.header("Indicadores de Momento y Osciladores")
        
        fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, 
                                subplot_titles=('RSI (Índice de Fuerza Relativa)', 'MACD (Convergencia/Divergencia de Medias Móviles)'))

        # 1. RSI (Row 1)
        fig_ind.add_trace(go.Scatter(x=df_signals.index, y=df_signals['RSI'], line=dict(color='orange'), name='RSI'), row=1, col=1)
        fig_ind.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig_ind.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig_ind.update_yaxes(range=[0, 100], row=1, col=1)

        # 2. MACD (Row 2)
        fig_ind.add_trace(go.Scatter(x=df_signals.index, y=df_signals['MACD'], line=dict(color='blue'), name='MACD'), row=2, col=1)
        fig_ind.add_trace(go.Scatter(x=df_signals.index, y=df_signals['Signal_Line'], line=dict(color='red'), name='Signal Line'), row=2, col=1)
        
        fig_ind.update_layout(height=600, template="plotly_dark")
        fig_ind.update_yaxes(title_text="RSI", row=1, col=1)
        fig_ind.update_yaxes(title_text="MACD", row=2, col=1)
        
        st.plotly_chart(fig_ind, use_container_width=True)

    # --- PESTAÑA 3: Modelos de Alto Nivel ---
    with tab_models:
        st.header("Resultados de Modelos de Machine Learning y Series de Tiempo")
        
        col1, col2 = st.columns(2)
        
        # 3.1. Volatilidad GARCH
        if show_garch and not model_results['garch_volatility'].empty:
            with col1:
                st.subheader("Pronóstico de Volatilidad (GARCH)")
                garch_fig = go.Figure()
                
                # Volatilidad Histórica (ATR como proxy)
                garch_fig.add_trace(go.Scatter(x=df_signals.index, y=df_signals['ATR'], mode='lines', name='ATR Histórico', line=dict(color='orange')))
                
                # Pronóstico de Volatilidad
                forecast_vol = model_results['garch_volatility']
                garch_fig.add_trace(go.Scatter(x=forecast_vol.index, y=forecast_vol.values, mode='lines', name='Volatilidad GARCH', line=dict(color='red', dash='dot')))
                
                garch_fig.update_layout(template="plotly_dark", height=400, showlegend=True)
                st.plotly_chart(garch_fig, use_container_width=True)
                
                st.markdown(f"**Volatilidad Pronosticada (Próximo Periodo):** `{forecast_vol.iloc[-1]:.4f}`")

        # 3.2. Señal de Machine Learning (Clasificación)
        with col2:
            st.subheader("Señal de Clasificación ML (Random Forest)")
            
            # Mapear la señal de ML a un color para el gráfico de barras
            ml_signal_map = {1: 'green', -1: 'red', 0: 'gray'}
            df_signals['ML_Color'] = df_signals['Signal_ML'].map(ml_signal_map)
            
            ml_fig = go.Figure()
            ml_fig.add_trace(go.Bar(
                x=df_signals.index, 
                y=df_signals['Signal_ML'], 
                marker_color=df_signals['ML_Color'],
                name='Señal ML (1: Compra, -1: Venta)'
            ))
            
            ml_fig.update_layout(
                template="plotly_dark", 
                height=400, 
                showlegend=False,
                yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Venta', 'Neutro', 'Compra'], title='Señal ML')
            )
            st.plotly_chart(ml_fig, use_container_width=True)
            
            # Mostrar el reporte de clasificación (para fines educativos)
            st.markdown("---")
            st.markdown("El modelo de ML predice si el precio subirá (1) o bajará (-1) en los próximos 5 periodos.")
            st.markdown("Para ver la precisión del modelo, se debe ejecutar el código y revisar el log de la consola.")


    # --- PESTAÑA 4: Datos y Ensamble ---
    with tab_data:
        st.header("Tabla de Datos, Indicadores y Señales Recientes")
        
        # Seleccionar columnas relevantes para la tabla
        display_cols = ['open', 'high', 'low', 'close', 'volume', 'SMA_50', 'RSI', 'MACD', 'Signal_MACD', 'Signal_RSI', 'Signal_ML', 'Ensemble_Score', 'Signal_Ensemble']
        
        # Mapear las señales numéricas a texto para mejor visualización
        signal_map = {1: 'COMPRA (BUY)', -1: 'VENTA (SELL)', 0: 'NEUTRO'}
        
        df_display = df_signals[display_cols].tail(30).copy()
        df_display['Signal_MACD'] = df_display['Signal_MACD'].map(signal_map)
        df_display['Signal_RSI'] = df_display['Signal_RSI'].map(signal_map)
        df_display['Signal_ML'] = df_display['Signal_ML'].map(signal_map)
        df_display['Signal_Ensemble'] = df_display['Signal_Ensemble'].map(signal_map)
        
        st.dataframe(df_display.style.format({
            'open': '{:.4f}', 'high': '{:.4f}', 'low': '{:.4f}', 'close': '{:.4f}', 
            'SMA_50': '{:.4f}', 'RSI': '{:.2f}', 'MACD': '{:.4f}', 'Ensemble_Score': '{:.0f}'
        }), use_container_width=True)

    # --- Sección Educacional en Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Objetivo Educacional:** Este proyecto demuestra la integración de:\n"
        "1. **Conectividad:** Uso de CCXT para datos en tiempo real.\n"
        "2. **Economía/Señales:** Implementación de indicadores técnicos (RSI, MACD, BB, Supertrend).\n"
        "3. **Machine Learning:** Uso de Random Forest para clasificación de tendencias.\n"
        "4. **Series de Tiempo:** Modelos ARIMA y GARCH para pronóstico de precios y volatilidad.\n"
        "5. **Visualización:** Dashboard interactivo con Streamlit."
    )
    
    st.markdown("---")
    st.markdown("### 📚 Lógica del Ensamble")
    st.markdown(
        "La señal final (`Signal_Ensemble`) se basa en un sistema de votación simple (`Ensemble_Score`) "
        "que combina las señales de MACD, RSI, Supertrend y el modelo de Machine Learning. "
        "Un puntaje de **+2 o más** genera una señal de **COMPRA**, y **-2 o menos** genera una señal de **VENTA**."
    )
    
else:
    st.warning("Por favor, verifique la configuración de su API en el archivo `.env` o intente con otro par/timeframe.")
