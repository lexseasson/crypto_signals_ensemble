# Crypto Signals Ensemble: Arquitectura Modular y Educacional

Este repositorio presenta una reestructuración modular de un sistema de generación de señales de trading de criptomonedas. El objetivo es ofrecer una solución **escalable**, **educacional** y **potente**, separando claramente las capas de datos, cálculo, modelado y visualización.

## Arquitectura Propuesta

Hemos adoptado una arquitectura de tres capas para maximizar la modularidad y la escalabilidad:

1.  **Capa de Datos y Conectividad (`data_layer/`):** Encargada de la conexión con APIs de *exchanges* (e.g., CCXT) y la gestión de la caché.
2.  **Capa de Lógica de Negocio (`logic_layer/`):** Contiene todos los cálculos de indicadores técnicos, la lógica de generación de señales y los modelos de Machine Learning/Series de Tiempo (ARIMA, GARCH, etc.).
3.  **Capa de Presentación (`dashboard/`):** Implementa el *dashboard* interactivo utilizando **Streamlit** para la visualización de datos, gráficos de velas y las señales generadas.

## Estructura del Repositorio

```
crypto_signals_ensemble/
├── README.md
├── requirements.txt
├── data_layer/
│   └── data_fetcher.py         # Conexión a CCXT, caché y preprocesamiento de datos.
├── logic_layer/
│   ├── indicators.py           # Funciones para calcular indicadores técnicos (SMA, RSI, MACD, etc.).
│   ├── models.py               # Implementación de modelos de ML y series de tiempo (ARIMA, GARCH, XGBoost).
│   └── signal_generator.py     # Lógica para combinar indicadores y modelos en señales de trading.
└── dashboard/
    └── app.py                  # Aplicación principal de Streamlit para la visualización.
```

## Características Clave

*   **Modularidad:** Cada componente es independiente y puede ser probado o reemplazado fácilmente.
*   **Escalabilidad:** Facilita la adición de nuevos pares de criptomonedas, *timeframes* o modelos de predicción.
*   **Educacional:** El código está diseñado para ser claro y bien documentado, ideal para demostrar habilidades en programación, economía y *data science*.
*   **Dashboards Interactivos:** Uso de Streamlit para una visualización rápida y atractiva.

## Configuración y Ejecución

1.  **Clonar el repositorio:**
    ```bash
    git clone [URL_DEL_REPOSITORIO]
    cd crypto_signals_ensemble
    ```
2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configurar credenciales:**
    *   **IMPORTANTE:** Nunca almacene claves API directamente en el código. Utilice variables de entorno o un archivo `.env`.
    *   Cree un archivo `.env` en la raíz del proyecto con sus claves de Binance (o el *exchange* que utilice):
        ```
        BINANCE_API_KEY=SU_CLAVE_API
        BINANCE_SECRET_KEY=SU_CLAVE_SECRETA
        ```
4.  **Ejecutar el Dashboard:**
    ```bash
    streamlit run dashboard/app.py
    ```
    El dashboard se abrirá automáticamente en su navegador.
