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
│   ├── risk_metrics.py         # Métricas de riesgo: Sharpe, Sortino, drawdown, retornos acumulados.
│   └── offline_training.py     # Pipeline offline con validación temporal para Random Forest.
├── scripts/
│   └── setup_env.py            # Script auxiliar para generar `.env` locales a partir del template.
└── dashboard/
    └── app.py                  # Aplicación principal de Streamlit para la visualización.

notebooks/
└── README.md                   # Guía de storytelling y ejercicios prácticos en Jupyter.
```

## Características Clave

*   **Modularidad:** Cada componente es independiente y puede ser probado o reemplazado fácilmente.
*   **Escalabilidad:** Facilita la adición de nuevos pares de criptomonedas, *timeframes* o modelos de predicción.
*   **Educacional:** El código está diseñado para ser claro y bien documentado, ideal para demostrar habilidades en programación, economía y *data science*.
*   **Dashboards Interactivos:** Uso de Streamlit para una visualización rápida y atractiva.
*   **Gestión de Riesgo Integrada:** Métricas clásicas (Sharpe, Sortino, drawdown) vinculadas a la señal de ensamble.
*   **Pipeline Educativo Offline:** Script dedicado para entrenar Random Forest con validación temporal y guardar modelos.

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
    *   El archivo real `.env` dejó de versionarse para proteger claves sensibles. En su lugar se mantiene un template seguro [`\.env.example`](\.env.example) del que puedes partir.
    *   Opción manual:
        ```bash
        cp .env.example .env
        ```
        BINANCE_API_KEY=SU_CLAVE_API
        BINANCE_SECRET_KEY=SU_CLAVE_SECRETA
    *   Opción asistida con script (recomendado para estudiantes):
        ```bash
        python scripts/setup_env.py
        ```
        Usa `python scripts/setup_env.py --force` si necesitas sobrescribir un `.env` existente.
    *   Completa las variables con tus credenciales reales (`BINANCE_API_KEY`, `BINANCE_API_SECRET`) **después** de copiar el archivo.


### Buenas prácticas para proteger tus credenciales

* El archivo [`.gitignore`](.gitignore) ya excluye `.env`, por lo que tus claves locales no se subirán al repositorio.
* Revisa `git status` antes de hacer *commit* para asegurarte de que `.env` sigue sin trackearse.
* Si alguna vez añades el archivo por accidente, elimínalo del *staging* con:
  ```bash
  git rm --cached .env
  ```
* Si necesitas recuperar la estructura base después de haber eliminado tu `.env`, vuelve a ejecutar `python scripts/setup_env.py` o copia otra vez el template manualmente.
* Para múltiples entornos, crea archivos como `.env.dev` o `.env.prod`; seguirán ignorados y podrás cargarlos manualmente con `load_dotenv`.

## Gestión de Riesgo Integrada

El módulo [`logic_layer/risk_metrics.py`](logic_layer/risk_metrics.py) añade un set de métricas cuantitativas al flujo principal:

* **Retornos logarítmicos y de estrategia:** permiten comparar la señal con el comportamiento del activo subyacente.
* **Sharpe & Sortino Ratio:** conectan la volatilidad con la rentabilidad ajustada por riesgo.
* **Drawdown máximo y curva histórica:** ideales para relatar eventos macro que impactan en el portafolio.

En el dashboard encontrarás una pestaña dedicada (**🛡️ Gestión de Riesgos**) con gráficos de retorno acumulado y drawdown, pensada para discutir storytelling financiero (halvings, subas de tasas, shocks geopolíticos, etc.).

## Entrenamiento Offline vs. Inferencia en Producción

Para resaltar buenas prácticas profesionales, el archivo [`logic_layer/offline_training.py`](logic_layer/offline_training.py) separa el pipeline de entrenamiento del flujo de inferencia online:

1. `load_historical_dataset`: carga datos desde CSV o desde la API para construir datasets reproducibles.
2. `build_supervised_dataset`: aplica los mismos indicadores del proyecto para generar features consistentes.
3. `train_random_forest_offline`: ejecuta validación temporal (TimeSeriesSplit) y guarda el modelo entrenado (`models/random_forest.joblib`).

Esta separación permite mostrar cómo documentar experimentos, comparar hiperparámetros y versionar modelos antes de exponerlos en el dashboard en vivo.

## Storytelling y Recursos Didácticos

La carpeta [`notebooks/`](notebooks/README.md) ofrece ideas para construir cuadernos educativos:

* Walkthroughs paso a paso de indicadores y modelos (ARIMA vs. GARCH).
* Estudios de caso con eventos de mercado (halvings, anuncios macro) para analizar drawdowns.
* Ejercicios guiados sobre gestión de riesgo, ajuste de portafolios y calibración del modelo de ML.

Completa los notebooks para reforzar la conexión entre código, teoría económica y toma de decisiones.

## ¿Cómo acepto los cambios y commits del mentor?

Si estás siguiendo estas guías como estudiante o instructor y quieres incorporar las recomendaciones del mentor en tu propia copia del repositorio, revisa la guía práctica en [`docs/git_workflow.md`](docs/git_workflow.md). Encontrarás instrucciones paso a paso para:

* Aprobar Pull Requests desde la interfaz web de GitHub.
* Traer los commits fusionados a tu máquina usando `git pull`.
* Aplicar parches puntuales con `git apply` cuando te compartan un diff específico.

El documento también resume buenas prácticas para evitar subir archivos sensibles (`.env`) y mantener un historial limpio.