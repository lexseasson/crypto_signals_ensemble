# Notebooks Educativos

Este directorio está pensado para alojar cuadernos Jupyter que acompañen las sesiones teóricas.

## Ideas de Cuadernos

1. **01_indicadores_basicos.ipynb**
   - Explica paso a paso cómo se calculan RSI, MACD, Bandas de Bollinger y Supertrend.
   - Referencia funciones en [`logic_layer/indicators.py`](../logic_layer/indicators.py).

2. **02_modelos_series_tiempo.ipynb**
   - Contrasta ARIMA vs. GARCH con ejemplos visuales.
   - Incluye ejercicios para probar diferentes horizontes de pronóstico.

3. **03_ml_pipeline.ipynb**
   - Usa [`logic_layer/offline_training.py`](../logic_layer/offline_training.py) para entrenar el Random Forest offline.
   - Documenta experimentos, hiperparámetros y métricas.

4. **04_storytelling_eventos.ipynb**
   - Revisa eventos macro (halvings, decisiones de tasas) y su impacto en drawdowns utilizando [`logic_layer/risk_metrics.py`](../logic_layer/risk_metrics.py).

## Recomendaciones Didácticas

* Añade celdas narrativas que conecten cada métrica con decisiones de inversión reales.
* Incluye preguntas guía al final de cada notebook para fomentar la discusión.
* Guarda las versiones entrenadas del modelo (`models/random_forest.joblib`) y compara resultados entre sesiones.
