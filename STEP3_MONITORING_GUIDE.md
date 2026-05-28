# Guia Paso 3 - Monitoreo y Drift

## Objetivo del flujo

Automatizar de forma repetible:
- Generacion de lotes de monitoreo.
- Corridas periodicas de drift sobre lotes distintos.
- Construccion de historico temporal.
- Visualizacion de resultados en Streamlit.

Este flujo ayuda a detectar cambios en la poblacion que puedan degradar el modelo de riesgo.

## Script principal

Archivo: src/model_monitoring.py

Ejemplo minimo:

python src/model_monitoring.py

Ejemplo recomendado (5 corridas y reinicio de historial):

python src/model_monitoring.py --batch-runs 5 --reset-history

Abrir Streamlit (en comando separado):

python -m streamlit run src/streamlit_monitoring_app.py

## Parametros utiles

- --batch-runs N: cantidad de corridas/lotes (recomendado entre 3 y 5).
- --periodicity W|M|Q: granularidad de muestreo para monitoreo.
- --sample-fraction X: porcentaje de cada periodo usado para drift.
- --min-sample-size N: piso de muestra por corrida.
- --reset-history: limpia el historico antes de correr.
- --batches-dir RUTA: directorio temporal de lotes batch (default: monitoring_batches).

## Archivos de salida y como leerlos

Carpeta de salida: src/monitoring_outputs

1. drift_metrics_latest.csv
- Una fila por variable monitoreada.
- Campos clave:
  - drift_score: score agregado para severidad.
  - severity: low, medium, high.
  - psi, js_divergence, ks_pvalue, chi2_pvalue: metricas base.

2. monitoring_scored_sample_latest.csv
- Muestra monitoreada con predicciones.
- Incluye:
  - pred_proba_risk
  - pred_label_risk

3. monitoring_history.csv
- Evolucion temporal por corrida.
- Campos clave:
  - run_timestamp_utc
  - global_drift_score
  - high_severity_ratio
  - status (ok, warning, critical)

4. monitoring_summary_latest.json
- Resumen ejecutivo de la ultima corrida.
- Incluye recomendaciones automaticas y top de variables con mayor drift.

## Que se espera en una ejecucion correcta

- El script finaliza sin errores.
- Se generan/actualizan los cuatro archivos de salida.
- monitoring_history.csv crece una fila por corrida.
- El dashboard abre con datos y tendencias visibles.

## Interpretacion operativa rapida

- status = ok:
  Estabilidad general. Continuar monitoreo.

- status = warning:
  Revisar variables con severidad media/alta y observar siguiente corrida.

- status = critical:
  Alto riesgo de degradacion. Priorizar analisis de causa raiz y evaluar retraining.

## Error comun: streamlit command not found

Causa:
- Streamlit no esta instalado en el entorno activo, o el ejecutable no esta en PATH.

Solucion recomendada:

python -m pip install -r requirements.txt
python -m streamlit run src/streamlit_monitoring_app.py

Usar python -m streamlit evita depender del comando streamlit en PATH.
