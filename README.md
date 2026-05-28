# Prediccion de Comportamiento de Credito

Este proyecto construye y opera un modelo de riesgo crediticio para anticipar el comportamiento de pago de nuevos clientes a partir de historicos de credito.

## Caso de negocio

En una entidad financiera, decidir bien en originacion reduce perdidas por incumplimiento y mejora la asignacion de capital. El objetivo de esta solucion es apoyar decisiones de riesgo con evidencia cuantitativa, trazable y monitoreable en el tiempo.

Objetivos de negocio:
- Estimar riesgo de no pago para nuevos solicitantes.
- Mejorar decisiones de aprobacion y priorizacion de revisiones.
- Reducir degradacion silenciosa del modelo mediante monitoreo de drift.

## Objetivos analiticos

- Preparar y transformar datos de credito con reglas reproducibles.
- Entrenar y evaluar modelos supervisados con metricas adecuadas para desbalance.
- Optimizar umbral de decision con enfoque de costo.
- Monitorear cambios de poblacion que puedan afectar desempeno.

## Principales hallazgos

- Se detecto y corrigio leakage en variables que producian metricas artificialmente perfectas.
- Al definir la clase positiva como evento de riesgo (no pago), las metricas se volvieron consistentes con el problema de negocio.
- El umbral por defecto 0.5 no era optimo para riesgo; la optimizacion por costo mejoro recuperacion de eventos riesgosos.
- El monitoreo de drift se traduce en alertas tempranas para evitar deterioro silencioso del modelo.

Resumen ejecutivo de data drift:
- Hallazgo: estado global ok y drift agregado bajo en las corridas iniciales.
	Impacto: no se observa deterioro inmediato que comprometa decisiones de originacion.
	Accion: mantener operacion normal con monitoreo periodico.
- Hallazgo: no hay concentracion critica de variables en severidad alta.
	Impacto: estabilidad poblacional suficiente en la ventana evaluada.
	Accion: sostener seguimiento y confirmar consistencia en nuevas corridas.
- Hallazgo: variables financieras como total de otros prestamos, promedio de ingresos reportados y salario muestran mayor sensibilidad al cambio.
	Impacto: principal riesgo de mediano plazo por deriva gradual, no por ruptura abrupta.
	Accion: priorizar estas variables en el tablero y activar recalibracion/retraining si la severidad aumenta de forma sostenida.

## Graficas ejecutivas recomendadas

Para comunicar hallazgos a nivel directivo, se recomienda mostrar solo 4 graficas que expliquen por si mismas el riesgo, el impacto y la accion.

1. Estabilidad global del modelo en el tiempo.
	- Tipo: serie temporal del drift agregado con bandas ok/warning/critical.
	- Titulo: Riesgo de deriva global: estable, con vigilancia activa.
	- Subtitulo: Estado actual OK; no hay evidencia de deterioro abrupto.
	- Mensaje clave: la situacion es estable hoy, pero debe vigilarse la tendencia.
	- Grafica:

![Riesgo de deriva global](fig_01_estabilidad_global.png)

2. Variables que explican el riesgo de deriva.
	- Tipo: barras horizontales Top 5 variables con mayor severidad de drift.
	- Titulo: La deriva se concentra en pocas variables criticas.
	- Subtitulo: El riesgo no es generalizado; se focaliza en variables financieras.
	- Mensaje clave: priorizar control sobre total de otros prestamos, ingresos y salario.
	- Grafica:

![Variables criticas de drift](fig_02_variables_criticas.png)

3. Evidencia de deriva gradual (sin ruptura).
	- Tipo: comparacion de distribuciones referencia vs actual para una variable critica (ejemplo: salario).
	- Titulo: Cambio gradual en el perfil financiero observado.
	- Subtitulo: Se aprecia desplazamiento progresivo, no quiebre de poblacion.
	- Mensaje clave: el principal riesgo es acumulativo en el tiempo.
	- Grafica:

![Deriva gradual en salario](fig_03_deriva_gradual_salario.png)

4. Impacto de decision por umbral operativo.
	- Tipo: comparativo 0.5 vs umbral optimo en recall de riesgo y costo esperado.
	- Titulo: Ajustar el umbral reduce costo de error en originacion.
	- Subtitulo: El umbral optimizado mejora captura de casos riesgosos.
	- Mensaje clave: la mejora no es solo estadistica; impacta decision de negocio.
	- Grafica:

![Impacto del umbral operativo](fig_04_impacto_umbral.png)

## Descripcion del proceso

El flujo se ejecuta como una sola unidad operativa:

1. Carga y validacion de datos.
2. Analisis exploratorio y definicion de reglas de calidad.
3. Feature engineering reproducible.
4. Entrenamiento y evaluacion con auditoria.
5. Monitoreo continuo de drift y alertas.
6. Visualizacion ejecutiva para seguimiento de estabilidad.

## Componentes principales

- Carga inicial: src/Cargar_datos.ipynb
- EDA y reglas de calidad: src/Comprension_eda.ipynb
- Feature engineering: src/ft_engineering.py
- Entrenamiento y evaluacion: src/model_training_evaluation.py
- Monitoreo de drift: src/model_monitoring.py
- Dashboard de monitoreo: src/streamlit_monitoring_app.py
- Auditoria de entrenamiento: src/model_training_evaluation_audit.json

## Ejecucion tecnica (resumen)

Primero activa el entorno virtual del proyecto.

Ejemplo con venv local:

```bash
source .venv/bin/activate
```

Si necesitas reconstruir entorno desde cero, usa:

```bash
bash setup.sh
```

Luego ejecuta los scripts principales:

```bash
python src/ft_engineering.py
python src/model_training_evaluation.py
python src/model_monitoring.py --batch-runs 5 --reset-history
python -m streamlit run src/streamlit_monitoring_app.py
```

## Interpretacion de resultados

Para entrenamiento:
- Revisar src/model_training_evaluation_audit.json y su guia en src/model_training_evaluation_audit_guide.txt.
- Priorizar PR-AUC, recall de clase de riesgo y balanced accuracy sobre accuracy aislada.

Para monitoreo:
- Revisar src/monitoring_outputs/monitoring_summary_latest.json.
- Estado ok/warning/critical orienta acciones de seguimiento, recalibracion o retraining.
- Analizar tendencia en monitoring_history.csv para detectar cambios persistentes.

## Notas operativas

- Base_de_datos.csv es una fuente de ejemplo no productiva.
- Los artefactos de ejecucion de monitoreo no se versionan.
- En produccion, la ingestion y curacion de datos deben venir de procesos corporativos (DWH/Data Lake).
