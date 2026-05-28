# Proyecto Data Science - Prediccion de Comportamiento de Credito

Pipeline inicial de analisis para desarrollar un modelo predictivo con datos historicos de creditos, orientado a anticipar el comportamiento de nuevos usuarios.

## Caso de negocio

Empresa financiera que requiere anticipar el comportamiento de pago de nuevos clientes para mejorar decisiones de originacion, priorizacion de revisiones y gestion de riesgo.

Objetivo analitico:
- Predecir riesgo de no pago usando historial de creditos.
- Optimizar umbral de decision con enfoque de costo de negocio.
- Monitorear drift de datos para evitar degradacion silenciosa del modelo.

## Objetivo de esta etapa

- Cargar y validar una fuente no productiva en CSV.
- Realizar comprension y analisis exploratorio de datos (EDA).
- Definir reglas de validacion de calidad para etapas posteriores.
- Identificar transformaciones y atributos derivados candidatos para modelado.
- Implementar monitoreo de drift y visualizacion ejecutiva en Streamlit.

## Estructura actual del repositorio

```text
mlops_pipeline/
├── Base_de_datos.csv
├── README.md
├── requirements.txt
├── setup.sh
├── mlops-venv/
└── src/
	├── Cargar_datos.ipynb
	├── Comprension_eda.ipynb
	├── ft_engineering.py
	├── model_training_evaluation.py
	├── model_monitoring.py
	├── streamlit_monitoring_app.py
	├── model_training_evaluation_audit.json
	└── model_training_evaluation_audit_guide.txt
```

## Notebooks de la etapa

### 1) src/Cargar_datos.ipynb

Notebook de carga y validacion inicial del dataset.

- Contextualiza por que en produccion la data deberia venir de DWH/Data Lake.
- Carga `Base_de_datos.csv` desde la raiz del proyecto.
- Ejecuta chequeos minimos de calidad: dimensiones, duplicados y nulos.
- Deja el DataFrame listo para pasar al notebook de EDA.

### 2) src/Comprension_eda.ipynb

Notebook de analisis experimental y exploratorio de datos.

- Exploracion inicial y caracterizacion de variables (numericas, categoricas, dicotomicas, fecha).
- Unificacion de representaciones de nulos.
- Eliminacion de variables irrelevantes por criterios objetivos (constantes o casi vacias).
- Correccion y conversion de tipos de datos por columna.
- Analisis univariable:
	- `describe()` para numericas y categoricas.
	- histogramas y boxplots para numericas.
	- countplots, `value_counts()` y tablas de frecuencia para categoricas.
	- estadisticos: tendencia central, dispersion, skewness y kurtosis.
- Analisis bivariable contra la variable objetivo (`Pago_atiempo`).
- Analisis multivariable: matriz de correlacion, pares con alta correlacion y `pairplot`.
- Propuesta de reglas de validacion de datos para pipeline.
- Identificacion de atributos derivados y transformaciones candidatas.

## Requisitos

La opcion mas estable y parsimoniosa para este proyecto es usar un unico entorno virtual `mlops-venv` con Python 3.11.x.

- Python 3.11.x
- manifiesto conda en `environment.yml`
- dependencias fijadas en `requirements.txt`

Motivo:
- `PyCaret 3.3.2` no es compatible de forma estable con Python 3.13.
- Con Python 3.11 se mantiene compatibilidad entre `PyCaret`, `scikit-learn`, `xgboost`, notebooks y el resto del stack actual.

## Manifiesto del entorno

`environment.yml` es el contrato declarativo del entorno conda del proyecto.

- Define version de Python y herramientas base (`python=3.11`, `pip`).
- Declara el origen de las dependencias de Python usando `pip` con `requirements.txt`.
- Permite reconstruir el mismo entorno de forma consistente en otras maquinas o en CI.

## Preparacion del entorno

Si vas a reconstruir el entorno original compatible, ejecuta:

```bash
bash setup.sh
```

Ese script reconstruye `mlops-venv` usando un flujo 100% conda, instala dependencias y registra el kernel de Jupyter.

Si tu binario de conda no esta en `/Users/amaury/miniconda3/bin/conda`, ejecuta:

```bash
CONDA_BIN=/ruta/a/conda bash setup.sh
```

## Como probar los scripts

Probar feature engineering:

```bash
"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" python src/ft_engineering.py
```

Ese comando:
- carga `Base_de_datos.csv`
- genera variables derivadas
- aplica imputacion y codificacion
- guarda `features_engineered.csv` en la raiz del proyecto

Probar modelamiento y evaluacion:

```bash
"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" python src/model_training_evaluation.py
```

Ese comando:
- selecciona candidatos iniciales con `PyCaret` si esta disponible
- evalua mezcla de modelos
- optimiza el candidato final con `scikit-learn`
- guarda el resumen en `src/model_training_evaluation_audit.json`

Probar monitoreo de drift (Paso 3):

```bash
"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" python src/model_monitoring.py
```

Ese comando:
- construye muestra periodica del lote actual
- adjunta pronosticos (o genera proxy auditable si no vienen del scoring productivo)
- calcula KS, PSI, Jensen-Shannon y Chi-cuadrado por variable
- guarda reportes en `src/monitoring_outputs/`

Levantar dashboard de monitoreo en Streamlit:

```bash
"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" streamlit run src/streamlit_monitoring_app.py
```

La app incluye:
- comparacion historico vs actual
- tabla de drift por variable con severidad
- evolucion temporal del drift
- recomendaciones automaticas (semaforo: ok / warning / critical)

Para notebooks:

```bash
"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" jupyter notebook
```

Orden recomendado:

1. `src/Cargar_datos.ipynb`
2. `src/Comprension_eda.ipynb`
3. `"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" python src/ft_engineering.py`
4. `"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" python src/model_training_evaluation.py`
5. `"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" python src/model_monitoring.py`
6. `"/Users/amaury/miniconda3/bin/conda" run -p "$PWD/mlops-venv" streamlit run src/streamlit_monitoring_app.py`

## Como leer el archivo de auditoria

`src/model_training_evaluation_audit.json` es el resumen estructurado de una corrida de entrenamiento.

Contiene:
- tamano total de la muestra, train y test
- numero de folds usados en validacion cruzada
- resultado de la seleccion inicial con `PyCaret` o fallback
- mejores hiperparametros encontrados en `scikit-learn`
- metricas CV con media y desviacion estandar
- metricas finales en holdout test
- ranking resumido de los mejores modelos evaluados

Para una guia campo por campo, revisa `src/model_training_evaluation_audit_guide.txt`.

## Notas importantes

- El CSV de este repositorio es una fuente de ejemplo no productiva.
- En una arquitectura empresarial, la capa de ingestion y curacion de datos se resuelve antes de la fase de modelado.
- Las reglas de validacion definidas en EDA deben migrarse a pruebas automatizadas de calidad en etapas siguientes.
- Si aparecen metricas perfectas (`1.0` en todo), revisa posible leakage de negocio o variables con informacion post-evento.
- Los reportes de monitoreo en `src/monitoring_outputs/` son artefactos generados y no deben versionarse.
