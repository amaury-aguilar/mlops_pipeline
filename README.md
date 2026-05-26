# Proyecto Data Science - Prediccion de Comportamiento de Credito

Pipeline inicial de analisis para desarrollar un modelo predictivo con datos historicos de creditos, orientado a anticipar el comportamiento de nuevos usuarios.

## Objetivo de esta etapa

- Cargar y validar una fuente no productiva en CSV.
- Realizar comprension y analisis exploratorio de datos (EDA).
- Definir reglas de validacion de calidad para etapas posteriores.
- Identificar transformaciones y atributos derivados candidatos para modelado.

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
		└── Comprension_eda.ipynb
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

Se recomienda tener instalado al menos:

- Python 3.13+
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

Las dependencias se gestionan desde `requirements.txt` y el entorno virtual del proyecto.

## Ejecucion

1. Activar entorno virtual:

```bash
source mlops-venv/bin/activate
```

2. Levantar Jupyter:

```bash
jupyter notebook
```

3. Ejecutar notebooks en este orden:

1. `src/Cargar_datos.ipynb`
2. `src/Comprension_eda.ipynb`

## Notas importantes

- El CSV de este repositorio es una fuente de ejemplo no productiva.
- En una arquitectura empresarial, la capa de ingestion y curacion de datos se resuelve antes de la fase de modelado.
- Las reglas de validacion definidas en EDA deben migrarse a pruebas automatizadas de calidad en etapas siguientes.
