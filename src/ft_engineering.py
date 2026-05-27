"""
ft_engineering.py
-----------------
Feature engineering pipeline para el modelo de prediccion de comportamiento
de credito (Pago_atiempo).

Responsabilidades:
    1. Cargar y convertir tipos de dato.
    2. Generar features derivados (temporales y financieros).
    3. Aplicar un ColumnTransformer con tres ramas:
       - Continuas    -> SimpleImputer(mean)
       - Nominales    -> SimpleImputer(most_frequent) + OneHotEncoder
       - Ordinales    -> SimpleImputer(most_frequent) + OrdinalEncoder
    4. Retornar (X_transformed, y, transformer) listos para entrenamiento.

Uso como modulo:
    from ft_engineering import run_feature_engineering
    X, y, transformer = run_feature_engineering(input_path=Path("..."))

Uso directo (script):
    python src/ft_engineering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ---------------------------------------------------------------------------
# Configuracion de columnas del dominio
# ---------------------------------------------------------------------------

# Columna objetivo (se separa antes de transformar)
TARGET_COL: str = "Pago_atiempo"

# Columna de fecha original (se parsea, se derivan features y se descarta)
DATE_COL: str = "fecha_prestamo"

# Variables continuas: imputacion por MEDIA
# Incluye las features derivadas que se generan en derive_features()
CONTINUOUS_COLS: list[str] = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    # Features derivados de fecha
    "anio_prestamo",
    "mes_prestamo",
    "dia_semana_prestamo",
    # Features derivados financieros
    "ratio_deuda_ingreso",
    "ratio_cuota_ingreso",
]

# Variables categoricas nominales: imputacion por MODA + OneHotEncoder
# tipo_credito tiene valores 4/7/9 que son codigos de tipo, no magnitudes ordinales
NOMINAL_COLS: list[str] = [
    "tipo_laboral",
    "tipo_credito",
]

# Variables categoricas ordinales: imputacion por MODA + OrdinalEncoder
# El orden de las categorias debe ir de menor a mayor intensidad
ORDINAL_COLS: list[str] = ["tendencia_ingresos"]
ORDINAL_CATEGORIES: list[list[str]] = [["Decreciente", "Estable", "Creciente"]]


# ---------------------------------------------------------------------------
# Paso 1: Carga de datos
# ---------------------------------------------------------------------------


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Lee el CSV de datos crudos y retorna un DataFrame.
    Lanza FileNotFoundError si la ruta no existe.
    """
    # Validamos existencia del archivo antes de leerlo
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {data_path}")

    # Cargamos sin inferir tipos para controlar conversion manualmente
    df = pd.read_csv(data_path, dtype=str, keep_default_na=True)
    return df


# ---------------------------------------------------------------------------
# Paso 2: Conversion de tipos
# ---------------------------------------------------------------------------


def preprocess_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte cada columna a su tipo de dato correcto:
    - fecha_prestamo  -> datetime (formato DD/MM/AA HH:MM)
    - tipo_credito    -> string/object (codigo nominal, no magnitud)
    - columnas numericas -> float
    - target          -> Int64 (entero nullable para conservar nulos)

    Ademas unifica representaciones vacias o textuales de nulo a np.nan.
    """
    df = df.copy()

    # Tokens que representan ausencia de valor en el CSV
    null_tokens = ["", " ", "NA", "N/A", "null", "NULL", "None", "none", "nan", "NaN"]
    df = df.replace(null_tokens, np.nan)

    # Convertimos la fecha con formato DD/MM/YY HH:MM (formato del CSV fuente)
    # Especificamos el formato para evitar inferencia lenta y warnings de ambiguedad
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(
            df[DATE_COL], format="%d/%m/%y %H:%M", errors="coerce"
        )

    # tipo_credito se mantiene como string para OneHotEncoder (es nominal)
    if "tipo_credito" in df.columns:
        df["tipo_credito"] = df["tipo_credito"].where(df["tipo_credito"].isna(), df["tipo_credito"].astype(str))

    # Convertimos todas las numericas esperadas a float
    numeric_cols_to_convert = [
        c for c in CONTINUOUS_COLS if c in df.columns
    ]
    for col in numeric_cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convertimos target a entero nullable (Int64 admite pd.NA)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")

    return df


# ---------------------------------------------------------------------------
# Paso 3: Generacion de features derivados
# ---------------------------------------------------------------------------


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera atributos nuevos a partir de columnas existentes:

    De fecha_prestamo:
        anio_prestamo       -> anio numerico del desembolso
        mes_prestamo        -> mes (1-12), captura estacionalidad
        dia_semana_prestamo -> dia de la semana (0=lunes), captura patron laboral

    Ratios financieros:
        ratio_deuda_ingreso  -> total_otros_prestamos / salario_cliente
        ratio_cuota_ingreso  -> cuota_pactada / salario_cliente (esfuerzo de pago)

    Elimina la columna de fecha original tras extraer los features utiles.
    """
    df = df.copy()

    # Extraemos componentes temporales del datetime ya convertido
    if DATE_COL in df.columns:
        df["anio_prestamo"] = df[DATE_COL].dt.year.astype("float64")
        df["mes_prestamo"] = df[DATE_COL].dt.month.astype("float64")
        df["dia_semana_prestamo"] = df[DATE_COL].dt.dayofweek.astype("float64")

        # Descartamos la fecha original: ya no aporta informacion adicional
        df = df.drop(columns=[DATE_COL])

    # Ratio deuda / ingreso: indica nivel de apalancamiento del cliente
    if {"total_otros_prestamos", "salario_cliente"}.issubset(df.columns):
        df["ratio_deuda_ingreso"] = np.where(
            df["salario_cliente"].notna() & (df["salario_cliente"] > 0),
            df["total_otros_prestamos"] / df["salario_cliente"],
            np.nan,
        )

    # Ratio cuota / ingreso: mide la carga de la cuota sobre el ingreso mensual
    if {"cuota_pactada", "salario_cliente"}.issubset(df.columns):
        df["ratio_cuota_ingreso"] = np.where(
            df["salario_cliente"].notna() & (df["salario_cliente"] > 0),
            df["cuota_pactada"] / df["salario_cliente"],
            np.nan,
        )

    return df


# ---------------------------------------------------------------------------
# Paso 4: Construccion del ColumnTransformer
# ---------------------------------------------------------------------------


def build_transformer(
    continuous_cols: list[str],
    nominal_cols: list[str],
    ordinal_cols: list[str],
    ordinal_categories: list[list[str]],
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer con tres ramas de transformacion:

    Rama "continuous":
        SimpleImputer(strategy="mean")
        -> rellena nulos de variables continuas con la media de entrenamiento

    Rama "nominal":
        SimpleImputer(strategy="most_frequent")
        -> rellena nulos categoricos con la moda
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        -> codifica categorias nominales en columnas binarias

    Rama "ordinal":
        SimpleImputer(strategy="most_frequent")
        -> rellena nulos ordinales con la moda
        OrdinalEncoder(categories=ordinal_categories)
        -> asigna entero segun el orden definido (ej. Decreciente=0, Estable=1, Creciente=2)

    remainder="drop": cualquier columna no declarada explicitamente se descarta.
    """

    # --- Rama continua: solo imputacion por media ---
    continuous_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    # --- Rama nominal: moda + codificacion OHE ---
    nominal_pipeline = Pipeline(steps=[
        # Imputamos con el valor mas frecuente antes de codificar
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # OHE genera una columna binaria por cada categoria observada
        # handle_unknown="ignore" descarta categorias nuevas en inferencia sin error
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # --- Rama ordinal: moda + codificacion ordinal con orden explicito ---
    ordinal_pipeline = Pipeline(steps=[
        # Imputamos con el valor mas frecuente antes de codificar
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # OrdinalEncoder respeta el orden de las categorias definido en ordinal_categories
        # unknown_value=-1 asigna -1 a categorias no vistas en entrenamiento
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, continuous_cols),
            ("nominal",    nominal_pipeline,    nominal_cols),
            ("ordinal",    ordinal_pipeline,    ordinal_cols),
        ],
        # Descartamos columnas no declaradas (ej. columnas auxiliares no usadas)
        remainder="drop",
        # verbose_feature_names_out=False genera nombres limpios sin prefijo del transformer
        verbose_feature_names_out=False,
    )

    return transformer


# ---------------------------------------------------------------------------
# Paso 5: Pipeline completo
# ---------------------------------------------------------------------------


def run_feature_engineering(
    input_path: Path,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Ejecuta el pipeline completo de feature engineering.

    Pasos:
        1. Carga datos crudos desde CSV.
        2. Convierte tipos de dato.
        3. Genera features derivados.
        4. Ajusta y aplica el ColumnTransformer.
        5. Retorna X transformado, y objetivo y el transformer ajustado.

    Parametros
    ----------
    input_path : Path
        Ruta al CSV de datos crudos.
    output_path : Path | None
        Si se indica, exporta el dataset procesado (X + y) a un CSV.

    Retorna
    -------
    X_transformed : pd.DataFrame
        Dataset con todos los features transformados e imputados.
    y : pd.Series
        Variable objetivo (Pago_atiempo) sin transformar.
    transformer : ColumnTransformer
        Transformer ajustado sobre los datos de entrenamiento.
        Se puede reutilizar con transformer.transform(X_new) en inferencia.
    """

    print("[1/5] Cargando datos crudos...")
    df = load_raw_data(input_path)
    print(f"      Dimensiones originales: {df.shape}")

    print("[2/5] Convirtiendo tipos de dato...")
    df = preprocess_dtypes(df)

    print("[3/5] Generando features derivados...")
    df = derive_features(df)
    print(f"      Columnas disponibles: {sorted(df.columns.tolist())}")

    # Separamos target antes de transformar para no incluirlo en X
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Columna objetivo '{TARGET_COL}' no encontrada. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # y: serie con la variable objetivo; se convierte a int estandar para sklearn
    y: pd.Series = df[TARGET_COL].copy()

    # X: matriz de features sin el target
    X: pd.DataFrame = df.drop(columns=[TARGET_COL])

    # Filtramos cada lista de columnas segun lo que realmente existe en X
    # (evita error si alguna columna derivada no pudo generarse)
    continuous_present = [c for c in CONTINUOUS_COLS if c in X.columns]
    nominal_present    = [c for c in NOMINAL_COLS    if c in X.columns]
    ordinal_present    = [c for c in ORDINAL_COLS    if c in X.columns]
    ordinal_cats_present = [
        cats
        for col, cats in zip(ORDINAL_COLS, ORDINAL_CATEGORIES)
        if col in X.columns
    ]

    print(
        f"[4/5] Construyendo ColumnTransformer...\n"
        f"      Continuas : {len(continuous_present)} columnas\n"
        f"      Nominales : {nominal_present}\n"
        f"      Ordinales : {ordinal_present}"
    )

    transformer = build_transformer(
        continuous_cols=continuous_present,
        nominal_cols=nominal_present,
        ordinal_cols=ordinal_present,
        ordinal_categories=ordinal_cats_present,
    )

    # Ajustamos y transformamos en un solo paso (fit_transform sobre datos de entrenamiento)
    X_array = transformer.fit_transform(X)

    # Recuperamos nombres de columnas generados por el transformer
    # get_feature_names_out() incluye los nombres OHE dinamicos (ej. tipo_laboral_Empleado)
    col_names: list[str] = list(transformer.get_feature_names_out())

    # Reconstruimos DataFrame con indices originales para trazabilidad
    X_transformed = pd.DataFrame(X_array, columns=col_names, index=X.index)

    print(f"[5/5] Transformacion completa.")
    print(f"      Shape final de X: {X_transformed.shape}")
    print(f"      Columnas generadas: {col_names}")

    # Exportamos si se indico ruta de salida
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Combinamos X transformado con target para tener un CSV autosuficiente
        export_df = X_transformed.copy()
        export_df[TARGET_COL] = y.values
        export_df.to_csv(output_path, index=False)
        print(f"      Dataset guardado en: {output_path}")

    return X_transformed, y, transformer


# ---------------------------------------------------------------------------
# Ejecucion directa como script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # La raiz del proyecto es un nivel arriba de src/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    INPUT_PATH  = PROJECT_ROOT / "Base_de_datos.csv"
    OUTPUT_PATH = PROJECT_ROOT / "features_engineered.csv"

    X, y, fitted_transformer = run_feature_engineering(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
    )

    print("\n--- Primeras filas del dataset transformado ---")
    print(X.head())

    print("\n--- Distribucion del target ---")
    print(y.value_counts())
