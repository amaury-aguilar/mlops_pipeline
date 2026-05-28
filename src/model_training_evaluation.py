# Declaramos el modulo con enfoque de entrenamiento y evaluacion de modelos de clasificacion.
"""
Script de modelamiento y evaluacion supervisada.

Objetivo:
- Usar PyCaret solo como punto de partida para comparar modelos y proponer candidato.
- Evaluar si mezclar modelos (blending) mejora el candidato inicial.
- Optimizar el modelo seleccionado de forma 100% auditable con scikit-learn
  (RandomizedSearchCV/GridSearchCV) evitando data leakage.
- Reportar validacion cruzada con media y desviacion estandar de metricas.

Notas de diseno para evitar data leakage:
- La separacion train/test se hace antes de cualquier ajuste de preprocesamiento.
- El preprocesamiento para el ajuste final vive dentro de un Pipeline de sklearn,
  de modo que en cada fold de CV se ajusta solo con datos de entrenamiento del fold.
- El conjunto de test se usa una sola vez al final para evaluacion holdout.
"""

# Habilitamos anotaciones modernas para tipado.
from __future__ import annotations

# Importamos librerias base para rutas y serializacion opcional.
from pathlib import Path
import json
import sys

# Importamos librerias numericas y tabulares.
import numpy as np
import pandas as pd

# Importamos utilidades de sklearn para modelamiento, validacion y metricas.
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# Importamos modelos auditables y comunes para clasificacion supervisada.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# Importamos XGBoost solo si esta disponible en requirements (ya existe en tu proyecto).
from xgboost import XGBClassifier

# Importamos piezas de feature engineering ya existentes en el proyecto.
from ft_engineering import (
    TARGET_COL,
    CONTINUOUS_COLS,
    NOMINAL_COLS,
    ORDINAL_COLS,
    ORDINAL_CATEGORIES,
    load_raw_data,
    preprocess_dtypes,
    derive_features,
    build_transformer,
)


# Definimos semilla global para reproducibilidad de splits, CV y busquedas.
RANDOM_STATE: int = 42


# Funcion para elegir numero de folds en funcion del tamano muestral y clase minoritaria.
def choose_folds(y_train: pd.Series) -> int:
    # Calculamos cantidad de filas de entrenamiento.
    n_samples = len(y_train)

    # Calculamos conteo por clase para respetar restriccion de StratifiedKFold.
    class_counts = y_train.value_counts(dropna=False)

    # Obtenemos tamano de clase minoritaria.
    minority_count = int(class_counts.min())

    # Definimos base de folds por tamano muestral.
    if n_samples < 1500:
        # Muestra pequena: menos folds para menor varianza por fold.
        base_folds = 5
    elif n_samples < 6000:
        # Muestra mediana: equilibrio costo/estabilidad.
        base_folds = 7
    else:
        # Muestra grande: maximo recomendado para mayor robustez.
        base_folds = 10

    # Ajustamos folds para no superar la cantidad de ejemplos de la clase minoritaria.
    folds = min(base_folds, minority_count)

    # Garantizamos minimo de 3 folds para tener estimacion de desviacion estandar util.
    folds = max(3, folds)

    # Devolvemos folds finales.
    return folds


# Funcion para cargar y preparar el dataset sin transformar (aun), evitando leakage.
def load_feature_frame(input_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    # Cargamos datos crudos desde el CSV.
    df = load_raw_data(input_path)

    # Estandarizamos tipos y nulos de forma consistente con el pipeline previo.
    df = preprocess_dtypes(df)

    # Generamos variables derivadas (temporales y ratios) antes del modelamiento.
    df = derive_features(df)

    # Validamos presencia de la variable objetivo.
    if TARGET_COL not in df.columns:
        raise ValueError(f"No existe la columna objetivo '{TARGET_COL}' en el dataset.")

    # Filtramos registros con objetivo no nulo para entrenamiento supervisado.
    df = df[df[TARGET_COL].notna()].copy()

    # Separamos matriz de features.
    X = df.drop(columns=[TARGET_COL])

    # Separamos vector objetivo y lo convertimos a entero clasico para sklearn.
    y = df[TARGET_COL].astype(int)

    # Retornamos features y target sin leakage.
    return X, y


# Funcion para construir listas de columnas presentes en la muestra de entrenamiento.
def get_present_columns(X_train: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    # Detectamos continuas presentes.
    continuous_present = [c for c in CONTINUOUS_COLS if c in X_train.columns]

    # Detectamos nominales presentes.
    nominal_present = [c for c in NOMINAL_COLS if c in X_train.columns]

    # Detectamos ordinales presentes.
    ordinal_present = [c for c in ORDINAL_COLS if c in X_train.columns]

    # Sincronizamos categorias ordinales segun columnas presentes.
    ordinal_cats_present = [
        cats
        for col, cats in zip(ORDINAL_COLS, ORDINAL_CATEGORIES)
        if col in X_train.columns
    ]

    # Retornamos estructura final de columnas para el preprocesador.
    return continuous_present, nominal_present, ordinal_present, ordinal_cats_present


# Funcion que ejecuta PyCaret solo para comparar candidatos iniciales y blending.
def pycaret_candidate_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int,
) -> tuple[object, pd.DataFrame, dict]:
    # Importamos PyCaret dentro de la funcion para mantener auditable el flujo y fallar con mensaje claro.
    try:
        # Importamos funciones de clasificacion necesarias en PyCaret.
        from pycaret.classification import setup, compare_models, blend_models, pull
    except Exception as exc:
        # Si PyCaret no esta disponible, usamos fallback auditable con sklearn.
        return sklearn_candidate_selection_fallback(X_train=X_train, y_train=y_train, folds=folds, import_error=exc)

    # Construimos dataframe de entrenamiento con target para PyCaret.
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    # Inicializamos setup de PyCaret solo con train para evitar tocar test.
    setup(
        # Entregamos dataset de entrenamiento.
        data=train_df,
        # Indicamos columna objetivo.
        target=TARGET_COL,
        # Usamos metrica principal por desbalance.
        session_id=RANDOM_STATE,
        # Definimos estrategia estratificada de CV.
        fold_strategy="stratifiedkfold",
        # Definimos folds adaptativos.
        fold=folds,
        # Silenciamos prompts interactivos para ejecucion automatizada.
        silent=True,
        # Desactivamos html para scripts no-notebook.
        html=False,
        # Mantenemos transform_target desactivado por tratarse de clasificacion.
        preprocess=True,
        # Evitamos logs automaticos externos para trazabilidad local simple.
        log_experiment=False,
        # Reducimos verbosidad.
        verbose=False,
    )

    # Comparamos modelos base y seleccionamos top 5 por AUC.
    top_models = compare_models(sort="AUC", n_select=5)

    # Extraemos leaderboard de comparacion.
    leaderboard_df = pull().copy()

    # Definimos mejor modelo base como el primero de la lista.
    best_base_model = top_models[0] if isinstance(top_models, list) else top_models

    # Recuperamos AUC base desde leaderboard (fila 0 = mejor segun compare_models).
    base_auc = float(leaderboard_df.iloc[0]["AUC"]) if "AUC" in leaderboard_df.columns else np.nan

    # Intentamos blending con los 3 mejores para evaluar mejora por mezcla.
    blend_model = None
    blend_auc = np.nan
    blending_used = False

    # Ejecutamos blend solo si hay al menos 2 modelos comparados.
    if isinstance(top_models, list) and len(top_models) >= 2:
        # Entrenamos blend de los top 3 optimizando por AUC con misma CV.
        blend_model = blend_models(
            estimator_list=top_models[:3],
            fold=folds,
            optimize="AUC",
            verbose=False,
        )

        # Obtenemos resumen de metricas del blend.
        blend_metrics_df = pull().copy()

        # Leemos AUC de blending si existe en salida.
        if "AUC" in blend_metrics_df.columns and len(blend_metrics_df) > 0:
            blend_auc = float(blend_metrics_df.iloc[0]["AUC"])

        # Marcamos que se intento blending.
        blending_used = True

    # Elegimos candidato final de PyCaret en funcion del mejor AUC.
    if blending_used and not np.isnan(blend_auc) and not np.isnan(base_auc) and blend_auc > base_auc:
        # Si blending mejora, se usa blend como candidato.
        selected_model = blend_model
        selected_origin = "blending_top3_pycaret"
        selected_auc = blend_auc
    else:
        # Si no mejora, se mantiene el mejor base.
        selected_model = best_base_model
        selected_origin = "best_single_model_pycaret"
        selected_auc = base_auc

    # Construimos metadata auditable de la seleccion inicial.
    selection_info = {
        "selected_origin": selected_origin,
        "selected_auc_pycaret": float(selected_auc) if not np.isnan(selected_auc) else None,
        "base_auc_pycaret": float(base_auc) if not np.isnan(base_auc) else None,
        "blend_auc_pycaret": float(blend_auc) if not np.isnan(blend_auc) else None,
        "blending_used": blending_used,
        "folds": folds,
    }

    # Retornamos candidato, leaderboard y metadata.
    return selected_model, leaderboard_df, selection_info


# Funcion fallback para seleccion inicial cuando PyCaret no esta disponible.
def sklearn_candidate_selection_fallback(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int,
    import_error: Exception,
) -> tuple[object, pd.DataFrame, dict]:
    # Detectamos columnas presentes para preprocesar correctamente dentro de CV.
    continuous_present, nominal_present, ordinal_present, ordinal_cats_present = get_present_columns(X_train)

    # Construimos preprocesador compartido para todos los candidatos.
    preprocessor = build_transformer(
        continuous_cols=continuous_present,
        nominal_cols=nominal_present,
        ordinal_cols=ordinal_present,
        ordinal_categories=ordinal_cats_present,
    )

    # Definimos candidatos base auditables y ligeros.
    candidates = {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "xgb": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1, use_label_encoder=False),
    }

    # Definimos validacion cruzada estratificada para comparacion justa.
    cv_strategy = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    # Lista para guardar resultados de comparacion tipo leaderboard.
    rows: list[dict] = []

    # Evaluamos cada candidato con ROC AUC en CV.
    for name, estimator in candidates.items():
        # Construimos pipeline para evitar leakage durante los folds.
        pipe = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )

        # Medimos AUC en CV del candidato.
        scores = cross_validate(
            estimator=pipe,
            X=X_train,
            y=y_train,
            scoring={"roc_auc": "roc_auc"},
            cv=cv_strategy,
            n_jobs=-1,
            return_train_score=False,
        )

        # Registramos media y desviacion estandar por modelo.
        rows.append(
            {
                "Model": name,
                "AUC": float(np.mean(scores["test_roc_auc"])),
                "AUC_std": float(np.std(scores["test_roc_auc"], ddof=1)),
            }
        )

    # Construimos leaderboard ordenado por AUC descendente.
    leaderboard_df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    # Recuperamos nombres de los top 3 para intentar blending.
    top3_names = leaderboard_df["Model"].head(3).tolist()

    # Construimos lista de estimadores para VotingClassifier.
    top3_estimators = [(name, clone(candidates[name])) for name in top3_names]

    # Definimos blend por votacion suave para comparar contra mejor individual.
    blend = VotingClassifier(estimators=top3_estimators, voting="soft", n_jobs=-1)

    # Construimos pipeline de blend con el mismo preprocesador.
    blend_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", blend),
        ]
    )

    # Medimos AUC del blend en CV.
    blend_scores = cross_validate(
        estimator=blend_pipe,
        X=X_train,
        y=y_train,
        scoring={"roc_auc": "roc_auc"},
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=False,
    )

    # Calculamos estadisticos del blending.
    blend_auc = float(np.mean(blend_scores["test_roc_auc"]))
    blend_auc_std = float(np.std(blend_scores["test_roc_auc"], ddof=1))

    # Leemos mejor AUC individual.
    best_single_auc = float(leaderboard_df.iloc[0]["AUC"])

    # Decidimos si mezcla mejora al mejor individual.
    if blend_auc > best_single_auc:
        # Si mejora, seleccionamos blend como candidato para etapa siguiente.
        selected_model = blend
        selected_origin = "blending_top3_sklearn_fallback"
        selected_auc = blend_auc
    else:
        # Si no mejora, seleccionamos mejor individual.
        best_name = str(leaderboard_df.iloc[0]["Model"])
        selected_model = candidates[best_name]
        selected_origin = "best_single_model_sklearn_fallback"
        selected_auc = best_single_auc

    # Agregamos fila de blend al leaderboard para trazabilidad.
    leaderboard_df = pd.concat(
        [
            leaderboard_df,
            pd.DataFrame([
                {
                    "Model": "blend_top3",
                    "AUC": blend_auc,
                    "AUC_std": blend_auc_std,
                }
            ]),
        ],
        ignore_index=True,
    )

    # Construimos metadata de seleccion inicial.
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    pycaret_recommendation = (
        "PyCaret 3.3.2 suele funcionar en Python 3.10/3.11. "
        "Si estas en Python 3.13, crea un venv Python 3.11 para habilitar PyCaret."
    )

    selection_info = {
        "selected_origin": selected_origin,
        "selected_auc_pycaret": None,
        "base_auc_pycaret": None,
        "blend_auc_pycaret": None,
        "blending_used": True,
        "folds": folds,
        "fallback_used": True,
        "fallback_reason": f"PyCaret no disponible en Python {py_ver}: {import_error}",
        "pycaret_recommendation": pycaret_recommendation,
        "selected_auc_fallback": selected_auc,
    }

    # Retornamos candidato y tabla de comparacion compatible con el flujo principal.
    return selected_model, leaderboard_df, selection_info


# Funcion para construir estimador sklearn auditable segun el candidato elegido por PyCaret.
def build_auditable_estimator(pycaret_model: object) -> object:
    # Obtenemos nombre de clase del modelo candidato para mapearlo a un estimador sklearn equivalente.
    model_name = pycaret_model.__class__.__name__

    # Si el candidato es un blend tipo VotingClassifier, tomamos el primer estimador base.
    # Esto permite mantener una optimizacion auditable sobre un modelo individual en sklearn.
    if isinstance(pycaret_model, VotingClassifier) and len(pycaret_model.estimators) > 0:
        first_base_estimator = pycaret_model.estimators[0][1]
        return build_auditable_estimator(first_base_estimator)

    # Mapeamos candidatos comunes a estimadores auditables explicitos.
    if "LogisticRegression" in model_name:
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)

    # Mapeamos random forest para clasificacion.
    if "RandomForest" in model_name:
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )

    # Mapeamos gradient boosting clasico.
    if "GradientBoosting" in model_name:
        return GradientBoostingClassifier(random_state=RANDOM_STATE)

    # Mapeamos XGBoost cuando el candidato venga de esa familia.
    if "XGB" in model_name or "XGBClassifier" in model_name:
        return XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            use_label_encoder=False,
        )

    # Fallback auditable a RandomForest si el tipo no se reconoce.
    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


# Funcion para definir espacio de busqueda sklearn segun el estimador final.
def build_search_space(estimator: object) -> tuple[str, dict, int]:
    # Caso LogisticRegression: grid pequeno, totalmente interpretable.
    if isinstance(estimator, LogisticRegression):
        # Definimos grid de parametros auditable.
        param_grid = {
            "model__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
        }

        # Retornamos tipo de busqueda, grid y numero de iteraciones no usado para grid.
        return "grid", param_grid, 0

    # Caso RandomForest: random search sobre espacio discreto controlado.
    if isinstance(estimator, RandomForestClassifier):
        # Definimos espacio de busqueda discreto.
        param_dist = {
            "model__n_estimators": [150, 250, 400, 600],
            "model__max_depth": [None, 4, 6, 8, 12],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        }

        # Retornamos random search con numero de iteraciones auditado.
        return "random", param_dist, 35

    # Caso GradientBoosting: random search compacto.
    if isinstance(estimator, GradientBoostingClassifier):
        # Definimos espacio de hiperparametros.
        param_dist = {
            "model__n_estimators": [100, 150, 250, 350],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.7, 0.85, 1.0],
        }

        # Retornamos random search.
        return "random", param_dist, 30

    # Caso XGBoost: random search controlado y auditable.
    if isinstance(estimator, XGBClassifier):
        # Definimos espacio discreto para control de complejidad.
        param_dist = {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [3, 4, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_alpha": [0.0, 0.1, 0.5],
            "model__reg_lambda": [1.0, 2.0, 5.0],
        }

        # Retornamos random search con iteraciones acotadas.
        return "random", param_dist, 35

    # Fallback: grid simple para cualquier estimador no contemplado.
    fallback_grid = {}
    return "grid", fallback_grid, 0


# Funcion para entrenar y optimizar modelo final con sklearn y CV sin leakage.
def train_and_optimize_with_sklearn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: int,
    pycaret_selected_model: object,
) -> tuple[Pipeline, dict, object]:
    # Detectamos columnas presentes para construir preprocesador.
    continuous_present, nominal_present, ordinal_present, ordinal_cats_present = get_present_columns(X_train)

    # Construimos preprocesador con mismas reglas de imputacion/codificacion.
    preprocessor = build_transformer(
        continuous_cols=continuous_present,
        nominal_cols=nominal_present,
        ordinal_cols=ordinal_present,
        ordinal_categories=ordinal_cats_present,
    )

    # Construimos estimador sklearn auditable equivalente al candidato PyCaret.
    base_estimator = build_auditable_estimator(pycaret_selected_model)

    # Encapsulamos preprocesamiento + modelo en un unico Pipeline para evitar leakage en CV.
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(base_estimator)),
        ]
    )

    # Definimos tipo de busqueda y espacio de hiperparametros.
    search_type, search_space, n_iter = build_search_space(base_estimator)

    # Definimos estrategia de validacion cruzada estratificada.
    cv_strategy = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    # Definimos metrica principal para optimizacion.
    primary_scoring = "roc_auc"

    # Elegimos GridSearchCV cuando corresponde.
    if search_type == "grid":
        # Construimos GridSearchCV auditable y deterministico.
        searcher = GridSearchCV(
            estimator=full_pipeline,
            param_grid=search_space,
            scoring=primary_scoring,
            cv=cv_strategy,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
    else:
        # Construimos RandomizedSearchCV auditable con semilla fija.
        searcher = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring=primary_scoring,
            cv=cv_strategy,
            n_jobs=-1,
            refit=True,
            random_state=RANDOM_STATE,
            verbose=1,
        )

    # Ajustamos la busqueda solo con train.
    searcher.fit(X_train, y_train)

    # Recuperamos mejor pipeline ya refiteado.
    best_pipeline = searcher.best_estimator_

    # Definimos conjunto de metricas para CV reportando media y desviacion estandar.
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }

    # Ejecutamos cross_validate sobre train para estimar estabilidad del pipeline final.
    cv_results = cross_validate(
        estimator=best_pipeline,
        X=X_train,
        y=y_train,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=False,
    )

    # Consolidamos metricas con media y desviacion estandar.
    cv_summary = {
        "folds": folds,
        "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_results["test_roc_auc"], ddof=1)),
        "f1_mean": float(np.mean(cv_results["test_f1"])),
        "f1_std": float(np.std(cv_results["test_f1"], ddof=1)),
        "precision_mean": float(np.mean(cv_results["test_precision"])),
        "precision_std": float(np.std(cv_results["test_precision"], ddof=1)),
        "recall_mean": float(np.mean(cv_results["test_recall"])),
        "recall_std": float(np.std(cv_results["test_recall"], ddof=1)),
        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"], ddof=1)),
    }

    # Retornamos mejor pipeline, resumen de CV y objeto de busqueda.
    return best_pipeline, cv_summary, searcher


# Funcion para evaluar en holdout (test) sin reentrenar ni recalibrar.
def evaluate_on_holdout(best_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    # Obtenemos probabilidades de la clase positiva cuando esten disponibles.
    if hasattr(best_pipeline, "predict_proba"):
        # Calculamos probabilidad de clase 1.
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    else:
        # Si no hay predict_proba, usamos decision_function y la normalizamos de forma simple.
        scores = best_pipeline.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    # Obtenemos predicciones binarias con umbral por defecto 0.5.
    y_pred = (y_proba >= 0.5).astype(int)

    # Calculamos metricas principales en test.
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }

    # Retornamos metricas de holdout.
    return metrics


# Funcion principal que orquesta seleccion inicial PyCaret + optimizacion sklearn auditable.
def run_training_and_evaluation(
    input_path: Path,
    test_size: float = 0.20,
    save_audit_json_path: Path | None = None,
) -> dict:
    # Cargamos y preparamos dataframe con features sin transformar.
    X, y = load_feature_frame(input_path)

    # Realizamos split estratificado antes de cualquier ajuste de modelos.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Elegimos folds de CV de forma adaptativa al tamano de train y clase minoritaria.
    folds = choose_folds(y_train)

    # Ejecutamos seleccion inicial con PyCaret (incluye evaluar blending).
    pycaret_selected_model, leaderboard_df, selection_info = pycaret_candidate_selection(
        X_train=X_train,
        y_train=y_train,
        folds=folds,
    )

    # Entrenamos y optimizamos candidato en sklearn (auditable) sin leakage.
    best_pipeline, cv_summary, searcher = train_and_optimize_with_sklearn(
        X_train=X_train,
        y_train=y_train,
        folds=folds,
        pycaret_selected_model=pycaret_selected_model,
    )

    # Evaluamos una unica vez en test holdout.
    holdout_metrics = evaluate_on_holdout(best_pipeline, X_test, y_test)

    # Consolidamos auditoria integral del experimento.
    result = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "cv_folds": int(folds),
        "pycaret_selection": selection_info,
        "best_sklearn_params": searcher.best_params_,
        "best_sklearn_score_cv_auc": float(searcher.best_score_),
        "cv_summary_mean_std": cv_summary,
        "holdout_metrics": holdout_metrics,
        "leaderboard_top_rows": leaderboard_df.head(10).to_dict(orient="records"),
        "selected_model_class_pycaret": pycaret_selected_model.__class__.__name__,
        "selected_model_class_sklearn": best_pipeline.named_steps["model"].__class__.__name__,
    }

    # Guardamos auditoria si se solicito ruta de salida.
    if save_audit_json_path is not None:
        # Creamos carpeta de salida solo si ya existe su estructura padre o es necesaria localmente.
        save_audit_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Escribimos JSON legible para auditoria y trazabilidad.
        with open(save_audit_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Devolvemos resumen completo.
    return result


# Punto de entrada para ejecutar el script desde terminal.
if __name__ == "__main__":
    # Definimos raiz del proyecto asumiendo que el archivo vive en src/.
    project_root = Path(__file__).resolve().parent.parent

    # Definimos ruta de entrada al dataset crudo.
    input_csv = project_root / "Base_de_datos.csv"

    # Definimos ruta opcional de auditoria en JSON.
    audit_json = project_root / "src" / "model_training_evaluation_audit.json"

    # Ejecutamos pipeline completo de entrenamiento y evaluacion.
    summary = run_training_and_evaluation(
        input_path=input_csv,
        test_size=0.20,
        save_audit_json_path=audit_json,
    )

    # Imprimimos resumen final de forma compacta.
    print("\n=== Resumen de entrenamiento y evaluacion ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
