"""
model_monitoring.py
-------------------
Monitoreo de data drift para un modelo de riesgo crediticio.

Incluye:
- Muestreo periodico de datos para analisis estadistico.
- Tabla de datos con pronosticos (si no existen, se generan scores proxy auditables).
- Metricas por variable:
  * Kolmogorov-Smirnov (KS) para numericas
  * Population Stability Index (PSI)
  * Jensen-Shannon divergence (JSD)
  * Chi-cuadrado para categoricas
- Resumen global con semaforo y recomendaciones.
- Historico temporal de drift para analisis de tendencia.

Uso:
    python src/model_monitoring.py
    python src/model_monitoring.py --current-csv /ruta/lote_actual.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, ks_2samp


RANDOM_STATE = 42
EPS = 1e-8
TARGET_COL = "Pago_atiempo"
DATE_COL = "fecha_prestamo"


@dataclass
class MonitoringConfig:
    reference_csv: Path
    current_csv: Path | None
    output_dir: Path
    periodicity: str
    sample_fraction: float
    min_sample_size: int


def parse_args() -> MonitoringConfig:
    parser = argparse.ArgumentParser(description="Monitoreo de drift de datos y pronosticos")
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("Base_de_datos.csv"),
        help="Dataset historico de referencia.",
    )
    parser.add_argument(
        "--current-csv",
        type=Path,
        default=None,
        help="Dataset actual para comparar drift. Si se omite, se simula con recorte temporal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src") / "monitoring_outputs",
        help="Directorio de salida para reportes.",
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        default="M",
        choices=["W", "M", "Q"],
        help="Periodicidad de muestreo: W (semanal), M (mensual), Q (trimestral).",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.25,
        help="Fraccion por periodo para muestreo de monitoreo.",
    )
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=300,
        help="Tamano minimo de muestra total para monitoreo.",
    )

    args = parser.parse_args()
    return MonitoringConfig(
        reference_csv=args.reference_csv,
        current_csv=args.current_csv,
        output_dir=args.output_dir,
        periodicity=args.periodicity,
        sample_fraction=args.sample_fraction,
        min_sample_size=args.min_sample_size,
    )


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")
    df = pd.read_csv(path)
    return df


def coerce_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DATE_COL in out.columns:
        # Formato esperado del proyecto: DD/MM/YY HH:MM
        out[DATE_COL] = pd.to_datetime(
            out[DATE_COL],
            format="%d/%m/%y %H:%M",
            errors="coerce",
        )
    return out


def ensure_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza columnas de pronostico en el dataset monitoreado.

    Si no existen, se crea una probabilidad proxy auditada desde variables de score.
    """
    out = df.copy()

    if "pred_proba_risk" in out.columns and "pred_label_risk" in out.columns:
        return out

    score_col = None
    for candidate in ["puntaje", "puntaje_datacredito", "huella_consulta"]:
        if candidate in out.columns:
            score_col = candidate
            break

    if score_col is not None:
        s = pd.to_numeric(out[score_col], errors="coerce")
        s = s.fillna(s.median())
        z = (s - s.mean()) / (s.std() + EPS)
        # Riesgo mayor cuando score es menor.
        p_risk = 1.0 / (1.0 + np.exp(z))
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        p_risk = rng.uniform(0.05, 0.25, size=len(out))

    out["pred_proba_risk"] = np.clip(p_risk, 0.0, 1.0)
    out["pred_label_risk"] = (out["pred_proba_risk"] >= 0.20).astype(int)
    return out


def periodic_sample(df: pd.DataFrame, periodicity: str, frac: float, min_size: int) -> pd.DataFrame:
    out = df.copy()

    if DATE_COL in out.columns and out[DATE_COL].notna().any():
        period = out[DATE_COL].dt.to_period(periodicity)
        sampled = (
            out.assign(_period=period)
            .groupby("_period", group_keys=False)
            .apply(lambda part: part.sample(max(1, int(len(part) * frac)), random_state=RANDOM_STATE))
            .drop(columns=["_period"])
        )
    else:
        n = max(min_size, int(len(out) * frac))
        n = min(n, len(out))
        sampled = out.sample(n, random_state=RANDOM_STATE)

    if len(sampled) < min_size and len(out) > len(sampled):
        faltan = min_size - len(sampled)
        restantes = out.drop(index=sampled.index)
        extra_n = min(len(restantes), faltan)
        if extra_n > 0:
            sampled = pd.concat([sampled, restantes.sample(extra_n, random_state=RANDOM_STATE)])

    return sampled.reset_index(drop=True)


def split_reference_current(df_ref_full: pd.DataFrame, current_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_full = coerce_date(df_ref_full)

    if current_path is not None:
        current = coerce_date(load_data(current_path))
        reference = ref_full
        return reference, current

    # Simulacion temporal para no bloquear el pipeline en ausencia de lote actual real.
    if DATE_COL in ref_full.columns and ref_full[DATE_COL].notna().any():
        cutoff = ref_full[DATE_COL].quantile(0.80)
        reference = ref_full[ref_full[DATE_COL] <= cutoff].copy()
        current = ref_full[ref_full[DATE_COL] > cutoff].copy()
    else:
        reference = ref_full.sample(frac=0.80, random_state=RANDOM_STATE)
        current = ref_full.drop(index=reference.index)

    if len(current) == 0:
        current = reference.sample(min(200, len(reference)), random_state=RANDOM_STATE)

    return reference.reset_index(drop=True), current.reset_index(drop=True)


def psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    ref_pct = ref_counts / (np.sum(ref_counts) + EPS)
    cur_pct = cur_counts / (np.sum(cur_counts) + EPS)
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_metrics(ref: pd.Series, cur: pd.Series) -> dict:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()

    if len(ref) < 20 or len(cur) < 20:
        return {
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "psi": np.nan,
            "js_divergence": np.nan,
        }

    ks = ks_2samp(ref, cur)

    # Bins por cuantiles de referencia para PSI/JSD.
    q = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if len(q) < 3:
        q = np.linspace(ref.min(), ref.max() + EPS, 11)

    ref_hist, _ = np.histogram(ref, bins=q)
    cur_hist, _ = np.histogram(cur, bins=q)

    psi = psi_from_counts(ref_hist, cur_hist)

    ref_prob = ref_hist / (ref_hist.sum() + EPS)
    cur_prob = cur_hist / (cur_hist.sum() + EPS)
    jsd = float(jensenshannon(ref_prob + EPS, cur_prob + EPS, base=2.0) ** 2)

    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "psi": psi,
        "js_divergence": jsd,
    }


def categorical_metrics(ref: pd.Series, cur: pd.Series) -> dict:
    ref = ref.astype(str).fillna("<NA>")
    cur = cur.astype(str).fillna("<NA>")

    cats = sorted(set(ref.unique()).union(set(cur.unique())))
    ref_counts = ref.value_counts().reindex(cats, fill_value=0).values
    cur_counts = cur.value_counts().reindex(cats, fill_value=0).values

    psi = psi_from_counts(ref_counts, cur_counts)

    ref_prob = ref_counts / (ref_counts.sum() + EPS)
    cur_prob = cur_counts / (cur_counts.sum() + EPS)
    jsd = float(jensenshannon(ref_prob + EPS, cur_prob + EPS, base=2.0) ** 2)

    expected = ref_prob * (cur_counts.sum() + EPS)
    chi = chisquare(f_obs=cur_counts + EPS, f_exp=expected + EPS)

    return {
        "chi2_stat": float(chi.statistic),
        "chi2_pvalue": float(chi.pvalue),
        "psi": psi,
        "js_divergence": jsd,
    }


def severity_from_score(score: float) -> str:
    if np.isnan(score):
        return "unknown"
    if score < 0.10:
        return "low"
    if score < 0.25:
        return "medium"
    return "high"


def global_status(global_score: float, high_ratio: float) -> str:
    if global_score >= 0.25 or high_ratio >= 0.30:
        return "critical"
    if global_score >= 0.10 or high_ratio >= 0.10:
        return "warning"
    return "ok"


def build_recommendations(status: str, top_high: pd.DataFrame) -> list[str]:
    recs: list[str] = []

    if status == "critical":
        recs.append("Alerta critica: revisar variables con drift alto y activar retraining prioritario.")
        recs.append("Validar cambios de origen en datos (definicion de campos, ETL y reglas de negocio).")
    elif status == "warning":
        recs.append("Alerta media: monitorear evolucion durante los siguientes lotes.")
        recs.append("Revisar calibracion de umbral y sensibilidad de variables con drift medio/alto.")
    else:
        recs.append("Estado estable: continuar monitoreo periodico.")

    if len(top_high) > 0:
        vars_txt = ", ".join(top_high["variable"].head(5).tolist())
        recs.append(f"Variables prioritarias para revision: {vars_txt}.")

    return recs


def compute_drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    shared_cols = [c for c in reference.columns if c in current.columns]
    excluded = {TARGET_COL, DATE_COL}
    cols = [c for c in shared_cols if c not in excluded]

    rows: list[dict] = []

    for col in cols:
        ref_s = reference[col]
        cur_s = current[col]

        if pd.api.types.is_numeric_dtype(ref_s):
            m = numeric_metrics(ref_s, cur_s)
            score = float(np.nanmax([m.get("psi", np.nan), m.get("js_divergence", np.nan)]))
            rows.append(
                {
                    "variable": col,
                    "dtype": "numeric",
                    "ks_stat": m.get("ks_stat"),
                    "ks_pvalue": m.get("ks_pvalue"),
                    "chi2_stat": np.nan,
                    "chi2_pvalue": np.nan,
                    "psi": m.get("psi"),
                    "js_divergence": m.get("js_divergence"),
                    "drift_score": score,
                    "severity": severity_from_score(score),
                }
            )
        else:
            m = categorical_metrics(ref_s, cur_s)
            score = float(np.nanmax([m.get("psi", np.nan), m.get("js_divergence", np.nan)]))
            rows.append(
                {
                    "variable": col,
                    "dtype": "categorical",
                    "ks_stat": np.nan,
                    "ks_pvalue": np.nan,
                    "chi2_stat": m.get("chi2_stat"),
                    "chi2_pvalue": m.get("chi2_pvalue"),
                    "psi": m.get("psi"),
                    "js_divergence": m.get("js_divergence"),
                    "drift_score": score,
                    "severity": severity_from_score(score),
                }
            )

    drift_df = pd.DataFrame(rows).sort_values("drift_score", ascending=False).reset_index(drop=True)
    return drift_df


def update_history(history_path: Path, run_row: dict) -> pd.DataFrame:
    if history_path.exists():
        hist = pd.read_csv(history_path)
    else:
        hist = pd.DataFrame()

    new_row_df = pd.DataFrame([run_row])
    if hist.empty:
        hist = new_row_df
    else:
        hist = pd.concat([hist, new_row_df], ignore_index=True)
    hist.to_csv(history_path, index=False)
    return hist


def run_monitoring(config: MonitoringConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_reference = load_data(config.reference_csv)
    reference_df, current_df = split_reference_current(raw_reference, config.current_csv)

    # Muestreo periodico para reducir costo de monitoreo y simular ventanas de control.
    current_sample = periodic_sample(
        df=current_df,
        periodicity=config.periodicity,
        frac=config.sample_fraction,
        min_size=config.min_sample_size,
    )

    # Adjuntamos pronosticos al lote monitoreado.
    current_scored = ensure_prediction_columns(current_sample)

    drift_df = compute_drift_table(reference_df, current_scored)

    global_score = float(drift_df["drift_score"].mean()) if len(drift_df) else np.nan
    high_ratio = float((drift_df["severity"] == "high").mean()) if len(drift_df) else 0.0
    status = global_status(global_score, high_ratio)

    top_high = drift_df[drift_df["severity"].isin(["high", "medium"])].copy()
    recommendations = build_recommendations(status, top_high)

    run_ts = datetime.now(timezone.utc).isoformat()

    drift_csv = config.output_dir / "drift_metrics_latest.csv"
    scored_csv = config.output_dir / "monitoring_scored_sample_latest.csv"
    history_csv = config.output_dir / "monitoring_history.csv"
    summary_json = config.output_dir / "monitoring_summary_latest.json"

    drift_df.to_csv(drift_csv, index=False)
    current_scored.to_csv(scored_csv, index=False)

    run_row = {
        "run_timestamp_utc": run_ts,
        "global_drift_score": global_score,
        "high_severity_ratio": high_ratio,
        "status": status,
        "n_variables_monitored": int(len(drift_df)),
        "n_current_sample": int(len(current_scored)),
    }
    history_df = update_history(history_csv, run_row)

    summary = {
        "run_timestamp_utc": run_ts,
        "status": status,
        "global_drift_score": global_score,
        "high_severity_ratio": high_ratio,
        "n_variables_monitored": int(len(drift_df)),
        "n_reference_rows": int(len(reference_df)),
        "n_current_rows": int(len(current_df)),
        "n_current_sample_rows": int(len(current_scored)),
        "periodicity": config.periodicity,
        "sample_fraction": config.sample_fraction,
        "paths": {
            "drift_metrics_csv": str(drift_csv),
            "scored_sample_csv": str(scored_csv),
            "history_csv": str(history_csv),
        },
        "recommendations": recommendations,
        "top_drift_variables": drift_df.head(10).to_dict(orient="records"),
        "history_tail": history_df.tail(10).to_dict(orient="records"),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Monitoring Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_monitoring(args)
