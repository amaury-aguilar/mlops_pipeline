"""
streamlit_monitoring_app.py
---------------------------
Aplicacion de visualizacion para monitoreo de drift.

Uso:
    streamlit run src/streamlit_monitoring_app.py
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px


OUTPUT_DIR = Path("src") / "monitoring_outputs"
SUMMARY_PATH = OUTPUT_DIR / "monitoring_summary_latest.json"
DRIFT_PATH = OUTPUT_DIR / "drift_metrics_latest.csv"
HISTORY_PATH = OUTPUT_DIR / "monitoring_history.csv"
SCORED_PATH = OUTPUT_DIR / "monitoring_scored_sample_latest.csv"


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def status_color(status: str) -> str:
    if status == "ok":
        return "green"
    if status == "warning":
        return "orange"
    if status == "critical":
        return "red"
    return "gray"


def main() -> None:
    st.set_page_config(page_title="Monitoreo de Drift", layout="wide")
    st.title("Monitoreo de Data Drift - Riesgo Crediticio")

    if not SUMMARY_PATH.exists() or not DRIFT_PATH.exists():
        st.warning(
            "No se encontraron reportes de monitoreo. Ejecuta primero: `python src/model_monitoring.py`"
        )
        st.stop()

    summary = load_json(SUMMARY_PATH)
    drift_df = load_csv(DRIFT_PATH)
    history_df = load_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()
    scored_df = load_csv(SCORED_PATH) if SCORED_PATH.exists() else pd.DataFrame()

    status = summary.get("status", "unknown")
    color = status_color(status)

    st.markdown(f"### Estado Global: :{color}[{status.upper()}]")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Drift Score", round(float(summary.get("global_drift_score", 0.0)), 4))
    c2.metric("High Severity Ratio", round(float(summary.get("high_severity_ratio", 0.0)), 4))
    c3.metric("Variables Monitoreadas", int(summary.get("n_variables_monitored", 0)))
    c4.metric("Muestra Actual", int(summary.get("n_current_sample_rows", 0)))

    tab1, tab2, tab3, tab4 = st.tabs([
        "Metricas de Drift",
        "Comparacion de Distribuciones",
        "Analisis Temporal",
        "Recomendaciones",
    ])

    with tab1:
        st.subheader("Tabla de drift por variable")
        show_cols = [
            "variable",
            "dtype",
            "drift_score",
            "severity",
            "psi",
            "js_divergence",
            "ks_pvalue",
            "chi2_pvalue",
        ]
        st.dataframe(drift_df[show_cols], use_container_width=True)

        fig_bar = px.bar(
            drift_df.head(20),
            x="variable",
            y="drift_score",
            color="severity",
            title="Top variables por drift score",
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("Historico vs Actual")
        if scored_df.empty:
            st.info("No hay muestra con pronosticos para comparar distribuciones.")
        else:
            num_cols = drift_df[drift_df["dtype"] == "numeric"]["variable"].tolist()
            if not num_cols:
                st.info("No hay variables numericas disponibles para graficar.")
            else:
                selected_col = st.selectbox("Selecciona variable numerica", num_cols)

                # Cargamos referencia desde Base_de_datos.csv para comparar distribuciones.
                ref_df = pd.read_csv("Base_de_datos.csv")
                cur_values = pd.to_numeric(scored_df[selected_col], errors="coerce").dropna()
                ref_values = pd.to_numeric(ref_df[selected_col], errors="coerce").dropna()

                hist_df = pd.DataFrame(
                    {
                        selected_col: pd.concat([ref_values, cur_values], ignore_index=True),
                        "dataset": ["historico"] * len(ref_values) + ["actual"] * len(cur_values),
                    }
                )

                fig_hist = px.histogram(
                    hist_df,
                    x=selected_col,
                    color="dataset",
                    barmode="overlay",
                    nbins=40,
                    opacity=0.6,
                    title=f"Distribucion historica vs actual - {selected_col}",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Evolucion temporal del drift")
        if history_df.empty:
            st.info("Aun no existe historico de monitoreo.")
        else:
            history_df["run_timestamp_utc"] = pd.to_datetime(history_df["run_timestamp_utc"], errors="coerce")
            history_df = history_df.sort_values("run_timestamp_utc")

            fig_line = px.line(
                history_df,
                x="run_timestamp_utc",
                y="global_drift_score",
                markers=True,
                title="Tendencia de drift global",
            )
            st.plotly_chart(fig_line, use_container_width=True)

            fig_ratio = px.line(
                history_df,
                x="run_timestamp_utc",
                y="high_severity_ratio",
                markers=True,
                title="Tendencia de ratio de variables en severidad alta",
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

    with tab4:
        st.subheader("Recomendaciones automaticas")
        for rec in summary.get("recommendations", []):
            st.write(f"- {rec}")

        if status == "critical":
            st.error("Se recomienda retraining o recalibracion inmediata.")
        elif status == "warning":
            st.warning("Se recomienda seguimiento estrecho y validacion de variables con drift.")
        else:
            st.success("Sin alertas criticas. Continuar monitoreo periodico.")


if __name__ == "__main__":
    main()
