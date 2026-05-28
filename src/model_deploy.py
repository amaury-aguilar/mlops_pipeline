"""FastAPI service for credit risk batch predictions.

The service loads a serialized training bundle that already contains the fitted
sklearn Pipeline used in training. Incoming raw records are normalized with the
same feature-engineering functions used during training and then scored in batch.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ft_engineering import TARGET_COL, derive_features, preprocess_dtypes


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_BUNDLE_PATH = PROJECT_ROOT / "model_artifacts" / "credit_risk_model_bundle.joblib"


class BatchPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    row_index: int
    risk_probability: float
    predicted_class: int
    predicted_label: str
    decision_threshold: float


class BatchPredictResponse(BaseModel):
    model_name: str
    artifact_path: str
    decision_threshold: float
    n_records: int
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    artifact_path: str
    model_name: Optional[str] = None
    decision_threshold: Optional[float] = None


app = FastAPI(
    title="Credit Risk Batch Prediction API",
    version="1.0.0",
    description="Batch scoring service for the credit-risk model.",
)


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_BUNDLE_PATH}. Run training to export it first."
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Invalid model bundle structure: expected a dict with a 'pipeline' key.")
    return bundle


def get_model_status() -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    try:
        bundle = load_model_bundle()
        return True, bundle, None
    except Exception as exc:
        return False, None, str(exc)


def normalize_input_frame(raw_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("No records were provided for scoring.")

    if TARGET_COL in raw_df.columns:
        raw_df = raw_df.drop(columns=[TARGET_COL])

    processed = preprocess_dtypes(raw_df)
    processed = derive_features(processed)

    if TARGET_COL in processed.columns:
        processed = processed.drop(columns=[TARGET_COL])

    required_columns = bundle.get("feature_columns", [])
    missing_columns = [col for col in required_columns if col not in processed.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns after feature engineering: "
            + ", ".join(missing_columns)
        )

    return processed


def predict_dataframe(raw_df: pd.DataFrame) -> BatchPredictResponse:
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("decision_threshold", 0.5))

    features = normalize_input_frame(raw_df, bundle)
    probabilities = pipeline.predict_proba(features)[:, 1]
    predicted_class = (probabilities >= threshold).astype(int)

    predictions = [
        PredictionItem(
            row_index=index,
            risk_probability=float(probabilities[index]),
            predicted_class=int(predicted_class[index]),
            predicted_label="risk" if int(predicted_class[index]) == 1 else "non_risk",
            decision_threshold=threshold,
        )
        for index in range(len(features))
    ]

    return BatchPredictResponse(
        model_name=str(bundle.get("selected_model_class", "unknown_model")),
        artifact_path=str(MODEL_BUNDLE_PATH),
        decision_threshold=threshold,
        n_records=len(predictions),
        predictions=predictions,
    )


@app.on_event("startup")
def startup_check() -> None:
    try:
        load_model_bundle()
    except Exception:
        # The service still starts so health checks can report the issue clearly.
        return


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready, bundle, _ = get_model_status()
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_ready=ready,
        artifact_path=str(MODEL_BUNDLE_PATH),
        model_name=None if bundle is None else str(bundle.get("selected_model_class", "unknown_model")),
        decision_threshold=None if bundle is None else float(bundle.get("decision_threshold", 0.5)),
    )


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    ready, bundle, error = get_model_status()
    if not ready or bundle is None:
        return {
            "model_ready": False,
            "artifact_path": str(MODEL_BUNDLE_PATH),
            "error": error,
        }

    return {
        "model_ready": True,
        "artifact_path": str(MODEL_BUNDLE_PATH),
        "artifact_type": bundle.get("artifact_type"),
        "artifact_version": bundle.get("artifact_version"),
        "created_at_utc": bundle.get("created_at_utc"),
        "target_column": bundle.get("target_column"),
        "positive_class_for_metrics": bundle.get("positive_class_for_metrics"),
        "decision_threshold": bundle.get("decision_threshold"),
        "expected_cost_per_row": bundle.get("expected_cost_per_row"),
        "selected_model_class": bundle.get("selected_model_class"),
        "feature_columns": bundle.get("feature_columns", []),
        "dropped_leakage_columns": bundle.get("dropped_leakage_columns", []),
    }


@app.post("/predict", response_model=BatchPredictResponse)
def predict(request: BatchPredictRequest) -> BatchPredictResponse:
    try:
        raw_df = pd.DataFrame(request.records)
        return predict_dataframe(raw_df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected scoring error: {exc}") from exc


@app.post("/predict/csv", response_model=BatchPredictResponse)
async def predict_csv(file: UploadFile = File(...)) -> BatchPredictResponse:
    try:
        csv_bytes = await file.read()
        raw_df = pd.read_csv(BytesIO(csv_bytes))
        return predict_dataframe(raw_df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected CSV scoring error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=False)