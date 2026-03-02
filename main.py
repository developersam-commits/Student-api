"""
Student Performance Prediction API
===================================
Endpoints:
  POST /train                    - AutoML training
  POST /predict                  - Single student prediction
  POST /predict/bulk             - Batch predictions
  POST /predict/trend            - Trend analysis + forecasting
  GET  /models                   - Inspect trained models
  GET  /health                   - Health check
  GET  /docs                     - Swagger UI (auto-generated)
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    PredictionRequest, BulkPredictionRequest, TrendRequest, TrainRequest,
    PredictionResult, BulkPredictionResult, TrendResult,
    TrainResult, HealthResponse, ModelComparisonEntry,
)
from ml.automl import run_automl, get_store
from services.prediction_service import predict_single, predict_bulk, analyze_trend

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Startup: train on first launch
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Student Performance API — running initial AutoML training…")
    try:
        run_automl(n_samples=1500, cv_folds=5)
        logger.info("✅ Initial training complete.")
    except Exception as exc:
        logger.error("❌ Initial training failed: %s", exc)
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "AutoML-powered API that predicts **final grade**, **pass/fail**, "
        "**at-risk status**, and **performance trends** for students."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check API health and whether models are loaded."""
    store = get_store()
    return HealthResponse(
        status="ok" if store.trained else "degraded",
        models_loaded={
            "regression": store.regression_pipeline is not None,
            "pass_fail_classification": store.classification_pipeline is not None,
            "at_risk_classification": store.risk_pipeline is not None,
        },
    )


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
@app.post("/train", response_model=TrainResult, tags=["Training"])
def train(req: TrainRequest = TrainRequest()):
    """
    Trigger AutoML re-training.

    Evaluates **8 regression models** and **7 classification models**, 
    cross-validates each, then selects the best performer for each task.
    Returns a full model comparison table.
    """
    logger.info("POST /train n_samples=%d cv=%d", req.n_samples, req.cv_folds)
    result = run_automl(
        n_samples=req.n_samples,
        cv_folds=req.cv_folds,
        test_size=req.test_size,
        random_state=req.random_state,
    )
    return result


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(req: PredictionRequest):
    """
    Predict all metrics for a **single student**.

    Returns:
    - **final_grade** – predicted grade out of 20
    - **pass_fail** – True if grade ≥ 10
    - **pass_probability** – model confidence for passing
    - **risk_level** – low / medium / high / critical
    - **risk_probability** – at-risk model probability
    - **confidence_interval** – 95% CI for the grade estimate
    - **recommendation** – actionable advice
    """
    return predict_single(
        req.student,
        include_confidence=req.include_confidence,
        include_feature_importance=req.include_feature_importance,
    )


@app.post("/predict/bulk", response_model=BulkPredictionResult, tags=["Prediction"])
def predict_bulk_endpoint(req: BulkPredictionRequest):
    """
    Batch predict for **up to 500 students** in one call.

    Also returns cohort summary statistics: mean grade, pass rate, 
    risk distribution.
    """
    return predict_bulk(req.students, include_confidence=req.include_confidence)


@app.post("/predict/trend", response_model=TrendResult, tags=["Prediction"])
def predict_trend(req: TrendRequest):
    """
    Analyze a student's **performance trajectory** across multiple time periods.

    Provide at least 2 historical snapshots (oldest → newest).  
    Returns trend direction, grade slope, risk trajectory, and grade forecasts.
    """
    if len(req.snapshots) < 2:
        raise HTTPException(status_code=422, detail="At least 2 snapshots required for trend analysis.")
    return analyze_trend(req.snapshots, forecast_periods=req.forecast_periods)


# ─────────────────────────────────────────────
# Model inspection
# ─────────────────────────────────────────────
@app.get("/models", tags=["System"])
def get_models():
    """Return info about the currently selected models."""
    store = get_store()
    if not store.trained:
        raise HTTPException(status_code=503, detail="Models not trained yet.")
    return {
        "regression_model": store.regression_model_name,
        "classification_model": store.classification_model_name,
        "risk_model": store.risk_model_name,
        "grade_residual_std": round(store.grade_std, 4),
        "top_features": sorted(
            store.feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:10] if store.feature_importances else [],
        "last_train_result": store.train_result,
    }


# ─────────────────────────────────────────────
# Feature reference
# ─────────────────────────────────────────────
@app.get("/features", tags=["System"])
def list_features():
    """Return the full feature schema with descriptions."""
    return {
        "raw_features": {
            "age": "Student age (10–25)",
            "gender": "0=Female, 1=Male",
            "address_urban": "1=Urban, 0=Rural",
            "parent_education": "0=None … 4=Higher education",
            "family_support": "1=Very Low … 5=Very High",
            "internet_access": "0/1",
            "past_failures": "Number of past class failures (0–3)",
            "absences": "Number of school absences (0–93)",
            "study_time": "Weekly study hours (0–10)",
            "extra_classes": "Paid tutoring 0/1",
            "extracurricular": "Extracurricular activities 0/1",
            "higher_edu_aspiration": "Plans for higher education 0/1",
            "motivation_score": "Self-reported motivation (0–10)",
            "grade_period_1": "First period grade (0–20)",
            "grade_period_2": "Second period grade (0–20)",
        },
        "engineered_features": [
            "grade_avg", "grade_delta", "grade_trend",
            "study_efficiency", "support_composite",
            "risk_composite", "aspiration_motivation",
            "grade_x_study", "absence_failure_interaction",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
