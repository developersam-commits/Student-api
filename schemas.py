from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


# ─────────────────────────────────────────────
# Input Schemas
# ─────────────────────────────────────────────

class StudentFeatures(BaseModel):
    """Core feature set for a single student observation."""

    # Demographics
    age: int = Field(..., ge=10, le=25, description="Student age")
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    address_urban: int = Field(..., ge=0, le=1, description="1=Urban, 0=Rural")

    # Family background
    parent_education: int = Field(..., ge=0, le=4, description="0=None … 4=Higher")
    family_support: int = Field(..., ge=1, le=5, description="1=Very Low … 5=Very High")
    internet_access: int = Field(..., ge=0, le=1)

    # Academic history
    past_failures: int = Field(..., ge=0, le=3)
    absences: int = Field(..., ge=0, le=93)
    study_time: float = Field(..., ge=0.0, le=10.0, description="Weekly study hours")
    extra_classes: int = Field(..., ge=0, le=1, description="Paid tutoring")

    # Engagement
    extracurricular: int = Field(..., ge=0, le=1)
    higher_edu_aspiration: int = Field(..., ge=0, le=1)
    motivation_score: float = Field(..., ge=0.0, le=10.0)

    # Prior grades (G1, G2 out of 20)
    grade_period_1: float = Field(..., ge=0.0, le=20.0)
    grade_period_2: float = Field(..., ge=0.0, le=20.0)

    @field_validator("grade_period_1", "grade_period_2")
    @classmethod
    def round_grades(cls, v: float) -> float:
        return round(v, 2)


class PredictionRequest(BaseModel):
    student: StudentFeatures
    include_confidence: bool = True
    include_feature_importance: bool = False


class BulkPredictionRequest(BaseModel):
    students: list[StudentFeatures] = Field(..., min_length=1, max_length=500)
    include_confidence: bool = True


class TrendRequest(BaseModel):
    """Historical snapshots for trend analysis (oldest → newest)."""
    snapshots: list[StudentFeatures] = Field(..., min_length=2, max_length=20)
    forecast_periods: int = Field(default=2, ge=1, le=5)


class TrainRequest(BaseModel):
    """Trigger re-training with optional hyperparameter overrides."""
    n_samples: int = Field(default=2000, ge=500, le=50_000)
    cv_folds: int = Field(default=5, ge=3, le=10)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = 42


# ─────────────────────────────────────────────
# Output Schemas
# ─────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    task_type: TaskType
    score_metric: str
    score: float
    trained_at: str


class PredictionResult(BaseModel):
    final_grade: float = Field(..., description="Predicted grade out of 20")
    pass_fail: bool = Field(..., description="True = Pass (grade ≥ 10)")
    pass_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: tuple[float, float] | None = None
    feature_importance: dict[str, float] | None = None
    recommendation: str


class BulkPredictionResult(BaseModel):
    results: list[PredictionResult]
    summary: dict[str, Any]


class TrendPoint(BaseModel):
    period: int
    predicted_grade: float
    risk_level: RiskLevel
    is_forecast: bool = False


class TrendResult(BaseModel):
    trend_direction: TrendDirection
    slope: float = Field(..., description="Grade change per period")
    current_grade: float
    forecasted_grades: list[float]
    trend_points: list[TrendPoint]
    risk_trajectory: str
    alert: str | None = None


class ModelComparisonEntry(BaseModel):
    model_name: str
    task: str
    cv_mean: float
    cv_std: float
    test_score: float
    selected: bool


class TrainResult(BaseModel):
    status: str
    models_evaluated: list[ModelComparisonEntry]
    best_regression_model: str
    best_classification_model: str
    training_samples: int
    test_samples: int
    duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    version: str = "1.0.0"
