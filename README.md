# 🎓 Student Performance Prediction API

An **AutoML-powered FastAPI** that predicts student outcomes across four dimensions:
final grade, pass/fail, at-risk status, and performance trends over time.

---

## Features

| Capability | Details |
|---|---|
| **AutoML** | Evaluates 8 regression + 7 classification models, cross-validates, selects best |
| **Final Grade** | Regression prediction (0–20 scale) with 95% confidence interval |
| **Pass / Fail** | Calibrated probability + binary classification |
| **At-Risk Detection** | Multi-factor risk scoring → LOW / MEDIUM / HIGH / CRITICAL |
| **Trend Forecasting** | Trajectory analysis from historical snapshots + future grade forecasts |
| **Bulk Prediction** | Score up to 500 students in a single request with cohort analytics |
| **Recommendations** | Actionable advice per student based on predicted risk |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API (auto-trains on startup)
python main.py
# → http://localhost:8000/docs
```

---

## API Endpoints

### `POST /train`
Trigger AutoML re-training with custom parameters.

```json
{
  "n_samples": 2000,
  "cv_folds": 5,
  "test_size": 0.2,
  "random_state": 42
}
```

Returns model comparison table with CV scores for every model evaluated.

---

### `POST /predict`
Predict all metrics for a single student.

```json
{
  "student": {
    "age": 17,
    "gender": 1,
    "address_urban": 1,
    "parent_education": 2,
    "family_support": 3,
    "internet_access": 1,
    "past_failures": 0,
    "absences": 5,
    "study_time": 4.0,
    "extra_classes": 0,
    "extracurricular": 1,
    "higher_edu_aspiration": 1,
    "motivation_score": 7.5,
    "grade_period_1": 13.0,
    "grade_period_2": 14.0
  },
  "include_confidence": true,
  "include_feature_importance": true
}
```

**Response:**
```json
{
  "final_grade": 14.87,
  "pass_fail": true,
  "pass_probability": 0.9421,
  "risk_level": "low",
  "risk_probability": 0.0831,
  "confidence_interval": [11.51, 18.23],
  "feature_importance": {"grade_period_2": 0.31, "grade_avg": 0.28, ...},
  "recommendation": "Performing well (predicted grade 14.9/20). Keep up current study habits."
}
```

---

### `POST /predict/bulk`
Batch prediction for up to 500 students.

```json
{
  "students": [ {...}, {...}, {...} ],
  "include_confidence": true
}
```

Returns per-student results + cohort summary (mean grade, pass rate, risk distribution).

---

### `POST /predict/trend`
Analyze performance trajectory from historical snapshots.

```json
{
  "snapshots": [
    { "grade_period_1": 8.0, "grade_period_2": 8.5, ... },
    { "grade_period_1": 9.5, "grade_period_2": 10.5, ... },
    { "grade_period_1": 11.0, "grade_period_2": 12.0, ... }
  ],
  "forecast_periods": 2
}
```

**Response:**
```json
{
  "trend_direction": "improving",
  "slope": 1.51,
  "current_grade": 11.23,
  "forecasted_grades": [12.68, 14.18],
  "risk_trajectory": "medium → low",
  "trend_points": [...],
  "alert": null
}
```

---

### `GET /models`
Inspect currently selected models, top feature importances, and last training result.

### `GET /health`
Returns API status and which models are loaded.

### `GET /features`
Returns the full feature schema with field descriptions.

---

## Feature Engineering

Raw features are automatically extended with 9 engineered features:

| Feature | Formula |
|---|---|
| `grade_avg` | `(G1 + G2) / 2` |
| `grade_delta` | `G2 - G1` |
| `grade_trend` | `sign(grade_delta)` |
| `study_efficiency` | `study_time / (absences + 1)` |
| `support_composite` | `family_support + internet + extra_classes` |
| `risk_composite` | `past_failures × 3 + absences / 10` |
| `aspiration_motivation` | `higher_edu × motivation_score` |
| `grade_x_study` | `grade_avg × study_time` |
| `absence_failure_interaction` | `absences × (past_failures + 1)` |

All features pass through a `PowerTransformer → StandardScaler` pipeline.

---

## Model Zoo

**Regression (grade prediction)**
Ridge · ElasticNet · RandomForest · ExtraTrees · GradientBoosting · SVR · KNN · MLP

**Classification (pass/fail + at-risk)**
LogisticRegression · RandomForest · ExtraTrees · GradientBoosting · SVC (calibrated) · KNN · MLP

AutoML selects the best model per task by cross-validated R² (regression) or ROC-AUC (classification).

---

## Project Structure

```
student_api/
├── main.py                        # FastAPI app + all endpoints
├── schemas.py                     # Pydantic input/output models
├── requirements.txt
├── ml/
│   ├── automl.py                  # AutoML engine
│   ├── features.py                # Feature engineering pipeline
│   └── data_generator.py          # Synthetic dataset generator
└── services/
    └── prediction_service.py      # Prediction + trend logic
```
