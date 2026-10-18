"""Configuration module for Diabetes Risk Prediction System."""

import os
from pathlib import Path
from typing import Final

# Project paths
PROJECT_ROOT: Final = Path(__file__).parent.parent
DATA_DIR: Final = PROJECT_ROOT / "data"
MODELS_DIR: Final = PROJECT_ROOT / "models"
REPORTS_DIR: Final = PROJECT_ROOT / "reports"

# Dataset configuration
DATASET_FILENAME: Final = "diabetes.csv"
DATASET_PATH: Final = DATA_DIR / DATASET_FILENAME

# Model configuration
MODEL_FILENAME: Final = "diabetes_model.joblib"
METRICS_FILENAME: Final = "evaluation_metrics.json"
MODEL_PATH: Final = MODELS_DIR / MODEL_FILENAME
METRICS_PATH: Final = REPORTS_DIR / METRICS_FILENAME

# Features
FEATURE_COLUMNS: Final = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN: Final = "Outcome"

# Features where zero values are treated as missing
MISSING_VALUE_FEATURES: Final = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Model selection
RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.2
CV_FOLDS: Final = 5

# Classification threshold
DEFAULT_THRESHOLD: Final = 0.5

# Input bounds for validation (based on clinical plausibility)
INPUT_BOUNDS: Final = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 70),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (0, 120),
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
