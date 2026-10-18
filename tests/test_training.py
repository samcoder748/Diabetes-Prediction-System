"""Tests for training module."""

import json
import pandas as pd
import pytest
from pathlib import Path

from src.config import METRICS_PATH, MODEL_PATH
from src.train import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample test data with enough samples for stratified k-fold."""
    return pd.DataFrame({
        "Pregnancies": [6, 1, 8, 1, 0, 5, 3, 10, 2, 4, 3, 7, 2, 5, 1, 4, 2, 6, 3, 1],
        "Glucose": [148, 85, 183, 89, 137, 116, 78, 115, 197, 110, 100, 120, 90, 140, 95, 105, 130, 125, 115, 135],
        "BloodPressure": [72, 66, 64, 66, 40, 74, 50, 0, 70, 76, 68, 75, 62, 80, 70, 72, 78, 74, 70, 68],
        "SkinThickness": [35, 29, 0, 23, 35, 0, 32, 0, 45, 20, 25, 30, 28, 33, 22, 26, 31, 34, 27, 24],
        "Insulin": [0, 0, 0, 94, 168, 0, 88, 0, 543, 100, 80, 120, 60, 150, 75, 95, 110, 130, 85, 105],
        "BMI": [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 27.2, 29.0, 32.0, 28.5, 34.0, 30.0, 31.5, 33.0, 32.5, 29.5, 31.0],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.519, 0.3, 0.4, 0.25, 0.45, 0.35, 0.38, 0.42, 0.4, 0.33, 0.37],
        "Age": [50, 31, 32, 21, 33, 30, 26, 29, 53, 41, 35, 45, 28, 48, 32, 40, 38, 42, 36, 34],
        "Outcome": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_create_models():
    """Test model creation."""
    trainer = ModelTrainer()
    models = trainer.create_models()

    assert len(models) == 3
    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models


def test_calculate_metrics():
    """Test metric calculation."""
    trainer = ModelTrainer()

    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]
    y_scores = [0.2, 0.8, 0.3, 0.4, 0.1, 0.9]

    metrics = trainer.calculate_metrics(y_true, y_pred, y_scores)

    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1-Score" in metrics
    assert "ROC-AUC" in metrics
    assert 0 <= metrics["Accuracy"] <= 1
    assert 0 <= metrics["Precision"] <= 1
    assert 0 <= metrics["Recall"] <= 1


def test_train_and_evaluate_model(sample_data):
    """Test training and evaluation of a single model."""
    trainer = ModelTrainer()

    # Prepare data
    X = sample_data.drop("Outcome", axis=1)
    y = sample_data["Outcome"]

    from sklearn.model_selection import train_test_split
    # Use smaller test size to ensure enough samples for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a model
    models = trainer.create_models()
    model = models["Logistic Regression"]

    # Train and evaluate
    result = trainer.train_and_evaluate_model(
        model, "Logistic Regression", X_train, X_test, y_train, y_test
    )

    assert "model" in result
    assert "model_name" in result
    assert "metrics" in result
    assert "y_pred" in result
    assert "y_scores" in result
    assert result["model_name"] == "Logistic Regression"
    assert len(result["y_pred"]) == len(y_test)


def test_select_best_model():
    """Test model selection."""
    trainer = ModelTrainer()

    # Create mock evaluation results with all required metrics
    evaluation_results = {
        "Model A": {
            "metrics": {
                "ROC-AUC": 0.85,
                "Accuracy": 0.80,
                "Precision": 0.75,
                "Recall": 0.70,
                "F1-Score": 0.72,
            },
        },
        "Model B": {
            "metrics": {
                "ROC-AUC": 0.90,
                "Accuracy": 0.82,
                "Precision": 0.78,
                "Recall": 0.75,
                "F1-Score": 0.76,
            },
        },
        "Model C": {
            "metrics": {
                "ROC-AUC": 0.88,
                "Accuracy": 0.81,
                "Precision": 0.77,
                "Recall": 0.73,
                "F1-Score": 0.75,
            },
        },
    }

    best_model = trainer.select_best_model(evaluation_results)

    assert best_model == "Model B"  # Highest ROC-AUC
