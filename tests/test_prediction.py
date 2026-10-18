"""Tests for prediction module."""

import pytest
from pathlib import Path

from src.predict import Predictor, PredictionError


def test_predict_single_without_model(tmp_path):
    """Test prediction when model doesn't exist."""
    predictor = Predictor(model_path=tmp_path / "nonexistent.joblib")

    input_data = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    with pytest.raises(PredictionError, match="Model file not found"):
        predictor.predict_single(input_data)


def test_predict_single_invalid_input(tmp_path):
    """Test prediction with invalid input."""
    # Create a mock model file
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    predictor = Predictor(model_path=model_path)

    # Invalid input (missing features)
    input_data = {
        "Pregnancies": 6,
        "Glucose": 148,
        # Missing other features
    }

    with pytest.raises(PredictionError, match="Invalid input"):
        predictor.predict_single(input_data)


def test_predict_single_out_of_bounds(tmp_path):
    """Test prediction with out-of-bounds input."""
    # Create a mock model file
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    predictor = Predictor(model_path=model_path)

    # Out of bounds input
    input_data = {
        "Pregnancies": 6,
        "Glucose": 999,  # Out of bounds
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    with pytest.raises(PredictionError, match="outside valid range"):
        predictor.predict_single(input_data)


def test_get_model_info_without_model(tmp_path):
    """Test getting model info when model doesn't exist."""
    predictor = Predictor(model_path=tmp_path / "nonexistent.joblib", metrics_path=tmp_path / "nonexistent_metrics.json")

    # Should return basic info even without model loaded
    info = predictor.get_model_info()

    assert info is not None
    assert "model_path" in info
    assert "metrics_path" in info
    assert info["model_path"] == str(tmp_path / "nonexistent.joblib")
