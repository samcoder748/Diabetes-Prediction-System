"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return pd.DataFrame({
        "Pregnancies": [6, 1, 8, 1, 0],
        "Glucose": [148, 85, 183, 89, 137],
        "BloodPressure": [72, 66, 64, 66, 40],
        "SkinThickness": [35, 29, 0, 23, 35],
        "Insulin": [0, 0, 0, 94, 168],
        "BMI": [33.6, 26.6, 23.3, 28.1, 43.1],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288],
        "Age": [50, 31, 32, 21, 33],
    })


def test_replace_zeros_with_nan(sample_data):
    """Test zero replacement with NaN."""
    preprocessor = DataPreprocessor()
    result = preprocessor.replace_zeros_with_nan(sample_data)

    # Check that zeros in MISSING_VALUE_FEATURES are replaced
    assert pd.isna(result.loc[2, "SkinThickness"])  # Row 2 has SkinThickness = 0
    assert pd.isna(result.loc[0, "Insulin"])  # Row 0 has Insulin = 0

    # Check that Pregnancies can be zero (not replaced)
    assert result.loc[4, "Pregnancies"] == 0


def test_create_pipeline():
    """Test pipeline creation."""
    preprocessor = DataPreprocessor()
    pipeline = preprocessor.create_pipeline()

    assert pipeline is not None
    assert len(pipeline.steps) == 2  # imputer and scaler


def test_fit_transform(sample_data):
    """Test fit_transform on training data."""
    preprocessor = DataPreprocessor()
    # sample_data already has only feature columns (no Outcome)
    result = preprocessor.fit_transform(sample_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (5, len(FEATURE_COLUMNS))  # 5 samples, 8 features
    assert not np.isnan(result).any()  # No NaN values after imputation


def test_transform_without_fit(sample_data):
    """Test transform without prior fit raises error."""
    preprocessor = DataPreprocessor()

    with pytest.raises(ValueError, match="Pipeline has not been fitted"):
        preprocessor.transform(sample_data)


def test_transform_after_fit(sample_data):
    """Test transform after fit_transform."""
    preprocessor = DataPreprocessor()

    # sample_data already has only feature columns (no Outcome)

    # Fit on training data
    train_result = preprocessor.fit_transform(sample_data)

    # Transform same data
    test_result = preprocessor.transform(sample_data)

    assert isinstance(test_result, np.ndarray)
    assert test_result.shape == (5, len(FEATURE_COLUMNS))


def test_get_feature_names():
    """Test getting feature names."""
    preprocessor = DataPreprocessor()
    names = preprocessor.get_feature_names_out()

    assert isinstance(names, list)
    assert len(names) == len(FEATURE_COLUMNS)
    assert "Glucose" in names
    assert "BMI" in names
