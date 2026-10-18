"""Tests for data_loader module."""

import pandas as pd
import pytest
from pathlib import Path

from src.config import FEATURE_COLUMNS
from src.data_loader import DataLoader, DatasetValidationError


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
        "Outcome": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def data_loader(tmp_path):
    """Create a DataLoader instance with temporary path."""
    loader = DataLoader(dataset_path=tmp_path / "test.csv")
    return loader


def test_validate_input_valid(data_loader):
    """Test validation of valid input."""
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

    is_valid, error_msg = data_loader.validate_input(input_data)
    assert is_valid is True
    assert error_msg is None


def test_validate_input_missing_feature(data_loader):
    """Test validation with missing feature."""
    input_data = {
        "Pregnancies": 6,
        "Glucose": 148,
        # Missing other features
    }

    is_valid, error_msg = data_loader.validate_input(input_data)
    assert is_valid is False
    assert "Missing features" in error_msg


def test_validate_input_out_of_bounds(data_loader):
    """Test validation with out-of-bounds value."""
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

    is_valid, error_msg = data_loader.validate_input(input_data)
    assert is_valid is False
    assert "outside valid range" in error_msg


def test_validate_input_non_numeric(data_loader):
    """Test validation with non-numeric value."""
    input_data = {
        "Pregnancies": "six",  # String instead of number
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    is_valid, error_msg = data_loader.validate_input(input_data)
    assert is_valid is False
    assert "must be numeric" in error_msg


def test_analyze_data_quality(data_loader, sample_data, tmp_path):
    """Test data quality analysis."""
    # Save sample data
    sample_path = tmp_path / "test.csv"
    sample_data.to_csv(sample_path, index=False)

    data_loader.dataset_path = sample_path
    data_loader.load_dataset()

    quality_report = data_loader.analyze_data_quality()

    assert quality_report["total_records"] == 5
    assert quality_report["total_features"] == 8
    assert "target_distribution" in quality_report
    assert "feature_statistics" in quality_report


def test_prepare_train_test_split(data_loader, sample_data, tmp_path):
    """Test train/test split preparation."""
    # Save sample data
    sample_path = tmp_path / "test.csv"
    sample_data.to_csv(sample_path, index=False)

    data_loader.dataset_path = sample_path
    data_loader.load_dataset()

    X_train, X_test, y_train, y_test = data_loader.prepare_train_test_split(test_size=0.4)

    assert len(X_train) == 3  # 60% of 5
    assert len(X_test) == 2  # 40% of 5
    assert len(y_train) == 3
    assert len(y_test) == 2
    assert list(X_train.columns) == FEATURE_COLUMNS


def test_validate_uploaded_csv_valid(data_loader, sample_data, tmp_path):
    """Test validation of uploaded CSV with valid data."""
    # Create a mock uploaded file
    csv_path = tmp_path / "upload.csv"
    sample_data.to_csv(csv_path, index=False)

    # Mock uploaded file object
    class MockUploadedFile:
        def __init__(self, path):
            self.path = path

        def read(self):
            with open(self.path, "rb") as f:
                return f.read()

        def getvalue(self):
            with open(self.path, "rb") as f:
                return f.read()

    uploaded = MockUploadedFile(csv_path)

    is_valid, error_msg, df = data_loader.validate_uploaded_csv(uploaded)

    assert is_valid is True
    assert error_msg is None
    assert df is not None
    assert len(df) == 5


def test_validate_uploaded_csv_missing_columns(data_loader, tmp_path):
    """Test validation of uploaded CSV with missing columns."""
    # Create CSV with missing columns
    bad_data = pd.DataFrame({
        "Pregnancies": [1, 2],
        "Glucose": [100, 120],
        # Missing other required columns
    })
    csv_path = tmp_path / "bad.csv"
    bad_data.to_csv(csv_path, index=False)

    class MockUploadedFile:
        def __init__(self, path):
            self.path = path

        def read(self):
            with open(self.path, "rb") as f:
                return f.read()

        def getvalue(self):
            with open(self.path, "rb") as f:
                return f.read()

    uploaded = MockUploadedFile(csv_path)

    is_valid, error_msg, df = data_loader.validate_uploaded_csv(uploaded)

    assert is_valid is False
    assert "Missing required columns" in error_msg
    assert df is None


def test_validate_uploaded_csv_empty(data_loader, tmp_path):
    """Test validation of empty uploaded CSV."""
    empty_data = pd.DataFrame(columns=FEATURE_COLUMNS)
    csv_path = tmp_path / "empty.csv"
    empty_data.to_csv(csv_path, index=False)

    class MockUploadedFile:
        def __init__(self, path):
            self.path = path

        def read(self):
            with open(self.path, "rb") as f:
                return f.read()

        def getvalue(self):
            with open(self.path, "rb") as f:
                return f.read()

    uploaded = MockUploadedFile(csv_path)

    is_valid, error_msg, df = data_loader.validate_uploaded_csv(uploaded)

    assert is_valid is False
    assert "empty" in error_msg.lower()
    assert df is None
