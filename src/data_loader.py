"""Data loading and validation module."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    INPUT_BOUNDS,
    MISSING_VALUE_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


class DatasetValidationError(Exception):
    """Custom exception for dataset validation errors."""

    pass


class DataLoader:
    """Handles loading, validation, and splitting of the diabetes dataset."""

    def __init__(self, dataset_path: Optional[Path] = None):
        """Initialize the DataLoader.

        Args:
            dataset_path: Path to the dataset CSV file. Defaults to config value.
        """
        self.dataset_path = dataset_path or DATASET_PATH
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from CSV file.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            DatasetValidationError: If dataset format is invalid.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Please download the Pima Indians Diabetes dataset and place it in the data directory."
            )

        try:
            self.data = pd.read_csv(self.dataset_path)
        except Exception as e:
            raise DatasetValidationError(f"Failed to read CSV file: {e}")

        self._validate_dataset()
        self._generate_metadata()

        return self.data

    def _validate_dataset(self) -> None:
        """Validate the dataset structure and content.

        Raises:
            DatasetValidationError: If validation fails.
        """
        if self.data is None:
            raise DatasetValidationError("No data loaded")

        # Check required columns
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = set(required_columns) - set(self.data.columns)

        if missing_columns:
            raise DatasetValidationError(
                f"Missing required columns: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )

        # Check for empty dataset
        if len(self.data) == 0:
            raise DatasetValidationError("Dataset is empty")

        # Check target values
        valid_targets = {0, 1}
        invalid_targets = set(self.data[TARGET_COLUMN].unique()) - valid_targets

        if invalid_targets:
            raise DatasetValidationError(
                f"Invalid target values found: {invalid_targets}. "
                f"Target column must contain only 0 and 1."
            )

    def _generate_metadata(self) -> None:
        """Generate dataset metadata."""
        if self.data is None:
            return

        # Calculate dataset hash for reproducibility
        dataset_hash = hashlib.md5(
            pd.util.hash_pandas_object(self.data).values
        ).hexdigest()

        self.metadata = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "zero_as_missing": {
                col: int((self.data[col] == 0).sum())
                for col in MISSING_VALUE_FEATURES
            },
            "target_distribution": self.data[TARGET_COLUMN].value_counts().to_dict(),
            "dataset_hash": dataset_hash,
        }

    def analyze_data_quality(self) -> Dict:
        """Perform comprehensive data quality analysis.

        Returns:
            Dictionary containing data quality metrics.
        """
        if self.data is None:
            raise DatasetValidationError("No data loaded")

        quality_report = {
            "total_records": len(self.data),
            "total_features": len(FEATURE_COLUMNS),
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicate_records": int(self.data.duplicated().sum()),
            "target_distribution": self.data[TARGET_COLUMN].value_counts().to_dict(),
            "class_balance_ratio": self.data[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
            "feature_statistics": self.data[FEATURE_COLUMNS].describe().to_dict(),
            "zero_as_missing": {
                col: {
                    "count": int((self.data[col] == 0).sum()),
                    "percentage": float((self.data[col] == 0).sum() / len(self.data) * 100),
                }
                for col in MISSING_VALUE_FEATURES
            },
        }

        return quality_report

    def prepare_train_test_split(
        self,
        data: Optional[pd.DataFrame] = None,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the dataset into training and testing sets.

        Args:
            data: DataFrame to split. If None, uses loaded data.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Raises:
            DatasetValidationError: If data is not available.
        """
        if data is None:
            data = self.data

        if data is None:
            raise DatasetValidationError("No data available for splitting")

        X = data[FEATURE_COLUMNS]
        y = data[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def validate_input(
        self, input_data: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """Validate a single prediction input.

        Args:
            input_data: Dictionary of feature names to values.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check all required features are present
        missing_features = set(FEATURE_COLUMNS) - set(input_data.keys())
        if missing_features:
            return False, f"Missing features: {missing_features}"

        # Validate each feature
        for feature, value in input_data.items():
            if feature not in FEATURE_COLUMNS:
                continue

            # Check if value is numeric
            if not isinstance(value, (int, float)):
                return False, f"Feature '{feature}' must be numeric"

            # Check bounds
            min_val, max_val = INPUT_BOUNDS[feature]
            if not (min_val <= value <= max_val):
                return False, (
                    f"Feature '{feature}' value {value} is outside valid range "
                    f"[{min_val}, {max_val}]"
                )

        return True, None

    def validate_uploaded_csv(self, uploaded_file) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
        """Validate an uploaded CSV file for batch evaluation.

        Args:
            uploaded_file: Streamlit uploaded file object.

        Returns:
            Tuple of (is_valid, error_message, dataframe).
        """
        try:
            # Handle both file-like objects and paths
            if hasattr(uploaded_file, 'read'):
                content = uploaded_file.read()
                import io
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            return False, f"Failed to read CSV: {e}", None

        # Check required columns
        missing_columns = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}", None

        # Validate row count
        if len(df) == 0:
            return False, "CSV file is empty", None

        # Validate data types
        for col in FEATURE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column '{col}' must be numeric", None

        return True, None, df
