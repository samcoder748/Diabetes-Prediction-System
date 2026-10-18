"""Data preprocessing module."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, MISSING_VALUE_FEATURES


class DataPreprocessor:
    """Handles data preprocessing for the diabetes dataset."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.pipeline = None
        self.feature_names = FEATURE_COLUMNS

    def create_pipeline(self) -> Pipeline:
        """Create the preprocessing pipeline.

        Returns:
            Scikit-learn Pipeline with imputation and scaling.
        """
        # Use default NaN imputation (after replace_zeros_with_nan)
        self.pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        return self.pipeline

    def replace_zeros_with_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replace zeros with NaN for features where zero indicates missing.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with zeros replaced by NaN for specified features.
        """
        data_copy = data.copy()

        for feature in MISSING_VALUE_FEATURES:
            if feature in data_copy.columns:
                data_copy[feature] = data_copy[feature].replace(0, np.nan)

        return data_copy

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessing pipeline on training data and transform it.

        Args:
            X_train: Training feature DataFrame.

        Returns:
            Transformed training data as numpy array.
        """
        if self.pipeline is None:
            self.create_pipeline()

        # Replace zeros with NaN for imputation
        X_train_processed = self.replace_zeros_with_nan(X_train)

        # Fit and transform
        X_train_transformed = self.pipeline.fit_transform(X_train_processed)

        return X_train_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using the fitted pipeline.

        Args:
            X: Feature DataFrame.

        Returns:
            Transformed data as numpy array.

        Raises:
            ValueError: If pipeline has not been fitted.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted. Call fit_transform first.")

        # Replace zeros with NaN for imputation
        X_processed = self.replace_zeros_with_nan(X)

        # Transform
        X_transformed = self.pipeline.transform(X_processed)

        return X_transformed

    def get_feature_names_out(self) -> list:
        """Get the feature names after preprocessing.

        Returns:
            List of feature names.
        """
        return self.feature_names
