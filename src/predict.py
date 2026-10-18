"""Prediction module for diabetes risk assessment."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
)
from src.preprocessing import DataPreprocessor


class PredictionError(Exception):
    """Custom exception for prediction errors."""

    pass


class Predictor:
    """Handles diabetes risk predictions using a trained model."""

    def __init__(self, model_path: Optional[Path] = None, metrics_path: Optional[Path] = None):
        """Initialize the predictor.

        Args:
            model_path: Path to the saved model pipeline.
            metrics_path: Path to the saved metrics file.
        """
        self.model_path = model_path or MODEL_PATH
        self.metrics_path = metrics_path or METRICS_PATH
        self.model = None
        self.metadata = None
        self.threshold = DEFAULT_THRESHOLD
        self.preprocessor = DataPreprocessor()

    def load_model(self) -> None:
        """Load the trained model and metadata.

        Raises:
            PredictionError: If model or metadata cannot be loaded.
        """
        if not self.model_path.exists():
            raise PredictionError(
                f"Model file not found at {self.model_path}. "
                "Please train the model first by running: python -m src.train"
            )

        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise PredictionError(f"Failed to load model: {e}")

        # Load metadata
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r") as f:
                    self.metadata = json.load(f)
                self.threshold = self.metadata.get("threshold", DEFAULT_THRESHOLD)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                self.metadata = None
        else:
            print(f"Warning: Metrics file not found at {self.metrics_path}")
            self.metadata = None

    def predict_single(
        self, input_data: Dict[str, float]
    ) -> Dict[str, any]:
        """Make a prediction for a single patient.

        Args:
            input_data: Dictionary of feature names to values.

        Returns:
            Dictionary containing prediction results.

        Raises:
            PredictionError: If prediction fails.
        """
        if self.model is None:
            self.load_model()

        # Validate input
        from src.data_loader import DataLoader

        data_loader = DataLoader()
        is_valid, error_msg = data_loader.validate_input(input_data)

        if not is_valid:
            raise PredictionError(f"Invalid input: {error_msg}")

        # Prepare input DataFrame
        input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

        # Replace zeros with NaN for features where zero indicates missing
        input_df = self.preprocessor.replace_zeros_with_nan(input_df)

        try:
            # Get prediction and probability
            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            positive_class_prob = probabilities[1]

            # Apply threshold
            predicted_class = 1 if positive_class_prob >= self.threshold else 0

            result = {
                "predicted_class": int(predicted_class),
                "positive_class_probability": float(positive_class_prob),
                "negative_class_probability": float(probabilities[0]),
                "threshold": self.threshold,
                "model_name": self.metadata.get("model_name", "Unknown") if self.metadata else "Unknown",
                "model_version": self.metadata.get("model_version", "Unknown") if self.metadata else "Unknown",
                "input_features": input_data,
            }

            return result

        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}")

    def predict_batch(
        self, input_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Make predictions for multiple patients.

        Args:
            input_data: DataFrame with feature columns.

        Returns:
            DataFrame with predictions added.

        Raises:
            PredictionError: If prediction fails.
        """
        if self.model is None:
            self.load_model()

        # Validate columns
        missing_columns = set(FEATURE_COLUMNS) - set(input_data.columns)
        if missing_columns:
            raise PredictionError(f"Missing columns: {missing_columns}")

        # Replace zeros with NaN for features where zero indicates missing
        input_data = self.preprocessor.replace_zeros_with_nan(input_data)

        try:
            # Get predictions
            predictions = self.model.predict(input_data[FEATURE_COLUMNS])
            probabilities = self.model.predict_proba(input_data[FEATURE_COLUMNS])

            # Add results to DataFrame
            result_df = input_data.copy()
            result_df["predicted_class"] = predictions
            result_df["positive_class_probability"] = probabilities[:, 1]
            result_df["negative_class_probability"] = probabilities[:, 0]

            return result_df

        except Exception as e:
            raise PredictionError(f"Batch prediction failed: {e}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        info = {
            "model_path": str(self.model_path),
            "metrics_path": str(self.metrics_path),
            "threshold": self.threshold,
        }

        # Only try to load model if it exists
        if self.model_path.exists():
            if self.model is None:
                self.load_model()

            if self.metadata:
                info.update({
                    "model_name": self.metadata.get("model_name"),
                    "model_version": self.metadata.get("model_version"),
                    "training_timestamp": self.metadata.get("training_timestamp"),
                    "feature_columns": self.metadata.get("feature_columns"),
                    "metrics": self.metadata.get("metrics"),
                })

        return info


def main():
    """Test the prediction module."""
    predictor = Predictor()

    # Example prediction
    example_input = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    try:
        result = predictor.predict_single(example_input)
        print("Prediction result:")
        print(json.dumps(result, indent=2))
    except PredictionError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
