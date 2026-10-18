"""Model explainability module."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH


class ExplainabilityError(Exception):
    """Custom exception for explainability errors."""

    pass


class ModelExplainer:
    """Provides model explanations for diabetes risk predictions."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the model explainer.

        Args:
            model_path: Path to the saved model pipeline.
        """
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.model_name = None

    def load_model(self) -> None:
        """Load the trained model.

        Raises:
            ExplainabilityError: If model cannot be loaded.
        """
        if not self.model_path.exists():
            raise ExplainabilityError(
                f"Model file not found at {self.model_path}. "
                "Please train the model first by running: python -m src.train"
            )

        try:
            self.model = joblib.load(self.model_path)
            # Try to get model name from metadata
            if hasattr(self.model, "named_steps") and "classifier" in self.model.named_steps:
                classifier = self.model.named_steps["classifier"]
                self.model_name = classifier.__class__.__name__
        except Exception as e:
            raise ExplainabilityError(f"Failed to load model: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model.

        Returns:
            Dictionary mapping feature names to importance values.

        Raises:
            ExplainabilityError: If model doesn't support feature importance.
        """
        if self.model is None:
            self.load_model()

        classifier = self.model.named_steps["classifier"]

        # Tree-based models
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            return dict(zip(FEATURE_COLUMNS, importances))

        # Logistic regression
        elif hasattr(classifier, "coef_"):
            coefficients = np.abs(classifier.coef_[0])
            return dict(zip(FEATURE_COLUMNS, coefficients))

        else:
            raise ExplainabilityError(
                f"Model type {type(classifier).__name__} does not support feature importance"
            )

    def get_coefficients(self) -> Dict[str, float]:
        """Get coefficients from logistic regression model.

        Returns:
            Dictionary mapping feature names to coefficient values.

        Raises:
            ExplainabilityError: If model is not logistic regression.
        """
        if self.model is None:
            self.load_model()

        classifier = self.model.named_steps["classifier"]

        if not hasattr(classifier, "coef_"):
            raise ExplainabilityError(
                f"Model type {type(classifier).__name__} does not have coefficients"
            )

        coefficients = classifier.coef_[0]
        return dict(zip(FEATURE_COLUMNS, coefficients))

    def explain_logistic_regression(self) -> Dict:
        """Provide detailed explanation for logistic regression model.

        Returns:
            Dictionary with coefficient interpretation.

        Raises:
            ExplainabilityError: If model is not logistic regression.
        """
        if self.model is None:
            self.load_model()

        classifier = self.model.named_steps["classifier"]

        if not hasattr(classifier, "coef_"):
            raise ExplainabilityError(
                "This method is only applicable to Logistic Regression models"
            )

        coefficients = classifier.coef_[0]
        feature_coef = dict(zip(FEATURE_COLUMNS, coefficients))

        # Sort by absolute coefficient value
        sorted_features = sorted(
            feature_coef.items(), key=lambda x: abs(x[1]), reverse=True
        )

        explanation = {
            "model_type": "Logistic Regression",
            "coefficients": feature_coef,
            "sorted_by_magnitude": [
                {"feature": feat, "coefficient": coef, "direction": "positive" if coef > 0 else "negative"}
                for feat, coef in sorted_features
            ],
            "interpretation": {
                "positive_coefficients": [
                    feat for feat, coef in sorted_features if coef > 0
                ],
                "negative_coefficients": [
                    feat for feat, coef in sorted_features if coef < 0
                ],
            },
            "limitations": [
                "Coefficients are standardized (features were scaled during preprocessing)",
                "These are statistical associations, not causal relationships",
                "Feature importance does not imply clinical significance",
                "Individual predictions depend on the combination of all features",
            ],
        }

        return explanation

    def explain_tree_model(self) -> Dict:
        """Provide detailed explanation for tree-based models.

        Returns:
            Dictionary with feature importance interpretation.

        Raises:
            ExplainabilityError: If model is not tree-based.
        """
        if self.model is None:
            self.load_model()

        classifier = self.model.named_steps["classifier"]

        if not hasattr(classifier, "feature_importances_"):
            raise ExplainabilityError(
                "This method is only applicable to tree-based models (Decision Tree, Random Forest)"
            )

        importances = classifier.feature_importances_
        feature_importance = dict(zip(FEATURE_COLUMNS, importances))

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        explanation = {
            "model_type": self.model_name,
            "feature_importance": feature_importance,
            "sorted_by_importance": [
                {"feature": feat, "importance": imp}
                for feat, imp in sorted_features
            ],
            "interpretation": {
                "most_important_features": [
                    feat for feat, imp in sorted_features[:3]
                ],
                "least_important_features": [
                    feat for feat, imp in sorted_features[-3:]
                ],
            },
            "limitations": [
                "Feature importance measures how useful a feature is for the model's predictions",
                "It does not indicate the direction of the relationship",
                "It does not imply causality",
                "Importance can be affected by feature correlations",
                "Different importance calculation methods may yield different rankings",
            ],
        }

        return explanation

    def get_explanation(self) -> Dict:
        """Get appropriate explanation based on model type.

        Returns:
            Dictionary with model explanation.
        """
        if self.model is None:
            self.load_model()

        classifier = self.model.named_steps["classifier"]

        if hasattr(classifier, "coef_"):
            return self.explain_logistic_regression()
        elif hasattr(classifier, "feature_importances_"):
            return self.explain_tree_model()
        else:
            return {
                "model_type": type(classifier).__name__,
                "message": "Explainability information not available for this model type",
            }


def main():
    """Test the explainability module."""
    explainer = ModelExplainer()

    try:
        explainer.load_model()
        explanation = explainer.get_explanation()
        print("Model explanation:")
        print(json.dumps(explanation, indent=2))
    except ExplainabilityError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
