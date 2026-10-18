"""Model training module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    CV_FOLDS,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    METRICS_PATH,
    METRICS_FILENAME,
    MODEL_PATH,
    MODEL_FILENAME,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.visualization import Visualizer


class ModelTrainer:
    """Handles model training and evaluation for the diabetes prediction system."""

    def __init__(self):
        """Initialize the model trainer."""
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.visualizer = Visualizer()
        self.models: Dict = {}
        self.evaluation_results: Dict = {}
        self.selected_model_name: Optional[str] = None
        self.final_pipeline: Optional[Pipeline] = None

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load dataset and prepare train/test split.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        print("Loading dataset...")
        data = self.data_loader.load_dataset()

        print("Performing data quality analysis...")
        quality_report = self.data_loader.analyze_data_quality()
        print(f"Dataset shape: {quality_report['total_records']} rows, {quality_report['total_features']} features")
        print(f"Target distribution: {quality_report['target_distribution']}")
        print(f"Duplicate records: {quality_report['duplicate_records']}")

        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = self.data_loader.prepare_train_test_split()

        # Replace zeros with NaN for imputation
        X_train = self.preprocessor.replace_zeros_with_nan(X_train)
        X_test = self.preprocessor.replace_zeros_with_nan(X_test)

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def create_models(self) -> Dict[str, Pipeline]:
        """Create the three required models with preprocessing pipelines.

        Returns:
            Dictionary of model name to pipeline.
        """
        models = {}

        # Common preprocessing pipeline
        preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # Logistic Regression with scaling
        models["Logistic Regression"] = Pipeline(
            [
                ("preprocessing", preprocessing),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=RANDOM_STATE,
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        # Decision Tree
        models["Decision Tree"] = Pipeline(
            [
                ("preprocessing", preprocessing),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        max_depth=10,
                        min_samples_split=5,
                    ),
                ),
            ]
        )

        # Random Forest
        models["Random Forest"] = Pipeline(
            [
                ("preprocessing", preprocessing),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                    ),
                ),
            ]
        )

        return models

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_scores: Predicted probabilities for positive class.

        Returns:
            Dictionary of metric names to values.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_scores),
            "Average Precision": average_precision_score(y_true, y_scores),
            "Brier Score": brier_score_loss(y_true, y_scores),
            "True Positives": int(tp),
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
        }

        return metrics

    def train_and_evaluate_model(
        self,
        model: Pipeline,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict:
        """Train and evaluate a single model.

        Args:
            model: Model pipeline to train.
            model_name: Name of the model.
            X_train: Training features.
            X_test: Test features.
            y_train: Training labels.
            y_test: Test labels.

        Returns:
            Dictionary containing evaluation results.
        """
        print(f"\nTraining {model_name}...")

        # Perform cross-validation on training set
        cv = StratifiedKFold(n_splits=CV_FOLDS, random_state=RANDOM_STATE, shuffle=True)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train on full training set
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_scores)

        # Add cross-validation information
        metrics["CV ROC-AUC Mean"] = float(cv_scores.mean())
        metrics["CV ROC-AUC Std"] = float(cv_scores.std())

        print(f"Test ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(f"Test Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Test F1-Score: {metrics['F1-Score']:.4f}")

        return {
            "model": model,
            "model_name": model_name,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_scores": y_scores,
        }

    def select_best_model(self, evaluation_results: Dict) -> str:
        """Select the best model based on ROC-AUC score.

        Args:
            evaluation_results: Dictionary of model evaluation results.

        Returns:
            Name of the selected model.
        """
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)

        comparison_data = []
        for model_name, results in evaluation_results.items():
            metrics = results["metrics"]
            comparison_data.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1-Score": metrics["F1-Score"],
                    "ROC-AUC": metrics["ROC-AUC"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Select based on ROC-AUC
        best_model_name = max(
            evaluation_results.keys(),
            key=lambda name: evaluation_results[name]["metrics"]["ROC-AUC"],
        )

        print(f"\nSelected model: {best_model_name}")
        print(f"Selection criterion: Highest ROC-AUC score")

        return best_model_name

    def train_all_models(self) -> None:
        """Train and evaluate all models, then select the best one."""
        # Load and prepare data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()

        # Create models
        models = self.create_models()

        # Train and evaluate each model
        for model_name, model in models.items():
            result = self.train_and_evaluate_model(
                model, model_name, X_train, X_test, y_train, y_test
            )
            self.evaluation_results[model_name] = result
            self.models[model_name] = result["model"]

        # Select best model
        self.selected_model_name = self.select_best_model(self.evaluation_results)
        self.final_pipeline = self.evaluation_results[self.selected_model_name]["model"]

        # Save model and metrics
        self.save_artifacts(X_test, y_test)

        print("\nTraining completed successfully!")

    def save_artifacts(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Save model pipeline and evaluation artifacts.

        Args:
            X_test: Test features for generating visualizations.
            y_test: Test labels for generating visualizations.
        """
        print("\nSaving model and artifacts...")

        # Save model pipeline
        joblib.dump(self.final_pipeline, MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

        # Prepare comprehensive metrics
        results = self.evaluation_results[self.selected_model_name]
        metrics = results["metrics"]

        # Add metadata
        artifacts_metadata = {
            "model_name": self.selected_model_name,
            "model_version": "1.0.0",
            "training_timestamp": datetime.now().isoformat(),
            "threshold": DEFAULT_THRESHOLD,
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
            "model_comparison": {
                name: {
                    "ROC-AUC": results["metrics"]["ROC-AUC"],
                    "Accuracy": results["metrics"]["Accuracy"],
                    "F1-Score": results["metrics"]["F1-Score"],
                }
                for name, results in self.evaluation_results.items()
            },
        }

        # Save metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(artifacts_metadata, f, indent=2)
        print(f"Metrics saved to: {METRICS_PATH}")

        # Generate and save visualizations
        y_pred = results["y_pred"]
        y_scores = results["y_scores"]

        # Confusion matrix
        cm_fig = self.visualizer.plot_confusion_matrix(y_test, y_pred)
        cm_fig.write_html(REPORTS_DIR / "confusion_matrix.html")

        # ROC curve
        roc_fig = self.visualizer.plot_roc_curve(
            y_test, y_scores, self.selected_model_name
        )
        roc_fig.write_html(REPORTS_DIR / "roc_curve.html")

        # Precision-recall curve
        pr_fig = self.visualizer.plot_precision_recall_curve(
            y_test, y_scores, self.selected_model_name
        )
        pr_fig.write_html(REPORTS_DIR / "precision_recall_curve.html")

        # Model comparison
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append(
                {
                    "Model": model_name,
                    "Accuracy": results["metrics"]["Accuracy"],
                    "Precision": results["metrics"]["Precision"],
                    "Recall": results["metrics"]["Recall"],
                    "F1-Score": results["metrics"]["F1-Score"],
                    "ROC-AUC": results["metrics"]["ROC-AUC"],
                }
            )
        comparison_df = pd.DataFrame(comparison_data)
        comp_fig = self.visualizer.plot_model_comparison(comparison_df)
        comp_fig.write_html(REPORTS_DIR / "model_comparison.html")

        # Feature importance (if applicable)
        if hasattr(self.final_pipeline.named_steps["classifier"], "feature_importances_"):
            importances = self.final_pipeline.named_steps["classifier"].feature_importances_
            fi_fig = self.visualizer.plot_feature_importance(
                FEATURE_COLUMNS, importances, self.selected_model_name
            )
            fi_fig.write_html(REPORTS_DIR / "feature_importance.html")
        elif hasattr(self.final_pipeline.named_steps["classifier"], "coef_"):
            # For logistic regression, use absolute coefficients as importance
            coef = np.abs(self.final_pipeline.named_steps["classifier"].coef_[0])
            fi_fig = self.visualizer.plot_feature_importance(
                FEATURE_COLUMNS, coef, self.selected_model_name
            )
            fi_fig.write_html(REPORTS_DIR / "feature_importance.html")

        print("Visualizations saved to reports directory")


def main():
    """Main training function."""
    trainer = ModelTrainer()
    trainer.train_all_models()


if __name__ == "__main__":
    main()
