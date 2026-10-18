"""Visualization module for exploratory data analysis and model results."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.config import FEATURE_COLUMNS, REPORTS_DIR, TARGET_COLUMN


class Visualizer:
    """Handles visualization creation for the diabetes prediction system."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save static plots. Defaults to config value.
        """
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def plot_feature_distributions(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interactive feature distribution plots.

        Args:
            data: DataFrame with features.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        n_features = len(FEATURE_COLUMNS)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=FEATURE_COLUMNS,
        )

        for idx, feature in enumerate(FEATURE_COLUMNS):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            fig.add_trace(
                go.Histogram(
                    x=data[feature],
                    name=feature,
                    nbinsx=30,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            height=300 * n_rows,
            title_text="Feature Distributions",
            showlegend=False,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_correlation_heatmap(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ) -> go.Figure:
        """Create correlation heatmap.

        Args:
            data: DataFrame with features.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        correlation_matrix = data[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()

        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_target_distribution(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ) -> go.Figure:
        """Create target class distribution plot.

        Args:
            data: DataFrame with target column.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        target_counts = data[TARGET_COLUMN].value_counts()
        target_labels = ["Negative (0)", "Positive (1)"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=target_labels,
                    y=[target_counts.get(0, 0), target_counts.get(1, 0)],
                    marker_color=["#3498db", "#e74c3c"],
                )
            ]
        )

        fig.update_layout(
            title="Target Class Distribution",
            xaxis_title="Diabetes Outcome",
            yaxis_title="Count",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_feature_comparison_by_target(
        self, data: pd.DataFrame, save_path: Optional[str] = None
    ) -> go.Figure:
        """Create feature comparison plots by target class.

        Args:
            data: DataFrame with features and target.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        n_features = len(FEATURE_COLUMNS)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=FEATURE_COLUMNS,
        )

        for idx, feature in enumerate(FEATURE_COLUMNS):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            for target_val in [0, 1]:
                fig.add_trace(
                    go.Box(
                        y=data[data[TARGET_COLUMN] == target_val][feature],
                        name=f"Class {target_val}",
                        showlegend=idx == 0,
                        legendgroup=f"Class {target_val}",
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            height=300 * n_rows,
            title_text="Feature Distributions by Target Class",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
    ) -> go.Figure:
        """Create confusion matrix visualization.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )

        fig.update_xaxes(ticktext=["Negative", "Positive"], tickvals=[0, 1])
        fig.update_yaxes(ticktext=["Negative", "Positive"], tickvals=[0, 1])

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create ROC curve visualization.

        Args:
            y_true: True labels.
            y_scores: Predicted probabilities or scores.
            model_name: Name of the model for the legend.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{model_name} (AUC = {roc_auc:.3f})",
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="Random Classifier",
                mode="lines",
                line=dict(dash="dash"),
            )
        )

        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create precision-recall curve visualization.

        Args:
            y_true: True labels.
            y_scores: Predicted probabilities or scores.
            model_name: Name of the model for the legend.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                name=f"{model_name} (AP = {avg_precision:.3f})",
                mode="lines",
            )
        )

        fig.update_layout(
            title=f"Precision-Recall Curve - {model_name}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_model_comparison(
        self, comparison_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> go.Figure:
        """Create model comparison bar chart.

        Args:
            comparison_df: DataFrame with model comparison metrics.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        available_metrics = [m for m in metrics if m in comparison_df.columns]

        fig = go.Figure()

        for metric in available_metrics:
            fig.add_trace(
                go.Bar(
                    x=comparison_df["Model"],
                    y=comparison_df[metric],
                    name=metric,
                )
            )

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create feature importance bar chart.

        Args:
            feature_names: List of feature names.
            importances: Array of importance values.
            model_name: Name of the model.
            save_path: Optional path to save the plot.

        Returns:
            Plotly figure object.
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_importances,
                    y=sorted_names,
                    orientation="h",
                )
            ]
        )

        fig.update_layout(
            title=f"Feature Importance - {model_name}",
            xaxis_title="Importance",
            yaxis_title="Feature",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def save_metrics_json(self, metrics: Dict, filename: str = "metrics.json") -> Path:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics.
            filename: Name of the output file.

        Returns:
            Path to the saved file.
        """
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return output_path


# Import make_subplots at module level
from plotly.subplots import make_subplots
