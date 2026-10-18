"""Streamlit application for Diabetes Risk Prediction System."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    INPUT_BOUNDS,
    METRICS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data_loader import DataLoader
from src.explainability import ModelExplainer
from src.predict import Predictor, PredictionError
from src.visualization import Visualizer

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_model_exists() -> bool:
    """Check if trained model exists."""
    return MODEL_PATH.exists()


def load_metrics() -> dict:
    """Load evaluation metrics."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


def dashboard_page():
    """Render the main dashboard page."""
    st.title("🏥 Diabetes Risk Prediction System")
    st.markdown("---")

    # Disclaimer
    st.warning(
        """
        ⚠️ **Medical Disclaimer**: This application is an academic decision-support demonstration, not a diagnostic tool.
        It does not replace laboratory testing or qualified healthcare professionals. A predicted probability is not
        necessarily a person's true clinical probability of developing diabetes.
        """
    )

    if not check_model_exists():
        st.error(
            """
            **Model not found!**

            Please train the model first by running:
            ```
            python -m src.train
            ```

            Make sure you have placed the dataset file at: `data/diabetes.csv`
            """
        )
        return

    # Load metrics
    metrics = load_metrics()

    if metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Model",
                metrics.get("model_name", "Unknown"),
            )

        with col2:
            st.metric(
                "Training Date",
                metrics.get("training_timestamp", "Unknown")[:10],
            )

        with col3:
            if "metrics" in metrics:
                st.metric(
                    "Test ROC-AUC",
                    f"{metrics['metrics']['ROC-AUC']:.4f}",
                )

        with col4:
            if "metrics" in metrics:
                st.metric(
                    "Test Accuracy",
                    f"{metrics['metrics']['Accuracy']:.4f}",
                )

        st.markdown("---")

        # Model comparison
        st.subheader("Model Comparison")
        if "model_comparison" in metrics:
            comparison_data = []
            for model_name, model_metrics in metrics["model_comparison"].items():
                comparison_data.append({
                    "Model": model_name,
                    "ROC-AUC": model_metrics["ROC-AUC"],
                    "Accuracy": model_metrics["Accuracy"],
                    "F1-Score": model_metrics["F1-Score"],
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

        # Test set performance
        st.subheader("Test Set Performance")
        if "metrics" in metrics:
            m = metrics["metrics"]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Precision", f"{m['Precision']:.4f}")
            with col2:
                st.metric("Recall", f"{m['Recall']:.4f}")
            with col3:
                st.metric("F1-Score", f"{m['F1-Score']:.4f}")
            with col4:
                st.metric("Specificity", f"{m['Specificity']:.4f}")

            # Confusion matrix values
            st.markdown("### Confusion Matrix")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Positives", m["True Positives"])
            with col2:
                st.metric("True Negatives", m["True Negatives"])
            with col3:
                st.metric("False Positives", m["False Positives"])
            with col4:
                st.metric("False Negatives", m["False Negatives"])

    else:
        st.warning("Metrics file not found. Please ensure the model has been trained properly.")


def prediction_page():
    """Render the prediction page."""
    st.title("🔮 Diabetes Risk Prediction")
    st.markdown("---")

    st.warning(
        """
        ⚠️ **Important**: This is a model prediction, not a medical diagnosis.
        Please consult a healthcare professional for individual medical concerns.
        """
    )

    if not check_model_exists():
        st.error("Model not found. Please train the model first.")
        return

    # Input form
    st.subheader("Enter Patient Measurements")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=INPUT_BOUNDS["Pregnancies"][0],
                max_value=INPUT_BOUNDS["Pregnancies"][1],
                value=0,
                help="Number of pregnancies"
            )

            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=INPUT_BOUNDS["Glucose"][0],
                max_value=INPUT_BOUNDS["Glucose"][1],
                value=120,
                help="Plasma glucose concentration (2 hours after oral glucose tolerance test)"
            )

            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=INPUT_BOUNDS["BloodPressure"][0],
                max_value=INPUT_BOUNDS["BloodPressure"][1],
                value=70,
                help="Diastolic blood pressure"
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=INPUT_BOUNDS["SkinThickness"][0],
                max_value=INPUT_BOUNDS["SkinThickness"][1],
                value=20,
                help="Triceps skin fold thickness"
            )

        with col2:
            insulin = st.number_input(
                "Insulin (mu U/ml)",
                min_value=INPUT_BOUNDS["Insulin"][0],
                max_value=INPUT_BOUNDS["Insulin"][1],
                value=80,
                help="2-Hour serum insulin"
            )

            bmi = st.number_input(
                "BMI",
                min_value=INPUT_BOUNDS["BMI"][0],
                max_value=INPUT_BOUNDS["BMI"][1],
                value=25.0,
                step=0.1,
                help="Body mass index"
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=INPUT_BOUNDS["DiabetesPedigreeFunction"][0],
                max_value=INPUT_BOUNDS["DiabetesPedigreeFunction"][1],
                value=0.5,
                step=0.001,
                help="Diabetes pedigree function (genetic score)"
            )

            age = st.number_input(
                "Age (years)",
                min_value=INPUT_BOUNDS["Age"][0],
                max_value=INPUT_BOUNDS["Age"][1],
                value=30,
                help="Age in years"
            )

        submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            input_data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }

            try:
                predictor = Predictor()
                result = predictor.predict_single(input_data)

                st.markdown("---")
                st.subheader("Prediction Results")

                # Display prediction
                pred_class = result["predicted_class"]
                prob = result["positive_class_probability"]

                if pred_class == 1:
                    st.error(
                        f"""
                        **Model predicted positive class**

                        Probability of positive class: {prob:.2%}

                        Threshold used: {result['threshold']}
                        """
                    )
                else:
                    st.success(
                        f"""
                        **Model predicted negative class**

                        Probability of positive class: {prob:.2%}

                        Threshold used: {result['threshold']}
                        """
                    )

                # Probability bar
                st.markdown("### Probability Distribution")
                prob_df = pd.DataFrame({
                    "Class": ["Negative", "Positive"],
                    "Probability": [
                        result["negative_class_probability"],
                        result["positive_class_probability"]
                    ]
                })

                fig = go.Figure(data=[
                    go.Bar(
                        x=prob_df["Class"],
                        y=prob_df["Probability"],
                        marker_color=["#3498db", "#e74c3c"]
                    )
                ])
                fig.update_layout(
                    yaxis_title="Probability",
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig, use_container_width=True)

                # Model info
                st.markdown("### Model Information")
                st.info(f"""
                - Model: {result['model_name']}
                - Version: {result['model_version']}
                - Threshold: {result['threshold']}
                """)

                # Disclaimer
                st.warning(
                    """
                    **Reminder**: This result is a model prediction, not a medical diagnosis.
                    Please consult a healthcare professional for individual medical concerns.
                    """
                )

            except PredictionError as e:
                st.error(f"Prediction error: {e}")


def model_performance_page():
    """Render the model performance page."""
    st.title("📊 Model Performance")
    st.markdown("---")

    if not check_model_exists():
        st.error("Model not found. Please train the model first.")
        return

    metrics = load_metrics()

    if not metrics:
        st.warning("Metrics file not found.")
        return

    # Model comparison table
    st.subheader("Model Comparison")
    if "model_comparison" in metrics:
        comparison_data = []
        for model_name, model_metrics in metrics["model_comparison"].items():
            comparison_data.append({
                "Model": model_name,
                "ROC-AUC": f"{model_metrics['ROC-AUC']:.4f}",
                "Accuracy": f"{model_metrics['Accuracy']:.4f}",
                "F1-Score": f"{model_metrics['F1-Score']:.4f}",
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

    # Detailed metrics
    st.subheader("Selected Model Performance")
    if "metrics" in metrics:
        m = metrics["metrics"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{m['Accuracy']:.4f}")
            st.metric("Precision", f"{m['Precision']:.4f}")
        with col2:
            st.metric("Recall/Sensitivity", f"{m['Recall']:.4f}")
            st.metric("Specificity", f"{m['Specificity']:.4f}")
        with col3:
            st.metric("F1-Score", f"{m['F1-Score']:.4f}")
            st.metric("ROC-AUC", f"{m['ROC-AUC']:.4f}")

        st.markdown("### Confusion Matrix")
        cm_fig_path = REPORTS_DIR / "confusion_matrix.html"
        if cm_fig_path.exists():
            with open(cm_fig_path, "r") as f:
                st.components.v1.html(f.read(), height=500)
        else:
            st.warning("Confusion matrix visualization not found.")

        st.markdown("### ROC Curve")
        roc_fig_path = REPORTS_DIR / "roc_curve.html"
        if roc_fig_path.exists():
            with open(roc_fig_path, "r") as f:
                st.components.v1.html(f.read(), height=500)
        else:
            st.warning("ROC curve visualization not found.")

        st.markdown("### Precision-Recall Curve")
        pr_fig_path = REPORTS_DIR / "precision_recall_curve.html"
        if pr_fig_path.exists():
            with open(pr_fig_path, "r") as f:
                st.components.v1.html(f.read(), height=500)
        else:
            st.warning("Precision-recall curve visualization not found.")

    # Training methodology
    st.subheader("Training and Validation Methodology")
    st.info(f"""
    - **Random State**: {metrics.get('random_state', 'N/A')}
    - **Test Size**: {metrics.get('test_size', 'N/A')}
    - **Cross-Validation Folds**: 5
    - **Features Used**: {', '.join(metrics.get('feature_columns', []))}
    - **Classification Threshold**: {metrics.get('threshold', 'N/A')}
    """)

    # Limitations
    st.subheader("Known Model Limitations")
    st.warning("""
    - The model was trained on the Pima Indians Diabetes dataset, which may not generalize to all populations
    - Certain clinical measurements with zero values were treated as missing and imputed
    - The model does not account for all risk factors for diabetes
    - Predictions are probabilistic and should be interpreted with caution
    - This is an academic demonstration, not a clinically validated tool
    """)


def dataset_explorer_page():
    """Render the dataset explorer page."""
    st.title("📁 Dataset Explorer")
    st.markdown("---")

    if not DATASET_PATH.exists():
        st.error(
            """
            Dataset not found!

            Please download the Pima Indians Diabetes dataset and place it at: `data/diabetes.csv`

            The dataset can be obtained from the UCI Machine Learning Repository or Kaggle.
            """
        )
        return

    try:
        data_loader = DataLoader()
        data = data_loader.load_dataset()

        # Dataset summary
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Total Features", len(FEATURE_COLUMNS))
        with col3:
            st.metric("Duplicate Records", data.duplicated().sum())

        # Target distribution
        st.subheader("Target Distribution")
        target_counts = data[TARGET_COLUMN].value_counts()
        st.bar_chart(target_counts)

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10), use_container_width=True)

        # Feature descriptions
        st.subheader("Feature Descriptions")
        feature_descriptions = {
            "Pregnancies": "Number of pregnancies",
            "Glucose": "Plasma glucose concentration (2 hours after oral glucose tolerance test)",
            "BloodPressure": "Diastolic blood pressure (mm Hg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (weight in kg / (height in m)^2)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function (genetic score)",
            "Age": "Age (years)",
        }

        for feature, description in feature_descriptions.items():
            st.markdown(f"**{feature}**: {description}")

        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(data[FEATURE_COLUMNS].describe(), use_container_width=True)

        # Missing value analysis
        st.subheader("Missing Value Analysis")
        st.info("""
        In this dataset, zero values in certain clinical measurements may represent missing or implausible measurements.
        The following features are treated as having missing values when zero:
        - Glucose
        - Blood Pressure
        - Skin Thickness
        - Insulin
        - BMI
        """)

        missing_analysis = data_loader.analyze_data_quality()
        missing_df = pd.DataFrame([
            {
                "Feature": feature,
                "Zero Count": missing_analysis["zero_as_missing"][feature]["count"],
                "Zero Percentage": f"{missing_analysis['zero_as_missing'][feature]['percentage']:.2f}%",
            }
            for feature in [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
            ]
        ])
        st.dataframe(missing_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def explainability_page():
    """Render the explainability page."""
    st.title("🔍 Model Explainability")
    st.markdown("---")

    st.warning(
        """
        ⚠️ **Important**: Feature importance measures how useful a feature is for the model's predictions.
        It does not indicate causal relationships or clinical significance. Individual predictions depend
        on the combination of all features.
        """
    )

    if not check_model_exists():
        st.error("Model not found. Please train the model first.")
        return

    try:
        explainer = ModelExplainer()
        explainer.load_model()
        explanation = explainer.get_explanation()

        st.subheader(f"Model Type: {explanation.get('model_type', 'Unknown')}")

        # Feature importance
        if "feature_importance" in explanation:
            st.subheader("Feature Importance")
            importance_data = []
            for feature, importance in explanation["feature_importance"].items():
                importance_data.append({
                    "Feature": feature,
                    "Importance": importance,
                })

            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values("Importance", ascending=False)
            st.dataframe(importance_df, use_container_width=True)

            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_df["Importance"],
                    y=importance_df["Feature"],
                    orientation="h",
                )
            ])
            fig.update_layout(
                xaxis_title="Importance",
                yaxis_title="Feature",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Coefficients (for logistic regression)
        if "coefficients" in explanation:
            st.subheader("Logistic Regression Coefficients")

            coef_data = []
            for feature, coef in explanation["coefficients"].items():
                coef_data.append({
                    "Feature": feature,
                    "Coefficient": coef,
                    "Direction": "Positive" if coef > 0 else "Negative",
                })

            coef_df = pd.DataFrame(coef_data)
            coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)
            st.dataframe(coef_df, use_container_width=True)

            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=coef_df["Coefficient"],
                    y=coef_df["Feature"],
                    orientation="h",
                    marker_color=["#e74c3c" if c < 0 else "#3498db" for c in coef_df["Coefficient"]]
                )
            ])
            fig.update_layout(
                xaxis_title="Coefficient Value",
                yaxis_title="Feature",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info("""
            **Coefficient Interpretation**:
            - Positive coefficients: Higher feature values are associated with higher predicted probability of diabetes
            - Negative coefficients: Higher feature values are associated with lower predicted probability of diabetes
            - Coefficients are standardized (features were scaled during preprocessing)
            """)

        # Limitations
        st.subheader("Limitations")
        if "limitations" in explanation:
            for limitation in explanation["limitations"]:
                st.markdown(f"- {limitation}")

    except Exception as e:
        st.error(f"Error loading explanation: {e}")


def project_info_page():
    """Render the project information page."""
    st.title("📖 Project Information")
    st.markdown("---")

    st.markdown("""
    ## Problem Statement

    Diabetes is a significant public health concern. Machine learning can help demonstrate how clinical
    and demographic measurements relate to diabetes outcomes. This project uses the Pima Indians Diabetes
    dataset to train and evaluate supervised learning models and provide an interactive educational
    risk-assessment interface.

    ## Objectives

    - Develop a web application that accepts patient measurements and validates inputs
    - Process measurements using a saved machine learning pipeline
    - Return calibrated predictions with supporting explanations
    - Provide educational insights into model behavior and feature importance

    ## System Architecture

    The system follows a modular architecture with clear separation of concerns:

    - **Data Loading**: Dataset ingestion, validation, and quality analysis
    - **Preprocessing**: Missing value imputation and feature scaling
    - **Training**: Model training with cross-validation and hyperparameter tuning
    - **Evaluation**: Comprehensive metric calculation and visualization
    - **Prediction**: Inference pipeline for new patient data
    - **Explainability**: Feature importance and coefficient interpretation

    ## Dataset

    - **Source**: Pima Indians Diabetes Database
    - **Records**: Approximately 768 records
    - **Features**: 8 clinical and demographic measurements
    - **Target**: Diabetes outcome (0 = negative, 1 = positive)

    ## Features Used

    1. Pregnancies: Number of pregnancies
    2. Glucose: Plasma glucose concentration (2 hours after oral glucose tolerance test)
    3. Blood Pressure: Diastolic blood pressure (mm Hg)
    4. Skin Thickness: Triceps skin fold thickness (mm)
    5. Insulin: 2-Hour serum insulin (mu U/ml)
    6. BMI: Body mass index
    7. Diabetes Pedigree Function: Genetic diabetes score
    8. Age: Age in years

    ## Algorithms

    Three supervised learning algorithms were trained and compared:

    1. **Logistic Regression**: Linear classifier with interpretable coefficients
    2. **Decision Tree**: Non-linear classifier with clear decision rules
    3. **Random Forest**: Ensemble method combining multiple decision trees

    ## Evaluation Methodology

    - **Train/Test Split**: Stratified 80/20 split
    - **Cross-Validation**: 5-fold stratified cross-validation on training set
    - **Metrics**: Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC
    - **Model Selection**: Based on ROC-AUC score

    ## Limitations

    - The dataset is specific to Pima Indian women and may not generalize to other populations
    - Missing values were imputed using median values
    - The model does not account for all diabetes risk factors
    - This is an academic demonstration, not a clinically validated diagnostic tool
    - Predictions should not replace professional medical advice

    ## Ethical Considerations

    - No personal identifiers are collected or stored
    - Patient measurements are not logged or saved by default
    - The application includes clear medical disclaimers
    - Model limitations are transparently communicated
    - The system is designed for educational purposes only

    ## Future Enhancements

    - Integration with additional clinical datasets
    - More sophisticated imputation strategies
    - Calibration of probability estimates
    - Support for temporal predictions
    - Integration with electronic health records
    - Prospective clinical validation

    ## References

    - UCI Machine Learning Repository: Pima Indians Diabetes Dataset
    - Scikit-learn Documentation
    - Streamlit Documentation
    """)


def main():
    """Main application entry point."""
    # Sidebar navigation
    with st.sidebar:
        st.title("🏥 Diabetes Risk")
        st.markdown("---")

        page = option_menu(
            menu_title=None,
            options=[
                "Dashboard",
                "Prediction",
                "Model Performance",
                "Dataset Explorer",
                "Explainability",
                "Project Info",
            ],
            icons=[
                "house",
                "magic",
                "graph-up",
                "database",
                "search",
                "info-circle",
            ],
            default_index=0,
        )

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #888; font-size: 12px;'>
            Academic Demonstration<br>
            Not a Medical Tool
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render selected page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Dataset Explorer":
        dataset_explorer_page()
    elif page == "Explainability":
        explainability_page()
    elif page == "Project Info":
        project_info_page()


if __name__ == "__main__":
    main()
