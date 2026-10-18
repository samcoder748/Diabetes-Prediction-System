# Models Directory

This directory contains trained model artifacts.

## Files

- `diabetes_model.joblib`: The trained machine learning pipeline (created after training)

## Model Persistence

Models are saved using Joblib and include:
- Preprocessing pipeline (imputation and scaling)
- Trained classifier
- Model configuration

## Training

To train a new model, run:
```bash
python -m src.train
```

This will:
1. Load the dataset from `data/diabetes.csv`
2. Train and evaluate three models
3. Select the best model based on ROC-AUC
4. Save the model pipeline to this directory

## Important Notes

- Do not commit model files to version control
- Models can be regenerated using the training script
- The model includes preprocessing steps for consistent inference
