# Reports Directory

This directory contains evaluation reports and visualizations.

## Files

- `evaluation_metrics.json`: Model evaluation metrics and metadata
- `confusion_matrix.html`: Confusion matrix visualization
- `roc_curve.html`: ROC curve visualization
- `precision_recall_curve.html`: Precision-recall curve visualization
- `model_comparison.html`: Model performance comparison chart
- `feature_importance.html`: Feature importance visualization

## Generation

These files are automatically generated during model training.

To regenerate reports, run:
```bash
python -m src.train
```

## Important Notes

- Reports are based on actual model evaluation results
- Do not commit report files to version control
- Reports can be regenerated using the training script
