# Data Directory

This directory contains the dataset files for the Diabetes Risk Prediction System.

## Dataset

The system uses the Pima Indians Diabetes Database.

### Source

The dataset can be obtained from:
- UCI Machine Learning Repository
- Kaggle

### File Format

Place the dataset file as `diabetes.csv` in this directory.

### Schema

The CSV file should contain the following columns:

- Pregnancies: Number of pregnancies
- Glucose: Plasma glucose concentration (2 hours after oral glucose tolerance test)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Target variable (0 = negative, 1 = positive)

### Important Notes

- Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are treated as missing values
- The system will validate the dataset structure before training
- Do not commit real patient datasets to version control
