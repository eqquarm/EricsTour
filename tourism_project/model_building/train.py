# for data manipulation
import pandas as pd
# for model serialization and experiment tracking
import mlflow

# Import the new utility function
from .train_utils import run_full_training_pipeline

# complete the code to set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Prediction_Experiment")     # complete the code to set the MLflow experiment name (same as the dev experimentation cell)

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
# For the utility, we need the original X and y before splitting
# Assuming the artifact download in GitHub Actions re-creates Xtrain.csv, Xtest.csv, etc.
# For local execution of train.py, we'll recreate X and y from combined data
# In actual GitHub Actions, these would be loaded from artifacts.

# Re-reading raw data to get full X, y for the utility function
# This is a temporary measure for demonstration. In a production pipeline,
# `prep.py` should ideally output the combined `X` and `y` or the utility should accept splits.
# For now, let's load the data similar to the notebook to get full X, y.
df = pd.read_csv("tourism_project/data/tourism.csv")
df = df.drop(columns=["CustomerID", "Unnamed: 0"])
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__colsample_bytree': [0.7, 1.0],
    'xgbclassifier__colsample_bylevel': [0.7, 1.0],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [1, 2],
}

# --- Call the new utility function ---
print("Calling run_full_training_pipeline from train_utils.py...")
run_full_training_pipeline(
    X=X,
    y=y,
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    param_grid=param_grid,
    classification_threshold=0.5,
    model_save_path="tourism_project/deployment/xgb_model.joblib",
    registered_model_name="XGBoostClassifierModel",
    cv_folds=4 # Ensure cv is set to 4
)
print("train.py execution complete using shared utility.")
