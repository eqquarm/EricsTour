# for data manipulation
import pandas as pd
# for model serialization and experiment tracking
import mlflow

# Import the new utility function
from .train_utils import run_full_training_pipeline

# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Prediction_Experiment")

# Load the dataset using a path relative to the tourism_project directory
# In the workflow, we run this from the project root using 'python -m model_building.train'
df = pd.read_csv("data/tourism.csv")
df = df.drop(columns=["CustomerID", "Unnamed: 0"])
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation'
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

# Call the utility function
print("Calling run_full_training_pipeline from train_utils.py...")
run_full_training_pipeline(
    X=X,
    y=y,
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    param_grid=param_grid,
    classification_threshold=0.5,
    model_save_path="deployment/xgb_model.joblib",
    registered_model_name="XGBoostClassifierModel",
    cv_folds=4
)
print("train.py execution complete using shared utility.")
