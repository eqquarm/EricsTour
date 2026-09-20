import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import classification_report
import mlflow
import joblib
from mlflow.models import infer_signature
import os

def run_full_training_pipeline(
    X,
    y,
    numeric_features,
    categorical_features,
    param_grid,
    classification_threshold=0.5,
    model_save_path="tourism_project/deployment/xgb_model.joblib",
    registered_model_name="XGBoostClassifierModel",
    cv_folds=4 # Default to 4 to avoid ValueError with small minority classes
):
    """
    Runs the full model training and MLflow logging pipeline.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.
        numeric_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        classification_threshold (float, optional): Threshold for binary classification.
                                                    Defaults to 0.5.
        model_save_path (str, optional): Local path to save the trained model.
                                         Defaults to 'tourism_project/deployment/xgb_model.joblib'.
        registered_model_name (str, optional): Name to register the model in MLflow Model Registry.
                                               Defaults to 'XGBoostClassifierModel'.
        cv_folds (int, optional): Number of cross-validation folds for GridSearchCV. Defaults to 4.
    """

    # Split data into training and testing sets
    # Ensure stratification on the target variable if it's imbalanced.
    if y.value_counts(normalize=True).min() < 0.2: # Example threshold for imbalance
        print("Target variable is imbalanced, using stratification for train-test split.")
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        print("Target variable is not significantly imbalanced, proceeding with regular train-test split.")
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {Xtrain.shape}")
    print(f"y_train shape: {ytrain.shape}")
    print(f"X_test shape: {Xtest.shape}")
    print(f"y_test shape: {ytest.shape}")

    # Handle class imbalance for XGBoost
    # This assumes binary classification and `1` is the positive class
    class_weight_value = ytrain.value_counts()[0] / ytrain.value_counts()[1] if ytrain.value_counts()[1] != 0 else 1.0

    # Define the preprocessing steps
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features)
    )

    # Define base XGBoost model
    xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight_value, random_state=42, eval_metric='logloss', use_label_encoder=False)

    # Model pipeline
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    with mlflow.start_run():
        print(f"Starting GridSearchCV with {cv_folds}-fold cross-validation...")
        # Hyperparameter tuning
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv_folds, n_jobs=-1, verbose=1)
        grid_search.fit(Xtrain, ytrain)
        print("GridSearchCV completed.")

        # Log all parameter combinations and their mean test scores
        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
                mlflow.log_metric("std_test_score", results["std_test_score"][i])

        # Log best parameters separately in main run
        mlflow.log_params(grid_search.best_params_)

        # Store and evaluate the best model
        best_model = grid_search.best_estimator_

        y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
        y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

        y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
        y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

        train_report = classification_report(ytrain, y_pred_train, output_dict=True)
        test_report = classification_report(ytest, y_pred_test, output_dict=True)

        mlflow.log_metrics({
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1-score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1-score": test_report["1"]["f1-score"]
        })

        # Infer model signature
        example_input = Xtrain.head(1) # Using Xtrain for input schema
        try:
            # Ensure the preprocessor and model steps are accessible by name
            # For a pipeline, you need to access steps by name or index
            # If preprocessor is directly applied, need to rebuild a column transformer for example_input
            # Let's directly apply preprocessor to the example_input if it's a make_column_transformer
            processed_example_input_df = pd.DataFrame(preprocessor.fit_transform(example_input),
                                                      columns=preprocessor.get_feature_names_out())
            predictions = best_model.predict(processed_example_input_df) # Predict on processed example
            signature = infer_signature(processed_example_input_df, predictions)
        except Exception as e:
            print(f"Could not infer signature: {e}")
            signature = None


        # Log the best model to MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=registered_model_name,
        )
        print(f"Model '{registered_model_name}' logged to MLflow and registered.")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")
        print(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

        # Save the model locally for deployment
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(best_model, model_save_path)
        print(f"Best model saved locally to {model_save_path}")
