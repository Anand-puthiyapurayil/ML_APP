import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score

def train_model(X, y=None, model_type="regression", model_dir="models"):
    """
    Train machine learning models for regression, classification, or clustering.

    Parameters:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable (for regression/classification).
        model_type (str): Type of task - 'regression', 'classification', 'clustering'.
        model_dir (str): Directory to save the trained models.

    Returns:
        dict: Training results, including metrics and the best model.
    """
    os.makedirs(model_dir, exist_ok=True)

    if model_type != "clustering":
        if y is None:
            raise ValueError("Target variable 'y' must be provided for regression and classification.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, None, None

    # Initialize models based on the task type
    if model_type == "regression":
        models = {
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
        }
    elif model_type == "classification":
        models = {
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression()
        }
    elif model_type == "clustering":
        models = {
            "KMeans": KMeans(n_clusters=3, random_state=42)  # Example: 3 clusters
        }
    else:
        raise ValueError("Unsupported model type. Choose from 'regression', 'classification', or 'clustering'.")

    results = {}
    best_model_name = None
    best_metric = float("inf") if model_type in ["regression", "classification"] else float("-inf")
    best_model_metrics = {}

    for model_name, model in models.items():
        if model_type != "clustering":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)

        if model_type == "regression":
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            metric = test_mse  # Lower is better for regression

            metrics = {
                "train_mse": train_mse,
                "train_r2": train_r2,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "sample_predictions": y_test_pred[:5].tolist() if y_test_pred is not None else []
            }

        elif model_type == "classification":
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            metric = -test_acc  # Higher is better for classification

            metrics = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "sample_predictions": y_test_pred[:5].tolist() if y_test_pred is not None else []
            }

        elif model_type == "clustering":
            y_pred = model.predict(X_train)
            silhouette = silhouette_score(X_train, y_pred)
            metric = silhouette  # Higher is better for clustering

            metrics = {
                "silhouette_score": silhouette
            }

        # Save the model
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)

        # Store metrics
        results[model_name] = {
            **metrics,
            "model_path": model_path
        }

        # Update the best model based on metric
        if (model_type == "clustering" and metric > best_metric) or (model_type != "clustering" and metric < best_metric):
            best_metric = metric
            best_model_name = model_name
            best_model_metrics = metrics

    return {
        "model_metrics": results,
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_metrics
    }
