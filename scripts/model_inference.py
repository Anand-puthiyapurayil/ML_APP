# model_inference.py
import joblib
import pandas as pd
import os

def load_model(model_dir, model_name):
    """
    Load a specified trained model from disk.
    """
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    return joblib.load(model_path)


def make_prediction(model, input_data: pd.DataFrame):
    """
    Make predictions using the loaded model.
    """
    predictions = model.predict(input_data)
    print("[DEBUG] Raw predictions:", predictions)
    return predictions