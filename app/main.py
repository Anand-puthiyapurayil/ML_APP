from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from scripts.data_processing import preprocess_data, preprocess_data_for_inference
from scripts.model_training import train_model
from scripts.model_inference import load_model, make_prediction
import os
import time
import joblib

app = FastAPI()

MODEL_DIR = "models"
ENCODER_DIR = "models/encoders"
SCALER_PATH = "models/scaler/scaler.joblib"
PROCESSED_DATA_PATH = "data/processed_data.csv"

# Define data classes for request bodies
class PreprocessRequest(BaseModel):
    data: list
    feature_columns: list
    target_column: str = None
    task_type: str  # "classification", "regression", or "clustering"

class TrainRequest(BaseModel):
    data: list
    feature_columns: list
    target_column: str = None
    task_type: str  # "classification", "regression", or "clustering"

class InferenceRequest(BaseModel):
    input_data: list
    model_name: str
    task_type: str  # "classification", "regression", or "clustering"


@app.post("/preprocess/")
async def preprocess(request: PreprocessRequest):
    print("Raw Request:", request.dict())  
    start_time = time.time()
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)

        # Validate columns
        missing_features = [col for col in request.feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature columns in the input data: {missing_features}",
            )
        if request.task_type != "clustering" and request.target_column and request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in the input data.",
            )

        # Preprocess data
        X, y, encoders, scaler, target_encoder, preprocessed_data = preprocess_data(
            df=df,
            feature_columns=request.feature_columns,
            target_column=request.target_column,
            task_type=request.task_type,
            encoder_dir=ENCODER_DIR,
            scaler_path=SCALER_PATH,
            processed_data_path=PROCESSED_DATA_PATH,
        )

        preprocessed_data.to_csv(PROCESSED_DATA_PATH, index=False)
        elapsed_time = time.time() - start_time
        print(f"Preprocessing completed in {elapsed_time:.2f} seconds")

        return {
            "message": "Data preprocessed successfully.",
            "preprocessed_data": preprocessed_data.to_dict(orient="records"),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")


@app.post("/train/")
async def train(request: TrainRequest):
    start_time = time.time()
    try:
        # Load preprocessed data if available
        if os.path.exists(PROCESSED_DATA_PATH):
            df = pd.read_csv(PROCESSED_DATA_PATH)
        else:
            # Otherwise preprocess the provided data
            df = pd.DataFrame(request.data)
            if request.task_type != "clustering" and request.target_column not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target column '{request.target_column}' not found in the input data.",
                )
            _, _, _, _, _, _ = preprocess_data(
                df=df,
                feature_columns=request.feature_columns,
                target_column=request.target_column,
                task_type=request.task_type,
                encoder_dir=ENCODER_DIR,
                scaler_path=SCALER_PATH,
                processed_data_path=PROCESSED_DATA_PATH,
            )

        # Split features and target
        X = df[request.feature_columns]
        y = None if request.task_type == "clustering" else df[request.target_column]

        # Train models
        results = train_model(X, y, model_type=request.task_type, model_dir=MODEL_DIR)
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")

        return {
            "message": f"{request.task_type.capitalize()} model(s) trained successfully.",
            "model_metrics": results["model_metrics"],
            "best_model_name": results["best_model_name"],
            "best_model_metrics": results["best_model_metrics"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.post("/predict/")
async def predict(request: InferenceRequest):
    start_time = time.time()
    try:
        # Load the model
        model = load_model(MODEL_DIR, request.model_name)

        # Convert input data to DataFrame
        input_df = pd.DataFrame(request.input_data)

        # Preprocess data for inference
        preprocessed_data = preprocess_data_for_inference(
            input_df=input_df,
            encoder_dir=ENCODER_DIR,
            scaler_path=SCALER_PATH,
            task_type=request.task_type,
        )

        # Make predictions
        predictions = make_prediction(model, preprocessed_data)

        # Decode predictions for classification tasks
        decoded_predictions = predictions
        if request.task_type == "classification":
            target_encoder_path = os.path.join(ENCODER_DIR, "target_encoder.joblib")
            if os.path.exists(target_encoder_path):
                target_encoder = joblib.load(target_encoder_path)
                decoded_predictions = target_encoder.inverse_transform(predictions)
            else:
                raise FileNotFoundError(f"Target encoder not found at {target_encoder_path}")

        elapsed_time = time.time() - start_time
        print(f"[INFO] Inference completed in {elapsed_time:.2f} seconds")

        return {
            "raw_predictions": predictions.tolist(),
            "decoded_predictions": [str(pred) for pred in decoded_predictions]
        }
    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=f"File not found: {fnf_error}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
