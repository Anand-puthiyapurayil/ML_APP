import streamlit as st
import pandas as pd
import requests
import numpy as np

API_URL = "http://localhost:8000"

st.title("Machine Learning Application")

# Step 1: Upload and Preview Data
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Initialize session state for task type, target, and feature columns
    if "task_type" not in st.session_state:
        st.session_state["task_type"] = "regression"
    if "target_column" not in st.session_state:
        st.session_state["target_column"] = None
    if "feature_columns" not in st.session_state:
        st.session_state["feature_columns"] = []

    # Step 2: Select Task Type
    task_type = st.selectbox(
        "Select Task Type",
        options=["regression", "classification", "clustering"],
        index=["regression", "classification", "clustering"].index(st.session_state["task_type"]),
    )
    st.session_state["task_type"] = task_type

    # Step 3: Select Features and Target
    feature_columns = st.multiselect(
        "Select Feature Columns",
        options=df.columns.tolist(),
        default=st.session_state["feature_columns"],
    )
    st.session_state["feature_columns"] = feature_columns

    available_target_columns = [col for col in df.columns if col not in feature_columns]

    if task_type != "clustering":
        target_column = st.selectbox(
            "Select Target Column",
            options=available_target_columns,
            index=available_target_columns.index(st.session_state["target_column"])
            if st.session_state["target_column"] in available_target_columns
            else 0,
        )
        st.session_state["target_column"] = target_column
    else:
        st.info("Clustering does not require a target column.")
        target_column = None

    # Step 4: Preprocess Data
    if st.button("Preprocess Data"):
        if not feature_columns:
            st.error("Please select at least one feature column.")
        else:
            sanitized_df = df.copy()
            sanitized_df = sanitized_df.replace([np.inf, -np.inf], np.nan).fillna(0)

            payload = {
                "data": sanitized_df.to_dict(orient="records"),
                "feature_columns": feature_columns,
                "target_column": target_column,
                "task_type": task_type,
            }

            try:
                response = requests.post(f"{API_URL}/preprocess/", json=payload)
                if response.status_code == 200:
                    st.success("Data preprocessed successfully!")
                    preprocessed_data = pd.DataFrame(response.json()["preprocessed_data"])
                    st.write("### Preprocessed Data", preprocessed_data.head())
                    st.session_state["preprocessed_data"] = preprocessed_data
                else:
                    st.error(f"Error during preprocessing: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

    # Step 5: Train Model
    if st.button("Train Model"):
        if "preprocessed_data" in st.session_state:
            preprocessed_data = st.session_state["preprocessed_data"]
            if not feature_columns:
                st.error("Please select at least one feature column.")
            elif task_type != "clustering" and not target_column:
                st.error("Please select a target column for regression or classification.")
            else:
                payload = {
                    "data": preprocessed_data.to_dict(orient="records"),
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "task_type": task_type,
                }
                try:
                    response = requests.post(f"{API_URL}/train/", json=payload)
                    if response.status_code == 200:
                        st.success("Model(s) trained successfully!")
                        result = response.json()
                        st.write("### Model Training Results", result)
                        st.session_state["model_names"] = list(result["model_metrics"].keys())
                    else:
                        st.error(f"Error during training: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
        else:
            st.error("Please preprocess the data first.")

    # Step 6: Make Predictions
    if "model_names" in st.session_state:
        st.header("Make Prediction")
        selected_model = st.selectbox("Select Model for Prediction", st.session_state["model_names"])
        input_data = {feature: st.text_input(f"Enter value for {feature}") for feature in feature_columns}

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])
                input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

                payload = {
                    "input_data": input_df.to_dict(orient="records"),
                    "model_name": selected_model,
                    "task_type": task_type,
                }
                response = requests.post(f"{API_URL}/predict/", json=payload)
                if response.status_code == 200:
                    response_data = response.json()
                    st.write("### Predictions")
                    st.write(response_data["decoded_predictions"])
                else:
                    st.error(f"Error during prediction: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
    else:
        st.info("Train a model first to enable predictions.")
