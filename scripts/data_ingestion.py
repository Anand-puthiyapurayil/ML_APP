# data_ingestion.py
# data_ingestion.py
import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    - file_path: str - Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error reading the file: {e}")