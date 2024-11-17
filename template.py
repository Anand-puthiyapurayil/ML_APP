# template.py
import os

# Define the project structure with FastAPI and Streamlit
project_structure = {
    "data": ["sample_data.csv", "README.md"],              # Folder for raw data files
    "scripts": [                                           # Folder for main Python scripts
        "data_ingestion.py",
        "data_processing.py",
        "model_training.py",
        "model_inference.py",
        "config.py"
    ],
    "models": ["README.md"],                               # Folder to store trained model files
    "app": ["main.py"],                                    # FastAPI app folder
    "streamlit_app": ["app.py"],                           # Streamlit app folder
    "": ["requirements.txt", "README.md", ".gitignore"]    # Root-level files
}

# Create the project structure
def create_project_structure(base_path="."):
    for folder, files in project_structure.items():
        folder_path = os.path.join(base_path, folder)
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Create each file in the folder
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    # Add default content for specific files
                    if file == "README.md":
                        f.write(f"# {folder.capitalize()}\n\nDescription for the {folder} folder.")
                    elif file == "requirements.txt":
                        f.write("pandas\nnumpy\nscikit-learn\nfastapi\nuvicorn\nstreamlit\njoblib")
                    elif file == ".gitignore":
                        f.write("__pycache__/\n*.pyc\nmodels/\n")
                    elif file == "main.py" and folder == "app":
                        f.write("from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\nasync def root():\n    return {'message': 'Hello, FastAPI!'}")
                    elif file == "app.py" and folder == "streamlit_app":
                        f.write("import streamlit as st\n\nst.title('Dynamic ML Application')\nst.write('Welcome to the Streamlit app')")
                    else:
                        f.write(f"# {file}\n\n# This file is part of the dynamicapp project.")
                print(f"Created file: {file_path}")
    print("Project structure created successfully.")

if __name__ == "__main__":
    create_project_structure()