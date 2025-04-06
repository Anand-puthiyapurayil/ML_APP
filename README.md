# 📊 Dynamic ML Application

This repository hosts a dynamic Machine Learning application designed to facilitate seamless data analysis and model building. It enables users to easily upload datasets, select desired columns and target variables, and perform various machine learning tasks including classification, prediction, and clustering—all through an interactive Streamlit interface.


## 🚀 Demo


![](assets/dynamic.png)

---

## 🚀 Features

- **Dynamic Data Upload**: Easily upload custom datasets for analysis.
- **Interactive Column Selection**: Select feature and target columns dynamically through the UI.
- **Versatile ML Tasks**: Perform classification, regression (prediction), and clustering with multiple model options.
- **Real-time Model Training**: Instantly train models based on your selected parameters and view results immediately.

---

## 📂 Project Structure

```bash
.
├── app                      # Streamlit frontend application
├── api                      # FastAPI backend
├── data                     # Directory for storing datasets
├── scripts                  # Data ingestion, processing, and model training scripts
├── streamlit_app            # Streamlit application scripts
├── requirements.txt         # Project dependencies
├── setup.py                 # Setup script
└── template.py              # Template for configuration
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: Scikit-learn (Classification, Regression, Clustering)
- **Dependency Management**: pip

---

## ⚙️ Setup

### **Installation**

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/dynamic-ml-app.git
cd dynamic-ml-app
pip install -r requirements.txt
```

### **Running the Application**

Start the Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

Start the API:

```bash
uvicorn api.main:app --reload
```

---

## 📖 Usage

1. **Upload Data**: Use the Streamlit interface to upload your dataset (CSV format).
2. **Select Columns**: Dynamically select feature columns and target variable.
3. **Choose Task**: Select the machine learning task (classification, prediction, clustering).
4. **Train & Evaluate**: Train your model and view performance metrics directly in the UI.

---

## 🤝 Contributing

Contributions are welcome:

- Fork the repository
- Create a feature branch (`git checkout -b feature/new-feature`)
- Commit changes (`git commit -m 'Add feature'`)
- Push to branch (`git push origin feature/new-feature`)
- Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for more information.
