# 💸 EMIPredict AI

**Intelligent Loan Eligibility & Risk Assessment Dashboard**

EMIPredict AI is an end-to-end machine learning pipeline that predicts EMI (Equated Monthly Installment) loan eligibility in real-time. Built with a locally tracked MLflow SQLite database and deployed via a fully interactive Streamlit web application.

## 🚀 Key Features
- **Large-Scale Data Handling:** Wrangled and preprocessed a 400,000+ record Kaggle financial dataset.
- **Memory-Safe Engineering:** Engineered derived financial ratios (expense-to-income, savings capacity) and utilized label encoding to prevent memory crashes during model training.
- **Experiment Tracking:** Integrated MLflow with a local SQLite backend (`mlflow.db`) to track hyperparameters, model artifacts, and evaluation metrics safely.
- **Interactive UI:** Deployed a Streamlit dashboard that dynamically scales applicant data, fetches the best-performing model from the database, and serves real-time risk confidence scores.

## 🧠 Model Performance
Three classification algorithms were trained and tracked to find the optimal risk assessor:
1. **XGBoost Classifier:** `95.33% Accuracy` 🏆 *(Deployed Model)*
2. **Random Forest:** `91.75% Accuracy`
3. **Logistic Regression (Baseline):** `88.81% Accuracy`

## 🛠️ Tech Stack
- **Python 3.12+**
- **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy
- **MLOps / Tracking:** MLflow, SQLite
- **Deployment:** Streamlit

## 💻 How to Run Locally

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/IshaCodex/EMIPredict-AI.git
cd EMIPredict-AI
pip install -r requirements.txt
```

**2. Launch the MLflow Tracking Server:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
(Leave this terminal running and open a new one for the next step).

**3. Run the Streamlit Dashboard:**
```bash
streamlit run app.py
```
