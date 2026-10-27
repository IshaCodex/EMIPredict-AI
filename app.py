import streamlit as st
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import random

# --- Page Config ---
st.set_page_config(page_title="EMIPredict AI", page_icon="💸", layout="wide")

@st.cache_resource
def load_ml_pipeline():
    """Loads data, rebuilds preprocessors, and fetches the best model from MLflow."""
    # 1. Load Data
    df = pd.read_csv('data/emi_prediction_dataset.csv', encoding='latin1')
    
    # 2. Feature Engineering (Matching your training script)
    expense_cols = ['school_fees', 'college_fees', 'travel_expenses', 
                    'groceries_utilities', 'other_monthly_expenses', 
                    'monthly_rent', 'current_emi_amount']
    
    for col in expense_cols + ['monthly_salary']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)
    df['monthly_salary_safe'] = df['monthly_salary'].replace(0, 1)
    df['expense_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary_safe']
    df['emi_to_income_ratio'] = df['current_emi_amount'] / df['monthly_salary_safe']
    df['savings_capacity'] = df['monthly_salary'] - df['total_monthly_expenses']
    df = df.drop(columns=['monthly_salary_safe'])
    
    # 3. Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 4. Scaling
    X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    y = df['emi_eligibility']
    scaler = StandardScaler()
    scaler.fit(X) # Fit scaler on all data for the app
    
    # 5. Fetch Best XGBoost Model from MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = mlflow.get_experiment_by_name("EMI_Eligibility_Classification")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string="params.model_type = 'XGBoost'",
        order_by=["metrics.accuracy DESC"]
    )
    best_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{best_run_id}/xgboost_model"
    model = mlflow.xgboost.load_model(model_uri)
    
    return df, X, scaler, model

# --- Load Pipeline ---
with st.spinner("Connecting to MLflow and loading AI pipeline..."):
    try:
        raw_df, X_features, scaler, xgb_model = load_ml_pipeline()
    except Exception as e:
        st.error(f"Failed to load model. Ensure MLflow database exists. Error: {e}")
        st.stop()

# --- UI Layout ---
st.title("💸 EMIPredict AI")
st.markdown("### Intelligent Loan Eligibility & Risk Assessment Dashboard")
st.write("This application uses an XGBoost classifier trained on 400,000 financial records and tracked via MLflow to predict EMI eligibility in real-time.")

st.sidebar.header("Applicant Selector")
st.sidebar.write("Because the dataset contains 27 features, manually typing them all is tedious. Click the button below to pull a random applicant from the database to test the AI.")

if st.sidebar.button("Fetch Random Applicant 🎲"):
    random_index = random.randint(0, len(X_features) - 1)
    st.session_state['applicant_index'] = random_index

# Default to applicant 0 if none selected
applicant_idx = st.session_state.get('applicant_index', 0)
applicant_data = X_features.iloc[[applicant_idx]]
raw_applicant_data = raw_df.iloc[applicant_idx]

st.divider()

# --- Financial Profile Display ---
st.subheader(f"Applicant Profile (ID: #{applicant_idx})")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Monthly Salary", f"₹{raw_applicant_data['monthly_salary']:,.2f}")
with col2:
    st.metric("Total Expenses", f"₹{raw_applicant_data['total_monthly_expenses']:,.2f}")
with col3:
    st.metric("Savings Capacity", f"₹{raw_applicant_data['savings_capacity']:,.2f}")
with col4:
    st.metric("Credit Score", int(raw_applicant_data.get('credit_score', 0)))

# --- AI Prediction ---
st.subheader("🤖 AI Assessment")
if st.button("Run XGBoost Prediction", type="primary"):
    with st.spinner("Analyzing risk factors..."):
        # Scale the single applicant's data
        scaled_data = scaler.transform(applicant_data)
        
        # Predict
        prediction = xgb_model.predict(scaled_data)[0]
        probability = xgb_model.predict_proba(scaled_data)[0]
        
        # FIX: Always grab the highest confidence score regardless of class order
        confidence = max(probability) * 100 
        
        # Display Results
        if prediction == 1:
            st.success(f"**APPROVED:** The AI predicts this applicant is ELIGIBLE for the EMI.")
            st.write(f"Confidence Score: {confidence:.1f}%")
        else:
            st.error(f"**REJECTED:** The AI predicts this applicant is INELIGIBLE.")
            st.write(f"Risk Confidence: {confidence:.1f}%")