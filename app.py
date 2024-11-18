import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit-authenticator as stauth
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# user authentication : 





@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('train.csv')
    
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    
    data['Dependents'] = data['Dependents'].replace('3+', '3').astype(int)
    
    data['LoanAmount'] = np.log1p(data['LoanAmount'])
    data['ApplicantIncome'] = np.log1p(data['ApplicantIncome'])
    data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])
    
    return data

@st.cache_resource
def get_model(data):
    # Prepare the data
    X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = data['Loan_Status']
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, feature_names
def predict_loan_approval(model, scaler, feature_names, input_data):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    adjusted_probability = max(probability, 0.3)
    
    adjusted_prediction = 'Y' if adjusted_probability >= 0.3 else 'N'
    
    return adjusted_prediction, adjusted_probability

# Streamlit app
def main():
    st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict", "Explore Data"])
    
    # Load data and model
    data = load_and_preprocess_data()
    model, scaler, feature_names = get_model(data)
    
    if page == "Predict":
        st.title("Loan Approval Predictor")
        st.write("Fill in the details below to predict your loan approval chances.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        
        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Income", min_value=0)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
        
        with col3:
            loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
            credit_history = st.selectbox("Credit History", [0, 1])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        if st.button("Predict"):
            input_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': np.log1p(applicant_income),
                'CoapplicantIncome': np.log1p(coapplicant_income),
                'LoanAmount': np.log1p(loan_amount),
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }
            
            prediction, probability = predict_loan_approval(model, scaler, feature_names, input_data)
            
            st.subheader("Prediction Result")
            if prediction == 'Y':
                st.success(f"Congratulations! Your loan is likely to be approved with a {probability:.2%} chance.")
            else:
                st.error(f"Sorry, your loan is likely to be rejected. The approval chance is {probability:.2%}.")
            
            # Visualization of prediction probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Approval Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            st.plotly_chart(fig)
    
    elif page == "Explore Data":
        st.title("Explore Loan Application Data")
        
        # Data overview
        st.subheader("Data Overview")
        st.write(data.head())
        st.write(f"Total number of records: {len(data)}")
        
        # Loan Status Distribution
        st.subheader("Loan Status Distribution")
        fig = px.pie(data, names='Loan_Status', title='Loan Status Distribution', hole=0.3,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig)
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
        st.plotly_chart(fig)
        
        # Loan Amount Distribution
        st.subheader("Loan Amount Distribution")
        fig = px.histogram(data, x="LoanAmount", nbins=50, title="Loan Amount Distribution",
                           color="Loan_Status", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig)
        
        # Applicant Income vs Loan Amount
        st.subheader("Applicant Income vs Loan Amount")
        fig = px.scatter(data, x="ApplicantIncome", y="LoanAmount", color="Loan_Status",
                         title="Applicant Income vs Loan Amount",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig)
        
        # Loan Status by Education and Credit History
        st.subheader("Loan Status by Education and Credit History")
        fig = px.sunburst(data, path=['Education', 'Credit_History', 'Loan_Status'],
                          title="Loan Status by Education and Credit History",
                          color='Loan_Status', color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
