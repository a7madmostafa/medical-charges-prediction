import streamlit as st
import pandas as pd
import joblib

# Load model and data
model = joblib.load('rf_model.joblib')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('insurance.csv')

df = load_data()

# UI
st.title("Medical Charges Prediction")

age = st.slider("Age", df['age'].min(), df['age'].max(), step=1)
sex = st.selectbox("Sex", df['sex'].unique())
bmi = st.slider("BMI", df['bmi'].min(), df['bmi'].max(), step=0.1)
children = st.selectbox("Number of Children", df['children'].unique())
smoker = st.selectbox("Smoker", df['smoker'].unique())
region = st.selectbox("Region", df['region'].unique())

# Prediction
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})
prediction = model.predict(input_df)

if st.button("Predict Charges"):
    
    st.success(f"Predicted Medical Charges: ${prediction[0]:,.2f}")
