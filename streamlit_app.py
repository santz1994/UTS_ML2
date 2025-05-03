import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load the saved TFLite model and preprocessing objects
interpreter = tf.lite.Interpreter(model_path='credit_risk_model.tflite')
interpreter.allocate_tensors()

# Get input and output details for the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit app
st.title("Credit Scoring Prediction App")
st.write("Predict whether a credit card applicant is a 'good' or 'bad' client.")

# Input fields for user data
st.sidebar.header("Applicant Information")
AMT_INCOME_TOTAL = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
age = st.sidebar.number_input("Age", min_value=18, value=35)
NAME_INCOME_TYPE = st.sidebar.selectbox("Income Type", label_encoders['NAME_INCOME_TYPE'].classes_)
NAME_EDUCATION_TYPE = st.sidebar.selectbox("Education Type", label_encoders['NAME_EDUCATION_TYPE'].classes_)
NAME_FAMILY_STATUS = st.sidebar.selectbox("Family Status", label_encoders['NAME_FAMILY_STATUS'].classes_)
NAME_HOUSING_TYPE = st.sidebar.selectbox("Housing Type", label_encoders['NAME_HOUSING_TYPE'].classes_)
OCCUPATION_TYPE = st.sidebar.selectbox("Occupation Type", label_encoders['OCCUPATION_TYPE'].classes_)
CODE_GENDER = st.sidebar.selectbox("Gender", label_encoders['CODE_GENDER'].classes_)
FLAG_OWN_CAR = st.sidebar.selectbox("Owns a Car?", label_encoders['FLAG_OWN_CAR'].classes_)
FLAG_OWN_REALTY = st.sidebar.selectbox("Owns Realty?", label_encoders['FLAG_OWN_REALTY'].classes_)
CNT_CHILDREN = st.sidebar.number_input("Number of Children", min_value=0, value=0)
AMT_CREDIT = st.sidebar.number_input("Credit Amount", min_value=0, value=100000)
AMT_ANNUITY = st.sidebar.number_input("Annuity Amount", min_value=0, value=5000)
AMT_GOODS_PRICE = st.sidebar.number_input("Goods Price", min_value=0, value=100000)
REGION_POPULATION_RELATIVE = st.sidebar.number_input("Region Population Relative", min_value=0.0, value=0.01)
DAYS_EMPLOYED = st.sidebar.number_input("Days Employed (negative value)", min_value=-10000, value=-1000)
credit_history_length = st.sidebar.number_input("Credit History Length (months)", min_value=0, value=12)

# Prepare input data
input_data = {
    'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
    'age': age,
    'NAME_INCOME_TYPE': label_encoders['NAME_INCOME_TYPE'].transform([NAME_INCOME_TYPE])[0],
    'NAME_EDUCATION_TYPE': label_encoders['NAME_EDUCATION_TYPE'].transform([NAME_EDUCATION_TYPE])[0],
    'NAME_FAMILY_STATUS': label_encoders['NAME_FAMILY_STATUS'].transform([NAME_FAMILY_STATUS])[0],
    'NAME_HOUSING_TYPE': label_encoders['NAME_HOUSING_TYPE'].transform([NAME_HOUSING_TYPE])[0],
    'OCCUPATION_TYPE': label_encoders['OCCUPATION_TYPE'].transform([OCCUPATION_TYPE])[0],
    'CODE_GENDER': label_encoders['CODE_GENDER'].transform([CODE_GENDER])[0],
    'FLAG_OWN_CAR': label_encoders['FLAG_OWN_CAR'].transform([FLAG_OWN_CAR])[0],
    'FLAG_OWN_REALTY': label_encoders['FLAG_OWN_REALTY'].transform([FLAG_OWN_REALTY])[0],
    'CNT_CHILDREN': CNT_CHILDREN,
    'AMT_CREDIT': AMT_CREDIT,
    'AMT_ANNUITY': AMT_ANNUITY,
    'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
    'REGION_POPULATION_RELATIVE': REGION_POPULATION_RELATIVE,
    'DAYS_EMPLOYED': DAYS_EMPLOYED,
    'credit_history_length': credit_history_length
}

input_df = pd.DataFrame([input_data])

# Ensure all columns match the training data
for col in scaler.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[scaler.feature_names_in_]
input_scaled = scaler.transform(input_df)

# Make prediction
if st.button("Predict"):
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_scaled.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the prediction result
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    if prediction > 0.5:
        st.error("The model predicts this applicant is a 'bad' client (high risk).")
    else:
        st.success("The model predicts this applicant is a 'good' client (low risk).")