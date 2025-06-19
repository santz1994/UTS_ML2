import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- Load Model and Preprocessing Objects (Cached) ---
@st.cache_resource
def load_assets():
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path='credit_risk_model.tflite')
        interpreter.allocate_tensors()

        # Load the preprocessing objects
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        training_columns = joblib.load('training_columns.pkl') # Load the saved training columns

        return interpreter, scaler, label_encoders, training_columns
    except FileNotFoundError as e:
        st.error(f"Error loading model or preprocessing files: {e}. Make sure 'credit_risk_model.tflite', 'scaler.pkl', 'label_encoders.pkl', and 'training_columns.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during asset loading: {e}")
        st.stop()

interpreter, scaler, label_encoders, training_columns = load_assets()

# Get input and output tensors
if interpreter: # Check if interpreter was loaded successfully
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    st.stop() # Stop if assets didn't load

# --- Data Preprocessing Function ---
def preprocess_input(input_data: pd.DataFrame, scaler, label_encoders, training_columns):
    # Apply label encoding
    for col, encoder in label_encoders.items():
        if col in input_data.columns:
            try:
                input_data[col] = encoder.transform(input_data[col].astype(str))
            except ValueError as e:
                st.error(f"Error encoding column '{col}': The value '{input_data[col].iloc[0]}' is not recognized. Please check the input.")
                st.stop() # Stop execution if encoding fails

    # Ensure all training columns are present, fill missing with 0
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[training_columns]

    # Apply scaling
    input_scaled = scaler.transform(input_data)
    return input_scaled.astype(np.float32)

# --- Prediction Function ---
def predict_risk(preprocessed_data, interpreter):
    interpreter.set_tensor(input_details[0]['index'], preprocessed_data)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    prediction_probability = output_tensor[0][0]
    predicted_class = 1 if prediction_probability > 0.5 else 0
    return predicted_class, prediction_probability

# --- Streamlit App Layout ---
st.title("Credit Risk Prediction")
st.write("Enter the applicant's details to predict their credit risk.")

# --- User Input Widgets ---
st.header("Applicant Information")

amt_income_total = st.number_input("Total Income", min_value=0.0, value=50000.0)
age = st.number_input("Age", min_value=0, value=35)

# Get the original labels for categorical features from the label encoders
income_type_options = label_encoders['NAME_INCOME_TYPE'].classes_.tolist() if 'NAME_INCOME_TYPE' in label_encoders else []
education_type_options = label_encoders['NAME_EDUCATION_TYPE'].classes_.tolist() if 'NAME_EDUCATION_TYPE' in label_encoders else []
family_status_options = label_encoders['NAME_FAMILY_STATUS'].classes_.tolist() if 'NAME_FAMILY_STATUS' in label_encoders else []
housing_type_options = label_encoders['NAME_HOUSING_TYPE'].classes_.tolist() if 'NAME_HOUSING_TYPE' in label_encoders else []
occupation_type_options = label_encoders['OCCUPATION_TYPE'].classes_.tolist() if 'OCCUPATION_TYPE' in label_encoders else []
gender_options = label_encoders['CODE_GENDER'].classes_.tolist() if 'CODE_GENDER' in label_encoders else []
car_ownership_options = label_encoders['FLAG_OWN_CAR'].classes_.tolist() if 'FLAG_OWN_CAR' in label_encoders else []
realty_ownership_options = label_encoders['FLAG_OWN_REALTY'].classes_.tolist() if 'FLAG_OWN_REALTY' in label_encoders else []

name_income_type = st.selectbox("Income Type", options=income_type_options)
name_education_type = st.selectbox("Education Type", options=education_type_options)
name_family_status = st.selectbox("Family Status", options=family_status_options)
name_housing_type = st.selectbox("Housing Type", options=housing_type_options)
occupation_type = st.selectbox("Occupation Type", options=occupation_type_options)
code_gender = st.radio("Gender", options=gender_options)
flag_own_car = st.radio("Own a Car?", options=car_ownership_options)
flag_own_realty = st.radio("Own Realty?", options=realty_ownership_options)

cnt_children = st.number_input("Number of Children", min_value=0, value=0)
amt_credit = st.number_input("Credit Amount", min_value=0.0, value=100000.0)
amt_annuity = st.number_input("Annuity Amount", min_value=0.0, value=5000.0)
amt_goods_price = st.number_input("Goods Price", min_value=0.0, value=100000.0)
region_population_relative = st.number_input("Region Population Relative", min_value=0.0, value=0.01, format="%.4f")
days_employed = st.number_input("Days Employed (-ve for current employment)", value=-1000)
credit_history_length = st.number_input("Credit History Length (Months)", min_value=0, value=12)

# --- Prediction Button ---
if st.button("Predict Credit Risk"):
    # Create a dictionary from user input
    user_input = {
        'AMT_INCOME_TOTAL': [amt_income_total],
        'age': [age],
        'NAME_INCOME_TYPE': [name_income_type],
        'NAME_EDUCATION_TYPE': [name_education_type],
        'NAME_FAMILY_STATUS': [name_family_status],
        'NAME_HOUSING_TYPE': [name_housing_type],
        'OCCUPATION_TYPE': [occupation_type],
        'CODE_GENDER': [code_gender],
        'FLAG_OWN_CAR': [flag_own_car],
        'FLAG_OWN_REALTY': [flag_own_realty],
        'CNT_CHILDREN': [cnt_children],
        'AMT_CREDIT': [amt_credit],
        'AMT_ANNUITY': [amt_annuity],
        'AMT_GOODS_PRICE': [amt_goods_price],
        'REGION_POPULATION_RELATIVE': [region_population_relative],
        'DAYS_EMPLOYED': [days_employed],
        'credit_history_length': [credit_history_length]
    }

    # Convert to DataFrame
    user_input_df = pd.DataFrame(user_input)

    # Preprocess the input
    preprocessed_input = preprocess_input(user_input_df, scaler, label_encoders, training_columns)

    # Make prediction
    predicted_class, prediction_probability = predict_risk(preprocessed_input, interpreter)

    # Display prediction result
    st.header("Prediction Result")
    if predicted_class == 1:
        st.error(f"The model predicts this applicant is a **'bad' client** (high risk).")
        st.write(f"Predicted probability of being 'bad': {prediction_probability:.2f}")
    else:
        st.success(f"The model predicts this applicant is a **'good' client** (low risk).")
        st.write(f"Predicted probability of being 'good': {1 - prediction_probability:.2f}")
