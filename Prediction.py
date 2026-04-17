from joblib import load
import pandas as pd
import sklearn
model_regression = load("C:\\Users\\ravit\\Downloads\\ML Projects\\Regression\\app\\Artifacts\\Regression_model.joblib")
model_scaler = load("C:\\Users\\ravit\\Downloads\\ML Projects\\Regression\\app\\Artifacts\\scaler_with_col.joblib")
scaler=model_scaler['scaler']
cols_to_scale=model_scaler['cols_to_scale']
def calculate_normalized_risk(medical_history):
    risk_scores={ "diabetes": 6, "heart disease": 8, "high blood pressure": 6, "thyroid": 5, "no disease": 0, "none": 0 }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    max_score = 14
    min_score = 0
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score
def preprocess_input(input_dict):
    expected_cols=['age',
 'number_of_dependants',
 'income_lakhs',
 'insurance_plan',
 'normalized_risk_score',
 'lifestyle_risk_score',
 'gender_Male',
 'region_Northwest',
 'region_Southeast',
 'region_Southwest',
 'marital_status_Unmarried',
 'bmi_category_Obesity',
 'bmi_category_Overweight',
 'bmi_category_Underweight',
 'smoking_status_Occasional',
 'smoking_status_Regular',
 'employment_status_Salaried',
 'employment_status_Self-Employed']
    insurance_plan_encoded = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_cols, index=[0])
    bmi = input_dict['BMI Category']
    for key, value in input_dict.items():
       if key == 'gender_Male':
          df['gender'] = 1
       elif key == 'Region':
           if value == 'Northwest':
               df['region_Northwest'] = 1
           elif value == 'Southeast':
              df['region_Southeast'] = 1
           elif value == 'Southwest':
              df['region_Southwest'] = 1
       elif key == 'marital_status_Unmarried':
              df['marital_status_Unmarried'] = 1
       elif key == 'BMI Category':
           if value == 'Obesity':
               df['bmi_category_Obesity'] = 1
           elif value == 'Overweight':
               df['bmi_category_Overweight'] = 1
           elif value == 'Underweight':
               df['bmi_category_Underweight'] = 1
       elif key == 'Smoking Status':
            if value == 'Occasional':
               df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
               df['smoking_status_Regular'] = 1
       elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self Employed':
                df['employment_status_Self-Employed'] = 1
            elif key == 'Insurance Plan':
                df['insurance_plan'] = insurance_plan_encoded.get(value, 1)
       elif key == 'Age':
                df['age'] = value
       elif key == 'Number of Dependants':
               df['number_of_dependants'] = value
       elif key == 'Income Lakhs':
           df['income_lakhs'] = value
       # elif key == 'Genetical Risk':
       #     df['genetical_risk'] = value

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(df)
    return df

def handle_scaling(df):
    scaler_object=model_scaler
    df['income_level']=None
    df['physical_activity'] = None
    df['stress_level'] = None
    df[cols_to_scale]=scaler.transform(df[cols_to_scale])
    df.drop('income_level',axis='columns',inplace=True)
    df.drop('physical_activity', axis='columns', inplace=True)
    df.drop('stress_level', axis='columns', inplace=True)
    return df

def predict(input_dict):
    input_df=preprocess_input(input_dict)
    prediction=model_regression.predict(input_df)
    return int(prediction[0])