import os
import json
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for caching model and preprocessor
_preprocessor = None
_local_model = None
_wml_client = None
_wml_config = None

def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        preprocessor_path = os.path.join('models', 'credit_preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            _preprocessor = joblib.load(preprocessor_path)
        else:
            # Try reloading component files if unified preprocessor file is missing
            from preprocess import CreditPreprocessor
            _preprocessor = CreditPreprocessor.load(os.path.join('models', 'credit'))
    return _preprocessor

def get_local_model():
    global _local_model
    if _local_model is None:
        model_path = os.path.join('models', 'best_model.pkl')
        if os.path.exists(model_path):
            _local_model = joblib.load(model_path)
        else:
            raise FileNotFoundError("Best model file not found. Please train models first by running src/train.py.")
    return _local_model

def get_wml_client_and_config():
    global _wml_client, _wml_config
    if _wml_config is None:
        wml_config_path = os.path.join('models', 'wml_config.json')
        if os.path.exists(wml_config_path):
            with open(wml_config_path, 'r') as f:
                _wml_config = json.load(f)
                
    wml_apikey = os.getenv("WML_APIKEY")
    wml_url = os.getenv("WML_URL")
    wml_space_id = os.getenv("WML_SPACE_ID")
    
    # Check if credentials and config exist
    if wml_apikey and wml_space_id and _wml_config:
        if _wml_client is None:
            try:
                from ibm_watson_machine_learning import APIClient
                wml_credentials = {
                    "url": wml_url or "https://us-south.ml.cloud.ibm.com",
                    "apikey": wml_apikey
                }
                _wml_client = APIClient(wml_credentials)
                _wml_client.set.default_space(wml_space_id)
            except Exception as e:
                print(f"Failed to initialize IBM Watson ML Client: {e}. Falling back to local model.")
                _wml_client = None
    else:
        _wml_client = None
        
    return _wml_client, _wml_config

def generate_verdict_explanation(data):
    """
    Generates a lists of positive and negative factors explaining the prediction.
    """
    explanations = []
    
    # Financial ratios
    income = float(data.get('Annual_Income', 0))
    debt = float(data.get('Existing_Debt', 0))
    loan = float(data.get('Loan_Amount', 0))
    credit_score = int(data.get('Credit_Score', 0))
    inquiries = int(data.get('Credit_Inquiries', 0))
    history = int(data.get('Credit_History', 0))
    payment = data.get('Payment_History_Status', 'On-time')
    duration = float(data.get('Employment_Duration_Years', 0))
    emp_type = data.get('Employment_Type', 'Unemployed')
    
    dti = debt / income if income > 0 else 0
    lti = loan / income if income > 0 else 0
    
    # Negative factors
    if credit_score < 600:
        explanations.append("CRITICAL: Credit score is low (under 600), representing high credit risk.")
    elif credit_score < 680:
        explanations.append("WARNING: Credit score is fair (600-680), which may require additional guarantees.")
        
    if history == 0:
        explanations.append("CRITICAL: History of credit defaults or lack of positive credit history.")
        
    if dti > 0.45:
        explanations.append(f"WARNING: High Debt-to-Income ratio ({dti*100:.1f}%), suggesting heavy debt load relative to income.")
    elif dti > 0.30:
        explanations.append(f"NOTE: Moderate Debt-to-Income ratio ({dti*100:.1f}%).")
        
    if lti > 0.50:
        explanations.append(f"WARNING: High Loan-to-Income ratio ({lti*100:.1f}%), requesting a loan amount larger than half of annual income.")
        
    if inquiries >= 4:
        explanations.append(f"WARNING: High number of recent credit inquiries ({inquiries}) indicates high credit-seeking behavior.")
        
    if payment == 'Delayed':
        explanations.append("WARNING: Recent payment delays or inconsistencies reported in payment history.")
        
    if emp_type == 'Unemployed':
        explanations.append("CRITICAL: Applicant is currently unemployed.")
    elif duration < 1.0 and emp_type != 'Retired':
        explanations.append("NOTE: Short employment duration (under 1 year) indicates potential income instability.")
        
    # Positive factors if nothing critical
    if not explanations:
        if credit_score >= 750:
            explanations.append("EXCELLENT: Superb credit score (above 750) indicating outstanding creditworthiness.")
        if history == 1:
            explanations.append("GOOD: Established positive credit history with no defaults.")
        if dti < 0.15:
            explanations.append("GOOD: Very low Debt-to-Income ratio, leaving substantial margin for loan servicing.")
        if duration > 5.0 and emp_type in ['Salaried', 'Self-Employed']:
            explanations.append(f"GOOD: Stable employment history ({duration:.1f} years).")
        if payment == 'On-time':
            explanations.append("GOOD: Flawless history of on-time payments.")
            
    return explanations

def predict_single(data_dict):
    """
    Predicts credit card approval for a single applicant.
    Combines preprocessing, local/WML scoring, confidence calculation, and explanations.
    """
    # 1. Clean and parse input keys
    parsed_data = {
        'Age': int(data_dict.get('Age', 35)),
        'Gender': data_dict.get('Gender', 'Male'),
        'Marital_Status': data_dict.get('Marital_Status', 'Single'),
        'Education_Level': data_dict.get('Education_Level', 'Graduate'),
        'Employment_Type': data_dict.get('Employment_Type', 'Salaried'),
        'Employment_Duration_Years': float(data_dict.get('Employment_Duration_Years', 3.0)),
        'Annual_Income': float(data_dict.get('Annual_Income', 50000.0)),
        'Loan_Amount': float(data_dict.get('Loan_Amount', 10000.0)),
        'Existing_Debt': float(data_dict.get('Existing_Debt', 5000.0)),
        'Credit_Score': int(data_dict.get('Credit_Score', 700)),
        'Credit_History': int(data_dict.get('Credit_History', 1)),
        'Payment_History_Status': data_dict.get('Payment_History_Status', 'On-time'),
        'Credit_Inquiries': int(data_dict.get('Credit_Inquiries', 1))
    }
    
    # Create DataFrame (1 row)
    df_raw = pd.DataFrame([parsed_data])
    
    # 2. Preprocess raw data
    preprocessor = get_preprocessor()
    df_processed = preprocessor.transform(df_raw)
    
    # 3. Check for IBM Watson Machine Learning Deployment
    wml_client, wml_config = get_wml_client_and_config()
    
    is_wml_used = False
    probability = 0.5
    prediction = 0
    
    if wml_client and wml_config:
        try:
            # Score using WML
            deployment_uid = wml_config['deployment_uid']
            
            # Format payload for WML
            fields = list(df_processed.columns)
            values = [df_processed.iloc[0].tolist()]
            
            payload = {
                "input_data": [{
                    "fields": fields,
                    "values": values
                }]
            }
            
            print(f"Scoring using WML deployment: {deployment_uid}")
            response = wml_client.deployments.score(deployment_uid, payload)
            
            # Parse WML response (Format depends on model type)
            predictions_raw = response['predictions'][0]
            
            # Determine indices
            pred_idx = predictions_raw['fields'].index('prediction')
            prob_idx = predictions_raw['fields'].index('values') if 'values' in predictions_raw['fields'] else -1
            
            prediction = int(predictions_raw['values'][0][pred_idx])
            
            if prob_idx != -1:
                # Probability distribution: [[prob_0, prob_1]]
                probs = predictions_raw['values'][0][prob_idx]
                probability = float(probs[1])
            else:
                probability = 1.0 if prediction == 1 else 0.0
                
            is_wml_used = True
            
        except Exception as e:
            print(f"WML Scoring failed: {e}. Falling back to local prediction.")
            is_wml_used = False
            
    if not is_wml_used:
        # Score using local model
        model = get_local_model()
        prediction = int(model.predict(df_processed)[0])
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(df_processed)[0]
            probability = float(probs[1])
        else:
            probability = 1.0 if prediction == 1 else 0.0
            
    # Calculate confidence level
    # If probability is near 0.5, confidence is Low. If near 0 or 1, confidence is High.
    distance_from_boundary = abs(probability - 0.5)
    if distance_from_boundary >= 0.35:
        confidence = "High"
    elif distance_from_boundary >= 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"
        
    verdict = "Approved" if prediction == 1 else "Rejected"
    explanations = generate_verdict_explanation(parsed_data)
    
    # Return formatted result
    return {
        'status': verdict,
        'prediction': prediction,
        'probability': round(probability, 4),
        'confidence': confidence,
        'explanations': explanations,
        'inference_source': 'IBM Watson ML' if is_wml_used else 'Local Engine'
    }

def test_local_prediction():
    """
    Tests prediction locally with dummy input to verify modules.
    """
    dummy_input = {
        'Age': 32,
        'Gender': 'Male',
        'Marital_Status': 'Married',
        'Education_Level': 'Graduate',
        'Employment_Type': 'Salaried',
        'Employment_Duration_Years': 5.0,
        'Annual_Income': 85000.0,
        'Loan_Amount': 15000.0,
        'Existing_Debt': 2000.0,
        'Credit_Score': 750,
        'Credit_History': 1,
        'Payment_History_Status': 'On-time',
        'Credit_Inquiries': 0
    }
    
    return predict_single(dummy_input)

if __name__ == '__main__':
    # Run test
    try:
        res = test_local_prediction()
        print("Test prediction result:")
        print(json.dumps(res, indent=4))
    except Exception as e:
        print(f"Error during test: {e}")
