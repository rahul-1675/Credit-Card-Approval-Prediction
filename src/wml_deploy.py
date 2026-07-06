import os
import joblib
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def deploy_to_wml():
    """
    Deploys the saved best model to IBM Watson Machine Learning.
    Returns the deployment UID or None if configuration is missing.
    """
    # 1. Read environment variables
    wml_apikey = os.getenv("WML_APIKEY")
    wml_url = os.getenv("WML_URL")
    wml_space_id = os.getenv("WML_SPACE_ID")
    
    if not wml_apikey or not wml_space_id:
        print("IBM Watson Machine Learning credentials not found in environment. Skipping cloud deployment.")
        print("Set WML_APIKEY, WML_URL, and WML_SPACE_ID in your .env file to enable cloud inference.")
        return None
        
    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        print("Error: ibm-watson-machine-learning SDK is not installed. Run 'pip install ibm-watson-machine-learning'.")
        return None
        
    print(f"Connecting to IBM Watson Machine Learning at {wml_url}...")
    wml_credentials = {
        "url": wml_url,
        "apikey": wml_apikey
    }
    
    client = APIClient(wml_credentials)
    client.set.default_space(wml_space_id)
    print("Successfully connected to WML deployment space.")
    
    # Check for existing local files
    model_path = os.path.join('models', 'best_model.pkl')
    results_path = os.path.join('models', 'evaluation_results.json')
    
    if not os.path.exists(model_path) or not os.path.exists(results_path):
        print("Error: Trained model or evaluation results not found. Train the model first.")
        return None
        
    with open(results_path, 'r') as f:
        results = json.load(f)
    best_model_name = results['best_model_name']
    
    # Load model and preprocessor to verify
    best_model = joblib.load(model_path)
    print(f"Loaded best local model: {best_model_name}")
    
    # Determine WML Software Specification and Model Type
    # scikit-learn models and xgboost models have different software specs in WML
    if best_model_name == 'XGBoost':
        model_type = 'xgboost_1.6'
        sw_spec_name = 'runtime-23.1-py3.10'
    else:
        model_type = 'scikit-learn_1.3'
        sw_spec_name = 'runtime-23.1-py3.10'
        
    sw_spec_id = client.software_specifications.get_id_by_name(sw_spec_name)
    
    # 2. Upload Model to WML Repository
    print(f"Uploading model to WML repository as type {model_type}...")
    model_meta = {
        client.repository.ModelMetaNames.NAME: f"Credit_Card_Approval_{best_model_name}",
        client.repository.ModelMetaNames.TYPE: model_type,
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
    }
    
    # For compatibility, store model object directly
    # Note: WML requires scikit-learn or xgboost object
    model_details = client.repository.store_model(
        model=best_model, 
        meta_props=model_meta
    )
    
    model_uid = client.repository.get_model_id(model_details)
    print(f"Model stored in WML with UID: {model_uid}")
    
    # 3. Create Online Deployment
    print("Creating online deployment for real-time predictions...")
    deploy_meta = {
        client.deployments.ConfigurationMetaNames.NAME: f"Credit_Card_Approval_Deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {}
    }
    
    deployment = client.deployments.create(
        artifact_uid=model_uid,
        meta_props=deploy_meta
    )
    
    deployment_uid = client.deployments.get_id(deployment)
    print(f"Deployment created successfully. Deployment UID: {deployment_uid}")
    
    # Save the deployment details locally so predict.py can read it
    wml_config = {
        'wml_url': wml_url,
        'space_id': wml_space_id,
        'model_uid': model_uid,
        'deployment_uid': deployment_uid,
        'model_name': best_model_name
    }
    
    wml_config_path = os.path.join('models', 'wml_config.json')
    with open(wml_config_path, 'w') as f:
        json.dump(wml_config, f, indent=4)
        
    print(f"WML deployment details saved to {wml_config_path}")
    return deployment_uid

if __name__ == '__main__':
    deploy_to_wml()
