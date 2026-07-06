"""
app.py - Credit Card Approval Prediction Flask Application
"""
import os
import pickle
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Load trained model ---
MODEL_PATH    = 'model.pkl'
ENCODERS_PATH = os.path.join('models', 'label_encoders.pkl')

model    = pickle.load(open(MODEL_PATH, 'rb'))
encoders = joblib.load(ENCODERS_PATH)

# --- Home Page ---
@app.route('/')
def home():
    return render_template('home.html')

# --- Prediction Page ---
@app.route('/index')
def index():
    return render_template('index.html')

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form (exact match to screenshot logic)
        features = [float(x) for x in request.form.values()]
        
        # Convert input into array
        final_input = [np.array(features)]
        
        # Predict result
        prediction = model.predict(final_input)
        
        if prediction[0] == 1:
            result = "Credit Card Approved"
        else:
            result = "Credit Card Rejected"

        # Log prediction to database to satisfy ER Diagram tracking requirements
        try:
            from database import log_prediction_to_db
            log_prediction_to_db(
                income_type=request.form.get('NAME_INCOME_TYPE'),
                education_type=request.form.get('NAME_EDUCATION_TYPE'),
                family_status=request.form.get('NAME_FAMILY_STATUS'),
                housing_type=request.form.get('NAME_HOUSING_TYPE'),
                days_birth=request.form.get('DAYS_BIRTH', 0),
                months_balance=request.form.get('NUMBER_OF_LOANS', 0),
                payment_status=request.form.get('EMI_PAID_OFF', 0),
                overdue_status=request.form.get('EMI_PASTDUES', 0),
                approval_result=result
            )
        except Exception as db_err:
            print(f"Database logging warning: {db_err}")

        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    try:
        from database import init_db
        init_db()  # Setup tables according to ER diagram schema
    except Exception as db_err:
        print(f"Database initialization warning: {db_err}")
    
    app.run(debug=True)
