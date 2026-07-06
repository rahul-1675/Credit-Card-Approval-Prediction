"""
database.py
===========
SQLite Database setup and management to implement the ER diagram architecture.
Tables:
  1. Users (UserID, Name, Email, Password, Role)
  2. Applicant_Details (ApplicantID, UserID, IncomeType, EducationType, FamilyStatus, HousingType, En, Days)
  3. Credit_History (HistoryID, ApplicantID, MonthsBalance, PaymentStatus, OverdueStatus)
  4. ML_Model (ModelID, ModelName, AlgorithmType, Accuracy, ModelFile)
  5. Approval_Prediction (PredictionID, ApplicantID, ModelID, ApprovalResult, RiskCategory, PredictionDate)
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'credit_card_prediction.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite tables based on the ER Diagram structure.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            UserID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Email TEXT UNIQUE NOT NULL,
            Password TEXT NOT NULL,
            Role TEXT NOT NULL DEFAULT 'User'
        )
    ''')

    # 2. Applicant_Details table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Applicant_Details (
            ApplicantID INTEGER PRIMARY KEY AUTOINCREMENT,
            UserID INTEGER,
            IncomeType TEXT,
            EducationType TEXT,
            FamilyStatus TEXT,
            HousingType TEXT,
            En INTEGER, -- e.g., Employment status or similar binary indicator
            Days INTEGER, -- DAYS_BIRTH or AGE in days
            FOREIGN KEY (UserID) REFERENCES Users (UserID)
        )
    ''')

    # 3. Credit_History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Credit_History (
            HistoryID INTEGER PRIMARY KEY AUTOINCREMENT,
            ApplicantID INTEGER,
            MonthsBalance INTEGER,
            PaymentStatus TEXT,
            OverdueStatus TEXT,
            FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details (ApplicantID)
        )
    ''')

    # 4. ML_Model table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ML_Model (
            ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
            ModelName TEXT NOT NULL,
            AlgorithmType TEXT NOT NULL,
            Accuracy REAL,
            ModelFile TEXT
        )
    ''')

    # 5. Approval_Prediction table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Approval_Prediction (
            PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
            ApplicantID INTEGER,
            ModelID INTEGER,
            ApprovalResult TEXT,
            RiskCategory TEXT,
            PredictionDate TEXT,
            FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details (ApplicantID),
            FOREIGN KEY (ModelID) REFERENCES ML_Model (ModelID)
        )
    ''')

    # Seed initial admin user if not exists
    cursor.execute("SELECT COUNT(*) FROM Users")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO Users (Name, Email, Password, Role)
            VALUES (?, ?, ?, ?)
        ''', ('Admin', 'admin@credaipulse.com', 'admin123', 'Admin'))

    # Seed ML Model record if not exists
    cursor.execute("SELECT COUNT(*) FROM ML_Model")
    if cursor.fetchone()[0] == 0:
        # Load details from evaluation results if they exist
        results_path = os.path.join(BASE_DIR, 'models', 'evaluation_results.json')
        model_name = "Logistic Regression"
        accuracy = 1.0000
        if os.path.exists(results_path):
            import json
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                    model_name = data.get('best_model_name', model_name)
                    accuracy = data.get('metrics', {}).get(model_name, {}).get('Accuracy', accuracy)
            except Exception:
                pass
        
        cursor.execute('''
            INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, ModelFile)
            VALUES (?, ?, ?, ?)
        ''', ('Best Model', model_name, accuracy, 'model.pkl'))

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def log_prediction_to_db(income_type, education_type, family_status, housing_type, days_birth,
                        months_balance, payment_status, overdue_status, approval_result):
    """
    Helper function to write prediction, applicant details, and credit history details into DB.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get current user ID (using default user 1 for demo purposes if not logged in)
    user_id = 1

    # 1. Insert into Applicant_Details
    cursor.execute('''
        INSERT INTO Applicant_Details (UserID, IncomeType, EducationType, FamilyStatus, HousingType, En, Days)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, income_type, education_type, family_status, housing_type, 1, int(days_birth)))
    applicant_id = cursor.lastrowid

    # 2. Insert into Credit_History
    cursor.execute('''
        INSERT INTO Credit_History (ApplicantID, MonthsBalance, PaymentStatus, OverdueStatus)
        VALUES (?, ?, ?, ?)
    ''', (applicant_id, int(months_balance), payment_status, overdue_status))

    # 3. Get best model ID
    cursor.execute("SELECT ModelID FROM ML_Model LIMIT 1")
    model_row = cursor.fetchone()
    model_id = model_row[0] if model_row else 1

    # 4. Insert into Approval_Prediction
    risk_category = "Low Risk" if approval_result == "Credit Card Approved" else "High Risk"
    prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO Approval_Prediction (ApplicantID, ModelID, ApprovalResult, RiskCategory, PredictionDate)
        VALUES (?, ?, ?, ?, ?)
    ''', (applicant_id, model_id, approval_result, risk_category, prediction_date))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
