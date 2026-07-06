import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class CreditPreprocessor:
    """
    Handles preprocessing, feature engineering, and scaling/encoding for the credit card dataset.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        # Handle unknown categories by ignoring them during transform
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.numerical_cols = [
            'Age', 'Employment_Duration_Years', 'Annual_Income', 
            'Loan_Amount', 'Existing_Debt', 'Credit_Score', 'Credit_Inquiries',
            'Debt_to_Income_Ratio', 'Loan_to_Income_Ratio', 'Risk_Score'
        ]
        self.categorical_cols = [
            'Gender', 'Marital_Status', 'Education_Level', 
            'Employment_Type', 'Payment_History_Status'
        ]
        self.binary_cols = ['Credit_History']
        self.encoded_feature_names_ = []

    def engineer_features(self, df):
        """
        Engineers features like Debt-to-Income, Loan-to-Income, and custom Risk Score.
        """
        df_copy = df.copy()
        
        # 1. Handle missing values (if any)
        # For numeric columns, fill with median
        for col in ['Age', 'Employment_Duration_Years', 'Annual_Income', 'Loan_Amount', 'Existing_Debt', 'Credit_Score', 'Credit_Inquiries']:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                
        # For categorical columns, fill with mode
        for col in self.categorical_cols + self.binary_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
        
        # 2. Feature Engineering
        # Debt-to-Income (DTI) Ratio
        df_copy['Debt_to_Income_Ratio'] = df_copy['Existing_Debt'] / (df_copy['Annual_Income'] + 1.0)
        
        # Loan-to-Income (LTI) Ratio
        df_copy['Loan_to_Income_Ratio'] = df_copy['Loan_Amount'] / (df_copy['Annual_Income'] + 1.0)
        
        # Risk Score (Combined score from Credit_Score, Inquiries, and Credit_History)
        # Standardize components to 0-1 scale
        cs_norm = (df_copy['Credit_Score'] - 300) / 550.0  # FICO range normalized
        inq_norm = 1.0 - (df_copy['Credit_Inquiries'] / 10.0)  # less inquiries = better
        ch_norm = df_copy['Credit_History']  # 1 = good history
        
        # Combine with weights
        df_copy['Risk_Score'] = (cs_norm * 0.5) + (ch_norm * 0.3) + (inq_norm * 0.2)
        
        return df_copy

    def fit(self, X):
        """
        Fits the scaler and encoder on the input features.
        """
        # First engineer features
        X_engineered = self.engineer_features(X)
        
        # Fit scaler on numerical columns
        self.scaler.fit(X_engineered[self.numerical_cols])
        
        # Fit encoder on categorical columns
        self.encoder.fit(X_engineered[self.categorical_cols])
        
        # Store categorical column names
        self.encoded_feature_names_ = list(self.encoder.get_feature_names_out(self.categorical_cols))
        
        return self

    def transform(self, X):
        """
        Transforms the input features using fitted scaler and encoder.
        """
        X_engineered = self.engineer_features(X)
        
        # Scale numerical columns
        scaled_nums = self.scaler.transform(X_engineered[self.numerical_cols])
        df_scaled = pd.DataFrame(scaled_nums, columns=self.numerical_cols, index=X.index)
        
        # Encode categorical columns
        encoded_cats = self.encoder.transform(X_engineered[self.categorical_cols])
        df_encoded = pd.DataFrame(encoded_cats, columns=self.encoded_feature_names_, index=X.index)
        
        # Binary features (already 0 or 1, don't need scaling or encoding)
        df_binaries = X_engineered[self.binary_cols].copy()
        
        # Combine everything
        X_processed = pd.concat([df_scaled, df_encoded, df_binaries], axis=1)
        
        return X_processed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, filepath_prefix):
        """
        Saves the preprocessor components (scaler, encoder, classes) to files.
        """
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
        joblib.dump(self.encoder, f"{filepath_prefix}_encoder.pkl")
        joblib.dump(self, f"{filepath_prefix}_preprocessor.pkl")
        print(f"Preprocessor saved to {filepath_prefix}_preprocessor.pkl")

    @classmethod
    def load(cls, filepath_prefix):
        """
        Loads preprocessor components from files.
        """
        preprocessor = joblib.load(f"{filepath_prefix}_preprocessor.pkl")
        preprocessor.scaler = joblib.load(f"{filepath_prefix}_scaler.pkl")
        preprocessor.encoder = joblib.load(f"{filepath_prefix}_encoder.pkl")
        return preprocessor
