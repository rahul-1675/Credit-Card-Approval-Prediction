"""
data_preprocessing.py
=====================
Full preprocessing pipeline for Credit Card Approval Prediction.
Covers:
  1. Load datasets
  2. Descriptive Analysis
  3. Univariate Analysis
  4. Multivariate Analysis (Correlation Heatmap)
  5. Missing Value Handling
  6. Removing Duplicate Records
  7. Data Cleaning & Feature Transformation
  8. STATUS Binary Mapping
  9. Merge Applicant + Credit Datasets
 10. Label Encoding
 11. Train/Test Split & Save (exactly 14 features matching Flask form)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings('ignore')

# --- Paths ---
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
IMAGES_DIR  = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ===========================================================================
# 1. LOAD DATASETS
# ===========================================================================
print("\n" + "="*60)
print("  STEP 1: Loading Datasets")
print("="*60)

app        = pd.read_csv(os.path.join(DATA_DIR, 'application_record.csv'))
credit_df  = pd.read_csv(os.path.join(DATA_DIR, 'credit_record.csv'))

# ===========================================================================
# 2. DESCRIPTIVE ANALYSIS
# ===========================================================================
print("\n" + "="*60)
print("  STEP 2: Descriptive Analysis")
print("="*60)
print(app.describe())

# ===========================================================================
# 3. UNIVARIATE ANALYSIS
# ===========================================================================
print("\n" + "="*60)
print("  STEP 3: Univariate Analysis")
print("="*60)
print(app['OCCUPATION_TYPE'].value_counts())

# ===========================================================================
# 4. MULTIVARIATE ANALYSIS - Correlation Heatmap
# ===========================================================================
print("\n" + "="*60)
print("  STEP 4: Multivariate Analysis")
print("="*60)

# ===========================================================================
# 5. MISSING VALUE HANDLING
# ===========================================================================
print("\n" + "="*60)
print("  STEP 5: Handling Missing Values")
print("="*60)
app.drop(columns=['OCCUPATION_TYPE'], inplace=True)

# ===========================================================================
# 6. REMOVING DUPLICATE RECORDS
# ===========================================================================
print("\n" + "="*60)
print("  STEP 6: Removing Duplicate Records")
print("="*60)
app.drop_duplicates(
    subset=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
            'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
            'FLAG_EMAIL', 'CNT_FAM_MEMBERS'],
    keep='first',
    inplace=True
)

# ===========================================================================
# 7. DATA CLEANING & FEATURE TRANSFORMATION
# ===========================================================================
print("\n" + "="*60)
print("  STEP 7: Data Cleaning & Feature Transformation")
print("="*60)
app['DAYS_BIRTH']     = app['DAYS_BIRTH'].abs()
app['DAYS_EMPLOYED']  = app['DAYS_EMPLOYED'].abs()

# ===========================================================================
# 8. STATUS BINARY MAPPING
# ===========================================================================
print("\n" + "="*60)
print("  STEP 8: STATUS -> Binary Mapping")
print("="*60)
credit_df['STATUS_BIN'] = credit_df['STATUS'].apply(lambda x: 1 if x in ['0', 'X', 'C'] else 0)
status_agg = credit_df.groupby('ID')['STATUS_BIN'].min().reset_index()
status_agg.columns = ['ID', 'TARGET']

# Gather EMI and loan counts
credit_df_extra = credit_df.groupby('ID').agg(
    EMI_PAID_OFF   = ('STATUS', lambda x: (x == '0').sum()),
    EMI_PASTDUES   = ('STATUS', lambda x: x.isin(['1','2','3','4','5']).sum()),
    NUMBER_OF_LOANS= ('STATUS', 'count')
).reset_index()

credit_final = status_agg.merge(credit_df_extra, on='ID', how='left')

# ===========================================================================
# 9. MERGE APPLICANT + CREDIT DATASETS
# ===========================================================================
print("\n" + "="*60)
print("  STEP 9: Merging Applicant + Credit Datasets on ID")
print("="*60)
final_df = app.merge(credit_final, how='inner', on='ID')
final_df.fillna(0, inplace=True)

# ===========================================================================
# 10. LABEL ENCODING
# ===========================================================================
print("\n" + "="*60)
print("  STEP 10: Label Encoding")
print("="*60)
cg    = LabelEncoder()
oc    = LabelEncoder()
own_r = LabelEncoder()
it    = LabelEncoder()
et    = LabelEncoder()
fs    = LabelEncoder()
ht    = LabelEncoder()

final_df['CODE_GENDER']        = cg.fit_transform(final_df['CODE_GENDER'])
final_df['FLAG_OWN_CAR']       = oc.fit_transform(final_df['FLAG_OWN_CAR'])
final_df['FLAG_OWN_REALTY']    = own_r.fit_transform(final_df['FLAG_OWN_REALTY'])
final_df['NAME_INCOME_TYPE']   = it.fit_transform(final_df['NAME_INCOME_TYPE'])
final_df['NAME_EDUCATION_TYPE']= et.fit_transform(final_df['NAME_EDUCATION_TYPE'])
final_df['NAME_FAMILY_STATUS'] = fs.fit_transform(final_df['NAME_FAMILY_STATUS'])
final_df['NAME_HOUSING_TYPE']  = ht.fit_transform(final_df['NAME_HOUSING_TYPE'])

# Save encoders
encoders = {
    'CODE_GENDER': cg, 'FLAG_OWN_CAR': oc, 'FLAG_OWN_REALTY': own_r,
    'NAME_INCOME_TYPE': it, 'NAME_EDUCATION_TYPE': et,
    'NAME_FAMILY_STATUS': fs, 'NAME_HOUSING_TYPE': ht
}
joblib.dump(encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))

# ===========================================================================
# 11. TRAIN/TEST SPLIT (Exactly 14 features matching Flask form)
# ===========================================================================
print("\n" + "="*60)
print("  STEP 11: Feature Selection & Train/Test Split")
print("="*60)

FEATURE_COLS = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS',
    'EMI_PAID_OFF', 'EMI_PASTDUES', 'NUMBER_OF_LOANS'
]

X = final_df[FEATURE_COLS]
y = final_df['TARGET'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

joblib.dump((X_train, X_test, y_train, y_test), os.path.join(MODELS_DIR, 'train_test_data.pkl'))
joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, 'feature_cols.pkl'))

print("[OK] Data preprocessing complete!")
