"""
train.py
========
Trains four classifiers on the preprocessed Credit Card Approval dataset:
  1. Logistic Regression
  2. Random Forest
  3. Decision Tree
  4. XGBoost

Evaluates each model, selects the best by F1-Score, and saves it as model.pkl.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.metrics          import (confusion_matrix, classification_report,
                                      accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score,
                                      roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Load preprocessed train/test data ---
print("Loading preprocessed data...")
data_path = os.path.join(MODELS_DIR, 'train_test_data.pkl')
if not os.path.exists(data_path):
    raise FileNotFoundError(
        "train_test_data.pkl not found. Run src/data_preprocessing.py first.")

X_train, X_test, y_train, y_test = joblib.load(data_path)
print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"Class balance (train) - 1:{(y_train==1).sum()} | 0:{(y_train==0).sum()}")


# ===========================================================================
# MODEL 1 - Logistic Regression
# ===========================================================================
def logistic_reg(X_train, X_test, y_train, y_test):
    """
    Builds, trains, tests, and evaluates a Logistic Regression classification model.
    Accepts pre-split training and testing data (X_train, X_test, y_train, y_test).
    Returns the trained model and its predictions.
    """
    print("\nInitializing Logistic Regression classifier...")
    # Initialize Logistic Regression with a fixed random state for reproducibility
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    print("Training the Logistic Regression model on training data...")
    # X_train contains input features (income, education, credit history, etc.)
    # y_train contains target labels (approval status: 1=Approved, 0=Not Approved)
    lr_model.fit(X_train, y_train)

    print("Generating predictions on the unseen test dataset...")
    # Predicts approval status for new applicants in X_test
    y_pred = lr_model.predict(X_test)

    print("\n== Model Evaluation Results ==")
    # 1. Confusion Matrix: Shows counts of True Positives, True Negatives,
    #    False Positives, and False Negatives.
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\n### Classification Report ###")
    print(classification_report(y_test, y_pred))

    return lr_model, y_pred


# ===========================================================================
# MODEL 2 - Random Forest
# ===========================================================================
def random_forest(X_train, X_test, y_train, y_test):
    """
    Builds, trains, and tests a Random Forest classification model,
    returning performance metrics.
    """
    # RandomForestClassifier() is initialized.
    # We use some common default hyperparameters for good initial performance.
    rf_model = RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=-1)

    # Trained on the training data.
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)

    # Tested on the test set.
    print("Generating predictions...")
    y_pred = rf_model.predict(X_test)

    print("\n" + "="*40)
    print("Random Forest Model Evaluation")
    print("="*40)

    # Classification Report provides Precision, Recall, F1-score, and Support for each class.
    print("\n### Classification Report ###")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return rf_model, y_pred


# ===========================================================================
# MODEL 3 - Decision Tree
# ===========================================================================
def d_tree(X_train, X_test, y_train, y_test):
    """
    Builds, trains, tests, and evaluates a Decision Tree classifier.
    """
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print('***DecisionTreeClassifier***')
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Classification report')
    print(classification_report(y_test, y_pred))
    return dt, y_pred


# ===========================================================================
# MODEL 4 - XGBoost
# ===========================================================================
def xgboost_model(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates an XGBoost classifier.
    """
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                        eval_metric='logloss', n_jobs=-1)
    print("\nTraining XGBoost model...")
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    print("\n" + "="*40)
    print("XGBoost Model Evaluation")
    print("="*40)
    print("\n### Classification Report ###")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    return xgb, y_pred


# ===========================================================================
# TRAIN ALL MODELS
# ===========================================================================
print("\n" + "="*60)
print("  Training All Models")
print("="*60)

lr_model,  lr_pred  = logistic_reg(X_train, X_test, y_train, y_test)
rf_model,  rf_pred  = random_forest(X_train, X_test, y_train, y_test)
dt_model,  dt_pred  = d_tree(X_train, X_test, y_train, y_test)
xgb_model, xgb_pred = xgboost_model(X_train, X_test, y_train, y_test)


# ===========================================================================
# COMPARE MODELS & SELECT BEST
# ===========================================================================
models = {
    'Logistic Regression': (lr_model,  lr_pred),
    'Random Forest'      : (rf_model,  rf_pred),
    'Decision Tree'      : (dt_model,  dt_pred),
    'XGBoost'            : (xgb_model, xgb_pred),
}

metrics = {}
for name, (model, y_pred) in models.items():
    metrics[name] = {
        'Accuracy' : round(accuracy_score(y_test, y_pred),  4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_test, y_pred,    zero_division=0), 4),
        'F1-Score' : round(f1_score(y_test, y_pred,        zero_division=0), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_pred),  4),
    }

print("\n" + "="*60)
print("  Model Comparison")
print("="*60)
for name, m in metrics.items():
    print(f"\nModel: {name}")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

# Best model by F1-Score
best_name = max(metrics, key=lambda k: metrics[k]['F1-Score'])
best_model = models[best_name][0]
print(f"\n[OK] Best Model: {best_name} (F1={metrics[best_name]['F1-Score']:.4f})")

# Save evaluation results
eval_results = {'best_model_name': best_name, 'metrics': metrics}
with open(os.path.join(MODELS_DIR, 'evaluation_results.json'), 'w') as f:
    json.dump(eval_results, f, indent=2)

# Save best model as model.pkl (for Flask app)
with open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)
print(f"-> Saved best model to models/model.pkl")

# Also save named pkl files for reference
with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)


# ===========================================================================
# PLOTS
# ===========================================================================

# 1. Confusion Matrices (all 4)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, (name, (model, y_pred)) in enumerate(models.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Approved','Approved'],
                yticklabels=['Not Approved','Approved'])
    axes[idx].set_title(f'{name}\nF1={metrics[name]["F1-Score"]:.4f}', fontweight='bold')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')
plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrices.png'), dpi=100, bbox_inches='tight')
plt.close()
print("-> Saved: confusion_matrices.png")

# 2. ROC Curves
fig, ax = plt.subplots(figsize=(10, 7))
for name, (model, _) in models.items():
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        prob = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'roc_curves.png'), dpi=100)
plt.close()
print("-> Saved: roc_curves.png")

# 3. Feature Importance (from best tree-based model)
tree_models = {k: v for k, v in models.items()
               if hasattr(v[0], 'feature_importances_')}
if tree_models:
    fi_name  = max(tree_models, key=lambda k: metrics[k]['F1-Score'])
    fi_model = tree_models[fi_name][0]
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))
    importances  = fi_model.feature_importances_
    fi_df = (pd.Series(importances, index=feature_cols)
               .sort_values(ascending=True)
               .tail(15))
    fig, ax = plt.subplots(figsize=(10, 7))
    fi_df.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Feature Importance - {fi_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=100)
    plt.close()
    print("-> Saved: feature_importance.png")

# 4. Class Distribution (TARGET)
fig, ax = plt.subplots(figsize=(6, 5))
labels = ['Not Approved (0)', 'Approved (1)']
counts = [int((y_test == 0).sum()), int((y_test == 1).sum())]
colors = ['#e74c3c', '#2ecc71']
ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.5)
ax.set_title('Class Distribution (Test Set)', fontsize=13, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(counts):
    ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'class_distribution.png'), dpi=100)
plt.close()
print("-> Saved: class_distribution.png")

print("\n[OK] Model training complete!")
print(f"   Best model : {best_name}")
print(f"   F1-Score   : {metrics[best_name]['F1-Score']:.4f}")
print(f"   Accuracy   : {metrics[best_name]['Accuracy']:.4f}")
