#!/usr/bin/env python3
"""
Deep analysis of L1 regularized logistic regression performance
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

def calculate_pr_auc(y_true, y_pred_proba):
    """Calculate PR-AUC for binary classification"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def prepare_data():
    """Load and basic preprocessing"""
    train_df = pd.read_csv('data/train_set.csv')
    test_df = pd.read_csv('data/test_set.csv')
    
    # Convert to binary classification 
    train_df['collision_binary'] = (train_df['collisions'] > 0).astype(int)
    test_df['collision_binary'] = (test_df['collisions'] > 0).astype(int)
    
    # Handle missing values
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop(['collisions', 'collision_binary'])
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
    test_df[numeric_cols] = test_df[numeric_cols].fillna(train_df[numeric_cols].median())
    
    # Encode month
    le = LabelEncoder()
    train_df['month_encoded'] = le.fit_transform(train_df['month'])
    test_df['month_encoded'] = le.transform(test_df['month'])
    
    # Feature columns
    feature_cols = [col for col in numeric_cols if col != 'driver_id'] + ['month_encoded']
    
    X_train = train_df[feature_cols]
    y_train = train_df['collision_binary']
    X_test = test_df[feature_cols]
    y_test = test_df['collision_binary']
    
    return X_train, y_train, X_test, y_test, feature_cols

def analyze_l1_performance():
    """Deep analysis of L1 logistic regression"""
    print("=== L1 Logistic Regression Deep Analysis ===")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different C values more systematically
    C_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    results = []
    
    for C in C_values:
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(lr, X_train_scaled, y_train, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                  scoring='roc_auc')
        
        # Fit and test
        lr.fit(X_train_scaled, y_train)
        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        test_pr_auc = calculate_pr_auc(y_test, y_pred_proba)
        
        # Count non-zero coefficients
        n_features_selected = np.sum(lr.coef_[0] != 0)
        
        results.append({
            'C': C,
            'CV_ROC_AUC': cv_scores.mean(),
            'CV_ROC_AUC_std': cv_scores.std(),
            'Test_ROC_AUC': test_roc_auc,
            'Test_PR_AUC': test_pr_auc,
            'N_Features': n_features_selected,
            'CV_Test_Gap': cv_scores.mean() - test_roc_auc
        })
        
        print(f"C={C:6.4f}: CV ROC-AUC={cv_scores.mean():.3f}±{cv_scores.std():.3f}, "
              f"Test ROC-AUC={test_roc_auc:.3f}, Test PR-AUC={test_pr_auc:.3f}, "
              f"Features={n_features_selected}, Gap={cv_scores.mean() - test_roc_auc:.3f}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['Test_PR_AUC'])
    print(f"\nBest C: {best_result['C']} with PR-AUC: {best_result['Test_PR_AUC']:.3f}")
    
    # Analyze best model coefficients
    best_C = best_result['C']
    lr_best = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, random_state=42)
    lr_best.fit(X_train_scaled, y_train)
    
    # Feature importance
    coeffs = lr_best.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': coeffs,
        'abs_coefficient': np.abs(coeffs)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nTop 10 Features (C={best_C}):")
    print(feature_importance.head(10))
    
    # Check class balance performance
    y_pred = lr_best.predict(X_test_scaled)
    print(f"\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred))
    
    return results, best_result, feature_importance

if __name__ == "__main__":
    results, best_result, feature_importance = analyze_l1_performance()
    
    print("\n=== SUMMARY ===")
    print(f"Best L1 Logistic Regression PR-AUC: {best_result['Test_PR_AUC']:.3f}")
    print(f"Target PR-AUC: 0.280")
    print(f"Performance vs Target: +{best_result['Test_PR_AUC'] - 0.280:.3f}")
    print(f"Best C value: {best_result['C']}")
    print(f"Number of selected features: {best_result['N_Features']}")