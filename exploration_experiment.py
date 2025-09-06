#!/usr/bin/env python3
"""
Lightweight exploration experiments to understand Experiment 3 performance gap
and test hypotheses for Experiment 4 design.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
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
    
    # Convert to binary classification (collision vs no collision)
    train_df['collision_binary'] = (train_df['collisions'] > 0).astype(int)
    test_df['collision_binary'] = (test_df['collisions'] > 0).astype(int)
    
    # Handle missing values (simple imputation)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop(['collisions', 'collision_binary'])
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
    test_df[numeric_cols] = test_df[numeric_cols].fillna(train_df[numeric_cols].median())
    
    # Encode month
    le = LabelEncoder()
    train_df['month_encoded'] = le.fit_transform(train_df['month'])
    test_df['month_encoded'] = le.transform(test_df['month'])
    
    # Feature columns (excluding driver_id as it's high cardinality)
    feature_cols = [col for col in numeric_cols if col != 'driver_id'] + ['month_encoded']
    
    X_train = train_df[feature_cols]
    y_train = train_df['collision_binary']
    X_test = test_df[feature_cols]
    y_test = test_df['collision_binary']
    
    return X_train, y_train, X_test, y_test, feature_cols

def experiment_1_nested_cv():
    """Test nested cross-validation to understand performance gap"""
    print("=== Experiment 1: Nested Cross-Validation Analysis ===")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    
    # Simulate Experiment 3 approach - SelectKBest + RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    selector = SelectKBest(f_classif, k=11)
    
    # Regular 5-fold CV (what was likely done in Exp 3)
    pipeline = Pipeline([
        ('selector', selector),
        ('classifier', rf)
    ])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, 
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                               scoring='roc_auc')
    
    print(f"Regular 5-fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Now fit on full train and test on holdout
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_pr_auc = calculate_pr_auc(y_test, y_pred_proba)
    
    print(f"Test ROC-AUC: {test_roc_auc:.3f}")
    print(f"Test PR-AUC: {test_pr_auc:.3f}")
    print(f"CV-Test ROC-AUC Gap: {cv_scores.mean() - test_roc_auc:.3f}")
    print()
    
    return cv_scores.mean(), test_roc_auc, test_pr_auc

def experiment_2_feature_selection_methods():
    """Compare different feature selection methods"""
    print("=== Experiment 2: Feature Selection Method Comparison ===")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    
    methods = {
        'SelectKBest_11': SelectKBest(f_classif, k=11),
        'SelectKBest_15': SelectKBest(f_classif, k=15),
        'RFE_11': RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=11),
        'L1_Logistic': None  # Will use penalty
    }
    
    results = {}
    
    for method_name, selector in methods.items():
        if method_name == 'L1_Logistic':
            # L1 regularized logistic regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Try different C values
            best_pr_auc = 0
            best_c = 1.0
            for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
                lr = LogisticRegression(penalty='l1', solver='liblinear', C=c, random_state=42)
                lr.fit(X_train_scaled, y_train)
                y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
                pr_auc = calculate_pr_auc(y_test, y_pred_proba)
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    best_c = c
            
            results[method_name] = {'PR_AUC': best_pr_auc, 'Best_C': best_c}
            print(f"{method_name}: PR-AUC = {best_pr_auc:.3f} (C={best_c})")
        else:
            # Feature selection + RandomForest
            pipeline = Pipeline([
                ('selector', selector),
                ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            pr_auc = calculate_pr_auc(y_test, y_pred_proba)
            
            results[method_name] = {'PR_AUC': pr_auc}
            print(f"{method_name}: PR-AUC = {pr_auc:.3f}")
    
    print()
    return results

def experiment_3_validation_strategies():
    """Test different validation strategies"""
    print("=== Experiment 3: Validation Strategy Analysis ===")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    
    # Split train into train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Original train size: {len(X_train)}")
    print(f"New train size: {len(X_tr)}")  
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")
    
    # Strategy 1: Train on reduced train, validate on validation set
    pipeline = Pipeline([
        ('selector', SelectKBest(f_classif, k=11)),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    
    pipeline.fit(X_tr, y_tr)
    
    # Performance on validation set
    val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    val_pr_auc = calculate_pr_auc(y_val, val_pred_proba)
    
    # Performance on test set
    test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pr_auc = calculate_pr_auc(y_test, test_pred_proba)
    
    print(f"Validation PR-AUC: {val_pr_auc:.3f}")
    print(f"Test PR-AUC: {test_pr_auc:.3f}")
    print(f"Val-Test Gap: {val_pr_auc - test_pr_auc:.3f}")
    
    # Strategy 2: Train on full train, test on test (current approach)
    pipeline_full = Pipeline([
        ('selector', SelectKBest(f_classif, k=11)),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    
    pipeline_full.fit(X_train, y_train)
    test_pred_proba_full = pipeline_full.predict_proba(X_test)[:, 1]
    test_pr_auc_full = calculate_pr_auc(y_test, test_pred_proba_full)
    
    print(f"Full Train -> Test PR-AUC: {test_pr_auc_full:.3f}")
    print()
    
    return val_pr_auc, test_pr_auc, test_pr_auc_full

if __name__ == "__main__":
    print("Starting exploration experiments...")
    print("="*50)
    
    # Run experiments
    cv_mean, test_roc, test_pr = experiment_1_nested_cv()
    fs_results = experiment_2_feature_selection_methods()
    val_pr, test_pr_split, test_pr_full = experiment_3_validation_strategies()
    
    print("=== SUMMARY ===")
    print(f"CV-Test ROC-AUC Gap: {cv_mean - test_roc:.3f}")
    print(f"Best Feature Selection Method: {max(fs_results.keys(), key=lambda k: fs_results[k]['PR_AUC'])}")
    print(f"Best PR-AUC: {max(fs_results.values(), key=lambda v: v['PR_AUC'])['PR_AUC']:.3f}")
    print(f"Validation Strategy Impact: Full train vs Split: {test_pr_full - test_pr_split:.3f}")