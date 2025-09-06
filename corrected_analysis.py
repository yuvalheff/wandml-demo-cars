#!/usr/bin/env python3
"""
Corrected analysis accounting for baseline PR-AUC
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
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
    
    # Convert to binary classification 
    train_df['collision_binary'] = (train_df['collisions'] > 0).astype(int)
    test_df['collision_binary'] = (test_df['collisions'] > 0).astype(int)
    
    # Handle missing values
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop(['collisions', 'collision_binary'])
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
    test_df[numeric_cols] = test_df[numeric_cols].fillna(train_df[numeric_cols].median())
    
    # Add feature engineering - exposure ratios
    for df in [train_df, test_df]:
        df['miles_per_trip'] = np.where(df['count_trip'] > 0, df['miles'] / df['count_trip'], 0)
        df['hours_per_trip'] = np.where(df['count_trip'] > 0, df['drive_hours'] / df['count_trip'], 0)
        df['brakes_per_mile'] = np.where(df['miles'] > 0, df['count_brakes'] / df['miles'], 0)
        df['speed_ratio'] = df['time_speeding_hours'] / np.maximum(df['drive_hours'], 0.001)
    
    # Encode month
    le = LabelEncoder()
    train_df['month_encoded'] = le.fit_transform(train_df['month'])
    test_df['month_encoded'] = le.transform(test_df['month'])
    
    # Feature columns
    feature_cols = [col for col in numeric_cols if col != 'driver_id'] + [
        'month_encoded', 'miles_per_trip', 'hours_per_trip', 'brakes_per_mile', 'speed_ratio'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['collision_binary']
    X_test = test_df[feature_cols]
    y_test = test_df['collision_binary']
    
    return X_train, y_train, X_test, y_test, feature_cols

def test_models():
    """Test different models and compare to baseline"""
    print("=== Model Comparison (Corrected) ===")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    
    # Calculate true baseline
    baseline_pr_auc = y_test.mean()  # This is the no-skill baseline
    print(f"No-skill baseline PR-AUC: {baseline_pr_auc:.3f}")
    
    models = {}
    
    # 1. Random Forest with SelectKBest (Experiment 3 approach)
    models['RF_SelectKBest'] = Pipeline([
        ('selector', SelectKBest(f_classif, k=11)),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    
    # 2. L1 Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models['L1_Logistic'] = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, random_state=42)
    
    # 3. Random Forest with RFE
    models['RF_RFE'] = Pipeline([
        ('selector', RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=11)),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    
    # 4. Random Forest with all features
    models['RF_All'] = RandomForestClassifier(n_estimators=200, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        if name == 'L1_Logistic':
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        pr_auc = calculate_pr_auc(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'PR_AUC': pr_auc,
            'ROC_AUC': roc_auc,
            'PR_AUC_vs_Baseline': pr_auc - baseline_pr_auc
        }
        
        print(f"{name:15}: PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f}, "
              f"vs Baseline={pr_auc - baseline_pr_auc:+.3f}")
    
    return results, baseline_pr_auc

def analyze_best_model():
    """Analyze the best performing model"""
    print("\n=== Best Model Analysis ===")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    
    # Based on previous experiments, RF with all features seems promising
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    pr_auc = calculate_pr_auc(y_test, y_pred_proba)
    
    print(f"Random Forest (all features) PR-AUC: {pr_auc:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return pr_auc, feature_importance

if __name__ == "__main__":
    results, baseline = test_models()
    pr_auc_best, feat_imp = analyze_best_model()
    
    print("\n=== SUMMARY ===")
    best_model = max(results.keys(), key=lambda k: results[k]['PR_AUC'])
    print(f"Best model: {best_model}")
    print(f"Best PR-AUC: {results[best_model]['PR_AUC']:.3f}")
    print(f"Improvement over baseline: {results[best_model]['PR_AUC_vs_Baseline']:+.3f}")
    print(f"Target PR-AUC: 0.280")
    print(f"Gap to target: {results[best_model]['PR_AUC'] - 0.280:+.3f}")