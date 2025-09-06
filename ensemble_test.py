#!/usr/bin/env python3
"""
Test ensemble methods and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def calculate_pr_auc(y_true, y_pred_proba):
    """Calculate PR-AUC for binary classification"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def prepare_data():
    """Load and preprocessing with feature engineering"""
    train_df = pd.read_csv('data/train_set.csv')
    test_df = pd.read_csv('data/test_set.csv')
    
    # Convert to binary classification 
    train_df['collision_binary'] = (train_df['collisions'] > 0).astype(int)
    test_df['collision_binary'] = (test_df['collisions'] > 0).astype(int)
    
    # Handle missing values
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop(['collisions', 'collision_binary'])
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
    test_df[numeric_cols] = test_df[numeric_cols].fillna(train_df[numeric_cols].median())
    
    # Feature engineering
    for df in [train_df, test_df]:
        df['miles_per_trip'] = np.where(df['count_trip'] > 0, df['miles'] / df['count_trip'], 0)
        df['hours_per_trip'] = np.where(df['count_trip'] > 0, df['drive_hours'] / df['count_trip'], 0)
        df['brakes_per_mile'] = np.where(df['miles'] > 0, df['count_brakes'] / df['miles'], 0)
        df['speed_ratio'] = df['time_speeding_hours'] / np.maximum(df['drive_hours'], 0.001)
        
        # Additional ratios
        df['accel_per_mile'] = np.where(df['miles'] > 0, df['count_accelarations'] / df['miles'], 0)
        df['highway_ratio'] = np.where(df['miles'] > 0, df['highway_miles'] / df['miles'], 0)
        df['night_ratio'] = df['night_drive_hrs'] / np.maximum(df['drive_hours'], 0.001)
    
    # Encode month
    le = LabelEncoder()
    train_df['month_encoded'] = le.fit_transform(train_df['month'])
    test_df['month_encoded'] = le.transform(test_df['month'])
    
    # Feature columns
    feature_cols = [col for col in numeric_cols if col != 'driver_id'] + [
        'month_encoded', 'miles_per_trip', 'hours_per_trip', 'brakes_per_mile', 
        'speed_ratio', 'accel_per_mile', 'highway_ratio', 'night_ratio'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['collision_binary']
    X_test = test_df[feature_cols]
    y_test = test_df['collision_binary']
    
    return X_train, y_train, X_test, y_test, feature_cols

def test_tuned_models():
    """Test hyperparameter tuned models"""
    print("=== Hyperparameter Tuned Models ===")
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data()
    print(f"Total features: {len(feature_cols)}")
    
    results = {}
    
    # 1. Tuned Random Forest
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print("Tuning Random Forest...")
    rf_grid.fit(X_train, y_train)
    rf_pred_proba = rf_grid.predict_proba(X_test)[:, 1]
    rf_pr_auc = calculate_pr_auc(y_test, rf_pred_proba)
    
    results['Tuned_RF'] = {
        'PR_AUC': rf_pr_auc,
        'params': rf_grid.best_params_
    }
    
    print(f"Tuned RF PR-AUC: {rf_pr_auc:.3f}")
    print(f"Best params: {rf_grid.best_params_}")
    
    # 2. Gradient Boosting
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }
    
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print("Tuning Gradient Boosting...")
    gb_grid.fit(X_train, y_train)
    gb_pred_proba = gb_grid.predict_proba(X_test)[:, 1]
    gb_pr_auc = calculate_pr_auc(y_test, gb_pred_proba)
    
    results['Tuned_GB'] = {
        'PR_AUC': gb_pr_auc,
        'params': gb_grid.best_params_
    }
    
    print(f"Tuned GB PR-AUC: {gb_pr_auc:.3f}")
    print(f"Best params: {gb_grid.best_params_}")
    
    return results

if __name__ == "__main__":
    results = test_tuned_models()
    
    print("\n=== FINAL SUMMARY ===")
    best_model = max(results.keys(), key=lambda k: results[k]['PR_AUC'])
    print(f"Best model: {best_model}")
    print(f"Best PR-AUC: {results[best_model]['PR_AUC']:.3f}")
    print(f"Target gap: {results[best_model]['PR_AUC'] - 0.280:+.3f}")
    print(f"Best parameters: {results[best_model]['params']}")