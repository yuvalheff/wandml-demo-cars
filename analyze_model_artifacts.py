#!/usr/bin/env python3
"""
Script to analyze model artifacts from experiment 3
"""

import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_and_analyze_artifacts(pickle_path):
    """Load and analyze model artifacts from pickle file"""
    
    print(f"Loading artifacts from: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        print(f"Successfully loaded artifacts. Type: {type(artifacts)}")
        
        # Analyze the structure of the artifacts
        print(f"\nAnalyzing ModelWrapper object...")
        
        # Get all attributes of the ModelWrapper object
        attributes = dir(artifacts)
        print(f"Available attributes: {len(attributes)}")
        
        # Filter out built-in methods and show relevant attributes
        relevant_attrs = [attr for attr in attributes if not attr.startswith('_')]
        print(f"Relevant attributes: {relevant_attrs}")
        
        print("\n" + "="*80)
        print("DETAILED ANALYSIS - ModelWrapper Object")
        print("="*80)
        
        # Analyze each attribute
        for attr_name in relevant_attrs:
            try:
                attr_value = getattr(artifacts, attr_name)
                print(f"\n--- {attr_name.upper()} ---")
                print(f"Type: {type(attr_value)}")
                
                # Analyze different types of attributes
                if callable(attr_value):
                    print("This is a method/function")
                    continue
                    
                if isinstance(attr_value, dict):
                    print(f"Dictionary with {len(attr_value)} keys: {list(attr_value.keys())}")
                    analyze_dict_attribute(attr_name, attr_value)
                elif isinstance(attr_value, pd.DataFrame):
                    print(f"DataFrame shape: {attr_value.shape}")
                    print("Columns:", list(attr_value.columns))
                    if attr_value.shape[0] < 20:
                        print("\nData preview:")
                        print(attr_value.to_string())
                elif isinstance(attr_value, (list, tuple)):
                    print(f"List/Tuple with {len(attr_value)} items")
                    if len(attr_value) < 10:
                        for i, item in enumerate(attr_value):
                            print(f"  [{i}]: {type(item)}")
                            if hasattr(item, 'shape'):
                                print(f"      Shape: {item.shape}")
                elif hasattr(attr_value, '__dict__'):
                    print(f"Object with attributes: {list(vars(attr_value).keys())}")
                elif hasattr(attr_value, 'shape'):
                    print(f"Array/Matrix shape: {attr_value.shape}")
                else:
                    print(f"Value: {attr_value}")
                    
            except Exception as e:
                print(f"Error accessing {attr_name}: {e}")
        
        # Try to access common model attributes
        print("\n" + "="*80)
        print("SEARCHING FOR COMMON MODEL ATTRIBUTES")
        print("="*80)
        
        common_attrs = [
            'models', 'cv_results', 'feature_importance', 'threshold_analysis',
            'performance_metrics', 'feature_selection_results', 'best_model',
            'results', 'metrics', 'scores', 'evaluation', 'validation_scores'
        ]
        
        for attr in common_attrs:
            if hasattr(artifacts, attr):
                try:
                    value = getattr(artifacts, attr)
                    print(f"\nFound {attr}: {type(value)}")
                    if isinstance(value, dict):
                        analyze_dict_attribute(attr, value)
                    elif hasattr(value, '__dict__'):
                        print(f"  Object attributes: {list(vars(value).keys())}")
                except Exception as e:
                    print(f"Error accessing {attr}: {e}")
        
        return artifacts
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None

def analyze_dict_attribute(attr_name, attr_value):
    """Analyze dictionary attributes in detail"""
    for key, value in attr_value.items():
        print(f"  {key}: {type(value)}")
        
        if isinstance(value, dict):
            print(f"    Sub-dictionary with keys: {list(value.keys())}")
            if attr_name in ['cv_results', 'performance_metrics']:
                analyze_metrics_dict(key, value)
        elif isinstance(value, pd.DataFrame):
            print(f"    DataFrame shape: {value.shape}")
            print(f"    Columns: {list(value.columns)}")
            if value.shape[0] < 10:
                print(f"    Data preview:\n{value.to_string()}")
        elif isinstance(value, (list, np.ndarray)):
            print(f"    Length: {len(value)}")
            if hasattr(value, 'shape'):
                print(f"    Shape: {value.shape}")
            if len(value) < 20:
                print(f"    Values: {value}")
        elif isinstance(value, (int, float)):
            print(f"    Value: {value}")
        elif hasattr(value, '__dict__'):
            print(f"    Object attributes: {list(vars(value).keys())}")

def analyze_metrics_dict(model_name, metrics):
    """Analyze metrics dictionary for a specific model"""
    print(f"    Metrics for {model_name}:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (list, np.ndarray)):
            mean_val = np.mean(metric_value)
            std_val = np.std(metric_value)
            print(f"      {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"      {metric_name}: {metric_value}")

def analyze_cv_results(cv_results):
    """Analyze cross-validation results"""
    print("Cross-validation results analysis:")
    
    if isinstance(cv_results, dict):
        for model_name, results in cv_results.items():
            print(f"\n  Model: {model_name}")
            if isinstance(results, dict):
                for metric, values in results.items():
                    if isinstance(values, (list, np.ndarray)):
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"    {metric}: {mean_val:.4f} ± {std_val:.4f}")
                    else:
                        print(f"    {metric}: {values}")
            else:
                print(f"    Results type: {type(results)}")

def analyze_feature_importance(feature_importance):
    """Analyze feature importance"""
    print("Feature importance analysis:")
    
    if isinstance(feature_importance, dict):
        for model_name, importance in feature_importance.items():
            print(f"\n  Model: {model_name}")
            if isinstance(importance, pd.DataFrame):
                print(f"    Top 10 features:")
                print(importance.head(10).to_string())
            elif isinstance(importance, dict):
                sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
                print(f"    Top 10 features:")
                for feat, imp in sorted_features[:10]:
                    print(f"      {feat}: {imp:.4f}")

def analyze_threshold_analysis(threshold_analysis):
    """Analyze threshold optimization results"""
    print("Threshold analysis:")
    
    if isinstance(threshold_analysis, dict):
        for model_name, analysis in threshold_analysis.items():
            print(f"\n  Model: {model_name}")
            if isinstance(analysis, dict):
                for metric, value in analysis.items():
                    print(f"    {metric}: {value}")

def analyze_models(models):
    """Analyze trained models"""
    print("Trained models analysis:")
    
    if isinstance(models, dict):
        for model_name, model in models.items():
            print(f"\n  Model: {model_name}")
            print(f"    Type: {type(model)}")
            if hasattr(model, 'get_params'):
                params = model.get_params()
                print(f"    Parameters: {list(params.keys())}")

def analyze_performance_metrics(metrics):
    """Analyze performance metrics"""
    print("Performance metrics analysis:")
    
    if isinstance(metrics, dict):
        for model_name, model_metrics in metrics.items():
            print(f"\n  Model: {model_name}")
            if isinstance(model_metrics, dict):
                for metric_name, metric_value in model_metrics.items():
                    print(f"    {metric_name}: {metric_value}")

def analyze_feature_selection(feature_selection):
    """Analyze feature selection results"""
    print("Feature selection analysis:")
    
    if isinstance(feature_selection, dict):
        for key, value in feature_selection.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, (list, np.ndarray)):
                print(f"    Length: {len(value)}")
                if len(value) < 50:
                    print(f"    Values: {value}")

if __name__ == "__main__":
    pickle_path = "/Users/yuvalheffetz/ds-agent-projects/session_5feb6ac6-f292-4d0c-9e41-ab6b3ffc14d6/experiments/experiment_3/model_artifacts/trained_models.pkl"
    artifacts = load_and_analyze_artifacts(pickle_path)