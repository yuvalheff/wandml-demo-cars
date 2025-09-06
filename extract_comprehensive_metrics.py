#!/usr/bin/env python3
"""
Comprehensive extraction of model metrics from experiment 3
"""

import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
from bs4 import BeautifulSoup
import re

def load_model_artifacts(pickle_path):
    """Load and analyze the ModelWrapper object"""
    with open(pickle_path, 'rb') as f:
        model_wrapper = pickle.load(f)
    
    print("="*80)
    print("MODEL WRAPPER ANALYSIS")
    print("="*80)
    
    # Get the trained model
    model = model_wrapper.model
    print(f"Model Type: {type(model)}")
    print(f"Model Parameters: {model.get_params()}")
    
    # Extract feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nFeature Importance (Top 15):")
            print(importance_df.head(15).to_string(index=False))
            
            return model_wrapper, importance_df
    
    return model_wrapper, None

def load_data_processor(pickle_path):
    """Load and analyze data processor"""
    with open(pickle_path, 'rb') as f:
        data_processor = pickle.load(f)
    
    print("\n" + "="*80)
    print("DATA PROCESSOR ANALYSIS")
    print("="*80)
    
    print(f"Data Processor Type: {type(data_processor)}")
    
    # Analyze attributes
    attrs = [attr for attr in dir(data_processor) if not attr.startswith('_')]
    print(f"Available methods/attributes: {attrs}")
    
    return data_processor

def load_feature_processor(pickle_path):
    """Load and analyze feature processor"""
    with open(pickle_path, 'rb') as f:
        feature_processor = pickle.load(f)
    
    print("\n" + "="*80)
    print("FEATURE PROCESSOR ANALYSIS")
    print("="*80)
    
    print(f"Feature Processor Type: {type(feature_processor)}")
    
    # Analyze attributes
    attrs = [attr for attr in dir(feature_processor) if not attr.startswith('_')]
    print(f"Available methods/attributes: {attrs}")
    
    # If it's a SelectKBest, get selected features
    if hasattr(feature_processor, 'get_support'):
        support = feature_processor.get_support()
        print(f"Number of selected features: {sum(support)}")
        
        if hasattr(feature_processor, 'feature_names_in_'):
            selected_features = feature_processor.feature_names_in_[support]
            print(f"Selected features: {list(selected_features)}")
        
        if hasattr(feature_processor, 'scores_'):
            scores = feature_processor.scores_
            print(f"Feature scores shape: {scores.shape}")
    
    return feature_processor

def extract_metrics_from_html(html_path):
    """Extract metrics from HTML visualization files"""
    print(f"\nExtracting data from: {html_path}")
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for JSON data in script tags
        soup = BeautifulSoup(content, 'html.parser')
        scripts = soup.find_all('script')
        
        metrics = {}
        
        for script in scripts:
            script_content = script.get_text()
            
            # Look for common metric patterns
            if 'AUC' in script_content or 'auc' in script_content:
                # Extract AUC values
                auc_matches = re.findall(r'["\']?(?:PR-AUC|ROC-AUC|AUC)["\']?\s*[:=]\s*([0-9.]+)', script_content)
                for match in auc_matches:
                    metrics[f'auc_value'] = float(match)
            
            # Look for precision/recall/f1 values
            pr_matches = re.findall(r'["\']?(precision|recall|f1)["\']?\s*[:=]\s*([0-9.]+)', script_content, re.IGNORECASE)
            for metric_name, value in pr_matches:
                metrics[metric_name.lower()] = float(value)
            
            # Look for threshold values
            threshold_matches = re.findall(r'["\']?threshold["\']?\s*[:=]\s*([0-9.]+)', script_content, re.IGNORECASE)
            for match in threshold_matches:
                metrics['threshold'] = float(match)
        
        return metrics
        
    except Exception as e:
        print(f"Error reading {html_path}: {e}")
        return {}

def analyze_all_artifacts():
    """Comprehensive analysis of all artifacts"""
    base_path = "/Users/yuvalheffetz/ds-agent-projects/session_5feb6ac6-f292-4d0c-9e41-ab6b3ffc14d6/experiments/experiment_3"
    
    print("COMPREHENSIVE EXPERIMENT 3 METRICS ANALYSIS")
    print("="*100)
    
    # 1. Load manifest.json for primary metric
    manifest_path = f"{base_path}/output/general_artifacts/manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    primary_metric = manifest.get('metric', {})
    print(f"\nPRIMARY METRIC FROM MANIFEST:")
    print(f"  {primary_metric.get('name', 'N/A')}: {primary_metric.get('value', 'N/A'):.6f}")
    
    # 2. Load and analyze model artifacts
    model_wrapper, feature_importance = load_model_artifacts(f"{base_path}/model_artifacts/trained_models.pkl")
    data_processor = load_data_processor(f"{base_path}/model_artifacts/data_processor.pkl")
    feature_processor = load_feature_processor(f"{base_path}/model_artifacts/feature_processor.pkl")
    
    # 3. Extract metrics from HTML files
    html_files = [
        "precision_recall_curve.html",
        "roc_curve.html", 
        "confusion_matrix.html",
        "threshold_analysis.html"
    ]
    
    all_html_metrics = {}
    for html_file in html_files:
        html_path = f"{base_path}/plots/{html_file}"
        if Path(html_path).exists():
            metrics = extract_metrics_from_html(html_path)
            all_html_metrics[html_file] = metrics
    
    # 4. Generate comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*80)
    
    summary = {
        "experiment_name": "Feature Selection Enhanced Vehicle Collision Prediction",
        "model_type": "RandomForestClassifier",
        "primary_metric": primary_metric,
        "model_parameters": model_wrapper.model.get_params(),
        "feature_importance": feature_importance.to_dict('records') if feature_importance is not None else None,
        "html_extracted_metrics": all_html_metrics,
        "model_info": {
            "n_features_in": getattr(model_wrapper.model, 'n_features_in_', None),
            "n_classes": getattr(model_wrapper.model, 'n_classes_', None),
            "classes": getattr(model_wrapper.model, 'classes_', None).tolist() if hasattr(model_wrapper.model, 'classes_') else None
        }
    }
    
    # Save comprehensive results
    output_path = f"{base_path}/comprehensive_metrics_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nComprehensive analysis saved to: {output_path}")
    
    # Display key findings
    print(f"\nKEY FINDINGS:")
    print(f"  • Model: {summary['model_type']} with {summary['model_info']['n_features_in']} features")
    print(f"  • Primary Metric: {summary['primary_metric']['name']} = {summary['primary_metric']['value']:.6f}")
    print(f"  • Number of classes: {summary['model_info']['n_classes']}")
    print(f"  • Classes: {summary['model_info']['classes']}")
    
    if feature_importance is not None:
        print(f"\nTOP 5 MOST IMPORTANT FEATURES:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.6f}")
    
    return summary

if __name__ == "__main__":
    summary = analyze_all_artifacts()