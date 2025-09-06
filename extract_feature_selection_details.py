#!/usr/bin/env python3
"""
Extract detailed feature selection information
"""

import pickle
import pandas as pd
import numpy as np
import json

def analyze_feature_selection():
    """Analyze feature selection details"""
    
    base_path = "/Users/yuvalheffetz/ds-agent-projects/session_5feb6ac6-f292-4d0c-9e41-ab6b3ffc14d6/experiments/experiment_3"
    
    # Load feature processor
    with open(f"{base_path}/model_artifacts/feature_processor.pkl", 'rb') as f:
        feature_processor = pickle.load(f)
    
    print("DETAILED FEATURE SELECTION ANALYSIS")
    print("="*80)
    
    # Get selected features
    if hasattr(feature_processor, 'selected_features'):
        selected_features = feature_processor.selected_features
        print(f"Selected features: {selected_features}")
        print(f"Number of selected features: {len(selected_features)}")
    
    # Get feature selector details
    if hasattr(feature_processor, 'feature_selector'):
        selector = feature_processor.feature_selector
        print(f"\nFeature selector type: {type(selector)}")
        
        if hasattr(selector, 'get_support'):
            support = selector.get_support()
            print(f"Feature support mask: {support}")
            print(f"Number of features selected: {sum(support)}")
        
        if hasattr(selector, 'scores_'):
            scores = selector.scores_
            print(f"\nFeature scores: {scores}")
            
            # If we can get feature names, create a detailed ranking
            if hasattr(selector, 'feature_names_in_'):
                feature_names = selector.feature_names_in_
                scores_df = pd.DataFrame({
                    'feature': feature_names,
                    'score': scores,
                    'selected': support
                }).sort_values('score', ascending=False)
                
                print(f"\nFEATURE RANKING BY F-SCORE:")
                print(scores_df.to_string(index=False))
                
                print(f"\nSELECTED FEATURES (Top {sum(support)}):")
                selected_df = scores_df[scores_df['selected']]
                print(selected_df.to_string(index=False))
                
                return scores_df
    
    return None

def cross_reference_with_model():
    """Cross-reference selected features with model feature importance"""
    
    base_path = "/Users/yuvalheffetz/ds-agent-projects/session_5feb6ac6-f292-4d0c-9e41-ab6b3ffc14d6/experiments/experiment_3"
    
    # Load model
    with open(f"{base_path}/model_artifacts/trained_models.pkl", 'rb') as f:
        model_wrapper = pickle.load(f)
    
    model = model_wrapper.model
    
    print("\n" + "="*80)
    print("MODEL FEATURE IMPORTANCE DETAILS")
    print("="*80)
    
    if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        importance_df = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("ALL FEATURES BY IMPORTANCE:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    return None

def analyze_model_performance_details():
    """Try to extract more performance details"""
    
    base_path = "/Users/yuvalheffetz/ds-agent-projects/session_5feb6ac6-f292-4d0c-9e41-ab6b3ffc14d6/experiments/experiment_3"
    
    # Check if there are any additional result files
    import os
    
    print("\n" + "="*80)
    print("SEARCHING FOR ADDITIONAL PERFORMANCE FILES")
    print("="*80)
    
    # Look in various directories for performance files
    search_paths = [
        f"{base_path}/",
        f"{base_path}/output/",
        f"{base_path}/results/",
        f"{base_path}/metrics/"
    ]
    
    performance_files = []
    for path in search_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if any(keyword in file.lower() for keyword in ['performance', 'metrics', 'results', 'cv', 'cross_validation', 'evaluation']):
                        full_path = os.path.join(root, file)
                        performance_files.append(full_path)
    
    print(f"Found performance-related files:")
    for file in performance_files:
        print(f"  - {file}")
    
    return performance_files

if __name__ == "__main__":
    # Analyze feature selection
    feature_scores = analyze_feature_selection()
    
    # Cross-reference with model
    feature_importance = cross_reference_with_model()
    
    # Look for additional performance files
    perf_files = analyze_model_performance_details()
    
    # Create combined analysis
    if feature_scores is not None and feature_importance is not None:
        print("\n" + "="*80)
        print("COMBINED FEATURE ANALYSIS")
        print("="*80)
        
        # Merge feature selection scores with model importance
        combined = pd.merge(
            feature_scores[['feature', 'score', 'selected']], 
            feature_importance[['feature', 'importance']], 
            on='feature', 
            how='outer'
        ).fillna(0)
        
        print("FEATURE SELECTION SCORES vs MODEL IMPORTANCE:")
        print(combined.sort_values('importance', ascending=False).to_string(index=False))