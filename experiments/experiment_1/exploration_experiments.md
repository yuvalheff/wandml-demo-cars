# Exploration Experiments Summary

## Overview
This document summarizes the lightweight exploration experiments conducted to understand the optimal approaches for the Vehicle Collision Prediction model before designing the comprehensive experiment plan.

## Experiments Conducted

### Experiment 1: Basic Preprocessing Baseline
**Approach**: Basic preprocessing without class balancing
- **Preprocessing**: Median imputation, standard scaling, label encoding for month
- **Model**: Random Forest with default settings
- **Results**: PR-AUC = 0.2107
- **Insights**: Model heavily biased toward majority class (0 collisions), poor recall for collision cases

### Experiment 2: SMOTE Class Balancing
**Approach**: Applied SMOTE oversampling to address class imbalance
- **Method**: SMOTE with k_neighbors=3 (reduced due to small minority class)
- **Results**: PR-AUC = 0.1788 (worse than baseline)
- **Insights**: SMOTE actually decreased performance, likely due to synthetic samples not representing realistic collision patterns

### Experiment 3: Feature Engineering Focus
**Approach**: Created domain-specific engineered features
- **New Features**: 
  - Exposure ratios: miles_per_trip, hours_per_trip, speed_ratio
  - Risk behavior ratios: brakes_per_mile, accel_per_mile, speeding_ratio
  - Composite scores: exposure_score, behavior_score
- **Results**: PR-AUC = 0.2480 (best so far)
- **Insights**: Feature engineering significantly improved performance; exposure-based features are most important

### Experiment 4: Model Comparison
**Approach**: Compared Random Forest, Gradient Boosting, and Logistic Regression
- **Results**:
  - Random Forest: PR-AUC = 0.2480 (best)
  - Gradient Boosting: PR-AUC = 0.1950
  - Logistic Regression: PR-AUC = 0.1863
- **Insights**: Random Forest with class_weight='balanced' performs best for this imbalanced problem

### Experiment 5: Binary vs Multi-class Strategy
**Approach**: Compared treating collision prediction as binary vs multi-class problem
- **Binary**: Collision (1,2) vs No Collision (0)
- **Multi-class**: Separate classes for 0, 1, 2 collisions
- **Results**:
  - Binary approach: PR-AUC = 0.2018
  - Multi-class approach: PR-AUC = 0.1755
- **Insights**: Binary classification performs better, likely due to very few class 2 samples

## Key Findings

1. **Feature Engineering is Critical**: Engineered features, especially exposure-based ratios, significantly improve performance
2. **Random Forest is Optimal**: Outperforms other algorithms for this imbalanced dataset
3. **Binary Classification Preferred**: Simpler binary approach (collision vs no collision) works better than multi-class
4. **Class Balancing**: Traditional SMOTE doesn't help; class_weight='balanced' is more effective
5. **Important Features**: miles, drive_hours, exposure_score, and engineered ratios are most predictive

## Recommendations for Final Experiment

Based on these findings, the optimal approach combines:
- Engineered exposure and behavior ratio features  
- Random Forest with balanced class weights
- Binary classification target
- Careful feature selection focusing on exposure metrics
- Robust evaluation strategy for imbalanced data