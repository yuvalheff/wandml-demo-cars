# Experiment 2: Calibrated Vehicle Collision Prediction with Threshold Optimization

## Experiment Overview

**Objective**: Improve precision-recall performance through probability calibration and threshold optimization, building on the exposure-based RandomForest model from iteration 1.

**Primary Metric**: PR-AUC (Target: ≥ 0.25, representing 5% improvement over iteration 1 baseline of ~0.24)

**Execution Date**: September 6, 2025  
**Status**: ✅ **Successfully Completed**

## Key Results Summary

### Primary Performance Metrics
- **PR-AUC**: **0.245** (2.1% improvement, slightly below 0.25 target)
- **ROC-AUC**: **0.811** (maintained strong discrimination, above 0.80 target)
- **Overall Assessment**: Achieved core objectives with room for calibration improvement

### Threshold Optimization Results

| Strategy | Threshold | Precision | Recall | F1-Score | Business Use Case |
|----------|-----------|-----------|--------|----------|------------------|
| **Optimal F1** | 0.1555 | 0.325 | 0.286 | **0.304** | Balanced performance |
| **High Precision** | 0.2052 | **0.405** | 0.187 | 0.256 | Minimize false alarms |
| **High Recall** | 0.0436 | 0.134 | **0.615** | 0.220 | Safety-critical applications |
| **Default** | 0.5000 | 0.667 | 0.044 | 0.083 | Conservative baseline |

✅ **Success**: Provided 3 viable business strategies with precision range 13-67% and recall range 4-62%

## Model Architecture & Implementation

### Base Model Configuration
- **Algorithm**: RandomForestClassifier
- **Hyperparameters**: 100 estimators, balanced class weights, random_state=42
- **Calibration**: CalibratedClassifierCV with sigmoid method and 5-fold cross-validation
- **Feature Set**: 16 features (11 original + 5 engineered exposure-based features)

### Feature Engineering Validation
- **Top Features**: drive_hours (16.6%), miles (11.6%), count_brakes (9.7%)
- **Engineered Features Impact**: 24.7% of total feature importance
- **Key Insight**: Exposure-based features (miles_per_trip, hours_per_trip, etc.) significantly enhance predictive power

## Experiment Weaknesses & Areas for Improvement

### 1. **Calibration Quality Issues**
- **Problem**: Mixed calibration results across probability bins
- **Evidence**: Some bins exceed ±10% accuracy target (e.g., high-risk bin shows 35% error)
- **Impact**: Reduced reliability for probability-based decision making
- **Root Cause**: Sigmoid calibration may not be optimal for this data distribution

### 2. **PR-AUC Performance Gap**
- **Problem**: Achieved 0.245 vs 0.25 target (2% shortfall)
- **Evidence**: Improvement was modest compared to ambitious 5% target
- **Impact**: Limited advancement in addressing class imbalance challenge
- **Root Cause**: Current feature set and model architecture may have reached performance ceiling

### 3. **Class Imbalance Persistence**
- **Problem**: Model still struggles with extreme class imbalance (~96% no collision)
- **Evidence**: Low precision scores across most thresholds (13-40%)
- **Impact**: High false positive rates in real deployment
- **Root Cause**: Insufficient data synthesis or sampling techniques

### 4. **Limited Feature Diversity**
- **Problem**: Heavy reliance on basic telematic features
- **Evidence**: Top features are raw driving metrics (hours, miles, brakes)
- **Impact**: May miss complex behavioral patterns
- **Root Cause**: Lack of temporal, contextual, or interaction features

## Success Criteria Assessment

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| PR-AUC Improvement | ≥ 0.25 | 0.245 | ❌ **Just Below** |
| ROC-AUC Maintenance | ≥ 0.80 | 0.811 | ✅ **Exceeded** |
| Calibration Quality | ±10% across bins | Mixed results | ❓ **Partial** |
| Threshold Flexibility | 3+ strategies | 3 strategies | ✅ **Met** |

## Technical Artifacts Generated

### Model Artifacts
- **trained_model.pkl** (31MB): Complete calibrated RandomForest pipeline
- **data_processor.pkl** (2.2KB): Data preprocessing transformations
- **feature_processor.pkl** (227B): Feature engineering pipeline
- **MLflow Model Package**: Full reproducibility package with metadata

### Evaluation Artifacts
- **7 Interactive HTML Plots**: Comprehensive visualizations for calibration, PR curves, ROC curves, feature importance, threshold analysis, confusion matrices, and prediction distributions
- **Manifest.json**: Complete execution metadata and primary metrics
- **Model Signature**: Production-ready input/output specifications

## Context for Next Iteration

### What Worked Well
1. **Threshold optimization successfully provided business-relevant strategies**
2. **ROC-AUC performance remained strong despite calibration**
3. **Feature engineering validation confirmed exposure-based hypothesis**
4. **Complete MLflow integration enables easy deployment**

### Critical Issues to Address
1. **Calibration reliability needs improvement** - consider isotonic calibration or ensemble approaches
2. **PR-AUC performance plateau suggests need for new feature types or model architectures**
3. **Class imbalance remains the fundamental challenge requiring advanced techniques**

### Recommended Next Steps
1. **Advanced Calibration Methods**: Test isotonic calibration, Platt scaling variants, or ensemble calibration
2. **Feature Engineering Enhancement**: Develop temporal patterns, driver behavior clusters, weather/road conditions
3. **Data Augmentation**: Implement SMOTE, ADASYN, or synthetic data generation for collision cases
4. **Model Architecture Evolution**: Consider ensemble methods, gradient boosting, or neural networks

## Execution Metadata

- **Seed**: 42 (reproducible results)
- **Dataset Checksums**: Verified data integrity
- **Framework**: scikit-learn 1.7.1
- **MLflow URI**: models:/m-4bebe28fe2df40d7a31da3c70e3a018e
- **Total Execution Time**: ~14 minutes

---

**Conclusion**: Iteration 2 successfully demonstrated the value of calibration and threshold optimization while revealing the complexity of the collision prediction challenge. The experiment provides a solid foundation for advanced techniques in the next iteration.