# Exploration Experiments Summary - Iteration 3

## Objective
Identify the most effective approach to improve upon the previous iteration's PR-AUC of 0.245 by focusing on advanced feature selection and engineering techniques, based on the insight that class imbalance handling methods showed mixed results.

## Background Context
- **Previous Best Performance**: PR-AUC 0.245, ROC-AUC 0.811
- **Key Challenge**: Severe class imbalance (95.2% no collision, 4.6% single collision, 0.2% double collision)
- **Previous Success**: Exposure-based feature engineering showed significant value (24.7% of model predictive power)

## Experiments Conducted

### 1. Class Imbalance Handling Comparison
**Methodology**: Tested various imbalance handling techniques with RandomForest baseline
**Results** (ROC-AUC/PR-AUC):
- **Baseline (no handling)**: 0.831 / 0.236
- **Class weighting**: 0.829 / 0.231
- **SMOTE**: 0.828 / 0.200
- **ADASYN**: 0.827 / 0.200
- **SMOTETomek**: 0.833 / 0.206

**Key Finding**: Surprisingly, no imbalance handling outperformed the baseline for PR-AUC, suggesting that the natural class distribution contains important signal.

### 2. Advanced Ensemble Methods
**Methodology**: Tested gradient boosting and XGBoost with class balancing
**Results** (PR-AUC):
- **GradientBoosting**: 0.208
- **XGBoost (balanced)**: 0.201

**Key Finding**: Tree-based ensemble methods didn't significantly improve over RandomForest baseline.

### 3. Feature Engineering Impact Analysis
**Methodology**: Added 5 exposure-based features from previous iteration
**Results** (PR-AUC):
- **RF + Feature Engineering**: 0.246 (+0.010 improvement)
- **RF + Feature Eng + Balance**: 0.232 (worse than unbalanced)

**Key Finding**: Feature engineering alone provides consistent improvement over baseline.

### 4. Advanced Imbalance Techniques
**Methodology**: Tested sophisticated sampling methods with feature engineering
**Results** (PR-AUC):
- **BorderlineSMOTE + Features**: 0.210
- **Random Undersampling + Features**: 0.214
- **Feature Selection (SelectKBest)**: **0.293** ⭐
- **RF(300) + Features**: 0.248

**Key Finding**: Feature selection dramatically outperformed all imbalance handling approaches!

### 5. Feature Selection Analysis
**Methodology**: Systematic testing of different numbers of selected features
**Results** (PR-AUC):
- **Top 10 features**: 0.280
- **Top 12 features**: 0.285
- **Top 15 features**: **0.288** (optimal)
- **Top 18 features**: 0.271
- **Top 20 features**: 0.273

**Key Finding**: Optimal performance achieved with top 15 features, showing clear improvement over using all features.

## Top 15 Selected Features (by F-score)
1. **drive_hours** (664.3) - Primary exposure metric
2. **miles** (634.3) - Secondary exposure metric  
3. **count_trip** (505.9) - Trip frequency
4. **count_brakes** (454.3) - Braking behavior
5. **count_accelarations** (454.3) - Acceleration behavior
6. **hours_per_trip** (361.9) - Engineered exposure ratio
7. **miles_per_trip** (346.7) - Engineered exposure ratio
8. **highway_miles** (268.0) - Road type exposure
9. **maximum_speed** (28.9) - Speed behavior
10. **time_speeding_hours** (11.4) - Risk behavior
11. **month_Nov-22** (3.9) - Temporal factor
12. **month_Aug-22** (2.7) - Temporal factor
13. **night_drive_hrs** (1.4) - Risk condition
14. **month_Oct-22** (0.8) - Temporal factor
15. **month_Apr-22** (0.6) - Temporal factor

## Conclusions and Recommendations

### Key Insights
1. **Feature selection is more impactful than class imbalance handling** for this dataset
2. **Natural class distribution contains valuable signal** - don't oversample
3. **Exposure-based features remain highly predictive** (6 of top 10 features)
4. **Temporal patterns have weak but measurable impact** (seasonal effects)

### Recommended Approach for Iteration 3
- **Focus on feature selection and engineering** rather than imbalance handling
- **Use SelectKBest with k=15** for optimal feature subset
- **Apply exposure-based feature engineering** (5 additional features)
- **Use RandomForest with increased trees** (200-300 estimators)
- **Target PR-AUC improvement**: From 0.245 → 0.290+ (18% improvement potential)

### Why This Approach Will Succeed
- **Evidence-based**: 0.288 PR-AUC achieved in cross-validation (vs 0.245 previous)
- **Theoretically sound**: Removes noise features while preserving predictive signal
- **Builds on previous success**: Incorporates proven exposure-based features
- **Addresses root cause**: Focuses on signal quality rather than class distribution