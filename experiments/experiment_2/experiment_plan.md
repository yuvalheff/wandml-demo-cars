# Experiment 2: Calibrated Vehicle Collision Prediction with Threshold Optimization

## Overview
Building on the successful exposure-based RandomForest model from iteration 1, this experiment focuses on improving precision-recall performance through **probability calibration** and **threshold optimization**. The goal is to address the poor probability calibration identified in iteration 1 while providing business stakeholders with actionable threshold recommendations.

## Experiment Details

### Task Configuration
- **Task Type**: Binary Classification  
- **Target Column**: collisions (converted to binary: collision vs no collision)
- **Primary Metric**: PR-AUC (target: ≥ 0.25)
- **Secondary Metrics**: ROC-AUC, Precision, Recall, F1-Score

### Data Preprocessing

#### 1. Missing Value Imputation
- **Columns**: count_trip, miles, drive_hours, count_brakes, count_accelarations, time_speeding_hours, time_phoneuse_hours, highway_miles, night_drive_hrs, maximum_speed
- **Method**: Median imputation
- **Rationale**: Preserves distribution shape and is robust to outliers in telematic data

#### 2. Target Transformation
- **Transformation**: Convert multi-class (0, 1, 2) to binary (0, 1) where collisions > 0 = 1
- **Rationale**: Binary classification improves model performance on imbalanced data and aligns with business use case

#### 3. Categorical Encoding
- **Column**: month → month_encoded
- **Method**: LabelEncoder
- **Rationale**: Numerical encoding enables tree-based models to capture temporal patterns

### Feature Engineering

#### Exposure-Based Features (Validated from Iteration 1)
1. **miles_per_trip** = miles / (count_trip + 1e-8)
   - *Rationale*: Trip distance patterns indicate driving behavior and risk exposure

2. **hours_per_trip** = drive_hours / (count_trip + 1e-8)  
   - *Rationale*: Trip duration patterns indicate driving patterns and exposure

3. **brakes_per_mile** = count_brakes / (miles + 1e-8)
   - *Rationale*: Braking intensity per distance indicates aggressive driving behavior

4. **accel_per_mile** = count_accelarations / (miles + 1e-8)
   - *Rationale*: Acceleration intensity per distance indicates aggressive driving behavior

5. **speed_per_mile** = time_speeding_hours / (miles + 1e-8)
   - *Rationale*: Speeding intensity per distance indicates risk-taking behavior

### Model Selection & Training

#### 1. Base Model Training
- **Algorithm**: RandomForestClassifier
- **Hyperparameters**:
  - n_estimators: 100
  - class_weight: 'balanced' 
  - random_state: 42
- **Rationale**: Proven effective in iteration 1 with strong discrimination (ROC-AUC=0.805)

#### 2. Probability Calibration ⭐ **NEW IN ITERATION 2**
- **Method**: Sigmoid calibration via CalibratedClassifierCV
- **Cross-validation**: 5 folds
- **Rationale**: Exploration experiments showed 8.5% PR-AUC improvement and better calibration quality than isotonic method

#### 3. Threshold Optimization ⭐ **NEW IN ITERATION 2**
Three threshold strategies for different business scenarios:

1. **Optimal F1 Threshold**: Maximize F1-score for balanced performance
2. **High Precision Threshold**: Achieve precision ≥ 40% to minimize false alarms  
3. **High Recall Threshold**: Achieve recall ≥ 60% for safety-critical applications

### Evaluation Strategy

#### Threshold-Independent Metrics
- **PR-AUC**: Primary metric, compare to iteration 1 baseline (0.238)
- **ROC-AUC**: Model discrimination, maintain ≥ 0.80

#### Calibration Assessment ⭐ **NEW IN ITERATION 2**
- **Reliability Diagram**: Compare predicted probabilities to actual collision rates
- **Method**: Bin predictions into 10 probability ranges, analyze prediction accuracy
- **Target**: Predicted probabilities within ±10% of actual rates

#### Threshold-Dependent Analysis ⭐ **NEW IN ITERATION 2** 
- **Performance Analysis**: Precision, Recall, F1 for each threshold strategy
- **Confusion Matrix**: Detailed error analysis for each threshold
- **Business Impact**: False positive/negative rates and their implications

#### Model Interpretation
- **Feature Importance**: Validate exposure-based features remain most predictive
- **Comparison**: Calibrated vs uncalibrated model performance

### Expected Outputs

1. **calibrated_collision_model.pkl**: Trained and calibrated model ready for deployment
2. **threshold_optimization_results.json**: Optimal thresholds with performance metrics
3. **model_evaluation_report.html**: Comprehensive evaluation with visualizations
4. **feature_importance_analysis.json**: Feature rankings and exposure hypothesis validation
5. **mlflow_experiment_tracking.json**: Complete MLflow tracking artifacts

### Success Criteria

1. **PR-AUC Improvement**: ≥ 0.25 (5% improvement over iteration 1)
2. **Calibration Quality**: Predicted probabilities within ±10% of actual rates
3. **ROC-AUC Maintenance**: ≥ 0.80 (preserve discrimination ability)  
4. **Threshold Flexibility**: 3+ options with precision 20-50% and recall 30-70%

### Key Changes from Iteration 1

| Change | Rationale |
|--------|-----------|
| **Added Probability Calibration** | Address poor calibration, improve decision-making reliability |
| **Implemented Threshold Optimization** | Provide actionable recommendations for different deployment scenarios |
| **Enhanced Calibration Evaluation** | Measure calibration quality and threshold effectiveness |

### Implementation Notes

- Use identical feature engineering as iteration 1 to isolate calibration impact
- Apply calibration with 5-fold cross-validation to prevent overfitting  
- Save both calibrated and uncalibrated models for comparison
- Document threshold selection rationale for business stakeholders
- Include confidence intervals for all reported metrics

---

## Expected Results

Based on exploration experiments:
- **PR-AUC**: Expected improvement to ~0.258 (8.5% gain)
- **Calibration**: Significantly improved probability reliability
- **Business Value**: Clear threshold recommendations for different risk tolerance levels
- **Decision Making**: Reliable probabilities enable evidence-based deployment strategies