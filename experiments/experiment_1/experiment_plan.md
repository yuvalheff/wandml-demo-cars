# Vehicle Collision Prediction - Experiment 1 Plan

## Experiment Overview
**Experiment Name**: Vehicle Collision Prediction - Binary Classification with Exposure-Based Features  
**Task Type**: Binary Classification  
**Target Variable**: `collisions_binary` (derived from `collisions`)  
**Primary Metric**: PR-AUC  
**Dataset**: 7,667 training records, 1,917 test records

## Experiment Rationale
Based on exploration experiments, this approach focuses on:
- **Binary classification** (collision vs no collision) performs better than multi-class
- **Feature engineering** with exposure-based ratios significantly improves performance  
- **Random Forest** with balanced class weights outperforms other algorithms
- **PR-AUC** is the appropriate metric for this severely imbalanced dataset (95.2% no collisions)

## Data Preprocessing Steps

### 1. Target Variable Transformation
Transform the multi-class target variable `collisions` (values: 0, 1, 2) into binary:
```python
collisions_binary = (collisions > 0).astype(int)
# 0 -> 0 (no collision)
# 1,2 -> 1 (collision occurred)
```

### 2. Missing Value Imputation
Apply median imputation to numerical features:
- **Columns**: `count_trip`, `miles`, `drive_hours`, `count_brakes`, `count_accelarations`, `time_speeding_hours`, `time_phoneuse_hours`, `highway_miles`, `night_drive_hrs`, `maximum_speed`
- **Strategy**: Median imputation (robust to outliers)
- **Implementation**: `SimpleImputer(strategy='median')`

### 3. Categorical Encoding  
Encode the `month` categorical variable:
- **Method**: Label encoding
- **Implementation**: `LabelEncoder().fit_transform(month)`

### 4. Infinite Values Handling
Handle infinite values created during feature engineering:
```python
data = data.replace([np.inf, -np.inf], np.nan)
# Then apply imputation
```

### 5. Feature Scaling
Standardize all numerical features:
- **Method**: Standard scaling (zero mean, unit variance)
- **Implementation**: `StandardScaler().fit_transform()`

## Feature Engineering Steps

### 1. Exposure-Based Features
Create features capturing driving exposure patterns (key insight from EDA):

- **`miles_per_trip`** = `miles / (count_trip + 1e-6)`
  - Rationale: Average trip length indicates driving patterns (commuting vs local)
  
- **`hours_per_trip`** = `drive_hours / (count_trip + 1e-6)`  
  - Rationale: Average trip duration indicates traffic conditions and driving style
  
- **`avg_speed`** = `miles / (drive_hours + 1e-6)`
  - Rationale: Overall average speed indicates driving environment (city vs highway)

### 2. Risk Behavior Ratios
Normalize risky behaviors by exposure for meaningful comparisons:

- **`brakes_per_mile`** = `count_brakes / (miles + 1e-6)`
  - Rationale: Hard braking rate per mile (more meaningful than absolute counts)
  
- **`accel_per_mile`** = `count_accelarations / (miles + 1e-6)`
  - Rationale: Hard acceleration rate per mile (normalized risk behavior)
  
- **`speeding_ratio`** = `time_speeding_hours / (drive_hours + 1e-6)`  
  - Rationale: Proportion of time spent speeding (percentage-based risk metric)

### 3. Driving Context Features  
Capture driving environment and conditions:

- **`highway_ratio`** = `highway_miles / (miles + 1e-6)`
  - Rationale: Highway vs surface street driving has different risk profiles
  
- **`night_ratio`** = `night_drive_hrs / (drive_hours + 1e-6)`
  - Rationale: Night driving proportion (higher risk conditions)
  
- **`phone_ratio`** = `time_phoneuse_hours / (drive_hours + 1e-6)`
  - Rationale: Distracted driving measure (proportion of time using phone)

### 4. Composite Risk Scores
Aggregate multiple indicators into composite features:

- **`exposure_score`** = `miles + drive_hours + count_trip`
  - Rationale: Total driving exposure combining distance, time, and frequency
  
- **`behavior_risk_score`** = `(brakes_per_mile + accel_per_mile + speeding_ratio + phone_ratio) / 4`
  - Rationale: Average normalized risk behavior score

## Model Selection and Training

### Primary Model: Random Forest
**Algorithm**: `RandomForestClassifier`
**Hyperparameters**:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Address class imbalance
    random_state=42,
    n_jobs=-1
)
```
**Rationale**: Best performance in exploration experiments, handles imbalanced data well, provides interpretable feature importance.

### Alternative Model: XGBoost
**Algorithm**: `XGBClassifier`
**Hyperparameters**:
```python
XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=20,  # Address class imbalance
    random_state=42
)
```
**Rationale**: Comparison model, potential ensemble candidate.

### Hyperparameter Tuning
- **Method**: 5-fold stratified cross-validation with `RandomizedSearchCV`
- **Scoring**: `average_precision` (PR-AUC)
- **CV Strategy**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

## Feature Selection Strategy
1. **Feature Importance Analysis**: Use Random Forest importance scores
2. **Cross-validation**: Validate feature selection with CV performance
3. **Target Features**: Select top 15 most important features
4. **Priority Features**: 
   - Core: `miles`, `drive_hours`, `exposure_score`
   - Behavior: `count_brakes`, `count_accelarations`, `brakes_per_mile`, `accel_per_mile` 
   - Context: `highway_miles`, `highway_ratio`, `miles_per_trip`

## Evaluation Strategy

### Primary Evaluation
- **Metric**: PR-AUC (Precision-Recall Area Under Curve)
- **Rationale**: Most appropriate for imbalanced datasets, focuses on positive class performance

### Secondary Metrics
- Precision, Recall, F1-Score at optimal threshold
- ROC-AUC for comparison
- Precision at 10% recall (business-relevant metric)

### Comprehensive Analysis

#### 1. Hold-out Test Evaluation
- Evaluate final model on held-out test set
- Generate all metrics and visualizations

#### 2. Cross-Validation Analysis  
- 5-fold stratified CV on training set
- Assess model stability and variance

#### 3. Threshold Optimization
- Find optimal classification threshold using validation set
- Maximize F1-score for balanced precision/recall

#### 4. Feature Importance Analysis
- Generate and interpret feature importance scores
- Validate against domain knowledge from EDA

#### 5. Error Analysis by Driver Segments
Segment analysis by exposure levels:
- **Low exposure**: < 200 miles/month
- **Medium exposure**: 200-800 miles/month  
- **High exposure**: > 800 miles/month

Analyze model performance within each segment to identify bias.

#### 6. Calibration Analysis
- Generate reliability diagrams
- Calculate Brier score  
- Assess if predicted probabilities are well-calibrated

#### 7. Temporal Analysis
- Analyze performance across months (Jan-22 to Dec-22)
- Identify seasonal patterns or temporal drift

## Expected Outputs

### Model Artifacts
- `trained_random_forest_model.pkl`
- `trained_xgboost_model.pkl`
- `feature_scaler.pkl` 
- `label_encoder.pkl`
- `feature_importance_scores.json`

### Evaluation Results  
- `test_set_predictions.csv` (predictions with probabilities)
- `evaluation_metrics.json` (all metrics)
- `confusion_matrix.png`
- `pr_curve.png` 
- `roc_curve.png`
- `feature_importance_plot.png`
- `calibration_plot.png`

### Analysis Reports
- `model_performance_report.md` (comprehensive performance analysis)
- `feature_analysis_report.md` (feature importance and selection analysis)
- `error_analysis_report.md` (detailed error analysis by segments)
- `final_experiment_summary.json` (structured summary of all results)

## Success Criteria
- **Minimum PR-AUC**: 0.25 (baseline improvement over exploration)
- **Target PR-AUC**: 0.30 (ambitious goal)  
- **Precision at 10% recall**: ≥ 0.15 (business relevance)
- **Model stability**: CV standard deviation < 0.05

## Implementation Notes
1. Use `driver_id` for grouping/stratification but not as a direct feature
2. Handle edge cases in feature engineering (division by zero with small epsilon)
3. Monitor for data leakage - ensure temporal consistency if using any temporal features
4. Document all preprocessing steps for reproducibility
5. Save all intermediate outputs for debugging and analysis