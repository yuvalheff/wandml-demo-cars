# Experiment 4: Enhanced Feature Engineering Vehicle Collision Prediction

## Experiment Overview

**Objective:** Address the critical performance regression identified in Experiment 3 by eliminating aggressive feature selection and implementing comprehensive feature engineering with Random Forest classifier.

**Key Change:** Remove SelectKBest feature selection entirely and use all features with enhanced feature engineering to recover the 7.4% performance loss from Experiment 3.

**Expected Performance:** PR-AUC in 0.240-0.250 range, closing the gap to baseline and moving toward the 0.280 target.

## Background & Rationale

Experiment 3 revealed a critical 21.3% performance gap between cross-validation exploration (0.288) and final results (0.227), along with a 7.4% regression from the previous baseline (0.245). Through systematic exploration experiments, I identified that **feature selection was the primary cause of performance degradation**. Random Forest with all features consistently outperformed any feature selection approach (SelectKBest, RFE, L1 regularization).

## Data Preprocessing Steps

### 1. Missing Value Handling
```
- Apply median imputation to numerical features:
  * count_trip, miles, drive_hours, count_brakes, count_accelarations
  * time_speeding_hours, highway_miles, night_drive_hrs, maximum_speed
- Special handling for time_phoneuse_hours (13% missing): median imputation
- No imputation needed for driver_id, month, collisions (no missing values)
```

### 2. Categorical Encoding
```
- Convert month to month_encoded using LabelEncoder
- Exclude driver_id (high cardinality, use for stratification only)
```

### 3. Target Transformation
```
- Convert multi-class collisions (0,1,2) to binary collision_binary (0,1)
- Transformation: collision_binary = (collisions > 0).astype(int)
- Rationale: Optimize for PR-AUC on binary classification task
```

## Feature Engineering Steps

### 1. Exposure Ratios (Critical for Performance)
```python
# Average trip characteristics - key exposure metrics
miles_per_trip = np.where(count_trip > 0, miles / count_trip, 0)
hours_per_trip = np.where(count_trip > 0, drive_hours / count_trip, 0)
```

### 2. Behavior Intensity Ratios  
```python
# Driving behavior normalized by exposure
brakes_per_mile = np.where(miles > 0, count_brakes / miles, 0)
accel_per_mile = np.where(miles > 0, count_accelarations / miles, 0)
```

### 3. Risk Factor Ratios
```python
# Proportion-based risk indicators
speed_ratio = time_speeding_hours / np.maximum(drive_hours, 0.001)
highway_ratio = np.where(miles > 0, highway_miles / miles, 0)  
night_ratio = night_drive_hrs / np.maximum(drive_hours, 0.001)
```

**Total Features:** 18 (11 original numerical + 1 encoded month + 6 engineered ratios)

**Critical Decision:** **NO FEATURE SELECTION** - Use all 18 features based on exploration evidence that feature selection consistently reduces performance.

## Model Selection Strategy

### Primary Algorithm: RandomForestClassifier

**Hyperparameters:**
```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=None,  # Default performs better than 'balanced'
    max_depth=None,     # No depth limit
    min_samples_split=2 # Default
)
```

**Rationale:** Exploration experiments showed default Random Forest parameters outperformed both tuned Random Forest and Gradient Boosting alternatives. The algorithm naturally handles the mixed feature types and feature interactions present in telematic data.

## Validation Strategy

### Cross-Validation
- **Method:** Stratified 5-fold cross-validation  
- **Stratification:** On collision_binary to preserve 95.2%/4.8% class balance
- **Monitoring Metric:** ROC-AUC during CV (stable metric for optimization)

### Final Evaluation
- **Method:** Single holdout test set evaluation
- **Primary Metric:** PR-AUC (optimized for imbalanced collision prediction)
- **Performance Gap Monitoring:** Track CV vs test performance to avoid Experiment 3's 21.3% gap

## Evaluation Strategy

### Primary Metrics
- **PR-AUC:** Primary metric for imbalanced classification
- **ROC-AUC:** Secondary discrimination metric
- **Precision, Recall, F1:** At optimal threshold

### Threshold Analysis
Generate precision-recall curve and analyze performance across business scenarios:

1. **High Precision Strategy** (30% precision target)
   - Expected recall: 10-20%
   - Use case: High-confidence collision alerts

2. **Balanced F1 Strategy** (maximize F1 score)
   - Expected: ~15% precision, 35% recall  
   - Use case: Balanced intervention approach

3. **High Recall Strategy** (60% recall target)
   - Expected precision: 8-12%
   - Use case: Comprehensive collision prevention

### Feature Importance Analysis
- Analyze RandomForest `feature_importances_` to validate engineered features
- Compare importance rankings to exploration experiment findings
- Identify top predictive features for business insights

### Error Analysis
- Analyze false positive patterns: What drives false alarms?
- Analyze false negative patterns: What collision cases are missed?
- Segment analysis by exposure levels (high/medium/low mileage drivers)

### Comparison Baselines
- **Experiment 3:** PR-AUC 0.227 (must exceed)
- **Previous baseline:** PR-AUC 0.245 (recovery target)
- **Target:** PR-AUC 0.280 (stretch goal)
- **No-skill baseline:** PR-AUC 0.047 (sanity check)

## Expected Outputs

### Model Artifacts
- Trained RandomForest model saved via MLflow
- Feature importance ranking (JSON + visualization)
- Model pipeline with preprocessing steps

### Performance Reports  
- `experiment_summary.json`: Structured metrics and results
- `experiment_summary.md`: Detailed analysis and business implications
- `threshold_analysis.json`: Business scenario performance

### Diagnostic Visualizations
- `precision_recall_curve.html`: PR curve with threshold analysis
- `roc_curve.html`: ROC analysis vs baselines  
- `feature_importance.html`: Feature ranking visualization
- `confusion_matrix.html`: Classification performance matrix
- `error_analysis.html`: FP/FN pattern analysis

## Success Criteria

### Primary Success
- **PR-AUC ≥ 0.245:** Recover to previous iteration baseline
- **Stability:** CV-Test performance gap < 5% (avoid Experiment 3's issues)

### Stretch Success  
- **PR-AUC ≥ 0.280:** Achieve project target

### Minimum Acceptable
- **PR-AUC ≥ 0.227:** No regression from Experiment 3
- **Model Stability:** Reproducible results with proper validation

## Risk Mitigation

1. **Performance Gap Risk:** Monitor CV vs test metrics throughout training
2. **Overfitting Risk:** Use out-of-bag error monitoring in Random Forest  
3. **Feature Engineering Validation:** Verify engineered features don't introduce leakage
4. **Baseline Comparison:** Ensure proper comparison methodology with previous experiments

## Implementation Notes

- Use existing train/test split (no re-splitting required)
- Maintain random_state=42 for reproducibility across all components
- Log all metrics and artifacts to MLflow for experiment tracking
- Generate comprehensive documentation for future iteration planning