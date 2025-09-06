# Experiment 3: Feature Selection Enhanced Vehicle Collision Prediction

## Overview
This experiment focuses on **advanced feature selection and engineering** to improve PR-AUC performance beyond the previous baseline of 0.245. Based on extensive exploration, this approach targets a **18% improvement** (PR-AUC ≥ 0.290) through systematic feature selection rather than class imbalance handling.

## Key Innovation: Evidence-Based Feature Selection
Unlike previous iterations that focused on calibration and threshold optimization, this experiment **prioritizes signal quality over class balancing**, based on evidence that natural class distribution preserves important predictive patterns.

---

## Data Preprocessing Steps

### 1. Missing Value Handling
- **Method**: Mean imputation using `df.fillna(df.mean())`
- **Columns**: All numerical features (`count_trip`, `miles`, `drive_hours`, `count_brakes`, `count_accelarations`, `time_speeding_hours`, `time_phoneuse_hours`, `highway_miles`, `night_drive_hrs`, `maximum_speed`)
- **Rationale**: Simple, robust approach that preserves feature distributions

### 2. Categorical Encoding  
- **Method**: One-hot encoding using `pd.get_dummies()`
- **Column**: `month` → 12 binary features (month_Jan-22, month_Feb-22, etc.)
- **Rationale**: Captures seasonal patterns identified in EDA

### 3. Target Transformation
- **Method**: Convert multi-class to binary classification
- **Formula**: `y_binary = (y > 0).astype(int)`
- **Result**: 0 = no collision, 1 = collision occurred (classes 1 and 2 combined)

### 4. Column Removal
- **Remove**: `driver_id` (high-cardinality identifier with no predictive value)

---

## Feature Engineering Steps

### 1. Exposure-Based Feature Creation (5 Features)
Based on proven success from Iteration 2, create normalized behavioral metrics:

```python
# Exposure-based features with collision-safe denominators
X['miles_per_trip'] = X['miles'] / (X['count_trip'] + 1e-6)
X['hours_per_trip'] = X['drive_hours'] / (X['count_trip'] + 1e-6)  
X['brakes_per_mile'] = X['count_brakes'] / (X['miles'] + 1e-6)
X['accel_per_mile'] = X['count_accelarations'] / (X['miles'] + 1e-6)
X['speed_per_mile'] = X['maximum_speed'] / (X['miles'] + 1e-6)
```

**Rationale**: These features normalize driving behaviors by exposure, capturing risk intensity rather than absolute counts.

### 2. Systematic Feature Selection
- **Method**: `SelectKBest` with `f_classif` scoring function  
- **Parameters**: `k=15` (optimal number determined through systematic testing)
- **Implementation**: Fit on training data only to prevent data leakage

**Expected Top 15 Features** (based on F-score ranking):
1. `drive_hours` (664.3) - Primary exposure metric
2. `miles` (634.3) - Secondary exposure metric
3. `count_trip` (505.9) - Trip frequency indicator
4. `count_brakes` (454.3) - Braking behavior
5. `count_accelarations` (454.3) - Acceleration behavior  
6. `hours_per_trip` (361.9) - **Engineered** exposure ratio
7. `miles_per_trip` (346.7) - **Engineered** exposure ratio
8. `highway_miles` (268.0) - Road type exposure
9. `maximum_speed` (28.9) - Speed behavior
10. `time_speeding_hours` (11.4) - Risk behavior
11. `month_Nov-22` (3.9) - Seasonal factor
12. `month_Aug-22` (2.7) - Seasonal factor  
13. `night_drive_hrs` (1.4) - Risk condition
14. `month_Oct-22` (0.8) - Seasonal factor
15. `month_Apr-22` (0.6) - Seasonal factor

---

## Model Selection Steps

### Primary Algorithm: RandomForestClassifier
**Configuration**:
```python
RandomForestClassifier(
    n_estimators=200,        # Increased from 100 for better performance
    random_state=42,         # Reproducibility
    max_depth=None,          # Allow trees to grow fully
    min_samples_split=5,     # Prevent overfitting
    min_samples_leaf=2,      # Prevent overfitting
    n_jobs=-1               # Parallel processing
)
```

### Class Imbalance Strategy: **No Artificial Balancing**
- **Rationale**: Exploration experiments showed natural class distribution (95.2% no collision) contains valuable signal that artificial balancing methods destroy
- **Evidence**: Baseline (no balancing) achieved PR-AUC 0.236 vs SMOTE 0.200, class weighting 0.231

---

## Evaluation Strategy

### Primary Metric: **PR-AUC** 
- **Target**: ≥ 0.290 (18% improvement from 0.245 baseline)
- **Justification**: Ideal for imbalanced classification, focuses on positive class performance

### Cross-Validation Protocol
- **Method**: 5-fold Stratified Cross-Validation
- **Settings**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Purpose**: Robust performance estimation maintaining class distribution

### Threshold Optimization Strategies
1. **Optimal F1**: Maximize F1-score for balanced precision-recall
2. **High Precision**: Target precision ≥ 0.40 to minimize false alarms  
3. **High Recall**: Target recall ≥ 0.60 for safety-critical detection

### Comprehensive Diagnostic Analysis

#### 1. Feature Importance Analysis
- **Method**: RandomForest `feature_importances_` attribute
- **Purpose**: Validate that selected features contribute meaningfully
- **Output**: Ranked importance scores and visualization

#### 2. Precision-Recall Analysis  
- **Method**: `precision_recall_curve()` with threshold annotations
- **Purpose**: Understand trade-offs and identify optimal operating points
- **Output**: Interactive PR curve with business threshold markers

#### 3. ROC Analysis
- **Method**: `roc_curve()` and `roc_auc_score()`
- **Purpose**: Assess discrimination ability between collision/non-collision
- **Target**: Maintain ROC-AUC ≥ 0.810

#### 4. Confusion Matrix Analysis
- **Method**: Confusion matrices at optimal thresholds
- **Purpose**: Quantify false positive/negative rates for business impact
- **Output**: Heat maps for each threshold strategy

#### 5. Prediction Distribution Analysis
- **Method**: Histogram of predicted probabilities by true class
- **Purpose**: Understand model confidence patterns and calibration needs
- **Output**: Distribution plots with class separation visualization

---

## Expected Outputs

### Model Artifacts
- `trained_model.pkl`: Complete RandomForest model with feature selection pipeline
- `feature_selector.pkl`: Fitted SelectKBest transformer  
- `feature_importance.json`: Importance scores and rankings

### Evaluation Reports
- `performance_metrics.json`: All metrics (PR-AUC, ROC-AUC, precision, recall, F1)
- `threshold_analysis.json`: Optimal thresholds for three business strategies
- `cross_validation_results.json`: Detailed CV scores with confidence intervals

### Interactive Visualizations  
- `precision_recall_curve.html`: PR curve with threshold annotations
- `roc_curve.html`: ROC curve with AUC score
- `feature_importance_plot.html`: Feature ranking visualization
- `confusion_matrix_heatmap.html`: Confusion matrices for different thresholds
- `prediction_distribution.html`: Probability distribution histograms

### Analysis Documents
- `experiment_summary.md`: Comprehensive results and business implications
- `model_performance_analysis.md`: Detailed performance breakdown
- `feature_analysis.md`: Selected features analysis and engineering impact

---

## Implementation Roadmap

### Phase 1: Data Pipeline (Steps 1-2)
1. Load training data from `data/train_set.csv`
2. Apply preprocessing pipeline (missing values, encoding, target transformation)

### Phase 2: Feature Engineering (Steps 3-4)  
3. Create 5 exposure-based engineered features using specified formulas
4. Apply SelectKBest with k=15 and f_classif scoring (**fit only on training data**)

### Phase 3: Model Development (Steps 5-6)
5. Train RandomForest with 200 estimators on selected feature subset
6. Perform 5-fold stratified cross-validation for performance estimation

### Phase 4: Business Analysis (Steps 7-8)
7. Optimize thresholds for three business strategies and generate evaluations  
8. Create comprehensive visualizations and package deployment artifacts

---

## Success Criteria & Business Impact

### Primary Success: PR-AUC ≥ 0.280
- **Improvement**: 14.3% minimum over 0.245 baseline
- **Stretch Goal**: 0.290 (18% improvement based on exploration results)

### Secondary Success Metrics
- **ROC-AUC ≥ 0.810**: Maintain discrimination ability
- **Actionable Thresholds**: Three distinct strategies for different business contexts
- **Feature Validation**: Comprehensive importance analysis supporting selection approach

### Business Value
- **Improved Collision Detection**: Higher precision and recall for safety applications
- **Operational Efficiency**: Optimized thresholds reduce false alarm costs
- **Model Interpretability**: Clear feature importance enables business understanding
- **Deployment Readiness**: Complete pipeline with robust evaluation metrics

---

## Critical Implementation Notes

⚠️ **Data Leakage Prevention**
- Feature selection MUST be fit only on training data
- Apply same preprocessing pipeline to test data during final evaluation

⚠️ **Cross-Validation Integrity**  
- Use stratified splits to maintain class distribution in each fold
- Ensure consistent random seeds for reproducibility

⚠️ **Business Communication**
- Generate interactive visualizations for stakeholder presentations
- Provide clear threshold recommendations with precision-recall trade-offs