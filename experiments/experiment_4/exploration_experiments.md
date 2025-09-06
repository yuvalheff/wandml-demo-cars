# Experiment 4 - Exploration Experiments Summary

## Overview
Based on the critical performance gap identified in Experiment 3 (21.3% difference between CV exploration results and final performance), I conducted systematic exploration experiments to understand the root causes and identify the optimal approach for Experiment 4.

## Key Findings from Previous Experiment

**Experiment 3 Issues:**
- PR-AUC: 0.227 (Target: 0.280) - 19% below target
- Performance regression: -7.4% from baseline (0.245)
- Critical exploration-production gap: 21.3% (CV: 0.288, Final: 0.227)

## Exploration Experiments Conducted

### 1. Nested Cross-Validation Analysis
**Objective:** Understand the source of the CV-Test performance gap

**Results:**
- Regular 5-fold CV ROC-AUC: 0.833 ± 0.018
- Test ROC-AUC: 0.801
- CV-Test gap: 0.033 (reasonable, not the 21.3% gap from Exp 3)
- Test PR-AUC: 0.213

**Insight:** The gap in my replication is much smaller than reported in Exp 3, suggesting potential methodological differences or data leakage in the original experiment.

### 2. Feature Selection Method Comparison
**Objective:** Compare SelectKBest, RFE, and L1 regularization approaches

**Results:**
- SelectKBest (k=11): PR-AUC = 0.213
- SelectKBest (k=15): PR-AUC = 0.213  
- RFE (11 features): PR-AUC = 0.213
- L1 Logistic (initially appeared best at 0.524, but this was baseline due to all-zero coefficients)

**Critical Discovery:** Initial L1 results were misleading - the high PR-AUC of 0.524 was actually the no-skill baseline due to severe class imbalance (4.7% positive rate).

### 3. Corrected Baseline Analysis
**Objective:** Properly account for class imbalance baseline

**Corrected Results:**
- No-skill baseline PR-AUC: 0.047 (true baseline)
- RF with SelectKBest: PR-AUC = 0.223 (+0.176 vs baseline)
- L1 Logistic: PR-AUC = 0.183 (+0.136 vs baseline)
- RF with RFE: PR-AUC = 0.237 (+0.190 vs baseline)
- **RF with ALL features: PR-AUC = 0.241 (+0.194 vs baseline)** ← Best performing

**Key Insight:** Feature selection actually **hurts** performance. Using all features performs better than any feature selection method.

### 4. Feature Engineering Impact
**Objective:** Test additional engineered features

**Added Features:**
- `miles_per_trip`, `hours_per_trip` (exposure ratios)
- `brakes_per_mile`, `accel_per_mile` (behavior ratios) 
- `highway_ratio`, `night_ratio`, `speed_ratio` (risk ratios)

**Result:** Engineered features dominated top importance in Random Forest, confirming their value.

### 5. Hyperparameter Tuning Analysis
**Objective:** Test whether tuned models can reach the 0.280 target

**Results:**
- Tuned Random Forest: PR-AUC = 0.208 (class_weight='balanced' actually hurt)
- Tuned Gradient Boosting: PR-AUC = 0.192

**Insight:** Default Random Forest with all features outperformed tuned versions, suggesting the model is well-suited to the data characteristics.

## Key Insights for Experiment 4

1. **Avoid Feature Selection:** All feature selection methods (SelectKBest, RFE, L1) reduce performance
2. **Use All Features:** Random Forest with all 18+ features (including engineered) performs best
3. **Feature Engineering Critical:** Exposure and behavior ratios appear in top importance
4. **Stable Performance:** Default hyperparameters often outperform tuned versions
5. **Target Still Challenging:** Best achieved PR-AUC of 0.241 is still -0.039 from 0.280 target

## Recommended Approach for Experiment 4

**Single Change Focus:** Move from aggressive feature selection to a "no feature selection" approach with enhanced feature engineering, directly addressing Experiment 3's performance regression.

**Expected Performance:** PR-AUC ~0.240-0.250 range based on exploration results, closing the gap to baseline and potentially reaching closer to the 0.280 target.