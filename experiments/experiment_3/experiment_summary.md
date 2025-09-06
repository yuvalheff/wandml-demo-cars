# Experiment 3: Feature Selection Enhanced Vehicle Collision Prediction - Analysis Report

## Executive Summary

**Experiment Outcome:** Performance significantly below expectations with PR-AUC of 0.227, falling short of the 0.280+ target by 19.0%.

This experiment implemented systematic feature selection using SelectKBest with F-score ranking, engineering 5 exposure-based features and selecting the top 15 features from a pool of 27. While the methodological approach was sound and feature selection identified meaningful predictors, the final model performance regressed from the previous iteration's baseline of 0.245, representing a 7.4% decrease rather than the targeted 14.3% improvement.

## Experiment Configuration

### Primary Approach
- **Algorithm:** RandomForestClassifier (200 estimators)
- **Feature Selection:** SelectKBest with F-score, k=15
- **Feature Engineering:** 5 exposure-based ratio features
- **Cross-Validation:** 5-fold stratified
- **Class Balancing:** None (natural distribution preserved)

### Key Hypotheses Tested
1. **Feature Selection Enhancement:** Systematic selection of top 15 features would improve signal-to-noise ratio
2. **Exposure-Based Engineering:** Ratio features (miles_per_trip, brakes_per_mile) would provide better normalization
3. **Optimal Feature Subset:** k=15 based on exploration showing PR-AUC of 0.288 in cross-validation

## Performance Results

### Primary Metrics
- **PR-AUC:** 0.227 (Target: ≥0.280) ❌
- **Performance Gap:** -19.0% vs target, -7.4% vs baseline (0.245)
- **ROC-AUC:** ~0.810 (maintained discrimination ability) ✅

### Feature Selection Success
**Top 11 Selected Features:**
1. **drive_hours** (F-score: 664.3) - Primary exposure indicator
2. **miles** (F-score: 634.3) - Secondary exposure measure
3. **count_trip** (F-score: 505.9) - Activity frequency
4. **count_brakes** (F-score: 454.3) - Safety behavior
5. **count_accelarations** (F-score: 454.3) - Driving aggressiveness
6. **hours_per_trip** (F-score: 361.9) - Engineered exposure ratio
7. **miles_per_trip** (F-score: 346.7) - Engineered trip intensity
8. **highway_miles** (F-score: 267.9) - Road type exposure
9. **maximum_speed** (F-score: 28.9) - Speed behavior
10. **time_speeding_hours** (F-score: 11.4) - Risk behavior
11. **night_drive_hrs** (F-score: N/A) - Temporal risk factor

### Model Feature Importance Distribution
- **Top 5 features:** 58.0% of total model importance
- **Exposure-based features:** 40% of selected features (6/15)
- **Engineered features:** Successfully ranked in top 10 (miles_per_trip, hours_per_trip)

## Critical Analysis

### What Worked Well
1. **Feature Selection Methodology:** SelectKBest successfully identified high-signal features with strong F-scores
2. **Feature Engineering Impact:** Exposure-based ratio features (miles_per_trip, hours_per_trip) ranked highly in both F-score and model importance
3. **Model Interpretability:** Clear feature importance hierarchy with exposure measures dominating
4. **Technical Implementation:** Model pipeline executed successfully with proper MLflow integration

### Major Issues Identified

#### 1. Exploration-to-Production Gap (Critical)
- **Cross-validation exploration:** PR-AUC ~0.288
- **Final model result:** PR-AUC 0.227
- **Performance gap:** -21.3% between exploration and final model
- **Root Cause:** Likely overfitting during hyperparameter exploration or data distribution differences

#### 2. Baseline Regression
- **Previous baseline:** 0.245 PR-AUC 
- **Current result:** 0.227 PR-AUC
- **Regression:** -7.4% performance loss
- **Implication:** Feature selection may have removed important signal despite high F-scores

#### 3. Target Miss Magnitude
- **Target performance:** 0.280+ (14.3% improvement)
- **Actual performance:** 0.227 (-7.4% regression)
- **Total gap:** 23.3% below target expectation

## Feature Analysis Deep Dive

### Exposure-Based Features Performance
The engineered exposure features showed strong predictive power:
- **miles_per_trip:** F-score 346.7, Model importance 11.1%
- **hours_per_trip:** F-score 361.9, Model importance 10.9%
- Combined contribution: 22.0% of total model importance

### Feature Selection Validation
The SelectKBest approach successfully prioritized:
- **Primary exposure measures:** drive_hours, miles (top 2 by F-score)
- **Behavioral indicators:** braking/acceleration patterns (high F-scores)
- **Engineered ratios:** both exposure ratios in top 7 selections

However, the strong statistical ranking didn't translate to expected performance gains.

## Business Impact Assessment

At PR-AUC 0.227 performance level:
- **High Precision (≥40%) Strategy:** Expect 10-20% recall, suitable for targeted interventions
- **Balanced F1 Strategy:** ~15% precision, ~35% recall for moderate alerting systems
- **High Recall (≥60%) Strategy:** 8-12% precision with high false alarm rates

## Technical Execution Assessment

### Implementation Strengths
- ✅ Proper data preprocessing pipeline
- ✅ Stratified cross-validation setup
- ✅ Feature engineering executed as planned
- ✅ SelectKBest implementation with appropriate scoring
- ✅ MLflow model packaging successful
- ✅ Comprehensive artifact generation

### Implementation Issues
- ❌ Multiple MLflow pipeline creation errors during execution
- ❌ Feature name mismatch issues requiring debugging iterations
- ❌ Performance significantly below exploration results

## Key Learnings & Insights

### 1. Exploration vs Production Performance Gap
The 21.3% gap between cross-validation exploration (0.288) and final results (0.227) indicates potential methodology issues:
- Possible data leakage in exploration phase
- Overfitting during feature selection optimization
- Train/test distribution differences

### 2. Feature Selection Limitations
Despite strong F-scores, feature selection may have:
- Removed important interaction effects
- Oversimplified complex relationships
- Reduced model's ability to capture minority class patterns

### 3. Baseline Maintenance Challenge
The regression from previous 0.245 baseline suggests:
- Feature selection approach may be too aggressive
- Previous feature set contained important signal not captured by F-scores
- Need for more conservative feature selection strategies

## Future Recommendations

### Immediate Priority (Next Iteration)
1. **Investigation Priority:** Diagnose exploration-to-production performance gap
   - Validate cross-validation methodology
   - Check for data leakage in exploration
   - Compare train/test distributions

### Alternative Approaches to Test
2. **Feature Selection Alternatives:**
   - Recursive Feature Elimination (RFE)
   - L1 regularization for implicit feature selection
   - Boruta algorithm for feature importance ranking

3. **Model Architecture Enhancements:**
   - Ensemble methods combining multiple feature subsets
   - XGBoost with built-in feature selection
   - Neural networks for complex interaction capture

### Methodological Improvements
4. **Cross-Validation Strategy:**
   - Time-aware validation splits if temporal patterns exist
   - Nested cross-validation for more robust estimates
   - Holdout validation set for final model assessment

## Conclusion

Experiment 3 demonstrated a methodologically sound approach to feature selection and engineering, successfully identifying exposure-based features as key predictors. However, the significant performance gap between exploration results and final model output (-21.3%) represents a critical issue requiring investigation. The 7.4% regression from the previous baseline, despite sophisticated feature selection, suggests that simpler approaches may sometimes outperform complex feature engineering strategies.

The experiment provides valuable insights about the challenges of translating cross-validation performance to real-world results and highlights the importance of robust validation methodologies in ML experimentation.

**Status:** Performance target missed - requires methodological review and alternative approach consideration.