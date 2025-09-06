# Vehicle Collision Prediction - EDA Report

## Dataset Overview
- **Dataset Name**: Telematic Vehicle Collision Prediction
- **Records**: 7,667 driver-month observations
- **Features**: 13 columns (10 numerical, 2 categorical, 1 target)
- **Task Type**: Multi-class Classification
- **Target Variable**: collisions (0, 1, 2 collision counts)

## Target Variable Analysis
### Collision Distribution
- **Class 0** (No collisions): 7,301 records (95.23%)
- **Class 1** (Single collision): 351 records (4.58%)
- **Class 2** (Double collisions): 15 records (0.20%)

**Key Insight**: Severe class imbalance with 95.2% no-collision cases, requiring specialized modeling techniques.

## Data Quality Assessment
### Missing Values
- **Total Missing**: 1,520 values (1.53% of all data)
- **Most Features**: 55 missing values each (0.72%)
- **Phone Use Feature**: 1,025 missing values (13.37%)

**Pattern**: Systematic missing data suggests collection methodology issues rather than random missingness.

## Feature Analysis

### Numerical Features (10 features)
**High-Variance Features** (by variance ranking):
1. **miles**: 735,891 variance - Primary exposure metric
2. **highway_miles**: 228,137 variance - Highway driving subset  
3. **count_accelerations**: 217,186 variance - Aggressive acceleration events
4. **count_brakes**: 217,162 variance - Hard braking events
5. **drive_hours**: 36,025 variance - Total driving time

### Categorical Features (2 features)
- **driver_id**: 7,664 unique values (high cardinality)
- **month**: 12 unique values (Jan-22 to Dec-22, balanced distribution)

## Correlation Analysis
### Features Most Correlated with Collisions:
1. **miles**: 0.297 (moderate positive)
2. **drive_hours**: 0.295 (moderate positive)  
3. **count_trip**: 0.254 (moderate positive)
4. **count_brakes**: 0.251 (moderate positive)
5. **count_accelerations**: 0.251 (moderate positive)

**Key Finding**: Exposure metrics (miles, hours) outperform behavior metrics (speed, braking) as collision predictors.

## Outlier Analysis
### Features with Highest Outlier Rates:
1. **time_speeding_hours**: 17.0% outliers (data quality concern)
2. **count_accelerations**: 12.4% outliers
3. **count_brakes**: 12.2% outliers
4. **highway_miles**: 12.2% outliers

**Interpretation**: High outlier rates may represent legitimate extreme driving patterns rather than errors.

## Target-Feature Relationships
### Miles Driven by Collision Class:
- **No Collisions (0)**: Mean = 425 miles
- **Single Collision (1)**: Mean = 1,294 miles (+205% increase)
- **Double Collisions (2)**: Mean = 3,445 miles (+166% from single)

**Clear Pattern**: Collision risk escalates dramatically with increased driving exposure.

## Key Insights & Findings

### 1. Exposure-Based Risk Model
The data strongly supports an **exposure hypothesis** where collision risk increases primarily with driving frequency/distance rather than poor driving behaviors.

### 2. Class Imbalance Challenge  
With 95.2% no-collision cases, this is a severely imbalanced classification problem requiring:
- Stratified sampling strategies
- Specialized metrics (PR-AUC over accuracy)
- Resampling techniques (SMOTE, ADASYN)

### 3. Feature Engineering Opportunities
- **Temporal patterns**: Month feature suggests seasonal effects
- **Rate features**: Create rates per mile (accidents/mile, brakes/mile)
- **Risk scores**: Combine exposure + behavior metrics

### 4. Data Quality Considerations
- Phone use feature missing 13.4% of data
- Time speeding has measurement reliability issues (17% outliers)
- Systematic missing pattern suggests collection methodology review needed

## Recommendations for ML Pipeline

### 1. Data Preprocessing
- **Imputation**: Handle phone use missing values with median/mode
- **Scaling**: Apply StandardScaler to numerical features (highly skewed)
- **Encoding**: Target encoding for high-cardinality driver_id

### 2. Class Imbalance Handling
- **Stratified splits** to preserve class distribution  
- **SMOTE** or ADASYN for oversampling minority classes
- **Class weights** in model training

### 3. Model Selection
- **Ensemble methods**: Random Forest, XGBoost (handle imbalance well)
- **Threshold tuning**: Optimize for PR-AUC rather than accuracy
- **Calibration**: Post-processing for probability calibration

### 4. Evaluation Strategy
- **Primary metric**: PR-AUC (ideal for imbalanced data)
- **Secondary metrics**: F1-score, Recall for each class
- **Stratified cross-validation**: Maintain class distribution

### 5. Feature Engineering
- Create exposure-normalized features (events per mile)
- Seasonal/temporal features from month
- Risk interaction terms (speed × miles, etc.)

## Business Context Alignment
This analysis confirms the **vehicle safety** focus - the model should predict collision risk to enable:
- **Proactive interventions** for high-exposure drivers
- **Risk-based insurance pricing** using telematics data
- **Safety coaching** programs targeting high-risk patterns

The findings suggest that simply driving more increases collision probability, supporting usage-based insurance models and targeted safety interventions for high-mileage drivers.