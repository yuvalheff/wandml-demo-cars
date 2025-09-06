from typing import Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from vehicle_collision_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config
        # No feature selection in Experiment 4 - use all features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.
        For Experiment 4: Create 6 enhanced engineered features without feature selection.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # No fitting required for Experiment 4 - just feature engineering without selection
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features by creating 6 enhanced engineered features.
        Experiment 4 uses ALL features (18 total: 11 original numerical + 1 encoded + 6 engineered).

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features with all original + engineered features.
        """
        # Create enhanced engineered features as per Experiment 4 plan
        X_engineered = self._create_enhanced_engineered_features(X)
        
        return X_engineered

    def _create_enhanced_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create 6 enhanced engineered features as specified in Experiment 4 plan:
        1) miles_per_trip = miles/count_trip (0 if count_trip == 0)
        2) hours_per_trip = drive_hours/count_trip (0 if count_trip == 0)
        3) brakes_per_mile = count_brakes/miles (0 if miles == 0)
        4) accel_per_mile = count_accelarations/miles (0 if miles == 0)
        5) speed_ratio = time_speeding_hours/max(drive_hours, 0.001)
        6) highway_ratio = highway_miles/miles (0 if miles == 0)
        
        Note: night_ratio excluded for exactly 6 features as per plan.
        """
        X_transformed = X.copy()
        
        # Exposure ratios (handle zero denominators by setting to 0)
        if 'miles' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['miles_per_trip'] = np.where(
                X_transformed['count_trip'] == 0, 
                0, 
                X_transformed['miles'] / X_transformed['count_trip']
            )
        
        if 'drive_hours' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['hours_per_trip'] = np.where(
                X_transformed['count_trip'] == 0, 
                0, 
                X_transformed['drive_hours'] / X_transformed['count_trip']
            )
        
        # Behavior ratios
        if 'count_brakes' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['brakes_per_mile'] = np.where(
                X_transformed['miles'] == 0, 
                0, 
                X_transformed['count_brakes'] / X_transformed['miles']
            )
        
        if 'count_accelarations' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['accel_per_mile'] = np.where(
                X_transformed['miles'] == 0, 
                0, 
                X_transformed['count_accelarations'] / X_transformed['miles']
            )
        
        # Risk ratios (using max(drive_hours, 0.001) to avoid division by zero)
        if 'time_speeding_hours' in X_transformed.columns and 'drive_hours' in X_transformed.columns:
            X_transformed['speed_ratio'] = X_transformed['time_speeding_hours'] / np.maximum(X_transformed['drive_hours'], 0.001)
        
        if 'highway_miles' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['highway_ratio'] = np.where(
                X_transformed['miles'] == 0, 
                0, 
                X_transformed['highway_miles'] / X_transformed['miles']
            )
        
        # Replace any infinite values that might have occurred  
        engineered_cols = ['miles_per_trip', 'hours_per_trip', 'brakes_per_mile', 'accel_per_mile', 
                          'speed_ratio', 'highway_ratio']
        existing_engineered_cols = [col for col in engineered_cols if col in X_transformed.columns]
        
        for col in existing_engineered_cols:
            X_transformed[col] = X_transformed[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input features.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the feature processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'FeatureProcessor':
        """
        Load the feature processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        FeatureProcessor: The loaded feature processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
