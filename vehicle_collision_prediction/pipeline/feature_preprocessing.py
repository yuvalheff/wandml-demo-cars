from typing import Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from vehicle_collision_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig):
        self.config: FeaturesConfig = config

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        # No fitting required for feature engineering
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features by creating engineered features.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features with engineered features.
        """
        X_transformed = X.copy()
        
        # Create exposure-based features with small epsilon to avoid division by zero
        eps = 1e-6
        
        if 'miles' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['miles_per_trip'] = X_transformed['miles'] / (X_transformed['count_trip'] + eps)
        
        if 'drive_hours' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['hours_per_trip'] = X_transformed['drive_hours'] / (X_transformed['count_trip'] + eps)
        
        if 'miles' in X_transformed.columns and 'drive_hours' in X_transformed.columns:
            X_transformed['avg_speed'] = X_transformed['miles'] / (X_transformed['drive_hours'] + eps)
        
        # Create risk behavior ratios
        if 'count_brakes' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['brakes_per_mile'] = X_transformed['count_brakes'] / (X_transformed['miles'] + eps)
        
        if 'count_accelarations' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['accel_per_mile'] = X_transformed['count_accelarations'] / (X_transformed['miles'] + eps)
        
        if 'time_speeding_hours' in X_transformed.columns and 'drive_hours' in X_transformed.columns:
            X_transformed['speeding_ratio'] = X_transformed['time_speeding_hours'] / (X_transformed['drive_hours'] + eps)
        
        # Create driving context features
        if 'highway_miles' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['highway_ratio'] = X_transformed['highway_miles'] / (X_transformed['miles'] + eps)
        
        if 'night_drive_hrs' in X_transformed.columns and 'drive_hours' in X_transformed.columns:
            X_transformed['night_ratio'] = X_transformed['night_drive_hrs'] / (X_transformed['drive_hours'] + eps)
        
        if 'time_phoneuse_hours' in X_transformed.columns and 'drive_hours' in X_transformed.columns:
            X_transformed['phone_ratio'] = X_transformed['time_phoneuse_hours'] / (X_transformed['drive_hours'] + eps)
        
        # Create composite scores
        if all(col in X_transformed.columns for col in ['miles', 'drive_hours', 'count_trip']):
            X_transformed['exposure_score'] = (
                X_transformed['miles'] + 
                X_transformed['drive_hours'] + 
                X_transformed['count_trip']
            )
        
        # Create behavior risk score from ratios (if they exist)
        ratio_cols = ['brakes_per_mile', 'accel_per_mile', 'speeding_ratio', 'phone_ratio']
        existing_ratio_cols = [col for col in ratio_cols if col in X_transformed.columns]
        
        if existing_ratio_cols:
            # Handle NaN and infinite values in ratios before computing mean
            ratio_data = X_transformed[existing_ratio_cols].copy()
            ratio_data = ratio_data.replace([np.inf, -np.inf], np.nan)
            X_transformed['behavior_risk_score'] = ratio_data.mean(axis=1, skipna=True)
        
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
