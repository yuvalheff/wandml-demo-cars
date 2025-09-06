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
        Uses identical feature engineering as iteration 1 to isolate calibration impact.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features with engineered features.
        """
        X_transformed = X.copy()
        
        # Create exposure-based features with small epsilon to avoid division by zero
        eps = self.config.eps_value  # Use config value: 1e-8
        
        # Key exposure-based features validated in iteration 1
        if 'miles' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['miles_per_trip'] = X_transformed['miles'] / (X_transformed['count_trip'] + eps)
        
        if 'drive_hours' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['hours_per_trip'] = X_transformed['drive_hours'] / (X_transformed['count_trip'] + eps)
        
        # Create risk behavior ratios - proven effective
        if 'count_brakes' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['brakes_per_mile'] = X_transformed['count_brakes'] / (X_transformed['miles'] + eps)
        
        if 'count_accelarations' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['accel_per_mile'] = X_transformed['count_accelarations'] / (X_transformed['miles'] + eps)
        
        if 'time_speeding_hours' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['speed_per_mile'] = X_transformed['time_speeding_hours'] / (X_transformed['miles'] + eps)
        
        # Replace infinite values with NaN and fill with 0
        feature_cols = ['miles_per_trip', 'hours_per_trip', 'brakes_per_mile', 'accel_per_mile', 'speed_per_mile']
        existing_feature_cols = [col for col in feature_cols if col in X_transformed.columns]
        
        for col in existing_feature_cols:
            X_transformed[col] = X_transformed[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
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
