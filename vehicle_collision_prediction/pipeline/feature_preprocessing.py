from typing import Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif

from vehicle_collision_prediction.config import FeaturesConfig


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeaturesConfig, n_features_select: Optional[int] = None):
        self.config: FeaturesConfig = config
        self.n_features_select = n_features_select or 15  # Default to 15 as per experiment plan
        self.feature_selector = None
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """
        Fit the feature processor to the data.
        Create exposure-based features then apply SelectKBest feature selection.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (required for SelectKBest).

        Returns:
        FeatureProcessor: The fitted processor.
        """
        if y is None:
            raise ValueError("Target variable y is required for fitting feature selector")
        
        # First create engineered features
        X_engineered = self._create_engineered_features(X)
        
        # Apply SelectKBest feature selection with f_classif scoring
        self.feature_selector = SelectKBest(score_func=f_classif, k=self.n_features_select)
        
        # Prepare features for selection (exclude target columns)
        feature_cols = [col for col in X_engineered.columns 
                       if col not in ['collisions', 'collisions_binary']]
        
        X_features = X_engineered[feature_cols]
        
        # Fit the feature selector
        self.feature_selector.fit(X_features, y)
        
        # Store selected feature names
        selected_indices = self.feature_selector.get_support()
        self.selected_features = [feature_cols[i] for i, selected in enumerate(selected_indices) if selected]
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features by creating engineered features and applying feature selection.
        Create 5 exposure-based features then select top k=15 features using SelectKBest.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features with engineered and selected features.
        """
        # First create engineered features
        X_engineered = self._create_engineered_features(X)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None and self.selected_features is not None:
            # Keep target columns if they exist (for consistency during pipeline)
            target_cols = [col for col in X_engineered.columns 
                          if col in ['collisions', 'collisions_binary']]
            
            # Select only the chosen features plus target columns
            X_selected = X_engineered[self.selected_features + target_cols]
            return X_selected
        
        return X_engineered

    def _create_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create 5 exposure-based features as specified in experiment plan.
        """
        X_transformed = X.copy()
        
        # Create exposure-based features with small epsilon to avoid division by zero
        eps = self.config.eps_value  # Use config value: 1e-6 (updated from experiment plan)
        
        # 5 exposure-based features as per experiment plan
        if 'miles' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['miles_per_trip'] = X_transformed['miles'] / (X_transformed['count_trip'] + eps)
        
        if 'drive_hours' in X_transformed.columns and 'count_trip' in X_transformed.columns:
            X_transformed['hours_per_trip'] = X_transformed['drive_hours'] / (X_transformed['count_trip'] + eps)
        
        if 'count_brakes' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['brakes_per_mile'] = X_transformed['count_brakes'] / (X_transformed['miles'] + eps)
        
        if 'count_accelarations' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['accel_per_mile'] = X_transformed['count_accelarations'] / (X_transformed['miles'] + eps)
        
        if 'maximum_speed' in X_transformed.columns and 'miles' in X_transformed.columns:
            X_transformed['speed_per_mile'] = X_transformed['maximum_speed'] / (X_transformed['miles'] + eps)
        
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
