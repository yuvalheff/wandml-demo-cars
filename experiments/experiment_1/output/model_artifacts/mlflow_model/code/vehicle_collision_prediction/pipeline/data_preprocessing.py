from typing import Optional
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from vehicle_collision_prediction.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        self.numerical_imputer = None
        self.label_encoder = None
        self.scaler = None
        self.numerical_features = [
            'count_trip', 'miles', 'drive_hours', 'count_brakes', 'count_accelarations',
            'time_speeding_hours', 'time_phoneuse_hours', 'highway_miles', 
            'night_drive_hrs', 'maximum_speed'
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        X_copy = X.copy()
        
        # Convert multi-class target to binary
        if 'collisions' in X_copy.columns:
            X_copy['collisions_binary'] = (X_copy['collisions'] > 0).astype(int)
        
        # Fit numerical imputer
        numerical_cols = [col for col in self.numerical_features if col in X_copy.columns]
        if numerical_cols:
            self.numerical_imputer = SimpleImputer(strategy='median')
            self.numerical_imputer.fit(X_copy[numerical_cols])
        
        # Fit label encoder for month
        if 'month' in X_copy.columns:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(X_copy['month'].astype(str))
        
        # Fit scaler on numerical features (after imputation and infinite replacement)
        if numerical_cols:
            X_imputed = X_copy.copy()
            X_imputed[numerical_cols] = self.numerical_imputer.transform(X_imputed[numerical_cols])
            
            # Replace infinite values with NaN, then impute again
            X_imputed[numerical_cols] = X_imputed[numerical_cols].replace([np.inf, -np.inf], np.nan)
            if X_imputed[numerical_cols].isnull().any().any():
                temp_imputer = SimpleImputer(strategy='median')
                X_imputed[numerical_cols] = temp_imputer.fit_transform(X_imputed[numerical_cols])
            
            self.scaler = StandardScaler()
            self.scaler.fit(X_imputed[numerical_cols])
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the preprocessing steps.

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        X_transformed = X.copy()
        
        # Convert multi-class target to binary
        if 'collisions' in X_transformed.columns:
            X_transformed['collisions_binary'] = (X_transformed['collisions'] > 0).astype(int)
        
        # Apply numerical imputation
        numerical_cols = [col for col in self.numerical_features if col in X_transformed.columns]
        if numerical_cols and self.numerical_imputer is not None:
            X_transformed[numerical_cols] = self.numerical_imputer.transform(X_transformed[numerical_cols])
        
        # Apply label encoding to month
        if 'month' in X_transformed.columns and self.label_encoder is not None:
            X_transformed['month_encoded'] = self.label_encoder.transform(X_transformed['month'].astype(str))
        
        # Replace infinite values with NaN and impute
        if numerical_cols:
            X_transformed[numerical_cols] = X_transformed[numerical_cols].replace([np.inf, -np.inf], np.nan)
            if X_transformed[numerical_cols].isnull().any().any():
                temp_imputer = SimpleImputer(strategy='median')
                X_transformed[numerical_cols] = temp_imputer.fit_transform(X_transformed[numerical_cols])
        
        # Apply scaling to numerical features
        if numerical_cols and self.scaler is not None:
            X_transformed[numerical_cols] = self.scaler.transform(X_transformed[numerical_cols])
        
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
