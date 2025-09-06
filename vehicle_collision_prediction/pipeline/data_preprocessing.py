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
        self.month_encoder = None
        self.numerical_features = [
            'count_trip', 'miles', 'drive_hours', 'count_brakes', 'count_accelarations',
            'time_speeding_hours', 'time_phoneuse_hours', 'highway_miles', 
            'night_drive_hrs', 'maximum_speed'
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data according to experiment 4 plan.
        - Apply median imputation to numerical features
        - Apply label encoding to month column  
        - Convert target to binary classification
        - Remove driver_id column

        Parameters:
        X (pd.DataFrame): The input features.
        y (Optional[pd.Series]): The target variable (not used in this processor).

        Returns:
        DataProcessor: The fitted processor.
        """
        X_copy = X.copy()
        
        # Convert multi-class target to binary
        if 'collisions' in X_copy.columns:
            X_copy['collision_binary'] = (X_copy['collisions'] > 0).astype(int)
        
        # Fit numerical imputer with median strategy (as per experiment 4 plan)
        numerical_cols = [col for col in self.numerical_features if col in X_copy.columns]
        if numerical_cols:
            self.numerical_imputer = SimpleImputer(strategy=self.config.imputation_strategy)
            self.numerical_imputer.fit(X_copy[numerical_cols])
        
        # Initialize label encoder for month (experiment 4 uses LabelEncoder instead of one-hot)
        if 'month' in X_copy.columns:
            self.month_encoder = LabelEncoder()
            self.month_encoder.fit(X_copy['month'])
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the preprocessing steps.
        - Apply median imputation to numerical features
        - Apply label encoding to month column creating single encoded feature
        - Convert target to binary classification using collision_binary = (collisions > 0).astype(int)
        - Remove driver_id column

        Parameters:
        X (pd.DataFrame): The input features to transform.

        Returns:
        pd.DataFrame: The transformed features.
        """
        X_transformed = X.copy()
        
        # Remove driver_id column as specified in experiment plan
        if 'driver_id' in X_transformed.columns:
            X_transformed = X_transformed.drop('driver_id', axis=1)
        
        # Convert multi-class target to binary
        if 'collisions' in X_transformed.columns:
            X_transformed['collision_binary'] = (X_transformed['collisions'] > 0).astype(int)
        
        # Apply numerical imputation with median strategy
        numerical_cols = [col for col in self.numerical_features if col in X_transformed.columns]
        if numerical_cols and self.numerical_imputer is not None:
            X_transformed[numerical_cols] = self.numerical_imputer.transform(X_transformed[numerical_cols])
        
        # Apply label encoding to month column (experiment 4 uses LabelEncoder)
        if 'month' in X_transformed.columns and self.month_encoder is not None:
            X_transformed['month_encoded'] = self.month_encoder.transform(X_transformed['month'])
            X_transformed = X_transformed.drop('month', axis=1)
        
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
