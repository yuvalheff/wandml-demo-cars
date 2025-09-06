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
        self.month_columns = None
        self.numerical_features = [
            'count_trip', 'miles', 'drive_hours', 'count_brakes', 'count_accelarations',
            'time_speeding_hours', 'time_phoneuse_hours', 'highway_miles', 
            'night_drive_hrs', 'maximum_speed'
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data according to experiment 3 plan.
        - Apply mean imputation to numerical features
        - Apply one-hot encoding to month column  
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
            X_copy['collisions_binary'] = (X_copy['collisions'] > 0).astype(int)
        
        # Fit numerical imputer with mean strategy (as per experiment plan)
        numerical_cols = [col for col in self.numerical_features if col in X_copy.columns]
        if numerical_cols:
            self.numerical_imputer = SimpleImputer(strategy=self.config.imputation_strategy)
            self.numerical_imputer.fit(X_copy[numerical_cols])
        
        # Store month columns for one-hot encoding (fit phase)
        if 'month' in X_copy.columns:
            month_dummies = pd.get_dummies(X_copy['month'], prefix='month')
            self.month_columns = month_dummies.columns.tolist()
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data based on the preprocessing steps.
        - Apply mean imputation to numerical features
        - Apply one-hot encoding to month column creating 12 binary features
        - Convert target to binary classification using y_binary = (collisions > 0).astype(int)
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
            X_transformed['collisions_binary'] = (X_transformed['collisions'] > 0).astype(int)
        
        # Apply numerical imputation with mean strategy
        numerical_cols = [col for col in self.numerical_features if col in X_transformed.columns]
        if numerical_cols and self.numerical_imputer is not None:
            X_transformed[numerical_cols] = self.numerical_imputer.transform(X_transformed[numerical_cols])
        
        # Apply one-hot encoding to month column
        if 'month' in X_transformed.columns:
            month_dummies = pd.get_dummies(X_transformed['month'], prefix='month')
            
            # Ensure all expected month columns are present (in case some months missing in transform data)
            if self.month_columns is not None:
                for col in self.month_columns:
                    if col not in month_dummies.columns:
                        month_dummies[col] = 0
                # Reorder columns to match training
                month_dummies = month_dummies[self.month_columns]
            
            # Drop original month column and add dummy columns
            X_transformed = X_transformed.drop('month', axis=1)
            X_transformed = pd.concat([X_transformed, month_dummies], axis=1)
        
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
