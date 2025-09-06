"""
Vehicle Collision Prediction Pipeline

Complete ML pipeline that combines data preprocessing, feature engineering, and model prediction.
Designed for MLflow deployment and end-to-end collision prediction.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

from vehicle_collision_prediction.pipeline.data_preprocessing import DataProcessor
from vehicle_collision_prediction.pipeline.feature_preprocessing import FeatureProcessor
from vehicle_collision_prediction.pipeline.model import ModelWrapper
from vehicle_collision_prediction.config import Config


class ModelPipeline:
    """
    Complete pipeline for vehicle collision prediction.
    
    Combines data preprocessing, feature engineering, and model prediction
    into a single deployable unit suitable for MLflow model registry.
    """
    
    def __init__(self, data_processor: DataProcessor = None, 
                 feature_processor: FeatureProcessor = None, 
                 model: ModelWrapper = None, 
                 config: Config = None):
        """
        Initialize the pipeline components.
        
        Parameters:
        data_processor: Fitted DataProcessor instance
        feature_processor: Fitted FeatureProcessor instance  
        model: Fitted ModelWrapper instance
        config: Configuration object
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model = model
        self.config = config
        
        # Store feature columns for consistent ordering
        self.feature_columns = None
        
    def _validate_components(self):
        """Validate that all pipeline components are available."""
        if self.data_processor is None:
            raise ValueError("data_processor is required")
        if self.feature_processor is None:
            raise ValueError("feature_processor is required")
        if self.model is None:
            raise ValueError("model is required")
    
    def _ensure_feature_order(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent feature ordering and handle missing columns."""
        if self.feature_columns is None:
            # First time - store the column order
            self.feature_columns = list(X.columns)
            return X
        
        # Ensure all expected features are present, fill missing with 0
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0.0
        
        # Return columns in the expected order
        return X[self.feature_columns]
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict collision outcomes for input data.
        
        Parameters:
        X: Input data (raw features)
        
        Returns:
        np.ndarray: Predicted class labels (0 or 1)
        """
        self._validate_components()
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            # Assume standard column order from training
            expected_cols = [
                'driver_id', 'month', 'count_trip', 'miles', 'drive_hours',
                'count_brakes', 'count_accelarations', 'time_speeding_hours',
                'time_phoneuse_hours', 'highway_miles', 'night_drive_hrs',
                'maximum_speed', 'collisions'
            ]
            X = pd.DataFrame(X, columns=expected_cols[:X.shape[1]])
        
        # Apply data preprocessing
        X_processed = self.data_processor.transform(X)
        
        # Apply feature engineering
        X_features = self.feature_processor.transform(X_processed)
        
        # Select numeric features for modeling (exclude non-feature columns but include month dummies)
        feature_cols = [col for col in X_features.columns 
                       if col not in ['driver_id', 'collisions', 'collisions_binary']]
        X_model_input = X_features[feature_cols].select_dtypes(include=[np.number])
        
        # Ensure consistent feature ordering
        X_model_input = self._ensure_feature_order(X_model_input)
        
        # Make predictions
        predictions = self.model.predict(X_model_input)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict collision probabilities for input data.
        
        Parameters:
        X: Input data (raw features)
        
        Returns:
        np.ndarray: Predicted class probabilities
        """
        self._validate_components()
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            # Assume standard column order from training
            expected_cols = [
                'driver_id', 'month', 'count_trip', 'miles', 'drive_hours',
                'count_brakes', 'count_accelarations', 'time_speeding_hours',
                'time_phoneuse_hours', 'highway_miles', 'night_drive_hrs',
                'maximum_speed', 'collisions'
            ]
            X = pd.DataFrame(X, columns=expected_cols[:X.shape[1]])
        
        # Apply data preprocessing
        X_processed = self.data_processor.transform(X)
        
        # Apply feature engineering
        X_features = self.feature_processor.transform(X_processed)
        
        # Select numeric features for modeling (exclude non-feature columns but include month dummies)
        feature_cols = [col for col in X_features.columns 
                       if col not in ['driver_id', 'collisions', 'collisions_binary']]
        X_model_input = X_features[feature_cols].select_dtypes(include=[np.number])
        
        # Ensure consistent feature ordering
        X_model_input = self._ensure_feature_order(X_model_input)
        
        # Make probability predictions
        probabilities = self.model.predict_proba(X_model_input)
        
        return probabilities
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features used by the model.
        
        Returns:
        List[str]: Feature names in the order expected by the model
        """
        if self.feature_columns is None:
            raise ValueError("Pipeline has not been fitted or used for prediction yet")
        return self.feature_columns.copy()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline components.
        
        Returns:
        Dict containing pipeline metadata
        """
        model_type = 'Unknown'
        if self.model and hasattr(self.model, 'config'):
            model_type = getattr(self.model.config, 'model_type', 'Unknown')
        
        return {
            'has_data_processor': self.data_processor is not None,
            'has_feature_processor': self.feature_processor is not None,
            'has_model': self.model is not None,
            'feature_columns': self.feature_columns,
            'model_type': model_type
        }
