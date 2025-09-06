import pandas as pd
import numpy as np
import pickle
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from vehicle_collision_prediction.config import ModelConfig

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model based on configuration - experiment 3: no class balancing"""
        if self.config.model_type == "RandomForest":
            self.model = RandomForestClassifier(**self.config.model_params)
        elif self.config.model_type == "XGBoost" and XGBClassifier is not None:
            self.model = XGBClassifier(**self.config.model_params)
        else:
            # Default to RandomForest with experiment 3 parameters (NO class balancing)
            self.model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.
        Experiment 3: No calibration, feature selection handled in FeatureProcessor.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        # Ensure we only use numeric features for modeling
        numeric_features = X.select_dtypes(include=[np.number])
        
        # Fit the model directly (feature selection handled in FeatureProcessor)
        self.model.fit(numeric_features, y)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Ensure we only use numeric features for modeling
        numeric_features = X.select_dtypes(include=[np.number])
        
        return self.model.predict(numeric_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Ensure we only use numeric features for modeling
        numeric_features = X.select_dtypes(include=[np.number])
        
        return self.model.predict_proba(numeric_features)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the trained model.
        
        Returns:
        np.ndarray: Feature importance scores, or None if not available.
        """
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None

    def save(self, path: str) -> None:
        """
        Save the model wrapper as an artifact.

        Parameters:
        path (str): The file path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'ModelWrapper':
        """
        Load the model wrapper from a saved artifact.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        ModelWrapper: The loaded model wrapper.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)