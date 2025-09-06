import pandas as pd
import numpy as np
import pickle
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from vehicle_collision_prediction.config import ModelConfig

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self.feature_selector = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model based on configuration"""
        if self.config.model_type == "RandomForest":
            self.model = RandomForestClassifier(**self.config.model_params)
        elif self.config.model_type == "XGBoost" and XGBClassifier is not None:
            self.model = XGBClassifier(**self.config.model_params)
        else:
            # Default to RandomForest with balanced class weights
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        # Ensure we only use numeric features for modeling
        numeric_features = X.select_dtypes(include=[np.number])
        
        # Feature selection - select top K features based on config or use all
        n_features_to_select = getattr(self.config, 'n_features_select', 15)
        if n_features_to_select and n_features_to_select < numeric_features.shape[1]:
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
            X_selected = self.feature_selector.fit_transform(numeric_features, y)
        else:
            X_selected = numeric_features
        
        # Fit the model
        self.model.fit(X_selected, y)
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
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(numeric_features)
        else:
            X_selected = numeric_features
        
        return self.model.predict(X_selected)

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
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(numeric_features)
        else:
            X_selected = numeric_features
        
        return self.model.predict_proba(X_selected)

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