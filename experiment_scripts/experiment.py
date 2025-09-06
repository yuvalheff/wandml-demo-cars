import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import sklearn
from pathlib import Path
from typing import Dict, Any

from vehicle_collision_prediction.pipeline.feature_preprocessing import FeatureProcessor
from vehicle_collision_prediction.pipeline.data_preprocessing import DataProcessor
from vehicle_collision_prediction.pipeline.model import ModelWrapper
from vehicle_collision_prediction.model_pipeline import ModelPipeline
from vehicle_collision_prediction.config import Config
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)
        self.data_processor = None
        self.feature_processor = None
        self.model = None
        self.pipeline = None

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        # Also set sklearn random state if needed

    def _load_data(self, train_path: str, test_path: str) -> tuple:
        """Load training and test datasets"""
        print(f"📂 Loading training data from: {train_path}")
        train_data = pd.read_csv(train_path)
        print(f"📂 Loading test data from: {test_path}")
        test_data = pd.read_csv(test_path)
        
        print(f"✅ Train set shape: {train_data.shape}")
        print(f"✅ Test set shape: {test_data.shape}")
        
        return train_data, test_data

    def _prepare_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
        """Prepare features and targets"""
        # Initialize and fit data processor on training data
        print("🔧 Initializing and fitting data processor...")
        self.data_processor = DataProcessor(self._config.data_prep)
        
        # Fit on training data
        train_processed = self.data_processor.fit_transform(train_data)
        test_processed = self.data_processor.transform(test_data)
        
        # Initialize and fit feature processor
        print("🔧 Initializing and fitting feature processor...")
        self.feature_processor = FeatureProcessor(self._config.feature_prep)
        
        # Apply feature engineering
        train_features = self.feature_processor.fit_transform(train_processed)
        test_features = self.feature_processor.transform(test_processed)
        
        # Prepare target variable (binary)
        y_train = train_features['collisions_binary'].values
        y_test = test_features['collisions_binary'].values
        
        # Prepare numeric features for modeling (exclude non-feature columns and non-numeric)
        feature_cols = [col for col in train_features.columns 
                       if col not in ['driver_id', 'collisions', 'collisions_binary', 'month']]
        
        X_train = train_features[feature_cols].select_dtypes(include=[np.number])
        X_test = test_features[feature_cols].select_dtypes(include=[np.number])
        
        print(f"✅ Training features shape: {X_train.shape}")
        print(f"✅ Training target distribution: {np.bincount(y_train)}")
        print(f"✅ Test features shape: {X_test.shape}")
        print(f"✅ Test target distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test

    def _train_model(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Train the model"""
        print("🏋️ Training model...")
        
        # Initialize model
        self.model = ModelWrapper(self._config.model)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        print("✅ Model training completed")

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """Evaluate the model and generate plots"""
        print("📊 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self._config.model_evaluation)
        
        # Calculate metrics
        metrics = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Generate plots
        plots_dir = os.path.join(output_dir, "plots")
        
        # Get feature importance if available
        feature_importance = self.model.get_feature_importance()
        feature_names = list(X_test.columns) if feature_importance is not None else None
        
        plot_files = evaluator.generate_all_plots(
            y_test, y_pred, y_pred_proba,
            feature_names=feature_names,
            feature_importance=feature_importance,
            output_dir=plots_dir
        )
        
        print(f"✅ Generated evaluation plots in: {plots_dir}")
        print(f"📈 Primary metric (PR-AUC): {metrics['pr_auc']:.4f}")
        
        return metrics, plot_files

    def _save_artifacts(self, output_dir: str) -> list:
        """Save model artifacts"""
        artifacts_dir = os.path.join(output_dir, "model_artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        artifact_files = []
        
        # Save individual components
        data_processor_path = os.path.join(artifacts_dir, "data_processor.pkl")
        self.data_processor.save(data_processor_path)
        artifact_files.append("data_processor.pkl")
        
        feature_processor_path = os.path.join(artifacts_dir, "feature_processor.pkl")
        self.feature_processor.save(feature_processor_path)
        artifact_files.append("feature_processor.pkl")
        
        model_path = os.path.join(artifacts_dir, "trained_model.pkl")
        self.model.save(model_path)
        artifact_files.append("trained_model.pkl")
        
        print(f"✅ Saved model artifacts to: {artifacts_dir}")
        
        return artifact_files

    def _create_pipeline(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """Create and save MLflow model pipeline"""
        print("🔗 Creating model pipeline...")
        
        # Create pipeline with fitted components
        self.pipeline = ModelPipeline(
            data_processor=self.data_processor,
            feature_processor=self.feature_processor,
            model=self.model,
            config=self._config
        )
        
        # Test pipeline with sample data to ensure it works
        print("🧪 Testing pipeline with sample data...")
        sample_predictions = self.pipeline.predict(X_sample.head(2))
        sample_probabilities = self.pipeline.predict_proba(X_sample.head(2))
        print(f"✅ Pipeline test successful - sample predictions: {sample_predictions}")
        
        # Define paths
        output_path = "/Users/yuvalheffetz/ds-agent-projects/session_5feb6ac6-f292-4d0c-9e41-ab6b3ffc14d6/experiments/experiment_1/output/model_artifacts/mlflow_model/"
        relative_path_for_return = "output/model_artifacts/mlflow_model/"
        
        # Create sample input for signature
        sample_input = X_sample.head(1)
        sample_output = self.pipeline.predict(sample_input)
        
        # Create signature
        signature = mlflow.models.infer_signature(sample_input, sample_output)
        
        # 1. Always save the model to the local path for harness validation
        print(f"💾 Saving model to local disk for harness: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        mlflow.sklearn.save_model(
            self.pipeline,
            path=output_path,
            code_paths=["vehicle_collision_prediction"],  # Bundle the custom code
            signature=signature
        )
        
        # 2. If an MLflow run ID is provided, reconnect and log the model as an artifact
        active_run_id = "5a75c53e2e2d49779d689f2a7a56b2f3"
        logged_model_uri = None  # Initialize to None
        
        if active_run_id and active_run_id != 'None' and active_run_id.strip():
            print(f"✅ Active MLflow run ID '{active_run_id}' detected. Reconnecting to log model as an artifact.")
            try:
                with mlflow.start_run(run_id=active_run_id):
                    logged_model_info = mlflow.sklearn.log_model(
                        self.pipeline,
                        artifact_path="model",  # Use a standard artifact path
                        code_paths=["vehicle_collision_prediction"],  # Bundle the custom code
                        signature=signature
                    )
                    logged_model_uri = logged_model_info.model_uri
            except Exception as e:
                print(f"⚠️ Warning: Could not log model to MLflow run: {e}")
        else:
            print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
        
        # Prepare model info for return
        mlflow_model_info = {
            "model_path": relative_path_for_return,
            "logged_model_uri": logged_model_uri,
            "model_type": "sklearn",
            "task_type": "classification",
            "signature": signature.to_dict() if signature else None,
            "input_example": sample_input.to_dict('records')[0],
            "framework_version": sklearn.__version__
        }
        
        print("✅ MLflow model pipeline created and saved")
        
        return mlflow_model_info

    def run(self, train_dataset_path: str, test_dataset_path: str, output_dir: str, seed: int = 42) -> Dict[str, Any]:
        """
        Run the complete experiment pipeline.
        
        Parameters:
        train_dataset_path: Path to training data
        test_dataset_path: Path to test data  
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility
        
        Returns:
        Dict containing experiment results
        """
        print("🚀 Starting Vehicle Collision Prediction Experiment")
        print(f"📁 Output directory: {output_dir}")
        
        # Set seed for reproducibility
        self._set_seed(seed)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "general_artifacts"), exist_ok=True)
        
        try:
            # 1. Load data
            train_data, test_data = self._load_data(train_dataset_path, test_dataset_path)
            
            # 2. Prepare data (preprocessing + feature engineering)
            X_train, X_test, y_train, y_test = self._prepare_data(train_data, test_data)
            
            # 3. Train model
            self._train_model(X_train, y_train)
            
            # 4. Evaluate model
            metrics, plot_files = self._evaluate_model(X_test, y_test, output_dir)
            
            # 5. Save artifacts
            artifact_files = self._save_artifacts(output_dir)
            
            # 6. Create MLflow pipeline
            # Use original test data (before processing) for pipeline testing
            mlflow_model_info = self._create_pipeline(test_data)
            
            # Prepare results
            results = {
                "metric_name": "average_precision",
                "metric_value": metrics["pr_auc"],
                "model_artifacts": artifact_files,
                "mlflow_model_info": mlflow_model_info
            }
            
            print("🎉 Experiment completed successfully!")
            print(f"📊 Final PR-AUC: {metrics['pr_auc']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment failed with error: {str(e)}")
            raise