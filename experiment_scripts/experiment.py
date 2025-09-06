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
        """Prepare features and targets according to experiment 3 plan"""
        # Initialize and fit data processor on training data
        print("🔧 Initializing and fitting data processor...")
        self.data_processor = DataProcessor(self._config.data_prep)
        
        # Transform training data
        X_train_processed = self.data_processor.fit_transform(train_data)
        y_train = (train_data['collisions'] > 0).astype(int)  # Binary target
        
        # Transform test data
        X_test_processed = self.data_processor.transform(test_data)
        y_test = (test_data['collisions'] > 0).astype(int)  # Binary target
        
        print("🔧 Initializing and fitting feature processor with SelectKBest...")
        self.feature_processor = FeatureProcessor(self._config.feature_prep, 
                                                  n_features_select=self._config.model.n_features_select)
        
        # Fit feature processor on training data (requires target for SelectKBest)
        X_train_features = self.feature_processor.fit_transform(X_train_processed, y_train)
        
        # Transform test data
        X_test_features = self.feature_processor.transform(X_test_processed)
        
        print(f"✅ Training features shape after preprocessing: {X_train_features.shape}")
        print(f"✅ Test features shape after preprocessing: {X_test_features.shape}")
        print(f"✅ Selected features: {self.feature_processor.selected_features}")
        
        return X_train_features, X_test_features, y_train, y_test

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model according to experiment 3 specifications"""
        print("🚀 Training RandomForest model (no class balancing)...")
        
        # Exclude target columns from training features
        feature_cols = [col for col in X_train.columns 
                       if col not in ['driver_id', 'collisions', 'collisions_binary']]
        X_train_clean = X_train[feature_cols].select_dtypes(include=[np.number])
        
        print(f"🔧 Training on {X_train_clean.shape[1]} features: {list(X_train_clean.columns)}")
        
        self.model = ModelWrapper(self._config.model)
        self.model.fit(X_train_clean, y_train)
        print("✅ Model training completed!")

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str) -> Dict[str, Any]:
        """Evaluate model and generate plots"""
        print("📊 Evaluating model performance...")
        
        # Exclude target columns from test features (same as training)
        feature_cols = [col for col in X_test.columns 
                       if col not in ['driver_id', 'collisions', 'collisions_binary']]
        X_test_clean = X_test[feature_cols].select_dtypes(include=[np.number])
        
        # Make predictions
        y_pred = self.model.predict(X_test_clean)
        y_pred_proba = self.model.predict_proba(X_test_clean)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self._config.model_evaluation)
        
        # Get comprehensive metrics
        metrics = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Create output directory for plots
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate evaluation plots
        print("📈 Generating evaluation plots...")
        plot_files = []
        
        try:
            plot_files.append(evaluator.create_pr_curve_plot(y_test, y_pred_proba, plots_dir))
            plot_files.append(evaluator.create_roc_curve_plot(y_test, y_pred_proba, plots_dir))
            plot_files.append(evaluator.create_confusion_matrix_plot(y_test, y_pred, plots_dir))
            
            # Feature importance plot if available
            feature_importance = self.model.get_feature_importance()
            if feature_importance is not None:
                feature_names = list(X_test_clean.columns)
                plot_files.append(evaluator.create_feature_importance_plot(feature_names, feature_importance, plots_dir))
            
            plot_files.append(evaluator.create_threshold_analysis_plot(y_test, y_pred_proba, plots_dir))
        except Exception as e:
            print(f"⚠️ Warning: Some plots could not be generated: {e}")
        
        metrics['plot_files'] = plot_files
        
        print(f"✅ Primary metric (PR-AUC): {metrics['pr_auc']:.4f}")
        print(f"✅ ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"✅ F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics

    def _save_artifacts(self, output_dir: str) -> list:
        """Save model artifacts"""
        print("💾 Saving model artifacts...")
        
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
        
        model_path = os.path.join(artifacts_dir, "trained_models.pkl")
        self.model.save(model_path)
        artifact_files.append("trained_models.pkl")
        
        print(f"✅ Saved {len(artifact_files)} model artifacts")
        return artifact_files

    def _create_mlflow_pipeline(self, X_sample: pd.DataFrame, X_train_features: pd.DataFrame) -> Dict[str, Any]:
        """Create MLflow model pipeline and save/log it"""
        print("🔧 Creating MLflow model pipeline...")
        
        # Create pipeline with fitted components
        self.pipeline = ModelPipeline(
            data_processor=self.data_processor,
            feature_processor=self.feature_processor,
            model=self.model,
            config=self._config
        )
        
        # Initialize feature columns from training data to ensure consistency
        feature_cols = [col for col in X_train_features.columns 
                       if col not in ['driver_id', 'collisions', 'collisions_binary']]
        training_features = X_train_features[feature_cols].select_dtypes(include=[np.number])
        self.pipeline.feature_columns = list(training_features.columns)
        
        # Test pipeline end-to-end with processed sample that matches training format
        print("🧪 Testing pipeline end-to-end...")
        sample_pred = self.pipeline.predict(X_sample)
        sample_proba = self.pipeline.predict_proba(X_sample)
        print(f"✅ Pipeline test successful - predictions: {sample_pred[:3]}")
        
        # Define paths
        base_output_path = "/Users/yuvalheffetz/ds-agent-projects/session_5feb6ac6-f292-4d0c-9e41-ab6b3ffc14d6/experiments/experiment_3/output/model_artifacts/mlflow_model/"
        relative_path = "output/model_artifacts/mlflow_model/"
        
        # Create signature
        signature = mlflow.models.infer_signature(X_sample, sample_pred)
        
        # Always save locally
        print(f"💾 Saving model to local disk: {base_output_path}")
        mlflow.sklearn.save_model(
            self.pipeline,
            path=base_output_path,
            code_paths=["vehicle_collision_prediction"],
            signature=signature
        )
        
        # Conditionally log to MLflow if run ID available
        logged_model_uri = None
        active_run_id = "771f363fd0da40d0b7f59349952caed2"
        
        if active_run_id and active_run_id != 'None' and active_run_id.strip():
            print(f"✅ Active MLflow run ID '{active_run_id}' detected. Reconnecting to log model.")
            try:
                with mlflow.start_run(run_id=active_run_id):
                    logged_model_info = mlflow.sklearn.log_model(
                        self.pipeline,
                        artifact_path="model",
                        code_paths=["vehicle_collision_prediction"],
                        signature=signature
                    )
                    logged_model_uri = logged_model_info.model_uri
                    print(f"✅ Model logged to MLflow: {logged_model_uri}")
            except Exception as e:
                print(f"⚠️ Warning: Could not log to MLflow: {e}")
        else:
            print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
        
        # Return model info
        return {
            "model_path": relative_path,
            "logged_model_uri": logged_model_uri,
            "model_type": "sklearn",
            "task_type": "classification", 
            "signature": {
                "inputs": signature.inputs.to_dict() if signature else None,
                "outputs": signature.outputs.to_dict() if signature else None
            },
            "framework_version": sklearn.__version__
        }

    def run(self, train_dataset_path: str, test_dataset_path: str, output_dir: str, seed: int = 42) -> Dict[str, Any]:
        """
        Run complete experiment according to experiment 3 plan
        
        Returns:
        Dict with mandatory format including mlflow_model_info
        """
        try:
            print("🎯 Starting Experiment 3: Feature Selection Enhanced Vehicle Collision Prediction")
            
            # Set seed for reproducibility
            self._set_seed(seed)
            
            # Load data
            train_data, test_data = self._load_data(train_dataset_path, test_dataset_path)
            
            # Prepare data with preprocessing and feature selection
            X_train, X_test, y_train, y_test = self._prepare_data(train_data, test_data)
            
            # Train model
            self._train_model(X_train, y_train)
            
            # Evaluate model
            metrics = self._evaluate_model(X_test, y_test, output_dir)
            
            # Save individual artifacts
            model_artifacts = self._save_artifacts(output_dir)
            
            # Create and save MLflow pipeline
            mlflow_model_info = self._create_mlflow_pipeline(train_data.head(3), X_train)
            
            print("🎉 Experiment completed successfully!")
            
            return {
                "metric_name": self._config.model_evaluation.evaluation_metric,
                "metric_value": metrics['pr_auc'],
                "model_artifacts": model_artifacts,
                "mlflow_model_info": mlflow_model_info
            }
            
        except Exception as e:
            print(f"❌ Experiment failed: {str(e)}")
            raise