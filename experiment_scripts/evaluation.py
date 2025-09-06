import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, 
    roc_curve, classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve

from vehicle_collision_prediction.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        
        # App color palette
        self.app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Parameters:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for positive class)
        
        Returns:
        Dict containing evaluation metrics
        """
        # Get probabilities for positive class (collision)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_proba_positive = y_pred_proba[:, 1]
        else:
            y_proba_positive = y_pred_proba.ravel()
        
        # Calculate metrics
        metrics = {
            'pr_auc': average_precision_score(y_true, y_proba_positive),
            'roc_auc': roc_auc_score(y_true, y_proba_positive),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics

    def create_pr_curve_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray, output_dir: str) -> str:
        """Create and save Precision-Recall curve plot"""
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_proba_positive = y_pred_proba[:, 1]
        else:
            y_proba_positive = y_pred_proba.ravel()
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
        pr_auc = average_precision_score(y_true, y_proba_positive)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.3f})',
            line=dict(color=self.app_color_palette[0], width=3)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        filename = "precision_recall_curve.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filename

    def create_roc_curve_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray, output_dir: str) -> str:
        """Create and save ROC curve plot"""
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_proba_positive = y_pred_proba[:, 1]
        else:
            y_proba_positive = y_pred_proba.ravel()
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
        roc_auc = roc_auc_score(y_true, y_proba_positive)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color=self.app_color_palette[1], width=3)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray', width=2)
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        filename = "roc_curve.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filename

    def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> str:
        """Create and save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title='Predicted',
            yaxis_title='Actual',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            )
        )
        
        filename = "confusion_matrix.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filename

    def create_feature_importance_plot(self, feature_names: list, feature_importance: np.ndarray, output_dir: str) -> str:
        """Create and save feature importance plot"""
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = feature_importance[sorted_idx]
        
        # Take top 15 features for readability
        top_n = min(15, len(sorted_features))
        top_features = sorted_features[:top_n]
        top_importance = sorted_importance[:top_n]
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_importance,
                y=top_features,
                orientation='h',
                marker_color=self.app_color_palette[2]
            )
        ])
        
        fig.update_layout(
            title='Feature Importance (Top 15)',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            )
        )
        
        filename = "feature_importance.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filename

    def create_calibration_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray, output_dir: str) -> str:
        """Create and save calibration plot"""
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_proba_positive = y_pred_proba[:, 1]
        else:
            y_proba_positive = y_pred_proba.ravel()
        
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba_positive, n_bins=10)
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfectly Calibrated',
            line=dict(dash='dash', color='gray', width=2)
        ))
        
        # Model calibration
        fig.add_trace(go.Scatter(
            x=mean_pred,
            y=fraction_pos,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color=self.app_color_palette[3], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Calibration Plot',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        filename = "calibration_plot.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filename

    def create_prediction_distribution_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray, output_dir: str) -> str:
        """Create and save prediction distribution plot"""
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_proba_positive = y_pred_proba[:, 1]
        else:
            y_proba_positive = y_pred_proba.ravel()
        
        # Create histogram data
        no_collision_probs = y_proba_positive[y_true == 0]
        collision_probs = y_proba_positive[y_true == 1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=no_collision_probs,
            name='No Collision',
            opacity=0.7,
            nbinsx=50,
            marker_color=self.app_color_palette[0]
        ))
        
        fig.add_trace(go.Histogram(
            x=collision_probs,
            name='Collision',
            opacity=0.7,
            nbinsx=50,
            marker_color=self.app_color_palette[1]
        ))
        
        fig.update_layout(
            title='Prediction Probability Distribution',
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        filename = "prediction_distribution.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filename

    def generate_all_plots(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, 
                          feature_names: list = None, feature_importance: np.ndarray = None, 
                          output_dir: str = None) -> Dict[str, str]:
        """
        Generate all evaluation plots and return filenames.
        
        Returns:
        Dict mapping plot type to filename
        """
        if output_dir is None:
            raise ValueError("output_dir must be specified")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = {}
        
        # Generate all plots
        plot_files['pr_curve'] = self.create_pr_curve_plot(y_true, y_pred_proba, output_dir)
        plot_files['roc_curve'] = self.create_roc_curve_plot(y_true, y_pred_proba, output_dir)
        plot_files['confusion_matrix'] = self.create_confusion_matrix_plot(y_true, y_pred, output_dir)
        plot_files['calibration'] = self.create_calibration_plot(y_true, y_pred_proba, output_dir)
        plot_files['prediction_distribution'] = self.create_prediction_distribution_plot(y_true, y_pred_proba, output_dir)
        
        if feature_names is not None and feature_importance is not None:
            plot_files['feature_importance'] = self.create_feature_importance_plot(feature_names, feature_importance, output_dir)
        
        return plot_files

    def optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          strategy: str = 'optimal_f1') -> Tuple[float, Dict[str, float]]:
        """
        Optimize classification threshold based on different strategies.
        
        Parameters:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        strategy: Threshold optimization strategy
        
        Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
        """
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_proba_positive = y_pred_proba[:, 1]
        else:
            y_proba_positive = y_pred_proba.ravel()
        
        thresholds = np.linspace(0.01, 0.99, 99)
        
        if strategy == 'optimal_f1':
            # Find threshold that maximizes F1 score
            best_threshold = 0.5
            best_f1 = 0
            
            for threshold in thresholds:
                y_pred_thresh = (y_proba_positive >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    
        elif strategy == 'high_precision':
            # Find threshold that achieves at least 40% precision with highest recall
            best_threshold = 0.5
            best_recall = 0
            target_precision = 0.40
            
            for threshold in thresholds:
                y_pred_thresh = (y_proba_positive >= threshold).astype(int)
                precision = precision_score(y_true, y_pred_thresh, zero_division=0)
                recall = recall_score(y_true, y_pred_thresh, zero_division=0)
                
                if precision >= target_precision and recall > best_recall:
                    best_recall = recall
                    best_threshold = threshold
                    
        elif strategy == 'high_recall':
            # Find threshold that achieves at least 60% recall with highest precision
            best_threshold = 0.5
            best_precision = 0
            target_recall = 0.60
            
            for threshold in thresholds:
                y_pred_thresh = (y_proba_positive >= threshold).astype(int)
                precision = precision_score(y_true, y_pred_thresh, zero_division=0)
                recall = recall_score(y_true, y_pred_thresh, zero_division=0)
                
                if recall >= target_recall and precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
        else:
            best_threshold = 0.5
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_proba_positive >= best_threshold).astype(int)
        metrics_at_threshold = {
            'threshold': best_threshold,
            'precision': precision_score(y_true, y_pred_optimal, zero_division=0),
            'recall': recall_score(y_true, y_pred_optimal, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_optimal, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_optimal)
        }
        
        return best_threshold, metrics_at_threshold

    def create_threshold_analysis_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                     output_dir: str) -> str:
        """Create threshold analysis plot showing precision/recall vs threshold"""
        # Get probabilities for positive class
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_proba_positive = y_pred_proba[:, 1]
        else:
            y_proba_positive = y_pred_proba.ravel()
        
        thresholds = np.linspace(0.01, 0.99, 99)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba_positive >= threshold).astype(int)
            precision = precision_score(y_true, y_pred_thresh, zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=precisions,
            mode='lines',
            name='Precision',
            line=dict(color=self.app_color_palette[0], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=recalls,
            mode='lines',
            name='Recall',
            line=dict(color=self.app_color_palette[1], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name='F1 Score',
            line=dict(color=self.app_color_palette[2], width=2)
        ))
        
        fig.update_layout(
            title='Threshold Analysis: Precision, Recall, and F1 Score',
            xaxis_title='Threshold',
            yaxis_title='Score',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B5CF6', size=12),
            title_font=dict(color='#7C3AED', size=16),
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),
                title_font=dict(color='#7C3AED', size=12)
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))
        )
        
        filename = "threshold_analysis.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        return filename
