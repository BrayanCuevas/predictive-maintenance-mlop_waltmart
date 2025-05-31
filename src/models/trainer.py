"""
Model training module for predictive maintenance.
Implements the selected Random Forest model for production.

Author: Brayan Cuevas
Created for technical challenge implementation.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, List, Any
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


class ModelTrainer:
    """Train the production Random Forest model for predictive maintenance."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.evaluation_results = None
        
    def prepare_model(self) -> RandomForestClassifier:
        """
        Prepare the production Random Forest model.
        Configuration based on notebook experiments.
        
        Returns:
            Configured RandomForestClassifier
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("Random Forest model prepared for training")
        return self.model
    
    def temporal_split(self, df: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal train/test split to avoid data leakage.
        
        Args:
            df: DataFrame with features and target
            feature_names: List of feature column names
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Sort by datetime for temporal split
        df_sorted = df.sort_values('datetime')
        
        # Use temporal split instead of random
        split_idx = int(len(df_sorted) * (1 - self.test_size))
        
        train_data = df_sorted.iloc[:split_idx]
        test_data = df_sorted.iloc[split_idx:]
        
        X_train = train_data[feature_names].fillna(0)
        X_test = test_data[feature_names].fillna(0)
        y_train = train_data['failure_within_window']
        y_test = test_data['failure_within_window']
        
        print(f"Temporal split completed:")
        print(f"  Training set: {len(X_train):,} samples ({y_train.mean()*100:.2f}% failures)")
        print(f"  Test set: {len(X_test):,} samples ({y_test.mean()*100:.2f}% failures)")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.prepare_model()
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Model training completed")
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Compile evaluation results
        evaluation = {
            'model_name': 'Random Forest',
            'auc_score': auc_score,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'classification_report': class_report,
            'business_metrics': {
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'miss_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            }
        }
        
        self.evaluation_results = evaluation
        
        print(f"Model evaluation completed:")
        print(f"  AUC Score: {auc_score:.4f}")
        print(f"  Precision: {evaluation['business_metrics']['precision']:.3f}")
        print(f"  Recall: {evaluation['business_metrics']['recall']:.3f}")
        
        return evaluation
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        importance_scores = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self, output_path: str, feature_names: List[str], 
                   evaluation: Dict, metadata: Dict = None) -> str:
        """
        Save the trained model with metadata.
        
        Args:
            output_path: Path to save the model
            feature_names: List of feature names
            evaluation: Evaluation results
            metadata: Additional metadata
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Prepare model package
        model_package = {
            'model': self.model,
            'model_name': 'Random Forest',
            'feature_names': feature_names,
            'feature_importance': self.get_feature_importance(),
            'evaluation': evaluation,
            'training_date': datetime.now().isoformat(),
            'metadata': metadata or {},
            'model_config': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'class_weight': 'balanced',
                'random_state': self.random_state
            }
        }
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model_package, output_path)
        
        print(f"Model saved to: {output_path}")
        return output_path
    
    def train_pipeline(self, df: pd.DataFrame, feature_names: List[str],
                      output_path: str = None) -> Tuple[Dict, str]:
        """
        Complete training pipeline for production model.
        
        Args:
            df: DataFrame with features and target
            feature_names: List of feature column names
            output_path: Path to save model (optional)
            
        Returns:
            Tuple of (evaluation_results, model_path)
        """
        print("Starting Random Forest training pipeline...")
        
        self.feature_names = feature_names
        
        # Create temporal split
        X_train, X_test, y_train, y_test = self.temporal_split(df, feature_names)
        
        # Prepare and train model
        self.prepare_model()
        self.train_model(X_train, y_train)
        
        # Evaluate model
        evaluation = self.evaluate_model(X_test, y_test)
        
        # Save model if path provided
        model_path = None
        if output_path:
            model_path = self.save_model(
                output_path, feature_names, evaluation,
                metadata={
                    'training_samples': len(X_train), 
                    'test_samples': len(X_test),
                    'data_split': 'temporal',
                    'test_size': self.test_size
                }
            )
        
        print("Training pipeline completed successfully!")
        
        return evaluation, model_path


def train_maintenance_model(df: pd.DataFrame, feature_names: List[str],
                          output_path: str = "models/baseline_model.joblib") -> Tuple[Dict, str]:
    """
    Convenience function to train maintenance prediction model.
    
    Args:
        df: DataFrame with features and target
        feature_names: List of feature column names
        output_path: Path to save the trained model
        
    Returns:
        Tuple of (evaluation_results, model_path)
    """
    trainer = ModelTrainer()
    return trainer.train_pipeline(df, feature_names, output_path)


if __name__ == "__main__":
    # Test the trainer
    from ..data import load_maintenance_data, create_maintenance_features
    
    print("Testing model trainer...")
    
    # Load and process data
    telemetry, failures = load_maintenance_data()
    processed_df, feature_names = create_maintenance_features(telemetry, failures)
    
    # Train model
    trainer = ModelTrainer()
    evaluation, model_path = trainer.train_pipeline(processed_df, feature_names)
    
    print(f"\nTraining test completed:")
    print(f"  Model: {evaluation['model_name']}")
    print(f"  AUC score: {evaluation['auc_score']:.4f}")
    print(f"  Precision: {evaluation['business_metrics']['precision']:.3f}")
    print(f"  Recall: {evaluation['business_metrics']['recall']:.3f}")
    
    # Show top features
    importance = trainer.get_feature_importance()
    print(f"\nTop 5 features:")
    for i, (feature, score) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feature}: {score:.4f}")