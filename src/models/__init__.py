"""
Model training and prediction modules - Brayan Cuevas Implementation
Handles model training, evaluation and inference for maintenance prediction.
"""

from .trainer import ModelTrainer, train_maintenance_model

__all__ = ['ModelTrainer', 'train_maintenance_model']