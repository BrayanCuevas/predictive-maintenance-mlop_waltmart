"""
End-to-end training pipeline for predictive maintenance model.
Orchestrates data loading, feature engineering, model training, and evaluation.

Author: Brayan Cuevas
Created for technical challenge implementation.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data import load_maintenance_data, create_maintenance_features
from src.models import train_maintenance_model


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def validate_environment():
    """Validate that all required files and directories exist."""
    logger = logging.getLogger(__name__)
    
    required_files = [
        "data/raw/PdM_telemetry.csv",
        "data/raw/PdM_failures.csv",
        "src/data/data_loader.py",
        "src/data/feature_engineering.py",
        "src/models/trainer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        raise FileNotFoundError("Required files missing")
    
    # Create output directories
    Path("models").mkdir(exist_ok=True)
    Path("models/registry").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Environment validation completed successfully")


def load_and_process_data(prediction_window_days: int = 3):
    """
    Load raw data and create features.
    
    Args:
        prediction_window_days: Days ahead to predict failures
        
    Returns:
        Tuple of (processed_dataframe, feature_names)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data loading and processing...")
    
    # Load raw data
    logger.info("Loading raw maintenance data...")
    telemetry, failures = load_maintenance_data()
    
    logger.info(f"Loaded {len(telemetry):,} telemetry records")
    logger.info(f"Loaded {len(failures)} failure events")
    
    # Create features
    logger.info(f"Creating features with {prediction_window_days}-day prediction window...")
    processed_df, feature_names = create_maintenance_features(
        telemetry, failures, prediction_window_days
    )
    
    logger.info(f"Feature engineering completed:")
    logger.info(f"  Total features: {len(feature_names)}")
    logger.info(f"  Total records: {len(processed_df):,}")
    logger.info(f"  Failure rate: {processed_df['failure_within_window'].mean()*100:.2f}%")
    
    return processed_df, feature_names


def train_and_evaluate_model(processed_df, feature_names, model_output_path: str):
    """
    Train model and evaluate performance.
    
    Args:
        processed_df: DataFrame with features and labels
        feature_names: List of feature column names
        model_output_path: Path to save trained model
        
    Returns:
        Dictionary with evaluation results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training and evaluation...")
    
    # Train model using the modular trainer
    evaluation, saved_model_path = train_maintenance_model(
        processed_df, feature_names, model_output_path
    )
    
    # Log results
    logger.info("Model training completed successfully!")
    logger.info(f"Model saved to: {saved_model_path}")
    logger.info(f"Performance metrics:")
    logger.info(f"  AUC Score: {evaluation['auc_score']:.4f}")
    logger.info(f"  Precision: {evaluation['business_metrics']['precision']:.3f}")
    logger.info(f"  Recall: {evaluation['business_metrics']['recall']:.3f}")
    logger.info(f"  False Alarm Rate: {evaluation['business_metrics']['false_alarm_rate']:.3f}")
    
    return evaluation


def register_trained_model(evaluation: Dict, feature_names: List[str], 
                          model_path: str, version: str = None) -> str:
    """Register newly trained model in registry."""
    from src.models.model_registry import model_registry
    
    logger = logging.getLogger(__name__)
    
    if version is None:
        # Auto-generate version based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v1.0.{timestamp}"
    
    logger.info(f"Registering model as version {version}...")
    
    training_metadata = {
        'feature_count': len(feature_names),
        'training_samples': evaluation.get('metadata', {}).get('training_samples', 0),
        'test_samples': evaluation.get('metadata', {}).get('test_samples', 0)
    }
    
    success = model_registry.register_model(
        model_path=model_path,
        version=version,
        evaluation_results=evaluation,
        training_metadata=training_metadata,
        notes=f"Automated training run with {len(feature_names)} features"
    )
    
    if success:
        logger.info(f"Model registered as {version}")
        
        # Auto-evaluate for promotion
        logger.info("Evaluating model for automatic promotion...")
        promoted = model_registry.auto_evaluate_and_promote(
            min_auc_threshold=0.75,
            min_improvement=0.005
        )
        
        if promoted:
            logger.info(f"Model {promoted} automatically promoted to active")
        else:
            logger.info("Model registered as candidate, manual review recommended")
        
        # Show registry summary
        summary = model_registry.get_registry_summary()
        logger.info(f"Registry Summary:")
        logger.info(f"  Total models: {summary['total_models']}")
        logger.info(f"  Active models: {summary['active_models']}")
        logger.info(f"  Candidate models: {summary['candidate_models']}")
        logger.info(f"  Best AUC: {summary['best_auc']:.4f}")
    
    return version


def generate_summary_report(evaluation, feature_names, model_version: str = None, 
                          output_path: str = "reports/training_summary.txt"):
    """
    Generate a summary report of the training results.
    
    Args:
        evaluation: Model evaluation results
        feature_names: List of feature names
        model_version: Version of registered model
        output_path: Path to save the report
    """
    logger = logging.getLogger(__name__)
    
    # Create reports directory
    Path(output_path).parent.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""
PREDICTIVE MAINTENANCE MODEL TRAINING SUMMARY
============================================

Training Date: {timestamp}
Author: Brayan Cuevas
Model Version: {model_version or 'Not versioned'}

MODEL CONFIGURATION
------------------
Algorithm: Random Forest Classifier
Features: {len(feature_names)} engineered features
Prediction Window: 3 days ahead
Train/Test Split: Temporal (80/20)

PERFORMANCE METRICS
------------------
AUC Score: {evaluation['auc_score']:.4f}
Precision: {evaluation['business_metrics']['precision']:.3f}
Recall: {evaluation['business_metrics']['recall']:.3f}
False Alarm Rate: {evaluation['business_metrics']['false_alarm_rate']:.3f}
Miss Rate: {evaluation['business_metrics']['miss_rate']:.3f}

CONFUSION MATRIX
---------------
True Negatives: {evaluation['confusion_matrix']['true_negatives']:,}
False Positives: {evaluation['confusion_matrix']['false_positives']:,}
False Negatives: {evaluation['confusion_matrix']['false_negatives']:,}
True Positives: {evaluation['confusion_matrix']['true_positives']:,}

BUSINESS IMPACT
--------------
- Model detects {evaluation['business_metrics']['recall']*100:.1f}% of actual failures
- {evaluation['business_metrics']['precision']*100:.1f}% of predictions are correct
- {evaluation['business_metrics']['false_alarm_rate']*100:.1f}% false alarm rate

MODEL REGISTRY
--------------
- Version: {model_version or 'Not registered'}
- Registry Status: Available for deployment
- Auto-promotion: Based on performance thresholds

DELIVERABLES
-----------
Trained model: models/baseline_model.joblib
Versioned model: models/registry/models/model_{model_version or 'unknown'}.joblib
Training logs: logs/training_[timestamp].log
Summary report: {output_path}
"""

    # Save report
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Training summary report saved to: {output_path}")
    
    return output_path


def main():
    """Main training pipeline execution."""
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument("--prediction-days", type=int, default=3,
                       help="Days ahead to predict failures (default: 3)")
    parser.add_argument("--model-output", type=str, default="models/baseline_model.joblib",
                       help="Path to save trained model")
    parser.add_argument("--version", type=str, default=None,
                       help="Model version (auto-generated if not provided)")
    parser.add_argument("--register", action="store_true", default=True,
                       help="Register model in registry (default: True)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("="*60)
        logger.info("PREDICTIVE MAINTENANCE TRAINING PIPELINE STARTED")
        logger.info("="*60)
        
        # Validate environment
        validate_environment()
        
        # Load and process data
        processed_df, feature_names = load_and_process_data(args.prediction_days)
        
        # Train and evaluate model
        evaluation = train_and_evaluate_model(processed_df, feature_names, args.model_output)
        
        # Register model in registry
        model_version = None
        if args.register:
            model_version = register_trained_model(
                evaluation, feature_names, args.model_output, args.version
            )
        
        # Generate summary report
        report_path = generate_summary_report(evaluation, feature_names, model_version)
        
        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Model AUC: {evaluation['auc_score']:.4f}")
        logger.info(f"Model saved: {args.model_output}")
        if model_version:
            logger.info(f"Model version: {model_version}")
        logger.info(f"Report saved: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.error("Check logs for detailed error information")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)