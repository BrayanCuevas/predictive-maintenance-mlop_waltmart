"""
Local Simulation of Vertex AI Pipeline.
Demonstrates how the current local pipeline maps to Vertex AI components
without requiring Google Cloud Platform access.

Author: Brayan Cuevas
Usage: python vertex_ai/local_simulation.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import existing local modules
try:
    from src.data import load_maintenance_data, create_maintenance_features
    from src.models import train_maintenance_model
    import pandas as pd
    import joblib
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


class VertexAISimulator:
    """Simulates Vertex AI pipeline execution using local code."""
    
    def __init__(self):
        """Initialize the simulator."""
        self.project_id = "demo-project"
        self.region = "us-central1"
        self.pipeline_run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Simulation state
        self.artifacts = {}
        self.metrics = {}
        
        print("=" * 60)
        print("VERTEX AI PIPELINE SIMULATION")
        print("=" * 60)
        print(f"Project ID: {self.project_id}")
        print(f"Region: {self.region}")
        print(f"Pipeline Run ID: {self.pipeline_run_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
    
    def simulate_component(self, component_name: str, func, *args, **kwargs):
        """
        Simulate a Vertex AI component execution.
        
        Args:
            component_name: Name of the component
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Component execution results
        """
        print(f"EXECUTING COMPONENT: {component_name}")
        print(f"   Container: python:3.9")
        print(f"   Resource: n1-standard-4 (4 vCPUs, 15GB RAM)")
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            print(f"   SUCCESS - Execution time: {execution_time:.2f}s")
            print(f"   Output: {type(result).__name__}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   FAILED - Execution time: {execution_time:.2f}s")
            print(f"   Error: {e}")
            raise
        finally:
            print()
    
    def data_ingestion_component(self) -> Dict[str, Any]:
        """
        Simulate Vertex AI Data Ingestion Component.
        Maps to: GCS data loading
        """
        def load_data():
            # Simulate GCS data loading with local file loading
            print("     Downloading from GCS: gs://bucket/raw/PdM_telemetry.csv")
            print("     Downloading from GCS: gs://bucket/raw/PdM_failures.csv")
            
            telemetry, failures = load_maintenance_data()
            
            # Simulate saving to component output artifacts
            stats = {
                "telemetry_records": len(telemetry),
                "failure_records": len(failures),
                "unique_machines": telemetry['machineID'].nunique(),
                "date_range_days": (telemetry['datetime'].max() - telemetry['datetime'].min()).days
            }
            
            print(f"     Telemetry records: {stats['telemetry_records']:,}")
            print(f"     Failure records: {stats['failure_records']:,}")
            print(f"     Unique machines: {stats['unique_machines']}")
            print(f"     Date range: {stats['date_range_days']} days")
            
            # Store in simulation artifacts
            self.artifacts['telemetry_data'] = telemetry
            self.artifacts['failures_data'] = failures
            
            return stats
        
        return self.simulate_component("data-ingestion", load_data)
    
    def feature_engineering_component(self, prediction_window_days: int = 3) -> Dict[str, Any]:
        """
        Simulate Vertex AI Feature Engineering Component.
        Maps to: src.data.feature_engineering
        """
        def create_features():
            print(f"     Creating rolling window features...")
            print(f"     Prediction window: {prediction_window_days} days")
            print(f"     Sensor columns: ['volt', 'rotate', 'pressure', 'vibration']")
            print(f"     Rolling windows: [3, 24] hours")
            
            telemetry = self.artifacts['telemetry_data']
            failures = self.artifacts['failures_data']
            
            processed_df, feature_names = create_maintenance_features(
                telemetry, failures, prediction_window_days
            )
            
            stats = {
                "total_features": len(feature_names),
                "total_records": len(processed_df),
                "failure_rate": processed_df['failure_within_window'].mean(),
                "feature_names_sample": feature_names[:5]
            }
            
            print(f"     Total features: {stats['total_features']}")
            print(f"     Total records: {stats['total_records']:,}")
            print(f"     Failure rate: {stats['failure_rate']:.4f}")
            print(f"     Sample features: {stats['feature_names_sample']}")
            
            # Store in simulation artifacts
            self.artifacts['processed_data'] = processed_df
            self.artifacts['feature_names'] = feature_names
            
            return stats
        
        return self.simulate_component("feature-engineering", create_features)
    
    def model_training_component(self) -> Dict[str, Any]:
        """
        Simulate Vertex AI Model Training Component.
        Maps to: src.models.trainer
        """
        def train_model():
            print(f"     Training Random Forest model...")
            print(f"     Algorithm: RandomForestClassifier")
            print(f"     Hyperparameters: n_estimators=100, max_depth=10")
            
            processed_df = self.artifacts['processed_data']
            feature_names = self.artifacts['feature_names']
            
            # Use temporary model path for simulation
            temp_model_path = "models/vertex_simulation_model.joblib"
            
            evaluation, model_path = train_maintenance_model(
                processed_df, feature_names, temp_model_path
            )
            
            stats = {
                "auc_score": evaluation['auc_score'],
                "precision": evaluation['business_metrics']['precision'],
                "recall": evaluation['business_metrics']['recall'],
                "training_samples": evaluation.get('metadata', {}).get('training_samples', 0),
                "model_path": model_path
            }
            
            print(f"     AUC Score: {stats['auc_score']:.4f}")
            print(f"     Precision: {stats['precision']:.4f}")
            print(f"     Recall: {stats['recall']:.4f}")
            print(f"     Training samples: {stats['training_samples']:,}")
            
            # Store in simulation artifacts
            self.artifacts['trained_model'] = evaluation
            self.artifacts['model_path'] = model_path
            
            return stats
        
        return self.simulate_component("model-training", train_model)
    
    def model_evaluation_component(self, auc_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Simulate Vertex AI Model Evaluation Component.
        Maps to: Model quality gates and deployment decisions
        """
        def evaluate_model():
            print(f"     Evaluating model against deployment criteria...")
            print(f"     AUC Threshold: {auc_threshold}")
            print(f"     Precision Threshold: 0.20")
            print(f"     Recall Threshold: 0.50")
            
            evaluation = self.artifacts['trained_model']
            
            auc_score = evaluation['auc_score']
            precision = evaluation['business_metrics']['precision']
            recall = evaluation['business_metrics']['recall']
            
            # Deployment decision logic
            auc_pass = auc_score >= auc_threshold
            precision_pass = precision >= 0.20
            recall_pass = recall >= 0.50
            
            deploy_ready = auc_pass and precision_pass and recall_pass
            
            evaluation_result = {
                "auc_score": auc_score,
                "precision": precision,
                "recall": recall,
                "auc_threshold": auc_threshold,
                "auc_pass": auc_pass,
                "precision_pass": precision_pass,
                "recall_pass": recall_pass,
                "deploy_ready": deploy_ready,
                "recommendation": "APPROVE" if deploy_ready else "REJECT"
            }
            
            print(f"     AUC Check: {auc_score:.4f} >= {auc_threshold} = {'PASS' if auc_pass else 'FAIL'}")
            print(f"     Precision Check: {precision:.4f} >= 0.20 = {'PASS' if precision_pass else 'FAIL'}")
            print(f"     Recall Check: {recall:.4f} >= 0.50 = {'PASS' if recall_pass else 'FAIL'}")
            print(f"     Final Decision: {evaluation_result['recommendation']}")
            
            # Store evaluation results
            self.artifacts['evaluation_results'] = evaluation_result
            
            return evaluation_result
        
        return self.simulate_component("model-evaluation", evaluate_model)
    
    def conditional_deployment_component(self) -> Dict[str, Any]:
        """
        Simulate Vertex AI Conditional Deployment Component.
        Maps to: Vertex AI Endpoint deployment with traffic management
        """
        def deploy_model():
            evaluation_results = self.artifacts['evaluation_results']
            
            if not evaluation_results['deploy_ready']:
                print("     Deployment skipped - Model did not pass evaluation")
                return {
                    "deployed": False,
                    "reason": "Model failed evaluation criteria",
                    "auc_score": evaluation_results['auc_score']
                }
            
            print("     Deploying to Vertex AI Endpoint...")
            print("     Endpoint: predictive-maintenance-endpoint")
            print("     Machine Type: n1-standard-2")
            print("     Min Replicas: 1, Max Replicas: 3")
            print("     Traffic Split: 100% to new model")
            
            # Simulate deployment process
            time.sleep(2)  # Simulate deployment time
            
            deployment_info = {
                "deployed": True,
                "endpoint_name": "predictive-maintenance-endpoint",
                "model_version": f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "machine_type": "n1-standard-2",
                "replicas": {"min": 1, "max": 3},
                "traffic_percentage": 100,
                "auc_score": evaluation_results['auc_score'],
                "deployment_time": datetime.now().isoformat()
            }
            
            print(f"     Model Version: {deployment_info['model_version']}")
            print(f"     Endpoint URL: https://us-central1-aiplatform.googleapis.com/v1/projects/demo-project/locations/us-central1/endpoints/123456789")
            print("     Deployment Status: SUCCESS")
            
            return deployment_info
        
        return self.simulate_component("conditional-deployment", deploy_model)
    
    def monitoring_setup_component(self) -> Dict[str, Any]:
        """
        Simulate Vertex AI Model Monitoring Setup.
        Maps to: Data drift detection and model performance monitoring
        """
        def setup_monitoring():
            print("     Configuring model monitoring...")
            print("     Data Drift Detection: Enabled")
            print("     Feature Monitoring: ['volt', 'rotate', 'pressure', 'vibration']")
            print("     Drift Thresholds: 0.3 for all features")
            print("     Monitoring Schedule: Every 6 hours")
            print("     Alert Emails: maintenance-team@company.com")
            
            monitoring_config = {
                "monitoring_job_id": f"monitoring-{self.pipeline_run_id}",
                "drift_detection": True,
                "monitored_features": ["volt", "rotate", "pressure", "vibration"],
                "drift_thresholds": {"volt": 0.3, "rotate": 0.3, "pressure": 0.3, "vibration": 0.3},
                "schedule": "0 */6 * * *",
                "alert_policies": [
                    "High Data Drift Alert",
                    "Low Prediction Volume Alert", 
                    "High Error Rate Alert"
                ],
                "status": "ACTIVE"
            }
            
            print(f"     Monitoring Job ID: {monitoring_config['monitoring_job_id']}")
            print(f"     Alert Policies: {len(monitoring_config['alert_policies'])} configured")
            
            return monitoring_config
        
        return self.simulate_component("monitoring-setup", setup_monitoring)
    
    def run_complete_pipeline(self):
        """Execute the complete Vertex AI pipeline simulation."""
        
        print("Starting Vertex AI Pipeline Execution...")
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Data Ingestion
            data_stats = self.data_ingestion_component()
            self.metrics['data_ingestion'] = data_stats
            
            # Step 2: Feature Engineering
            feature_stats = self.feature_engineering_component(prediction_window_days=3)
            self.metrics['feature_engineering'] = feature_stats
            
            # Step 3: Model Training
            training_stats = self.model_training_component()
            self.metrics['model_training'] = training_stats
            
            # Step 4: Model Evaluation
            evaluation_stats = self.model_evaluation_component(auc_threshold=0.75)
            self.metrics['model_evaluation'] = evaluation_stats
            
            # Step 5: Conditional Deployment
            deployment_stats = self.conditional_deployment_component()
            self.metrics['deployment'] = deployment_stats
            
            # Step 6: Monitoring Setup
            monitoring_stats = self.monitoring_setup_component()
            self.metrics['monitoring'] = monitoring_stats
            
            # Pipeline Summary
            pipeline_execution_time = time.time() - pipeline_start_time
            
            print("=" * 60)
            print("PIPELINE EXECUTION COMPLETED")
            print("=" * 60)
            print(f"Total Execution Time: {pipeline_execution_time:.2f} seconds")
            print(f"Pipeline Status: SUCCESS")
            print()
            
            self.print_pipeline_summary()
            
        except Exception as e:
            pipeline_execution_time = time.time() - pipeline_start_time
            print("=" * 60)
            print("PIPELINE EXECUTION FAILED")
            print("=" * 60)
            print(f"Total Execution Time: {pipeline_execution_time:.2f} seconds")
            print(f"Error: {e}")
            raise
    
    def print_pipeline_summary(self):
        """Print a summary of the pipeline execution."""
        
        print("PIPELINE SUMMARY")
        print("=" * 30)
        
        # Data Summary
        data_stats = self.metrics.get('data_ingestion', {})
        print(f"Data Records: {data_stats.get('telemetry_records', 'N/A'):,}")
        print(f"Failure Events: {data_stats.get('failure_records', 'N/A'):,}")
        print(f"Unique Machines: {data_stats.get('unique_machines', 'N/A')}")
        
        # Feature Summary
        feature_stats = self.metrics.get('feature_engineering', {})
        print(f"Total Features: {feature_stats.get('total_features', 'N/A')}")
        print(f"Failure Rate: {feature_stats.get('failure_rate', 0):.4f}")
        
        # Model Summary
        training_stats = self.metrics.get('model_training', {})
        print(f"Model AUC: {training_stats.get('auc_score', 'N/A'):.4f}")
        print(f"Precision: {training_stats.get('precision', 'N/A'):.4f}")
        print(f"Recall: {training_stats.get('recall', 'N/A'):.4f}")
        
        # Deployment Summary
        deployment_stats = self.metrics.get('deployment', {})
        deployed = deployment_stats.get('deployed', False)
        print(f"Deployment Status: {'SUCCESS' if deployed else 'SKIPPED'}")
        
        if deployed:
            print(f"Endpoint: {deployment_stats.get('endpoint_name', 'N/A')}")
            print(f"Model Version: {deployment_stats.get('model_version', 'N/A')}")
        
        


def main():
    """Main function to run the simulation."""
    print("Vertex AI Pipeline Local Simulation")
    print("Validating migration strategy without cloud costs")
    print()
    
    # Check if data is available
    try:
        from src.data import load_maintenance_data
        telemetry, failures = load_maintenance_data()
        print(f"Data validation: Found {len(telemetry):,} telemetry records")
        print()
    except Exception as e:
        print(f"Error: Cannot load data - {e}")
        print("Please ensure data files are in data/raw/ directory")
        return 1
    
    # Run simulation
    try:
        simulator = VertexAISimulator()
        simulator.run_complete_pipeline()
        
        print("Simulation completed successfully!")
        print("The local pipeline is ready for Vertex AI migration.")
        return 0
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)