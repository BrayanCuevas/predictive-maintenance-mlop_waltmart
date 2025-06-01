"""
Vertex AI Model Monitoring Configuration.
Sets up data drift detection, model performance monitoring, and alerting.

Author: Brayan Cuevas
"""

from google.cloud import aiplatform
from google.cloud.monitoring_v3 import AlertPolicyServiceClient, NotificationChannelServiceClient
from typing import Dict, List, Optional
import logging
import json


class VertexAIMonitoring:
    """Manages Vertex AI model monitoring and alerting."""
    
    def __init__(self, project_id: str, region: str):
        """
        Initialize monitoring manager.
        
        Args:
            project_id: GCP project ID
            region: GCP region
        """
        self.project_id = project_id
        self.region = region
        
        aiplatform.init(project=project_id, location=region)
        self.logger = logging.getLogger(__name__)
    
    def setup_model_monitoring(self,
                              endpoint: aiplatform.Endpoint,
                              training_dataset_uri: str,
                              monitoring_config: Optional[Dict] = None) -> Dict:
        """
        Configure model monitoring for data drift and performance.
        
        Args:
            endpoint: Vertex AI endpoint to monitor
            training_dataset_uri: GCS URI of training dataset for baseline
            monitoring_config: Optional monitoring configuration
            
        Returns:
            Dictionary with monitoring job information
        """
        try:
            if monitoring_config is None:
                monitoring_config = self._get_default_monitoring_config()
            
            self.logger.info(f"Setting up monitoring for endpoint: {endpoint.display_name}")
            
            # Create monitoring job
            monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
                display_name=f"monitoring-{endpoint.display_name}",
                endpoint=endpoint,
                logging_sampling_strategy=aiplatform.SamplingStrategy(
                    random_sample_config=aiplatform.SamplingStrategy.RandomSampleConfig(
                        sample_rate=monitoring_config.get("sampling_rate", 0.1)
                    )
                ),
                schedule_config=aiplatform.ScheduleConfig(
                    cron_schedule=monitoring_config.get("schedule", "0 */6 * * *")  # Every 6 hours
                ),
                alert_config=aiplatform.EmailAlertConfig(
                    user_emails=monitoring_config.get("alert_emails", [])
                ),
                # Data drift detection
                drift_detection_config=self._create_drift_config(
                    training_dataset_uri, 
                    monitoring_config.get("drift_thresholds", {})
                ),
                # Prediction drift detection
                explanation_config=aiplatform.ExplanationConfig(
                    enable_feature_attributes=True
                )
            )
            
            monitoring_info = {
                "monitoring_job_id": monitoring_job.resource_name,
                "endpoint_id": endpoint.resource_name,
                "sampling_rate": monitoring_config.get("sampling_rate", 0.1),
                "schedule": monitoring_config.get("schedule"),
                "status": "ACTIVE"
            }
            
            self.logger.info(f"Model monitoring configured: {monitoring_info}")
            return monitoring_info
            
        except Exception as e:
            self.logger.error(f"Failed to setup model monitoring: {e}")
            raise
    
    def _get_default_monitoring_config(self) -> Dict:
        """Get default monitoring configuration for predictive maintenance."""
        return {
            "sampling_rate": 0.1,  # Monitor 10% of predictions
            "schedule": "0 */6 * * *",  # Every 6 hours
            "drift_thresholds": {
                "volt": 0.3,
                "rotate": 0.3,
                "pressure": 0.3,
                "vibration": 0.3
            },
            "performance_thresholds": {
                "prediction_drift": 0.2,
                "feature_attribution_drift": 0.3
            },
            "alert_emails": ["maintenance-team@company.com"]
        }
    
    def _create_drift_config(self, 
                            training_dataset_uri: str,
                            drift_thresholds: Dict[str, float]) -> aiplatform.DriftDetectionConfig:
        """
        Create data drift detection configuration.
        
        Args:
            training_dataset_uri: GCS URI of training dataset
            drift_thresholds: Thresholds for each feature
            
        Returns:
            Drift detection configuration
        """
        feature_configs = []
        
        # Configure drift detection for each sensor
        sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
        
        for feature in sensor_features:
            threshold = drift_thresholds.get(feature, 0.3)
            
            feature_config = aiplatform.DriftDetectionConfig.FeatureConfig(
                feature_name=feature,
                drift_threshold=threshold,
                attribution_score_drift_threshold=threshold * 1.5
            )
            feature_configs.append(feature_config)
        
        return aiplatform.DriftDetectionConfig(
            drift_thresholds=feature_configs,
            training_dataset=aiplatform.Dataset(training_dataset_uri)
        )
    
    def create_alerting_policies(self, monitoring_job_id: str) -> List[Dict]:
        """
        Create Cloud Monitoring alert policies for model monitoring.
        
        Args:
            monitoring_job_id: Model monitoring job resource ID
            
        Returns:
            List of created alert policies
        """
        try:
            client = AlertPolicyServiceClient()
            project_name = f"projects/{self.project_id}"
            
            alert_policies = []
            
            # Alert 1: Data Drift Detection
            drift_policy = {
                "display_name": "Predictive Maintenance - Data Drift Alert",
                "conditions": [
                    {
                        "display_name": "High data drift detected",
                        "condition_threshold": {
                            "filter": f'resource.type="vertex_ai_model_deployment_monitoring_job" AND resource.labels.job_id="{monitoring_job_id}"',
                            "comparison": "COMPARISON_GREATER_THAN",
                            "threshold_value": 0.3,
                            "duration": "300s",
                            "aggregations": [
                                {
                                    "alignment_period": "300s",
                                    "per_series_aligner": "ALIGN_MEAN"
                                }
                            ]
                        }
                    }
                ],
                "alert_strategy": {
                    "auto_close": "1800s"  # Auto-close after 30 minutes
                },
                "enabled": True
            }
            
            # Alert 2: Prediction Volume Drop
            volume_policy = {
                "display_name": "Predictive Maintenance - Low Prediction Volume",
                "conditions": [
                    {
                        "display_name": "Prediction volume below threshold",
                        "condition_threshold": {
                            "filter": f'resource.type="vertex_ai_endpoint" AND metric.type="aiplatform.googleapis.com/prediction/online/prediction_count"',
                            "comparison": "COMPARISON_LESS_THAN",
                            "threshold_value": 10,  # Less than 10 predictions per hour
                            "duration": "3600s",
                            "aggregations": [
                                {
                                    "alignment_period": "3600s",
                                    "per_series_aligner": "ALIGN_RATE"
                                }
                            ]
                        }
                    }
                ]
            }
            
            # Alert 3: High Error Rate
            error_policy = {
                "display_name": "Predictive Maintenance - High Error Rate",
                "conditions": [
                    {
                        "display_name": "Prediction error rate above threshold",
                        "condition_threshold": {
                            "filter": f'resource.type="vertex_ai_endpoint" AND metric.type="aiplatform.googleapis.com/prediction/online/error_count"',
                            "comparison": "COMPARISON_GREATER_THAN",
                            "threshold_value": 0.05,  # 5% error rate
                            "duration": "600s",
                            "aggregations": [
                                {
                                    "alignment_period": "300s",
                                    "per_series_aligner": "ALIGN_RATE"
                                }
                            ]
                        }
                    }
                ]
            }
            
            policies = [drift_policy, volume_policy, error_policy]
            
            for policy_config in policies:
                policy = client.create_alert_policy(
                    name=project_name,
                    alert_policy=policy_config
                )
                alert_policies.append({
                    "policy_id": policy.name,
                    "display_name": policy.display_name
                })
                
                self.logger.info(f"Created alert policy: {policy.display_name}")
            
            return alert_policies
            
        except Exception as e:
            self.logger.error(f"Failed to create alert policies: {e}")
            raise
    
    def create_monitoring_dashboard(self) -> Dict:
        """
        Create Cloud Monitoring dashboard for model metrics.
        
        Returns:
            Dictionary with dashboard information
        """
        dashboard_config = {
            "displayName": "Predictive Maintenance - Model Monitoring",
            "mosaicLayout": {
                "tiles": [
                    {
                        "width": 6,
                        "height": 4,
                        "widget": {
                            "title": "Prediction Volume",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": 'resource.type="vertex_ai_endpoint" AND metric.type="aiplatform.googleapis.com/prediction/online/prediction_count"',
                                                "aggregation": {
                                                    "alignmentPeriod": "300s",
                                                    "perSeriesAligner": "ALIGN_RATE"
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    {
                        "width": 6,
                        "height": 4,
                        "xPos": 6,
                        "widget": {
                            "title": "Prediction Latency",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": 'resource.type="vertex_ai_endpoint" AND metric.type="aiplatform.googleapis.com/prediction/online/response_latencies"',
                                                "aggregation": {
                                                    "alignmentPeriod": "300s",
                                                    "perSeriesAligner": "ALIGN_MEAN"
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    },
                    {
                        "width": 12,
                        "height": 4,
                        "yPos": 4,
                        "widget": {
                            "title": "Data Drift Metrics",
                            "xyChart": {
                                "dataSets": [
                                    {
                                        "timeSeriesQuery": {
                                            "timeSeriesFilter": {
                                                "filter": 'resource.type="vertex_ai_model_deployment_monitoring_job" AND metric.type="aiplatform.googleapis.com/model_deployment_monitoring/drift_detection"',
                                                "aggregation": {
                                                    "alignmentPeriod": "3600s",
                                                    "perSeriesAligner": "ALIGN_MEAN"
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        }
        
        return {
            "dashboard_config": dashboard_config,
            "status": "READY_FOR_CREATION"
        }


def setup_model_monitoring(project_id: str,
                          region: str,
                          endpoint_name: str,
                          training_dataset_uri: str) -> Dict:
    """
    Utility function to setup complete model monitoring.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        endpoint_name: Name of the endpoint to monitor
        training_dataset_uri: GCS URI of training dataset
        
    Returns:
        Dictionary with monitoring setup information
    """
    monitoring_manager = VertexAIMonitoring(project_id, region)
    
    # Get endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    if not endpoints:
        raise ValueError(f"Endpoint {endpoint_name} not found")
    
    endpoint = endpoints[0]
    
    # Setup monitoring
    monitoring_info = monitoring_manager.setup_model_monitoring(
        endpoint=endpoint,
        training_dataset_uri=training_dataset_uri
    )
    
    # Create alert policies
    alert_policies = monitoring_manager.create_alerting_policies(
        monitoring_info["monitoring_job_id"]
    )
    
    return {
        "monitoring": monitoring_info,
        "alerts": alert_policies
    }


def create_monitoring_dashboard(project_id: str) -> Dict:
    """
    Utility function to create monitoring dashboard.
    
    Args:
        project_id: GCP project ID
        
    Returns:
        Dashboard configuration
    """
    monitoring_manager = VertexAIMonitoring(project_id, "us-central1")
    return monitoring_manager.create_monitoring_dashboard()


if __name__ == "__main__":
    # Example usage
    PROJECT_ID = "your-project-id"
    REGION = "us-central1"
    ENDPOINT_NAME = "predictive-maintenance-endpoint"
    TRAINING_DATA_URI = "gs://your-bucket/training-data/baseline.csv"
    
    # Setup monitoring
    monitoring_result = setup_model_monitoring(
        PROJECT_ID, REGION, ENDPOINT_NAME, TRAINING_DATA_URI
    )
    
    print(f"Monitoring setup completed: {monitoring_result}")