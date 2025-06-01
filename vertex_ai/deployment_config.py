"""
Vertex AI Model Deployment Configuration.
Handles endpoint creation, model deployment, and traffic management.

Author: Brayan Cuevas
"""

from google.cloud import aiplatform
from typing import Dict, List, Optional
import logging


class VertexAIDeployment:
    """Manages Vertex AI model deployment and endpoints."""
    
    def __init__(self, project_id: str, region: str):
        """
        Initialize deployment manager.
        
        Args:
            project_id: GCP project ID
            region: GCP region for deployment
        """
        self.project_id = project_id
        self.region = region
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        self.logger = logging.getLogger(__name__)
    
    def create_endpoint(self, 
                       endpoint_name: str,
                       description: str = "Predictive maintenance endpoint") -> aiplatform.Endpoint:
        """
        Create or get existing Vertex AI endpoint.
        
        Args:
            endpoint_name: Display name for the endpoint
            description: Description of the endpoint
            
        Returns:
            Vertex AI Endpoint object
        """
        try:
            # Try to find existing endpoint
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            
            if endpoints:
                self.logger.info(f"Using existing endpoint: {endpoint_name}")
                return endpoints[0]
            else:
                self.logger.info(f"Creating new endpoint: {endpoint_name}")
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_name,
                    description=description,
                    labels={
                        "use_case": "predictive_maintenance",
                        "model_type": "random_forest",
                        "deployment_type": "production"
                    }
                )
                return endpoint
                
        except Exception as e:
            self.logger.error(f"Failed to create endpoint: {e}")
            raise
    
    def deploy_model(self,
                    model: aiplatform.Model,
                    endpoint: aiplatform.Endpoint,
                    traffic_percentage: int = 100,
                    machine_type: str = "n1-standard-2",
                    min_replica_count: int = 1,
                    max_replica_count: int = 3) -> Dict:
        """
        Deploy model to endpoint with specified configuration.
        
        Args:
            model: Vertex AI Model to deploy
            endpoint: Target endpoint
            traffic_percentage: Percentage of traffic for this model
            machine_type: Machine type for serving
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            
        Returns:
            Dictionary with deployment information
        """
        try:
            deployed_model_display_name = f"predictive-maintenance-{model.create_time.strftime('%Y%m%d-%H%M%S')}"
            
            self.logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")
            
            # Deploy model to endpoint
            endpoint.deploy(
                model=model,
                deployed_model_display_name=deployed_model_display_name,
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                traffic_percentage=traffic_percentage,
                deploy_request_timeout=1800,  # 30 minutes timeout
                service_account=None  # Use default service account
            )
            
            deployment_info = {
                "model_id": model.resource_name,
                "endpoint_id": endpoint.resource_name,
                "deployed_model_name": deployed_model_display_name,
                "machine_type": machine_type,
                "min_replicas": min_replica_count,
                "max_replicas": max_replica_count,
                "traffic_percentage": traffic_percentage,
                "deployment_status": "SUCCESS"
            }
            
            self.logger.info(f"Model deployed successfully: {deployment_info}")
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            raise
    
    def setup_traffic_split(self,
                           endpoint: aiplatform.Endpoint,
                           model_traffic_map: Dict[str, int]) -> Dict:
        """
        Configure traffic splitting between multiple models.
        
        Args:
            endpoint: Target endpoint
            model_traffic_map: Map of model IDs to traffic percentages
            
        Returns:
            Dictionary with traffic configuration
        """
        try:
            self.logger.info(f"Configuring traffic split: {model_traffic_map}")
            
            # Update traffic allocation
            endpoint.update_traffic(model_traffic_map)
            
            # Verify configuration
            updated_endpoint = aiplatform.Endpoint(endpoint.resource_name)
            current_traffic = {
                deployed_model.id: deployed_model.traffic_percentage
                for deployed_model in updated_endpoint.list_models()
            }
            
            self.logger.info(f"Traffic split configured: {current_traffic}")
            return {"traffic_configuration": current_traffic}
            
        except Exception as e:
            self.logger.error(f"Traffic split configuration failed: {e}")
            raise
    
    def blue_green_deployment(self,
                             endpoint: aiplatform.Endpoint,
                             new_model: aiplatform.Model,
                             validation_traffic: int = 10) -> Dict:
        """
        Perform blue-green deployment with gradual traffic shift.
        
        Args:
            endpoint: Target endpoint
            new_model: New model version to deploy
            validation_traffic: Initial traffic percentage for new model
            
        Returns:
            Dictionary with deployment steps
        """
        try:
            self.logger.info("Starting blue-green deployment")
            
            # Step 1: Deploy new model with minimal traffic
            self.deploy_model(
                model=new_model,
                endpoint=endpoint,
                traffic_percentage=validation_traffic
            )
            
            # Step 2: Get current models
            deployed_models = endpoint.list_models()
            
            # Step 3: Adjust traffic (new model gets validation_traffic, others split remainder)
            total_existing_traffic = 100 - validation_traffic
            existing_models = [m for m in deployed_models if m.model != new_model.resource_name]
            
            traffic_map = {}
            if existing_models:
                traffic_per_existing = total_existing_traffic // len(existing_models)
                for model in existing_models:
                    traffic_map[model.id] = traffic_per_existing
            
            # Add new model
            new_deployed_model = [m for m in deployed_models if m.model == new_model.resource_name][0]
            traffic_map[new_deployed_model.id] = validation_traffic
            
            # Apply traffic split
            self.setup_traffic_split(endpoint, traffic_map)
            
            deployment_result = {
                "deployment_type": "blue_green",
                "new_model_id": new_model.resource_name,
                "validation_traffic": validation_traffic,
                "traffic_map": traffic_map,
                "status": "VALIDATION_PHASE"
            }
            
            self.logger.info(f"Blue-green deployment initiated: {deployment_result}")
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            raise


def create_vertex_endpoint(project_id: str, 
                          region: str,
                          endpoint_name: str = "predictive-maintenance-endpoint") -> aiplatform.Endpoint:
    """
    Utility function to create Vertex AI endpoint.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        endpoint_name: Name for the endpoint
        
    Returns:
        Created or existing endpoint
    """
    deployment_manager = VertexAIDeployment(project_id, region)
    return deployment_manager.create_endpoint(endpoint_name)


def deploy_model_to_endpoint(project_id: str,
                            region: str,
                            model_id: str,
                            endpoint_name: str,
                            deployment_config: Optional[Dict] = None) -> Dict:
    """
    Utility function to deploy model to endpoint.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        model_id: Vertex AI model resource ID
        endpoint_name: Target endpoint name
        deployment_config: Optional deployment configuration
        
    Returns:
        Deployment information
    """
    deployment_manager = VertexAIDeployment(project_id, region)
    
    # Get model and endpoint
    model = aiplatform.Model(model_id)
    endpoint = deployment_manager.create_endpoint(endpoint_name)
    
    # Use default config if none provided
    if deployment_config is None:
        deployment_config = {
            "machine_type": "n1-standard-2",
            "min_replica_count": 1,
            "max_replica_count": 3,
            "traffic_percentage": 100
        }
    
    return deployment_manager.deploy_model(model, endpoint, **deployment_config)


if __name__ == "__main__":
    # Example usage
    PROJECT_ID = "your-project-id"
    REGION = "us-central1"
    
    # Create deployment manager
    deployment = VertexAIDeployment(PROJECT_ID, REGION)
    
    # Create endpoint
    endpoint = deployment.create_endpoint("predictive-maintenance-demo")
    
    print(f"Endpoint created: {endpoint.resource_name}")