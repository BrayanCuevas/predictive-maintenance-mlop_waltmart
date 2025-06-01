"""
Vertex AI Pipeline Definition for Predictive Maintenance MLOps.
Migrates the current local pipeline to Google Cloud Vertex AI.

Author: Brayan Cuevas
"""

from kfp.v2 import dsl
from kfp.v2.dsl import component, pipeline, Artifact, Output, Input, Dataset, Model
from google.cloud import aiplatform
import os


# Component 1: Data Ingestion
@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==1.5.3",
        "google-cloud-storage==2.10.0",
        "pyarrow==12.0.1"
    ]
)
def data_ingestion_component(
    project_id: str,
    bucket_name: str,
    telemetry_output: Output[Dataset],
    failures_output: Output[Dataset]
) -> dict:
    """
    Ingest telemetry and failure data from Google Cloud Storage.
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket containing raw data
        telemetry_output: Output artifact for telemetry data
        failures_output: Output artifact for failures data
        
    Returns:
        Dictionary with ingestion statistics
    """
    import pandas as pd
    from google.cloud import storage
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize GCS client
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    # Download telemetry data
    logger.info("Downloading telemetry data from GCS...")
    telemetry_blob = bucket.blob("raw/PdM_telemetry.csv")
    telemetry_content = telemetry_blob.download_as_text()
    
    # Download failures data
    logger.info("Downloading failures data from GCS...")
    failures_blob = bucket.blob("raw/PdM_failures.csv")
    failures_content = failures_blob.download_as_text()
    
    # Parse CSV data
    telemetry_df = pd.read_csv(pd.StringIO(telemetry_content))
    failures_df = pd.read_csv(pd.StringIO(failures_content))
    
    # Convert datetime columns
    telemetry_df['datetime'] = pd.to_datetime(telemetry_df['datetime'])
    failures_df['datetime'] = pd.to_datetime(failures_df['datetime'])
    
    # Save to output artifacts
    telemetry_df.to_parquet(telemetry_output.path, index=False)
    failures_df.to_parquet(failures_output.path, index=False)
    
    stats = {
        "telemetry_records": len(telemetry_df),
        "failure_records": len(failures_df),
        "unique_machines": telemetry_df['machineID'].nunique(),
        "date_range_days": (telemetry_df['datetime'].max() - telemetry_df['datetime'].min()).days
    }
    
    logger.info(f"Data ingestion completed: {stats}")
    return stats


# Component 2: Feature Engineering
@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==1.5.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0"
    ]
)
def feature_engineering_component(
    telemetry_data: Input[Dataset],
    failures_data: Input[Dataset],
    processed_data: Output[Dataset],
    prediction_window_days: int = 3
) -> dict:
    """
    Create rolling window features and failure labels.
    
    Args:
        telemetry_data: Input telemetry dataset
        failures_data: Input failures dataset
        processed_data: Output processed dataset with features
        prediction_window_days: Days ahead to predict failures
        
    Returns:
        Dictionary with feature engineering statistics
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load input data
    logger.info("Loading input datasets...")
    telemetry_df = pd.read_parquet(telemetry_data.path)
    failures_df = pd.read_parquet(failures_data.path)
    
    # Feature engineering parameters
    sensor_columns = ['volt', 'rotate', 'pressure', 'vibration']
    windows = [3, 24]  # hours
    
    logger.info("Creating rolling window features...")
    
    # Sort data
    telemetry_df = telemetry_df.sort_values(['machineID', 'datetime'])
    
    # Create rolling features
    for window in windows:
        logger.info(f"Processing {window}h window...")
        
        for sensor in sensor_columns:
            rolling_group = telemetry_df.groupby('machineID')[sensor].rolling(
                window=window, min_periods=1
            )
            
            telemetry_df[f'{sensor}_{window}h_mean'] = rolling_group.mean().reset_index(0, drop=True)
            telemetry_df[f'{sensor}_{window}h_std'] = rolling_group.std().reset_index(0, drop=True)
            telemetry_df[f'{sensor}_{window}h_max'] = rolling_group.max().reset_index(0, drop=True)
            telemetry_df[f'{sensor}_{window}h_min'] = rolling_group.min().reset_index(0, drop=True)
    
    # Fill NaN values
    telemetry_df = telemetry_df.ffill().fillna(0)
    
    # Create failure labels
    logger.info(f"Creating failure labels with {prediction_window_days}-day window...")
    
    telemetry_df['failure_within_window'] = 0
    window_hours = prediction_window_days * 24
    
    for _, failure in failures_df.iterrows():
        machine_id = failure['machineID']
        failure_time = pd.to_datetime(failure['datetime'])
        
        mask = (
            (telemetry_df['machineID'] == machine_id) &
            (telemetry_df['datetime'] >= failure_time - pd.Timedelta(hours=window_hours)) &
            (telemetry_df['datetime'] < failure_time)
        )
        
        telemetry_df.loc[mask, 'failure_within_window'] = 1
    
    # Get feature names
    exclude_cols = ['datetime', 'machineID', 'failure_within_window']
    feature_names = [col for col in telemetry_df.columns if col not in exclude_cols]
    
    # Save processed data
    telemetry_df.to_parquet(processed_data.path, index=False)
    
    stats = {
        "total_features": len(feature_names),
        "total_records": len(telemetry_df),
        "failure_rate": telemetry_df['failure_within_window'].mean(),
        "feature_names": feature_names[:10]  # First 10 for logging
    }
    
    logger.info(f"Feature engineering completed: {stats}")
    return stats


# Component 3: Model Training
@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==1.5.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "joblib==1.3.1"
    ]
)
def model_training_component(
    processed_data: Input[Dataset],
    trained_model: Output[Model],
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Train Random Forest model for failure prediction.
    
    Args:
        processed_data: Input processed dataset
        trained_model: Output trained model artifact
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with training results and metrics
    """
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    import joblib
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load processed data
    logger.info("Loading processed dataset...")
    df = pd.read_parquet(processed_data.path)
    
    # Prepare features and target
    exclude_cols = ['datetime', 'machineID', 'failure_within_window']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['failure_within_window']
    
    logger.info(f"Training with {len(feature_cols)} features on {len(df)} samples")
    logger.info(f"Failure rate: {y.mean():.4f}")
    
    # Temporal split (last 20% by time)
    df_sorted = df.sort_values('datetime')
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['failure_within_window']
    X_test = test_df[feature_cols]
    y_test = test_df['failure_within_window']
    
    # Train Random Forest model
    logger.info("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate business metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Save model
    model_package = {
        'model': model,
        'feature_names': feature_cols,
        'model_name': 'Random Forest',
        'auc_score': auc_score,
        'training_date': pd.Timestamp.now().isoformat(),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'business_metrics': {
            'precision': precision,
            'recall': recall,
            'false_alarm_rate': false_alarm_rate
        }
    }
    
    joblib.dump(model_package, trained_model.path)
    
    results = {
        "auc_score": float(auc_score),
        "precision": float(precision),
        "recall": float(recall),
        "false_alarm_rate": float(false_alarm_rate),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "top_features": [f"{name}: {importance:.4f}" for name, importance in top_features]
    }
    
    logger.info(f"Training completed. AUC: {auc_score:.4f}")
    return results


# Component 4: Model Evaluation
@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==1.5.3",
        "scikit-learn==1.3.0",
        "joblib==1.3.1"
    ]
)
def model_evaluation_component(
    trained_model: Input[Model],
    evaluation_results: Output[Artifact],
    auc_threshold: float = 0.75
) -> dict:
    """
    Evaluate trained model and determine deployment readiness.
    
    Args:
        trained_model: Input trained model
        evaluation_results: Output evaluation artifact
        auc_threshold: Minimum AUC score for deployment
        
    Returns:
        Dictionary with evaluation decision
    """
    import joblib
    import json
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info("Loading trained model for evaluation...")
    model_package = joblib.load(trained_model.path)
    
    auc_score = model_package['auc_score']
    precision = model_package['business_metrics']['precision']
    recall = model_package['business_metrics']['recall']
    
    # Evaluation criteria
    deploy_ready = (
        auc_score >= auc_threshold and
        precision >= 0.2 and  # At least 20% precision
        recall >= 0.5         # At least 50% recall
    )
    
    evaluation = {
        "auc_score": auc_score,
        "precision": precision,
        "recall": recall,
        "auc_threshold": auc_threshold,
        "deploy_ready": deploy_ready,
        "evaluation_date": pd.Timestamp.now().isoformat(),
        "deployment_recommendation": "APPROVE" if deploy_ready else "REJECT"
    }
    
    # Save evaluation results
    with open(evaluation_results.path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    logger.info(f"Model evaluation: {'APPROVED' if deploy_ready else 'REJECTED'}")
    logger.info(f"AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return evaluation


# Component 5: Model Deployment
@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform==1.34.0",
        "joblib==1.3.1"
    ]
)
def model_deployment_component(
    trained_model: Input[Model],
    evaluation_results: Input[Artifact],
    project_id: str,
    region: str,
    endpoint_display_name: str = "predictive-maintenance-endpoint"
) -> dict:
    """
    Deploy model to Vertex AI Endpoint if evaluation passes.
    
    Args:
        trained_model: Input trained model
        evaluation_results: Input evaluation results
        project_id: GCP project ID
        region: GCP region
        endpoint_display_name: Name for the endpoint
        
    Returns:
        Dictionary with deployment information
    """
    import json
    from google.cloud import aiplatform
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load evaluation results
    with open(evaluation_results.path, 'r') as f:
        evaluation = json.load(f)
    
    if not evaluation['deploy_ready']:
        logger.info("Model did not pass evaluation criteria. Skipping deployment.")
        return {
            "deployed": False,
            "reason": "Model failed evaluation criteria",
            "auc_score": evaluation['auc_score']
        }
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    logger.info("Uploading model to Vertex AI Model Registry...")
    
    # Upload model
    model = aiplatform.Model.upload(
        display_name=f"predictive-maintenance-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}",
        artifact_uri=trained_model.uri,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest",
        description=f"Predictive maintenance model with AUC: {evaluation['auc_score']:.4f}"
    )
    
    logger.info("Creating or updating endpoint...")
    
    # Create or get endpoint
    try:
        endpoint = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )[0]
        logger.info(f"Using existing endpoint: {endpoint.display_name}")
    except (IndexError, Exception):
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            description="Predictive maintenance endpoint"
        )
        logger.info(f"Created new endpoint: {endpoint.display_name}")
    
    # Deploy model
    logger.info("Deploying model to endpoint...")
    
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"model-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100
    )
    
    deployment_info = {
        "deployed": True,
        "model_id": model.name,
        "endpoint_id": endpoint.name,
        "endpoint_name": endpoint.display_name,
        "auc_score": evaluation['auc_score'],
        "deployment_date": pd.Timestamp.now().isoformat()
    }
    
    logger.info(f"Model deployed successfully: {deployment_info}")
    return deployment_info


# Main Pipeline Definition
@pipeline(
    name="predictive-maintenance-pipeline",
    description="End-to-end predictive maintenance pipeline on Vertex AI",
    pipeline_root="gs://your-bucket/pipeline-root"
)
def predictive_maintenance_pipeline(
    project_id: str = "your-project-id",
    region: str = "us-central1",
    bucket_name: str = "your-data-bucket",
    prediction_window_days: int = 3,
    auc_threshold: float = 0.75
):
    """
    Complete predictive maintenance pipeline for Vertex AI.
    
    Args:
        project_id: GCP project ID
        region: GCP region for resources
        bucket_name: GCS bucket containing data
        prediction_window_days: Days ahead to predict failures
        auc_threshold: Minimum AUC score for deployment
    """
    
    # Step 1: Data Ingestion
    data_ingestion_task = data_ingestion_component(
        project_id=project_id,
        bucket_name=bucket_name
    )
    
    # Step 2: Feature Engineering
    feature_engineering_task = feature_engineering_component(
        telemetry_data=data_ingestion_task.outputs["telemetry_output"],
        failures_data=data_ingestion_task.outputs["failures_output"],
        prediction_window_days=prediction_window_days
    )
    
    # Step 3: Model Training
    model_training_task = model_training_component(
        processed_data=feature_engineering_task.outputs["processed_data"]
    )
    
    # Step 4: Model Evaluation
    model_evaluation_task = model_evaluation_component(
        trained_model=model_training_task.outputs["trained_model"],
        auc_threshold=auc_threshold
    )
    
    # Step 5: Conditional Deployment
    with dsl.Condition(
        model_evaluation_task.outputs["Output"]["deploy_ready"] == True,
        name="deploy-if-approved"
    ):
        deployment_task = model_deployment_component(
            trained_model=model_training_task.outputs["trained_model"],
            evaluation_results=model_evaluation_task.outputs["evaluation_results"],
            project_id=project_id,
            region=region
        )


# Utility function to compile and submit pipeline
def compile_and_run_pipeline(
    project_id: str,
    region: str,
    pipeline_display_name: str = "predictive-maintenance-run"
):
    """
    Compile and submit the pipeline to Vertex AI.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        pipeline_display_name: Display name for the pipeline run
    """
    from kfp.v2 import compiler
    from google.cloud import aiplatform
    
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=predictive_maintenance_pipeline,
        package_path="predictive_maintenance_pipeline.json"
    )
    
    # Submit to Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    job = aiplatform.PipelineJob(
        display_name=pipeline_display_name,
        template_path="predictive_maintenance_pipeline.json",
        pipeline_root=f"gs://{project_id}-pipeline-root",
        parameter_values={
            "project_id": project_id,
            "region": region,
            "bucket_name": f"{project_id}-data-bucket"
        }
    )
    
    job.run(sync=False)
    print(f"Pipeline submitted: {job.resource_name}")
    return job


if __name__ == "__main__":
    # Example usage
    PROJECT_ID = "your-project-id"
    REGION = "us-central1"
    
    # Compile and run pipeline
    job = compile_and_run_pipeline(
        project_id=PROJECT_ID,
        region=REGION,
        pipeline_display_name="predictive-maintenance-demo"
    )
    
    print(f"Pipeline job created: {job.resource_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/pipelines")