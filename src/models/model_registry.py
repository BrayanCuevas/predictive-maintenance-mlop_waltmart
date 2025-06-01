"""
Model Registry for predictive maintenance models.
Handles model versioning, comparison, and automated rollback.

Author: Brayan Cuevas
"""

import json
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class ModelMetadata:
    """Model metadata structure."""
    version: str
    model_name: str
    algorithm: str
    auc_score: float
    precision: float
    recall: float
    training_date: str
    training_samples: int
    feature_count: int
    file_path: str
    status: str  # 'active', 'candidate', 'retired'
    deployment_date: Optional[str] = None
    notes: str = ""


class ModelRegistry:
    """Centralized model registry with versioning and comparison."""
    
    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to store model registry data
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.registry_path / "registry.json"
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load existing model registry."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    version: ModelMetadata(**model_data) 
                    for version, model_data in data.items()
                }
        return {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        data = {
            version: asdict(metadata) 
            for version, metadata in self.models.items()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, 
                      model_path: str,
                      version: str,
                      evaluation_results: Dict,
                      training_metadata: Dict,
                      notes: str = "") -> bool:
        """
        Register a new model version.
        
        Args:
            model_path: Path to trained model file
            version: Model version (e.g., "v1.0.0", "v1.1.0")
            evaluation_results: Model evaluation metrics
            training_metadata: Training process metadata
            notes: Optional notes about the model
            
        Returns:
            True if registration successful
        """
        try:
            # Copy model file to registry
            model_filename = f"model_{version}.joblib"
            registry_model_path = self.models_dir / model_filename
            shutil.copy2(model_path, registry_model_path)
            
            # Create metadata
            metadata = ModelMetadata(
                version=version,
                model_name=evaluation_results.get('model_name', 'Unknown'),
                algorithm=evaluation_results.get('model_name', 'Unknown'),
                auc_score=evaluation_results.get('auc_score', 0.0),
                precision=evaluation_results.get('business_metrics', {}).get('precision', 0.0),
                recall=evaluation_results.get('business_metrics', {}).get('recall', 0.0),
                training_date=datetime.now().isoformat(),
                training_samples=training_metadata.get('training_samples', 0),
                feature_count=training_metadata.get('feature_count', 0),
                file_path=str(registry_model_path),
                status='candidate',  # New models start as candidates
                notes=notes
            )
            
            # Add to registry
            self.models[version] = metadata
            self._save_registry()
            
            print(f"✓ Model {version} registered successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to register model {version}: {e}")
            return False
    
    def get_active_model(self) -> Optional[ModelMetadata]:
        """Get currently active model."""
        active_models = [m for m in self.models.values() if m.status == 'active']
        
        if not active_models:
            return None
        
        # Return the most recent active model
        return max(active_models, key=lambda m: m.training_date)
    
    def get_best_candidate(self) -> Optional[ModelMetadata]:
        """Get best candidate model based on AUC score."""
        candidates = [m for m in self.models.values() if m.status == 'candidate']
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda m: m.auc_score)
    
    def compare_models(self, version1: str, version2: str) -> Dict:
        """
        Compare two model versions.
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison results dictionary
        """
        if version1 not in self.models or version2 not in self.models:
            raise ValueError("One or both model versions not found")
        
        model1 = self.models[version1]
        model2 = self.models[version2]
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'metrics_comparison': {
                'auc_score': {
                    'model1': model1.auc_score,
                    'model2': model2.auc_score,
                    'difference': model2.auc_score - model1.auc_score,
                    'improvement': ((model2.auc_score - model1.auc_score) / model1.auc_score * 100) if model1.auc_score > 0 else 0
                },
                'precision': {
                    'model1': model1.precision,
                    'model2': model2.precision,
                    'difference': model2.precision - model1.precision
                },
                'recall': {
                    'model1': model1.recall,
                    'model2': model2.recall,
                    'difference': model2.recall - model1.recall
                }
            },
            'recommendation': self._get_recommendation(model1, model2)
        }
        
        return comparison
    
    def _get_recommendation(self, model1: ModelMetadata, model2: ModelMetadata) -> str:
        """Get deployment recommendation based on model comparison."""
        auc_improvement = model2.auc_score - model1.auc_score
        precision_improvement = model2.precision - model1.precision
        recall_improvement = model2.recall - model1.recall
        
        # Decision logic
        if auc_improvement > 0.01:  # 1% improvement threshold
            return "DEPLOY - Significant AUC improvement"
        elif auc_improvement > 0.005 and precision_improvement > 0.02:
            return "DEPLOY - Good AUC and precision improvement"
        elif auc_improvement < -0.01:
            return "REJECT - AUC degradation too high"
        elif precision_improvement < -0.05:
            return "REJECT - Precision degradation too high"
        else:
            return "HOLD - Marginal improvement, consider more data"
    
    def promote_model(self, version: str) -> bool:
        """
        Promote a candidate model to active status.
        
        Args:
            version: Model version to promote
            
        Returns:
            True if promotion successful
        """
        if version not in self.models:
            print(f"✗ Model {version} not found")
            return False
        
        # Retire current active model
        current_active = self.get_active_model()
        if current_active:
            self.models[current_active.version].status = 'retired'
            print(f"✓ Model {current_active.version} retired")
        
        # Promote new model
        self.models[version].status = 'active'
        self.models[version].deployment_date = datetime.now().isoformat()
        
        self._save_registry()
        print(f"✓ Model {version} promoted to active")
        
        return True
    
    def auto_evaluate_and_promote(self, 
                                 min_auc_threshold: float = 0.75,
                                 min_improvement: float = 0.005) -> Optional[str]:
        """
        Automatically evaluate candidates and promote if criteria met.
        
        Args:
            min_auc_threshold: Minimum AUC score required
            min_improvement: Minimum improvement over current active model
            
        Returns:
            Version of promoted model or None
        """
        current_active = self.get_active_model()
        best_candidate = self.get_best_candidate()
        
        if not best_candidate:
            print("No candidate models available")
            return None
        
        # Check minimum threshold
        if best_candidate.auc_score < min_auc_threshold:
            print(f"Candidate AUC {best_candidate.auc_score:.3f} below threshold {min_auc_threshold}")
            return None
        
        # Compare with active model
        if current_active:
            improvement = best_candidate.auc_score - current_active.auc_score
            
            if improvement < min_improvement:
                print(f"Improvement {improvement:.3f} below threshold {min_improvement}")
                return None
            
            print(f"Candidate shows {improvement:.3f} AUC improvement")
        
        # Promote the candidate
        if self.promote_model(best_candidate.version):
            return best_candidate.version
        
        return None
    
    def get_registry_summary(self) -> Dict:
        """Get summary of model registry."""
        if not self.models:
            return {"total_models": 0, "active_models": 0, "candidate_models": 0}
        
        status_counts = {}
        for model in self.models.values():
            status_counts[model.status] = status_counts.get(model.status, 0) + 1
        
        best_model = max(self.models.values(), key=lambda m: m.auc_score)
        
        return {
            "total_models": len(self.models),
            "active_models": status_counts.get('active', 0),
            "candidate_models": status_counts.get('candidate', 0),
            "retired_models": status_counts.get('retired', 0),
            "best_auc": best_model.auc_score,
            "best_model_version": best_model.version,
            "latest_training": max(m.training_date for m in self.models.values())
        }
    
    def list_models(self) -> List[Dict]:
        """List all models with key information."""
        return [
            {
                "version": model.version,
                "algorithm": model.algorithm,
                "auc_score": model.auc_score,
                "precision": model.precision,
                "recall": model.recall,
                "status": model.status,
                "training_date": model.training_date[:10],  # Date only
                "notes": model.notes
            }
            for model in sorted(self.models.values(), 
                              key=lambda m: m.training_date, reverse=True)
        ]


# Global registry instance
model_registry = ModelRegistry()