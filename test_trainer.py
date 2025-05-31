"""
Quick test for model trainer

Author: Brayan Cuevas
"""

def test_trainer_quick():
    """Quick test of trainer functionality."""
    print("TESTING MODEL TRAINER")
    print("=" * 30)
    
    try:
        # Test imports
        from src.data import load_maintenance_data, create_maintenance_features
        from src.models import ModelTrainer, train_maintenance_model
        print("PASS: All imports successful")
        
        # Load small sample for quick test
        print("\nLoading data...")
        telemetry, failures = load_maintenance_data()
        
        # Take small sample for speed
        sample_size = 10000
        telemetry_sample = telemetry.head(sample_size)
        
        print(f"Processing {sample_size} records for quick test...")
        processed_df, feature_names = create_maintenance_features(
            telemetry_sample, failures
        )
        
        # Test trainer
        print("Testing trainer...")
        trainer = ModelTrainer()
        evaluation, model_path = trainer.train_pipeline(
            processed_df, feature_names, 
            output_path="models/test_model.joblib"
        )
        
        print(f"\nQUICK TEST COMPLETED:")
        print(f"  AUC: {evaluation['auc_score']:.4f}")
        print(f"  Model saved: {model_path}")
        print(f"  Ready for production!")
        
        return True
        
    except Exception as e:
        print(f"FAIL: {e}")
        return False

if __name__ == "__main__":
    test_trainer_quick()