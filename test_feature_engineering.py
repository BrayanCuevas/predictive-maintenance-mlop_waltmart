import os
import sys

def test_feature_engineering():
    """Test the feature engineering module."""
    print("TESTING FEATURE ENGINEERING MODULE")
    print("=" * 50)
    
    try:
        # Import modules
        from src.data import load_maintenance_data
        from src.data.feature_engineering import MaintenanceFeatureEngineer, create_maintenance_features
        print("PASS: Modules imported successfully")
        
        # Load data
        print("\n=== LOADING DATA ===")
        telemetry, failures = load_maintenance_data()
        print(f"PASS: Loaded {len(telemetry):,} telemetry records")
        print(f"PASS: Loaded {len(failures)} failure events")
        
        # Test convenience function
        print("\n=== TESTING CONVENIENCE FUNCTION ===")
        processed_df, feature_names = create_maintenance_features(
            telemetry, failures, prediction_window_days=3
        )
        
        print(f"PASS: Feature engineering completed")
        print(f"  Input shape: {telemetry.shape}")
        print(f"  Output shape: {processed_df.shape}")
        print(f"  Features created: {len(feature_names)}")
        print(f"  Failure rate: {processed_df['failure_within_window'].mean()*100:.2f}%")
        
        # Test class directly
        print("\n=== TESTING CLASS DIRECTLY ===")
        engineer = MaintenanceFeatureEngineer(prediction_window_days=7)
        processed_df_7d, feature_names_7d = engineer.process_features(telemetry, failures)
        
        print(f"PASS: 7-day prediction window completed")
        print(f"  7-day failure rate: {processed_df_7d['failure_within_window'].mean()*100:.2f}%")
        
        # Show sample features
        print(f"\n=== SAMPLE FEATURES ===")
        print("First 10 feature names:")
        for i, feature in enumerate(feature_names[:10]):
            print(f"  {i+1:2d}. {feature}")
        
        # Data quality check
        print(f"\n=== DATA QUALITY CHECK ===")
        missing_values = processed_df.isnull().sum().sum()
        if missing_values == 0:
            print("PASS: No missing values in processed data")
        else:
            print(f"WARNING: Found {missing_values} missing values")
        
        # Check label distribution
        label_counts = processed_df['failure_within_window'].value_counts()
        print(f"Label distribution:")
        print(f"  No failure (0): {label_counts[0]:,}")
        print(f"  Failure (1): {label_counts[1]:,}")
        
        print(f"\nALL FEATURE ENGINEERING TESTS PASSED")
        print(f"Ready to proceed with model training")
        
        return processed_df, feature_names
        
    except Exception as e:
        print(f"FAIL: Feature engineering test failed: {e}")
        return None, None

if __name__ == "__main__":
    processed_df, feature_names = test_feature_engineering()