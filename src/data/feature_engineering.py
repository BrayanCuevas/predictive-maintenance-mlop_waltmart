"""
Feature engineering module for predictive maintenance.
Creates rolling window features and failure labels.

Author: Brayan Cuevas
Created for technical challenge implementation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import timedelta


class MaintenanceFeatureEngineer:
    """Create features for predictive maintenance model."""
    
    def __init__(self, prediction_window_days: int = 3):
        """
        Initialize feature engineer.
        
        Args:
            prediction_window_days: Days ahead to predict failures
        """
        self.prediction_window_days = prediction_window_days
        self.sensor_columns = ['volt', 'rotate', 'pressure', 'vibration']
        
    def create_rolling_features(self, df: pd.DataFrame, 
                              windows: List[int] = [3, 24]) -> pd.DataFrame:
        """
        Create rolling window features for sensor data.
        
        Args:
            df: Telemetry dataframe with sensor readings
            windows: List of window sizes in hours
            
        Returns:
            DataFrame with additional rolling features
        """
        df_features = df.copy()
        df_features = df_features.sort_values(['machineID', 'datetime'])
        
        print(f"Creating rolling features for {len(self.sensor_columns)} sensors...")
        
        for window in windows:
            print(f"  Processing {window}h window...")
            
            for sensor in self.sensor_columns:
                # Rolling statistics
                rolling_group = df_features.groupby('machineID')[sensor].rolling(
                    window=window, min_periods=1
                )
                
                df_features[f'{sensor}_{window}h_mean'] = rolling_group.mean().reset_index(0, drop=True)
                df_features[f'{sensor}_{window}h_std'] = rolling_group.std().reset_index(0, drop=True)
                df_features[f'{sensor}_{window}h_max'] = rolling_group.max().reset_index(0, drop=True)
                df_features[f'{sensor}_{window}h_min'] = rolling_group.min().reset_index(0, drop=True)
        
        # Fill NaN values
        df_features = df_features.ffill().fillna(0)
        
        feature_count = len([col for col in df_features.columns if any(f'{h}h_' in col for h in windows)])
        print(f"Created {feature_count} rolling features")
        
        return df_features
    
    def create_failure_labels(self, telemetry_df: pd.DataFrame, 
                            failures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary failure labels for prediction.
        
        Args:
            telemetry_df: Telemetry data with features
            failures_df: Failure events data
            
        Returns:
            DataFrame with failure_within_window labels
        """
        df_labeled = telemetry_df.copy()
        df_labeled['failure_within_window'] = 0
        
        window_hours = self.prediction_window_days * 24
        
        print(f"Creating failure labels with {self.prediction_window_days}-day prediction window...")
        
        failure_count = 0
        for _, failure in failures_df.iterrows():
            machine_id = failure['machineID']
            failure_time = pd.to_datetime(failure['datetime'])
            
            # Mark records within prediction window before failure
            mask = ((df_labeled['machineID'] == machine_id) & 
                    (df_labeled['datetime'] >= failure_time - pd.Timedelta(hours=window_hours)) &
                    (df_labeled['datetime'] < failure_time))
            
            affected_records = mask.sum()
            if affected_records > 0:
                df_labeled.loc[mask, 'failure_within_window'] = 1
                failure_count += affected_records
        
        positive_rate = df_labeled['failure_within_window'].mean() * 100
        print(f"Labeled {failure_count:,} records as positive ({positive_rate:.2f}%)")
        
        return df_labeled
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of engineered feature column names.
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['datetime', 'machineID', 'failure_within_window']
        feature_names = [col for col in df.columns if col not in exclude_cols]
        return feature_names
    
    def process_features(self, telemetry_df: pd.DataFrame, 
                        failures_df: pd.DataFrame,
                        windows: List[int] = [3, 24]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete feature engineering pipeline.
        
        Args:
            telemetry_df: Raw telemetry data
            failures_df: Failure events data
            windows: Rolling window sizes in hours
            
        Returns:
            Tuple of (processed_dataframe, feature_names)
        """
        print("Starting feature engineering pipeline...")
        
        # Create rolling features
        df_with_features = self.create_rolling_features(telemetry_df, windows)
        
        # Create failure labels
        df_final = self.create_failure_labels(df_with_features, failures_df)
        
        # Get feature names
        feature_names = self.get_feature_names(df_final)
        
        print(f"Feature engineering completed:")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Total records: {len(df_final):,}")
        print(f"  Failure rate: {df_final['failure_within_window'].mean()*100:.2f}%")
        
        return df_final, feature_names


def create_maintenance_features(telemetry_df: pd.DataFrame, 
                              failures_df: pd.DataFrame,
                              prediction_window_days: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function for feature engineering.
    
    Args:
        telemetry_df: Raw telemetry data
        failures_df: Failure events data
        prediction_window_days: Days ahead to predict failures
        
    Returns:
        Tuple of (processed_dataframe, feature_names)
    """
    engineer = MaintenanceFeatureEngineer(prediction_window_days)
    return engineer.process_features(telemetry_df, failures_df)


if __name__ == "__main__":
    # Test feature engineering
    from .data_loader import load_maintenance_data
    
    print("Testing feature engineering...")
    telemetry, failures = load_maintenance_data()
    
    engineer = MaintenanceFeatureEngineer(prediction_window_days=3)
    processed_df, feature_names = engineer.process_features(telemetry, failures)
    
    print(f"\nFeature engineering test completed:")
    print(f"  Input shape: {telemetry.shape}")
    print(f"  Output shape: {processed_df.shape}")
    print(f"  Features created: {len(feature_names)}")
    print(f"  Sample features: {feature_names[:5]}")