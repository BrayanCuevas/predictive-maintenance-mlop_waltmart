"""
Data loading module for predictive maintenance project.
Handles loading and basic validation of maintenance datasets.

Author: Brayan Cuevas
Created for technical challenge implementation.
"""

import pandas as pd
import os
from typing import Tuple, Dict, Optional
from pathlib import Path


class MaintenanceDataLoader:
    """Load and validate maintenance datasets."""

    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize data loader.

        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load telemetry and failures datasets.

        Returns:
            Tuple of (telemetry_df, failures_df)

        Raises:
            FileNotFoundError: If required files are missing
        """
        # Define required files
        telemetry_file = self.data_path / "PdM_telemetry.csv"
        failures_file = self.data_path / "PdM_failures.csv"

        # Check file existence
        for file_path in [telemetry_file, failures_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load datasets
        print(f"Loading data from {self.data_path}...")
        telemetry_df = pd.read_csv(telemetry_file)
        failures_df = pd.read_csv(failures_file)

        # Convert datetime columns
        telemetry_df["datetime"] = pd.to_datetime(telemetry_df["datetime"])
        failures_df["datetime"] = pd.to_datetime(failures_df["datetime"])

        print(f"✓ Telemetry: {telemetry_df.shape}")
        print(f"✓ Failures: {failures_df.shape}")

        return telemetry_df, failures_df

    def validate_data(self, telemetry_df: pd.DataFrame, failures_df: pd.DataFrame) -> Dict:
        """
        Validate data quality and return summary.

        Args:
            telemetry_df: Telemetry sensor data
            failures_df: Failure events data

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "telemetry_shape": telemetry_df.shape,
            "failures_shape": failures_df.shape,
            "date_range": (telemetry_df["datetime"].min(), telemetry_df["datetime"].max()),
            "unique_machines": telemetry_df["machineID"].nunique(),
            "total_failures": len(failures_df),
            "missing_values": telemetry_df.isnull().sum().to_dict(),
            "failure_types": failures_df["failure"].value_counts().to_dict(),
        }

        return validation_results

    def load_and_validate(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load datasets and return with validation summary.

        Returns:
            Tuple of (telemetry_df, failures_df, validation_results)
        """
        telemetry_df, failures_df = self.load_datasets()
        validation_results = self.validate_data(telemetry_df, failures_df)

        print(f"✓ Loaded {validation_results['telemetry_shape'][0]:,} telemetry records")
        print(f"✓ Loaded {validation_results['total_failures']} failure events")
        print(f"✓ Monitoring {validation_results['unique_machines']} machines")

        return telemetry_df, failures_df, validation_results


def load_maintenance_data(data_path: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load maintenance data.

    Args:
        data_path: Path to raw data directory

    Returns:
        Tuple of (telemetry_df, failures_df)
    """
    loader = MaintenanceDataLoader(data_path)
    telemetry_df, failures_df = loader.load_datasets()
    return telemetry_df, failures_df


if __name__ == "__main__":
    # Test the data loader
    loader = MaintenanceDataLoader()
    telemetry, failures, validation = loader.load_and_validate()

    print("\nValidation Summary:")
    print(
        f"  Date range: {validation['date_range'][0].date()} to {validation['date_range'][1].date()}"
    )
    print(f"  Machines: {validation['unique_machines']}")
    print(f"  Failure types: {len(validation['failure_types'])}")
