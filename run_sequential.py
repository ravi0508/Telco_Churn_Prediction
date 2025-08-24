#!/usr/bin/env python3
"""
Sequential Pipeline Runner
Executes each module step by step with proper logging and feature validation
"""

import subprocess
import sys
import time
import pandas as pd
from pathlib import Path


def run_command(description, command):
    """Run a command and log the results."""
    print(f"\n� {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully in {end_time - start_time:.2f}s")
        if result.stdout:
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"   {line}")
    else:
        print(f"❌ {description} failed!")
        print(f"Error: {result.stderr}")
        return False
    
    return True

def check_feature_count(file_path):
    """Check the number of features in a dataset."""
    try:
        if Path(file_path).exists():
            df = pd.read_csv(file_path, nrows=1)
            return len(df.columns)
    except:
        return 0
    return 0

def main():
    """Run the complete pipeline sequentially."""
    print("🚀 Starting Sequential ML Pipeline Execution")
    print("=" * 60)
    
    # Change to pipeline directory
    pipeline_dir = "/home/jupyter/DMML/churn_prediction_pipeline"
    import os
    os.chdir(pipeline_dir)
    
    steps = [
        ("Stage 1: Data Ingestion", "python3 Directory_2_Ingestion/ingestion.py"),
        ("Stage 2: Raw Storage", "python3 Directory_3_RawStorage/raw_storage.py"),
        ("Stage 3: Data Validation", "python3 Directory_4_Validation/validation.py"),
        ("Stage 4: Data Preparation (Fixed Encoding)", "python3 Directory_5_Preparation/preparation.py"),
        ("Stage 5: Data Transformation", "python3 Directory_6_Transformation/transformation.py"),
        ("Stage 6: Feature Store Population", "python3 Directory_7_FeatureStore/feature_store.py"),
        ("Stage 7: Data Versioning", "python3 Directory_8_Versioning/versioning.py"),
        ("Stage 8: Model Building", "python3 Directory_9_ModelBuilding/model_building.py"),
    ]
    
    failed_steps = []
    
    for step_name, command in steps:
        if not run_command(step_name, command):
            failed_steps.append(step_name)
            
        # Special check after data preparation
        if "Data Preparation" in step_name:
            print("\n📊 Checking feature count after preparation...")
            prep_files = list(Path("Directory_5_Preparation/cleaned").glob("prepared_churn_data_*.csv"))
            if prep_files:
                latest_file = max(prep_files, key=lambda x: x.stat().st_mtime)
                feature_count = check_feature_count(latest_file)
                if feature_count > 100:
                    print(f"⚠️  Warning: {feature_count} features detected - may be too many!")
                else:
                    print(f"✅ Good: {feature_count} features detected - reasonable count")
        
        # Special check after feature store
        if "Feature Store" in step_name:
            print("\n📈 Checking feature store status...")
            # Run feature store again to see the summary
            result = subprocess.run("python3 Directory_7_FeatureStore/feature_store.py", 
                                   shell=True, capture_output=True, text=True)
            if "Number of feature sets:" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Number of feature sets:" in line or "features," in line:
                        print(f"   {line}")
    
    print("\n" + "=" * 60)
    if failed_steps:
        print(f"❌ Pipeline completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            print(f"   - {step}")
    else:
        print("🎉 Pipeline completed successfully!")
    
    print("\n📋 Final Status Check:")
    print(f"   - Models directory: {len(list(Path('models').glob('*.joblib')))} model files")
    print(f"   - Reports directory: {len(list(Path('reports').glob('*.xlsx')))} report files")
    print(f"   - Logs: {Path('logs/main_pipeline.log').exists()}")

if __name__ == "__main__":
    main()
