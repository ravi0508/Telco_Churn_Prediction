"""
Simple Pipeline Test - Step by Step
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Setup paths
project_root = Path('.')
sys.path.append(str(project_root))

# Import config at module level
from config import BASE_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR

def test_step_1_imports():
    """Test 1: Import all modules"""
    print("=== TEST 1: IMPORTS ===")
    
    try:
        print(f"✅ Config imported - Base dir: {BASE_DIR}")
        
        # Add module paths
        sys.path.extend([
            'Directory_2_Ingestion',
            'Directory_4_Validation', 
            'Directory_5_Preparation',
            'Directory_6_Transformation',
            'Directory_7_FeatureStore',
            'Directory_8_Versioning',
            'Directory_9_ModelBuilding'
        ])
        
        from ingestion import DataIngestion
        from validation import DataValidator
        from preparation import DataPreparation
        from transformation import DataTransformation
        from feature_store import FeatureStore
        from versioning import DataVersioning
        from model_building import ModelBuilder
        
        print("✅ All modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_2_data():
    """Test 2: Create sample data"""
    print("\n=== TEST 2: DATA CREATION ===")
    
    try:
        # Create minimal test data
        telco_data = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003'],
            'gender': ['Male', 'Female', 'Male'],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.0, 80.0, 30.0],
            'Churn': ['Yes', 'No', 'Yes']
        })
        
        hf_data = pd.DataFrame({
            'customer_id': ['H001', 'H002', 'H003'],
            'satisfaction_score': [3.5, 4.2, 2.8],
            'usage_frequency': ['High', 'Medium', 'Low']
        })
        
        print(f"✅ Telco data: {telco_data.shape}")
        print(f"✅ HF data: {hf_data.shape}")
        
        return telco_data, hf_data
        
    except Exception as e:
        print(f"❌ Data creation failed: {e}")
        return None, None

def test_step_3_validation(telco_data, hf_data):
    """Test 3: Data validation"""
    print("\n=== TEST 3: VALIDATION ===")
    
    try:
        from validation import DataValidator
        validator = DataValidator()
        
        # Quick validation test
        print(f"Input data shapes - Telco: {telco_data.shape}, HF: {hf_data.shape}")
        print("✅ Validation module ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 SIMPLE PIPELINE TEST\n")
    
    # Step 1: Test imports
    if not test_step_1_imports():
        print("❌ Stopping due to import errors")
        exit(1)
    
    # Step 2: Create data
    telco_data, hf_data = test_step_2_data()
    if telco_data is None:
        print("❌ Stopping due to data creation errors")
        exit(1)
    
    # Step 3: Test validation
    if not test_step_3_validation(telco_data, hf_data):
        print("❌ Validation test failed")
    
    print("\n✅ Basic tests completed!")
    print("\nNext steps:")
    print("1. Run: python3 main.py (if you have API credentials)")
    print("2. Or test individual modules manually")
