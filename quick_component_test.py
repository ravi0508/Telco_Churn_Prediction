#!/usr/bin/env python3
"""
Quick Pipeline Component Test
Tests each component independently with minimal dependencies
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
project_root = Path('.')
sys.path.append(str(project_root))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing module imports...")
    
    tests = []
    
    # Test config
    try:
        import config
        tests.append(("config", "✅ PASS"))
    except Exception as e:
        tests.append(("config", f"❌ FAIL: {e}"))
    
    # Test ingestion
    try:
        sys.path.append('Directory_2_Ingestion')
        from ingestion import DataIngestion
        tests.append(("DataIngestion", "✅ PASS"))
    except Exception as e:
        tests.append(("DataIngestion", f"❌ FAIL: {e}"))
    
    # Test validation
    try:
        sys.path.append('Directory_4_Validation')
        from validation import DataValidator
        tests.append(("DataValidator", "✅ PASS"))
    except Exception as e:
        tests.append(("DataValidator", f"❌ FAIL: {e}"))
    
    # Test preparation
    try:
        sys.path.append('Directory_5_Preparation')
        from preparation import DataPreparation
        tests.append(("DataPreparation", "✅ PASS"))
    except Exception as e:
        tests.append(("DataPreparation", f"❌ FAIL: {e}"))
    
    # Test transformation
    try:
        sys.path.append('Directory_6_Transformation')
        from transformation import DataTransformation
        tests.append(("DataTransformation", "✅ PASS"))
    except Exception as e:
        tests.append(("DataTransformation", f"❌ FAIL: {e}"))
    
    # Test feature store
    try:
        sys.path.append('Directory_7_FeatureStore')
        from feature_store import FeatureStore
        tests.append(("FeatureStore", "✅ PASS"))
    except Exception as e:
        tests.append(("FeatureStore", f"❌ FAIL: {e}"))
    
    # Test versioning
    try:
        sys.path.append('Directory_8_Versioning')
        from versioning import DataVersioning
        tests.append(("DataVersioning", "✅ PASS"))
    except Exception as e:
        tests.append(("DataVersioning", f"❌ FAIL: {e}"))
    
    # Test model building
    try:
        sys.path.append('Directory_9_ModelBuilding')
        from model_building import ModelBuilder
        tests.append(("ModelBuilder", "✅ PASS"))
    except Exception as e:
        tests.append(("ModelBuilder", f"❌ FAIL: {e}"))
    
    print("\n=== IMPORT TEST RESULTS ===")
    for module, result in tests:
        print(f"{module:20} {result}")
    
    passed = sum(1 for _, result in tests if "✅" in result)
    total = len(tests)
    print(f"\nPassed: {passed}/{total}")
    
    return passed == total

def create_test_data():
    """Create simple test data."""
    print("\nCreating test data...")
    
    # Simple telco data
    telco_data = pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003'],
        'gender': ['Male', 'Female', 'Male'],
        'tenure': [12, 24, 6],
        'MonthlyCharges': [50.0, 75.0, 40.0],
        'TotalCharges': [600.0, 1800.0, 240.0],
        'Churn': ['No', 'Yes', 'No']
    })
    
    # Simple HF data
    hf_data = pd.DataFrame({
        'customer_id': ['H001', 'H002', 'H003'],
        'satisfaction_score': [4.5, 2.1, 3.8],
        'usage_frequency': ['High', 'Low', 'Medium']
    })
    
    print(f"✅ Test data created - Telco: {telco_data.shape}, HF: {hf_data.shape}")
    return telco_data, hf_data

def test_directories():
    """Test if required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/transformed',
        'logs',
        'models',
        'reports',
        'Directory_2_Ingestion',
        'Directory_4_Validation',
        'Directory_5_Preparation',
        'Directory_6_Transformation',
        'Directory_7_FeatureStore',
        'Directory_8_Versioning',
        'Directory_9_ModelBuilding'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ➕ Created {dir_path}")

def test_credentials():
    """Test if credentials are available."""
    print("\nTesting credentials...")
    
    kaggle_file = 'kaggle.json'
    hf_file = 'hugging_face.json'
    
    if os.path.exists(kaggle_file):
        print(f"✅ {kaggle_file} exists")
    else:
        print(f"❌ {kaggle_file} missing")
    
    if os.path.exists(hf_file):
        print(f"✅ {hf_file} exists")
    else:
        print(f"❌ {hf_file} missing")

def main():
    """Run quick tests."""
    print("🚀 QUICK PIPELINE COMPONENT TEST")
    print("=" * 50)
    
    # Test directory structure
    test_directories()
    
    # Test credentials
    test_credentials()
    
    # Test imports
    imports_ok = test_imports()
    
    # Create test data
    telco_data, hf_data = create_test_data()
    
    print("\n" + "=" * 50)
    if imports_ok:
        print("✅ All imports successful! Pipeline components are ready.")
        print("💡 You can now run the full pipeline with: python3 main.py")
    else:
        print("❌ Some imports failed. Check the error messages above.")
    
    print("\n📝 Test data shapes:")
    print(f"   Telco: {telco_data.shape}")
    print(f"   HF: {hf_data.shape}")

if __name__ == "__main__":
    main()
