#!/usr/bin/env python3
"""
Safe Pipeline Test Runner
Tests pipeline with lazy imports and timeouts
"""

import sys
import os
import signal
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import time

def run_with_timeout(command, timeout=10):
    """Run a command with timeout."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", 124

def test_basic_imports():
    """Test basic imports without hanging libraries."""
    print("=== Testing Basic Imports ===")
    
    tests = [
        ("pandas", "import pandas as pd; print('pandas version:', pd.__version__)"),
        ("numpy", "import numpy as np; print('numpy version:', np.__version__)"),
        ("sklearn", "import sklearn; print('sklearn version:', sklearn.__version__)"),
        ("config", "import sys; sys.path.append('.'); import config; print('config loaded')"),
    ]
    
    for name, cmd in tests:
        stdout, stderr, code = run_with_timeout(f"python3 -c \"{cmd}\"", 5)
        if code == 0:
            print(f"✅ {name}: {stdout.strip()}")
        elif code == 124:
            print(f"⏰ {name}: TIMEOUT")
        else:
            print(f"❌ {name}: {stderr.strip()}")

def test_directory_structure():
    """Test if directories exist."""
    print("\n=== Testing Directory Structure ===")
    
    dirs = [
        'data/raw', 'data/transformed', 'logs', 'models', 'reports',
        'Directory_2_Ingestion', 'Directory_4_Validation', 'Directory_5_Preparation',
        'Directory_6_Transformation', 'Directory_7_FeatureStore', 
        'Directory_8_Versioning', 'Directory_9_ModelBuilding'
    ]
    
    for d in dirs:
        if os.path.exists(d):
            print(f"✅ {d}")
        else:
            print(f"❌ {d} (missing)")
            os.makedirs(d, exist_ok=True)
            print(f"   ➕ Created {d}")

def test_credentials():
    """Test credentials files."""
    print("\n=== Testing Credentials ===")
    
    files = ['kaggle.json', 'hugging_face.json']
    for f in files:
        if os.path.exists(f):
            print(f"✅ {f}")
            try:
                with open(f, 'r') as file:
                    import json
                    data = json.load(file)
                    print(f"   📋 Keys: {list(data.keys())}")
            except Exception as e:
                print(f"   ⚠️  File exists but invalid JSON: {e}")
        else:
            print(f"❌ {f} (missing)")

def create_synthetic_data():
    """Create synthetic data for testing."""
    print("\n=== Creating Synthetic Data ===")
    
    # Create synthetic telco data
    np.random.seed(42)
    n = 100
    
    telco_data = pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(n)],
        'gender': np.random.choice(['Male', 'Female'], n),
        'tenure': np.random.randint(1, 73, n),
        'MonthlyCharges': np.random.uniform(18.0, 120.0, n),
        'TotalCharges': np.random.uniform(18.0, 8500.0, n),
        'Churn': np.random.choice(['Yes', 'No'], n, p=[0.27, 0.73])
    })
    
    # Save to data directory
    os.makedirs('data/raw', exist_ok=True)
    telco_data.to_csv('data/raw/telco_synthetic.csv', index=False)
    print(f"✅ Synthetic telco data created: {telco_data.shape}")
    
    # Create synthetic HF data
    hf_data = pd.DataFrame({
        'customer_id': [f'H{i:04d}' for i in range(50)],
        'satisfaction_score': np.random.uniform(1.0, 5.0, 50),
        'usage_frequency': np.random.choice(['Low', 'Medium', 'High'], 50)
    })
    
    hf_data.to_csv('data/raw/hf_synthetic.csv', index=False)
    print(f"✅ Synthetic HF data created: {hf_data.shape}")
    
    return telco_data, hf_data

def test_pipeline_modules():
    """Test each pipeline module individually with timeouts."""
    print("\n=== Testing Pipeline Modules ===")
    
    modules = [
        ("config", "import sys; sys.path.append('.'); import config"),
        ("validation", "import sys; sys.path.append('Directory_4_Validation'); from validation import DataValidator"),
        ("preparation", "import sys; sys.path.append('Directory_5_Preparation'); from preparation import DataPreparation"),
        ("transformation", "import sys; sys.path.append('Directory_6_Transformation'); from transformation import DataTransformation"),
        ("feature_store", "import sys; sys.path.append('Directory_7_FeatureStore'); from feature_store import FeatureStore"),
        ("versioning", "import sys; sys.path.append('Directory_8_Versioning'); from versioning import DataVersioning"),
        ("model_building", "import sys; sys.path.append('Directory_9_ModelBuilding'); from model_building import ModelBuilder"),
    ]
    
    for name, cmd in modules:
        stdout, stderr, code = run_with_timeout(f"python3 -c \"{cmd}\"", 10)
        if code == 0:
            print(f"✅ {name}: Import successful")
        elif code == 124:
            print(f"⏰ {name}: Import timeout (likely network issue)")
        else:
            print(f"❌ {name}: {stderr.strip()}")

def test_simple_pipeline():
    """Test a simple pipeline without API calls."""
    print("\n=== Testing Simple Pipeline ===")
    
    # Create synthetic data
    telco_data, hf_data = create_synthetic_data()
    
    pipeline_script = '''
import sys
import pandas as pd
import numpy as np
sys.path.append('.')

# Test data loading
telco = pd.read_csv('data/raw/telco_synthetic.csv')
hf = pd.read_csv('data/raw/hf_synthetic.csv')

print(f"Loaded telco data: {telco.shape}")
print(f"Loaded HF data: {hf.shape}")

# Test basic processing
telco_clean = telco.dropna()
features = telco_clean.select_dtypes(include=[np.number])
print(f"Features shape: {features.shape}")

print("✅ Simple pipeline test completed successfully!")
'''
    
    stdout, stderr, code = run_with_timeout(f"python3 -c '{pipeline_script}'", 15)
    if code == 0:
        print("✅ Simple pipeline test successful")
        print(stdout)
    else:
        print(f"❌ Simple pipeline test failed: {stderr}")

def main():
    """Run all tests."""
    print("🚀 SAFE PIPELINE TEST RUNNER")
    print("=" * 60)
    
    # Change to project directory
    os.chdir('/home/jupyter/DMML/churn_prediction_pipeline')
    
    # Run tests
    test_basic_imports()
    test_directory_structure()
    test_credentials()
    test_pipeline_modules()
    test_simple_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ Safe pipeline testing completed!")
    print("\n💡 Next steps:")
    print("   1. If all imports work, try running individual modules")
    print("   2. Use synthetic data to test the pipeline logic")
    print("   3. Once stable, try with real API data")

if __name__ == "__main__":
    main()
