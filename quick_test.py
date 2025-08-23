#!/usr/bin/env python3
"""
Quick Pipeline Test
"""

print("🧪 QUICK PIPELINE TEST")
print("=" * 50)

# Test 1: Basic Python environment
print("\n1. Testing Python environment...")
try:
    import pandas as pd
    import numpy as np
    import sklearn
    print(f"✅ pandas: {pd.__version__}")
    print(f"✅ numpy: {np.__version__}")
    print(f"✅ sklearn: {sklearn.__version__}")
except Exception as e:
    print(f"❌ Package error: {e}")
    exit(1)

# Test 2: Directory structure
print("\n2. Testing directory structure...")
from pathlib import Path
pipeline_dirs = [
    "Directory_2_Ingestion",
    "Directory_4_Validation", 
    "Directory_5_Preparation",
    "Directory_6_Transformation",
    "Directory_7_FeatureStore",
    "Directory_8_Versioning",
    "Directory_9_ModelBuilding"
]

for dir_name in pipeline_dirs:
    dir_path = Path(dir_name)
    py_file = dir_path / f"{dir_name.split('_')[-1].lower()}.py"
    if dir_path.exists() and py_file.exists():
        print(f"✅ {dir_name}")
    else:
        print(f"❌ {dir_name} - missing directory or Python file")

# Test 3: Config import
print("\n3. Testing config...")
try:
    from config import BASE_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR
    print(f"✅ Config imported - Base: {BASE_DIR}")
except Exception as e:
    print(f"❌ Config error: {e}")

# Test 4: Create sample data
print("\n4. Testing sample data creation...")
try:
    telco_sample = pd.DataFrame({
        'customerID': ['C001', 'C002'],
        'Churn': ['Yes', 'No'],
        'tenure': [12, 24],
        'MonthlyCharges': [50.0, 80.0]
    })
    
    hf_sample = pd.DataFrame({
        'customer_id': ['H001', 'H002'],
        'satisfaction_score': [3.5, 4.2]
    })
    
    print(f"✅ Sample data created:")
    print(f"   Telco: {telco_sample.shape}")
    print(f"   HF: {hf_sample.shape}")
    
except Exception as e:
    print(f"❌ Data creation error: {e}")

# Test 5: Basic ML test
print("\n5. Testing basic ML functionality...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Create simple test data
    X = np.random.rand(100, 5)
    y = np.random.choice([0, 1], 100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ ML test completed - Accuracy: {accuracy:.3f}")
    
except Exception as e:
    print(f"❌ ML test error: {e}")

print("\n" + "=" * 50)
print("🎉 QUICK TEST COMPLETED!")
print("\nTo run the full pipeline:")
print("1. Ensure you have valid API credentials (kaggle.json, hugging_face.json)")
print("2. Run: python3 main.py")
print("\nAlternatively, test individual modules:")
print("- python3 Directory_2_Ingestion/ingestion.py")
print("- python3 Directory_4_Validation/validation.py")
print("- etc.")
