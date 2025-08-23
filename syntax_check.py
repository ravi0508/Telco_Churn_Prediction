#!/usr/bin/env python3

# Minimal test - just check syntax of all modules
import sys
from pathlib import Path

print("Checking module syntax...")

modules_to_check = [
    "config.py",
    "main.py", 
    "Directory_2_Ingestion/ingestion.py",
    "Directory_4_Validation/validation.py",
    "Directory_5_Preparation/preparation.py",
    "Directory_6_Transformation/transformation.py",
    "Directory_7_FeatureStore/feature_store.py",
    "Directory_8_Versioning/versioning.py",
    "Directory_9_ModelBuilding/model_building.py"
]

import py_compile

for module in modules_to_check:
    try:
        py_compile.compile(module, doraise=True)
        print(f"✅ {module}")
    except py_compile.PyCompileError as e:
        print(f"❌ {module}: {e}")
    except FileNotFoundError:
        print(f"❓ {module}: File not found")

print("Syntax check completed!")
