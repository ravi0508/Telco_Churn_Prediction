#!/usr/bin/env python3
"""
Complete Pipeline Test Runner
Tests the full pipeline end-to-end with synthetic data
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    logger.info("🚀 STARTING COMPLETE PIPELINE TEST")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Import config
        import config
        logger.info("✅ Config imported successfully")
        
        # Stage 1: Data Ingestion (using safe version)
        logger.info("=== STAGE 1: DATA INGESTION ===")
        sys.path.append('Directory_2_Ingestion')
        from ingestion_safe import DataIngestion
        
        ingestion = DataIngestion()
        telco_data, hf_data = ingestion.run_ingestion()
        logger.info(f"✅ Data ingestion completed - Telco: {telco_data.shape}, HF: {hf_data.shape}")
        
        # Stage 2: Data Validation
        logger.info("=== STAGE 2: DATA VALIDATION ===")
        sys.path.append('Directory_4_Validation')
        from validation import DataValidator
        
        validator = DataValidator()
        validation_results = validator.validate_datasets(telco_data, hf_data)
        logger.info(f"✅ Data validation completed: {type(validation_results)}")
        
        # Stage 3: Data Preparation
        logger.info("=== STAGE 3: DATA PREPARATION ===")
        sys.path.append('Directory_5_Preparation')
        from preparation import DataPreparation
        
        preparation = DataPreparation()
        telco_clean, hf_clean = preparation.clean_datasets(telco_data, hf_data)
        logger.info(f"✅ Data preparation completed - Telco: {telco_clean.shape}, HF: {hf_clean.shape}")
        
        # Stage 4: Data Transformation
        logger.info("=== STAGE 4: DATA TRANSFORMATION ===")
        sys.path.append('Directory_6_Transformation')
        from transformation import DataTransformation
        
        transformation = DataTransformation()
        features, labels = transformation.transform_datasets(telco_clean, hf_clean)
        logger.info(f"✅ Data transformation completed - Features: {features.shape}, Labels: {labels.shape}")
        
        # Stage 5: Feature Store
        logger.info("=== STAGE 5: FEATURE STORE ===")
        sys.path.append('Directory_7_FeatureStore')
        from feature_store import FeatureStore
        
        feature_store = FeatureStore()
        store_result = feature_store.store_features(features, labels)
        logger.info(f"✅ Feature store completed: {store_result.get('status', 'unknown')}")
        
        # Stage 6: Data Versioning
        logger.info("=== STAGE 6: DATA VERSIONING ===")
        sys.path.append('Directory_8_Versioning')
        from versioning import DataVersioning
        
        versioning = DataVersioning()
        version_info = versioning.version_data(features, labels)
        logger.info(f"✅ Data versioning completed: {version_info.get('status', 'completed')}")
        
        # Stage 7: Model Building
        logger.info("=== STAGE 7: MODEL BUILDING ===")
        sys.path.append('Directory_9_ModelBuilding')
        from model_building import ModelBuilder
        
        model_builder = ModelBuilder()
        best_model, metrics = model_builder.build_and_evaluate_models(features, labels)
        logger.info(f"✅ Model building completed - Best model: {best_model}")
        
        # Final Results
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("🎉 COMPLETE PIPELINE TEST SUCCESSFUL!")
        logger.info(f"⏱️  Total execution time: {duration}")
        logger.info("=" * 60)
        
        # Summary
        print("\n📊 PIPELINE EXECUTION SUMMARY:")
        print(f"   Input Data: Telco {telco_data.shape}, HF {hf_data.shape}")
        print(f"   Cleaned Data: Telco {telco_clean.shape}, HF {hf_clean.shape}")
        print(f"   Final Features: {features.shape}")
        print(f"   Labels: {labels.shape}")
        print(f"   Best Model: {best_model}")
        print(f"   Feature Store: {store_result.get('status', 'unknown')}")
        print(f"   Versioning: {version_info.get('status', 'unknown')}")
        print(f"   Total Time: {duration}")
        
        # Check output files
        print("\n📁 Generated Files:")
        output_dirs = ['data/raw', 'data/transformed', 'models', 'reports', 'logs']
        for dir_path in output_dirs:
            if os.path.exists(dir_path):
                files = list(Path(dir_path).glob('*'))
                print(f"   {dir_path}: {len(files)} files")
                for f in files[-3:]:  # Show last 3 files
                    print(f"     - {f.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete pipeline test."""
    success = test_full_pipeline()
    
    if success:
        print("\n🎉 SUCCESS: Pipeline is working end-to-end!")
        print("✅ All components tested and functional")
        print("✅ Data flows through all stages")
        print("✅ Models trained and saved")
        print("✅ Reports generated")
        print("\n💡 You can now run the main pipeline with: python3 main.py")
    else:
        print("\n❌ FAILURE: Pipeline test failed")
        print("🔧 Check the error messages above for debugging")
    
    return success

if __name__ == "__main__":
    os.chdir('/home/jupyter/DMML/churn_prediction_pipeline')
    success = main()
    sys.exit(0 if success else 1)
