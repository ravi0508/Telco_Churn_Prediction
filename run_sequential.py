#!/usr/bin/env python3
"""
Sequential Pipeline Runner
Executes each module step by step with proper error handling
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sequential_execution.log')
    ]
)
logger = logging.getLogger(__name__)

def run_module(module_path, module_name, step_number):
    """Run a single module and handle errors."""
    try:
        logger.info(f"🚀 Step {step_number}: {module_name}")
        logger.info(f"Running: {module_path}")
        
        result = subprocess.run([
            sys.executable, module_path
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            logger.info(f"✅ Step {step_number} completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout[:500]}...")  # First 500 chars
        else:
            logger.error(f"❌ Step {step_number} failed with return code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Step {step_number} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Step {step_number} failed with exception: {e}")
        return False

def main():
    """Execute the pipeline sequentially."""
    start_time = datetime.now()
    logger.info("🚀 Starting Sequential Pipeline Execution")
    logger.info("=" * 50)
    
    # Change to pipeline directory
    pipeline_dir = Path(__file__).parent
    os.chdir(pipeline_dir)
    
    # Define pipeline steps
    steps = [
        ("Directory_2_Ingestion/ingestion.py", "Data Ingestion", 1),
        ("Directory_3_RawStorage/storage_integration.py", "Raw Data Storage", 2),
        ("Directory_4_Validation/validation.py", "Data Validation", 3),
        ("Directory_5_Preparation/preparation.py", "Data Preparation", 4),
        ("Directory_6_Transformation/transformation.py", "Data Transformation", 5),
        ("Directory_7_FeatureStore/feature_store.py", "Feature Store", 6),
        ("Directory_8_Versioning/versioning.py", "Data Versioning", 7),
        ("Directory_9_ModelBuilding/model_building.py", "Model Building", 8),
    ]
    
    successful_steps = 0
    failed_steps = []
    
    # Execute each step
    for module_path, module_name, step_number in steps:
        if not Path(module_path).exists():
            logger.warning(f"⚠️ Module not found: {module_path}, skipping...")
            continue
            
        success = run_module(module_path, module_name, step_number)
        
        if success:
            successful_steps += 1
        else:
            failed_steps.append((step_number, module_name))
            
        logger.info("-" * 30)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("🎉 Sequential Pipeline Execution Summary")
    logger.info("=" * 50)
    logger.info(f"Total steps: {len(steps)}")
    logger.info(f"Successful: {successful_steps}")
    logger.info(f"Failed: {len(failed_steps)}")
    logger.info(f"Duration: {duration}")
    
    if failed_steps:
        logger.error("❌ Failed steps:")
        for step_num, step_name in failed_steps:
            logger.error(f"  - Step {step_num}: {step_name}")
    else:
        logger.info("✅ All steps completed successfully!")

if __name__ == "__main__":
    main()
