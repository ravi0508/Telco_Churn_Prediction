#!/usr/bin/env python3
"""
Storage Integration Script
Handles integration between ingestion and raw storage modules
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Tuple

# Add parent directory to path for config import
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import LOG_FORMAT
from raw_storage import RawDataStorage

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def integrate_ingestion_with_storage() -> Dict[str, Any]:
    """
    Integrate ingestion module with raw storage.
    
    Returns:
        Dict[str, Any]: Integration results
    """
    try:
        # Import ingestion module
        ingestion_dir = parent_dir / 'Directory_2_Ingestion'
        sys.path.append(str(ingestion_dir))
        
        from ingestion import DataIngestion
        
        logger.info("Starting ingestion-storage integration")
        
        # Initialize components
        ingestion = DataIngestion()
        storage = RawDataStorage()
        
        # Run ingestion
        logger.info("Running data ingestion...")
        telco_data, hf_data = ingestion.run_ingestion()
        
        # Store in organized structure
        logger.info("Storing data in organized storage structure...")
        
        # Store Kaggle/Telco data
        kaggle_result = storage.store_raw_data(
            data=telco_data,
            source='kaggle', 
            data_type='telco_churn',
            filename='telco_churn_data.csv'
        )
        
        # Store Hugging Face data  
        hf_result = storage.store_raw_data(
            data=hf_data,
            source='huggingface',
            data_type='churn_prediction', 
            filename='hf_churn_data.csv'
        )
        
        # Prepare results
        results = {
            'status': 'success',
            'ingestion': {
                'telco_shape': telco_data.shape,
                'hf_shape': hf_data.shape
            },
            'storage': {
                'kaggle_result': kaggle_result,
                'hf_result': hf_result
            },
            'summary': {
                'total_files_stored': 2,
                'storage_locations': [
                    kaggle_result.get('file_path', 'Unknown'),
                    hf_result.get('file_path', 'Unknown')
                ]
            }
        }
        
        logger.info("Ingestion-storage integration completed successfully")
        logger.info(f"Telco data: {telco_data.shape} -> {kaggle_result.get('file_path', 'Unknown')}")
        logger.info(f"HF data: {hf_data.shape} -> {hf_result.get('file_path', 'Unknown')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Ingestion-storage integration failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

def main():
    """Main function to run integration."""
    print("🔄 Starting Ingestion-Storage Integration...")
    
    result = integrate_ingestion_with_storage()
    
    if result['status'] == 'success':
        print("✅ Integration completed successfully!")
        print(f"Telco data: {result['ingestion']['telco_shape']}")
        print(f"HF data: {result['ingestion']['hf_shape']}")
        print(f"Files stored: {result['summary']['total_files_stored']}")
        print("\nStorage locations:")
        for location in result['summary']['storage_locations']:
            print(f"  📁 {location}")
    else:
        print(f"❌ Integration failed: {result['error']}")

if __name__ == "__main__":
    main()
