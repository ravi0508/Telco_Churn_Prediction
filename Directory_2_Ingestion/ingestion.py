#!/usr/bin/env python3
"""
Modified Data Ingestion Module with Lazy Imports
Avoids hanging on datasets import
"""

import os
import json
import pandas as pd
from pathlib import Path
import requests
import time
import logging
from typing import Dict, Any, Tuple

# Add parent directory to path for config import
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import (
    KAGGLE_CONFIG_PATH, HUGGINGFACE_CONFIG_PATH, RAW_DATA_DIR, 
    KAGGLE_DATASET, HUGGINGFACE_DATASET, LOG_FORMAT
)

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Class to handle data ingestion from multiple sources."""
    
    def __init__(self):
        self.kaggle_config = self._load_kaggle_config()
        self.hf_config = self._load_huggingface_config()
        
    def _load_kaggle_config(self) -> Dict[str, str]:
        """Load Kaggle API configuration."""
        try:
            with open(KAGGLE_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info("Kaggle configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load Kaggle configuration: {e}")
            return {}
    
    def _load_huggingface_config(self) -> Dict[str, str]:
        """Load Hugging Face API configuration."""
        try:
            with open(HUGGINGFACE_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info("Hugging Face configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load Hugging Face configuration: {e}")
            return {}
    
    def _setup_kaggle_credentials(self):
        """Setup Kaggle API credentials."""
        try:
            # Create .kaggle directory
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_dir.mkdir(exist_ok=True)
            
            # Copy kaggle.json to ~/.kaggle/
            dest_path = kaggle_dir / 'kaggle.json'
            with open(KAGGLE_CONFIG_PATH, 'r') as src, open(dest_path, 'w') as dest:
                dest.write(src.read())
            
            # Set proper permissions
            dest_path.chmod(0o600)
            logger.info("Kaggle credentials setup completed")
            return True
        except Exception as e:
            logger.error(f"Failed to setup Kaggle credentials: {e}")
            return False
    
    def ingest_kaggle_data(self) -> pd.DataFrame:
        """Ingest data from Kaggle using kagglehub with timeout and fallback."""
        try:
            # Setup credentials
            if not self._setup_kaggle_credentials():
                raise Exception("Failed to setup Kaggle credentials")
            
            # Import kagglehub for modern API
            try:
                import kagglehub
                from kagglehub import KaggleDatasetAdapter
            except ImportError:
                logger.warning("kagglehub not available, falling back to legacy kaggle API")
                return self._ingest_kaggle_legacy()
            
            # Download dataset using kagglehub
            logger.info(f"Downloading Kaggle dataset using kagglehub: {KAGGLE_DATASET}")
            
            try:
                # Load dataset directly as pandas DataFrame
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    KAGGLE_DATASET,
                    "",  # Empty file_path to get the main dataset file
                )
                
                logger.info(f"Kaggle data loaded successfully using kagglehub: {df.shape}")
                logger.info(f"First 5 records:\n{df.head()}")
                
                # Save to raw data directory for pipeline consistency
                output_path = RAW_DATA_DIR / "kaggle_churn.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Data saved to: {output_path}")
                
                return df
                
            except Exception as kagglehub_error:
                logger.warning(f"kagglehub method failed: {kagglehub_error}")
                logger.info("Falling back to legacy kaggle API")
                return self._ingest_kaggle_legacy()
                    
        except Exception as e:
            logger.warning(f"Kaggle data ingestion failed: {e}")
            logger.info("Using existing Kaggle dataset as fallback")
            return pd.read_csv(Path(RAW_DATA_DIR) / 'kaggle_churn.csv')
    
    def _ingest_kaggle_legacy(self) -> pd.DataFrame:
        """Legacy Kaggle API ingestion method as fallback."""
        try:
            import kaggle
            import signal
            
            logger.info("Using legacy Kaggle API for dataset download")
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Kaggle download timeout")
            
            # Set alarm for 20 seconds total timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(20)
            
            try:
                kaggle.api.dataset_download_files(
                    KAGGLE_DATASET, 
                    path=RAW_DATA_DIR, 
                    unzip=True
                )
                
                # Cancel alarm
                signal.alarm(0)
                
                # Find the CSV file
                csv_files = list(Path(RAW_DATA_DIR).glob('*.csv'))
                if not csv_files:
                    raise Exception("No CSV files found after download")
                
                data = pd.read_csv(csv_files[0])
                logger.info(f"Kaggle data loaded successfully via legacy API: {data.shape}")
                return data
                
            except (TimeoutError, Exception) as e:
                signal.alarm(0)  # Cancel alarm
                raise e
                    
        except Exception as e:
            logger.warning(f"Kaggle data API ingestion failed: {e}")
            logger.info("Using existing Kaggle dataset as fallback")
            return pd.read_csv(Path(RAW_DATA_DIR) / 'kaggle_churn.csv')
    
    def ingest_huggingface_data(self) -> pd.DataFrame:
        """Ingest data from Hugging Face using hf:// protocol with fallback."""
        try:
            logger.info(f"Loading Hugging Face dataset using hf:// protocol: {HUGGINGFACE_DATASET}")
            
            # Use modern hf:// protocol to load dataset directly
            try:
                df = pd.read_csv("hf://datasets/scikit-learn/churn-prediction/dataset.csv")
                
                logger.info(f"Hugging Face data loaded successfully using hf:// protocol: {df.shape}")
                logger.info(f"First 5 records:\n{df.head()}")
                
                # Save to raw data directory for pipeline consistency
                output_path = RAW_DATA_DIR / "hf_data.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"HF data saved to: {output_path}")
                
                return df
                
            except Exception as hf_error:
                logger.warning(f"hf:// protocol method failed: {hf_error}")
                logger.info("Falling back to existing HF dataset")
                return pd.read_csv(Path(RAW_DATA_DIR) / 'hf_data.csv')
            
        except Exception as e:
            logger.warning(f"Hugging Face data ingestion failed: {e}")
            logger.info("Using existing HF dataset as fallback")
            return pd.read_csv(Path(RAW_DATA_DIR) / 'hf_data.csv')

    
    def run_ingestion(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run complete data ingestion process."""
        logger.info("Starting data ingestion process")
        
        # Create raw data directory
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        
        # Check if data already exists to avoid re-downloading
        telco_path = Path(RAW_DATA_DIR) / 'kaggle_churn.csv'
        hf_path = Path(RAW_DATA_DIR) / 'hf_data.csv'
        
        if telco_path.exists() and hf_path.exists():
            logger.info("Using existing data files")
            telco_data = pd.read_csv(telco_path)
            hf_data = pd.read_csv(hf_path)
            logger.info(f"Loaded existing data - Telco: {telco_data.shape}, HF: {hf_data.shape}")
        else:
            # Ingest data from both sources
            telco_data = self.ingest_kaggle_data()
            hf_data = self.ingest_huggingface_data()
            
            # Save raw data
            telco_data.to_csv(telco_path, index=False)
            hf_data.to_csv(hf_path, index=False)
            
            logger.info(f"Data ingestion completed - Telco: {telco_data.shape}, HF: {hf_data.shape}")
        
        return telco_data, hf_data

# Test the module
if __name__ == "__main__":
    print("Testing modified ingestion module...")
    ingestion = DataIngestion()
    telco_data, hf_data = ingestion.run_ingestion()
    print(f"✅ Ingestion test completed - Telco: {telco_data.shape}, HF: {hf_data.shape}")
