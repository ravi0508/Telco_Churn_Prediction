"""
Data Ingestion Module - Directory 2
End-to-End Data Management Pipeline for Machine Learning
Handles data ingestion from Kaggle and Hugging Face APIs
"""

import os
import json
import pandas as pd
import kaggle
from datasets import load_dataset
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
            raise
    
    def _load_huggingface_config(self) -> Dict[str, str]:
        """Load Hugging Face API configuration."""
        try:
            with open(HUGGINGFACE_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info("Hugging Face configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load Hugging Face configuration: {e}")
            raise
    
    def setup_kaggle_api(self):
        """Setup Kaggle API authentication."""
        try:
            # Create .kaggle directory in home if it doesn't exist
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            
            # Also try the .config/kaggle directory
            config_kaggle_dir = Path.home() / ".config" / "kaggle"
            config_kaggle_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy kaggle.json to both locations
            kaggle_json_path = kaggle_dir / "kaggle.json"
            config_kaggle_json_path = config_kaggle_dir / "kaggle.json"
            
            with open(kaggle_json_path, 'w') as f:
                json.dump(self.kaggle_config, f)
            
            with open(config_kaggle_json_path, 'w') as f:
                json.dump(self.kaggle_config, f)
            
            # Set permissions
            os.chmod(kaggle_json_path, 0o600)
            os.chmod(config_kaggle_json_path, 0o600)
            
            # Configure kaggle environment variables
            os.environ['KAGGLE_USERNAME'] = self.kaggle_config['username']
            os.environ['KAGGLE_KEY'] = self.kaggle_config['key']
            os.environ['KAGGLE_CONFIG_DIR'] = str(config_kaggle_dir)
            
            logger.info("Kaggle API setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup Kaggle API: {e}")
            raise
    
    def ingest_kaggle_data(self) -> str:
        """
        Ingest data from Kaggle.
        
        Returns:
            str: Path to the downloaded data file
        """
        try:
            logger.info("Starting Kaggle data ingestion...")
            
            # Setup Kaggle API
            self.setup_kaggle_api()
            
            # Download dataset
            kaggle.api.dataset_download_files(
                KAGGLE_DATASET, 
                path=RAW_DATA_DIR, 
                unzip=True
            )
            
            # Find the downloaded CSV file
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            if csv_files:
                kaggle_file_path = csv_files[0]
                # Rename to a standard name
                standard_path = RAW_DATA_DIR / "kaggle_churn.csv"
                if kaggle_file_path != standard_path:
                    kaggle_file_path.rename(standard_path)
                
                logger.info(f"Kaggle data successfully downloaded and stored in {standard_path}")
                return str(standard_path)
            else:
                raise FileNotFoundError("No CSV file found after Kaggle download")
                
        except Exception as e:
            logger.error(f"Failed to ingest Kaggle data: {e}")
            raise
    
    def ingest_huggingface_data(self) -> str:
        """
        Ingest data from Hugging Face.
        
        Returns:
            str: Path to the downloaded data file
        """
        try:
            logger.info("Starting Hugging Face data ingestion...")
            
            # Set up authentication
            os.environ['HUGGINGFACE_HUB_TOKEN'] = self.hf_config['value']
            
            # Load dataset
            dataset = load_dataset(HUGGINGFACE_DATASET)
            
            # Convert to pandas DataFrame (assuming it has train split)
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # If no train split, use the first available split
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()
            
            # Save to CSV
            hf_file_path = RAW_DATA_DIR / "huggingface_churn.csv"
            df.to_csv(hf_file_path, index=False)
            
            logger.info(f"Hugging Face data successfully downloaded and stored in {hf_file_path}")
            return str(hf_file_path)
            
        except Exception as e:
            logger.error(f"Failed to ingest Hugging Face data: {e}")
            raise
    
    def ingest_all_data(self) -> Dict[str, str]:
        """
        Ingest data from all sources.
        
        Returns:
            Dict[str, str]: Dictionary with source names and file paths
        """
        try:
            results = {}
            
            # Ingest Kaggle data
            kaggle_path = self.ingest_kaggle_data()
            results['kaggle'] = kaggle_path
            
            # Add delay between API calls
            time.sleep(2)
            
            # Ingest Hugging Face data
            hf_path = self.ingest_huggingface_data()
            results['huggingface'] = hf_path
            
            logger.info("All data sources ingested successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to ingest all data: {e}")
            raise

    def run_ingestion(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete ingestion process and return dataframes.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Kaggle and Hugging Face dataframes
        """
        try:
            logger.info("Starting complete data ingestion process...")
            
            # Ingest all data sources
            file_paths = self.ingest_all_data()
            
            # Load the data into DataFrames
            telco_df = pd.read_csv(file_paths['kaggle'])
            hf_df = pd.read_csv(file_paths['huggingface'])
            
            logger.info(f"Kaggle dataset shape: {telco_df.shape}")
            logger.info(f"Hugging Face dataset shape: {hf_df.shape}")
            
            return telco_df, hf_df
            
        except Exception as e:
            logger.error(f"Failed to run ingestion: {e}")
            raise

def main():
    """Main function to run data ingestion."""
    try:
        ingestion = DataIngestion()
        results = ingestion.ingest_all_data()
        
        print("Data Ingestion Completed Successfully!")
        print("Downloaded files:")
        for source, path in results.items():
            print(f"  {source}: {path}")
            
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()
