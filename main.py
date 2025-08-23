"""
Main Pipeline Execution Script
End-to-End Data Management Pipeline for Machine Learning - Customer Churn Prediction
Runs the complete pipeline from data ingestion to model building
"""

import sys
import os
from pathlib import Path
import logging
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import configuration
from config import *

# Import pipeline modules from their respective directories
sys.path.append(str(INGESTION_DIR))
sys.path.append(str(VALIDATION_DIR))
sys.path.append(str(PREPARATION_DIR))
sys.path.append(str(TRANSFORMATION_DIR))
sys.path.append(str(FEATURE_STORE_DIR))
sys.path.append(str(VERSIONING_DIR))
sys.path.append(str(MODEL_BUILDING_DIR))

from ingestion import DataIngestion
from validation import DataValidator
from preparation import DataPreparation
from transformation import DataTransformation
from feature_store import FeatureStore
from versioning import DataVersioning
from model_building import ModelBuilder

def setup_logging():
    """Set up logging configuration."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOGS_DIR / "main_pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Execute the complete churn prediction pipeline."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting End-to-End Data Management Pipeline for Customer Churn Prediction")
        
        # Stage 1: Data Ingestion
        logger.info("Stage 1: Data Ingestion")
        ingestion = DataIngestion()
        telco_data, hf_data = ingestion.run_ingestion()
        logger.info("Data ingestion completed successfully")
        
        # Stage 2: Data Validation
        logger.info("Stage 2: Data Validation")
        validator = DataValidator()
        validation_results = validator.validate_datasets(telco_data, hf_data)
        logger.info(f"Data validation completed: {validation_results}")
        
        # Stage 3: Data Preparation
        logger.info("Stage 3: Data Preparation")
        preparation = DataPreparation()
        telco_clean, hf_clean = preparation.clean_datasets(telco_data, hf_data)
        logger.info("Data preparation completed successfully")
        
        # Stage 4: Data Transformation
        logger.info("Stage 4: Data Transformation")
        transformation = DataTransformation()
        features, labels = transformation.transform_datasets(telco_clean, hf_clean)
        logger.info("Data transformation completed successfully")
        
        # Stage 5: Feature Store Management
        logger.info("Stage 5: Feature Store Management")
        feature_store = FeatureStore()
        feature_store.store_features(features, labels)
        logger.info("Feature store management completed successfully")
        
        # Stage 6: Data Versioning
        logger.info("Stage 6: Data Versioning")
        versioning = DataVersioning()
        version_info = versioning.version_data(features, labels)
        logger.info(f"Data versioning completed: {version_info}")
        
        # Stage 7: Model Building
        logger.info("Stage 7: Model Building")
        model_builder = ModelBuilder()
        best_model, metrics = model_builder.build_and_evaluate_models(features, labels)
        logger.info(f"Model building completed - Best model: {best_model}")
        logger.info(f"Model metrics: {metrics}")
        
        logger.info("Pipeline completed successfully!")
        
        return {
            'status': 'success',
            'best_model': best_model,
            'metrics': metrics,
            'version_info': version_info
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    setup_logging()
    result = main()
    print(f"Pipeline execution result: {result}")
