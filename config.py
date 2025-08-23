"""
Configuration file for End-to-End Data Management Pipeline for Machine Learning
Customer Churn Prediction Pipeline - Assignment I
"""

import os
import logging
from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Main data directories (matching the reference structure)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"

# Directory structure following the reference project
INGESTION_DIR = BASE_DIR / "Directory_2_Ingestion"
STAGING_DIR = INGESTION_DIR / "staging"

RAW_STORAGE_DIR = BASE_DIR / "Directory_3_RawStorage"
RAW_ARCHIVE_DIR = RAW_STORAGE_DIR / "raw_archive"

VALIDATION_DIR = BASE_DIR / "Directory_4_Validation"

PREPARATION_DIR = BASE_DIR / "Directory_5_Preparation"
CLEANED_DATA_DIR = PREPARATION_DIR / "cleaned"

TRANSFORMATION_DIR = BASE_DIR / "Directory_6_Transformation"

FEATURE_STORE_DIR = BASE_DIR / "Directory_7_FeatureStore"

VERSIONING_DIR = BASE_DIR / "Directory_8_Versioning"

MODEL_BUILDING_DIR = BASE_DIR / "Directory_9_ModelBuilding"

ORCHESTRATION_DIR = BASE_DIR / "Directory_10_Orchestration"

# Legacy directories (maintain for backward compatibility)
PROCESSED_DATA_DIR = CLEANED_DATA_DIR  # Alias for backward compatibility
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Database and file paths
FEATURE_STORE_DB = FEATURE_STORE_DIR / "feature_store.db"
DATABASE_PATH = CLEANED_DATA_DIR / "churn_data.db"

# API Configuration
KAGGLE_CONFIG_PATH = "/home/jupyter/DMML/kaggle.json"
HUGGINGFACE_CONFIG_PATH = "/home/jupyter/DMML/hugging_face.json"

# Dataset configurations
KAGGLE_DATASET = "blastchar/telco-customer-churn"
HUGGINGFACE_DATASET = "scikit-learn/churn-prediction"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

# Model configuration
MODEL_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
RANDOM_STATE = 42

# File naming patterns (following reference project)
TELCO_RAW_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
HF_RAW_FILE = "huggingface_churn.csv"
TELCO_CLEAN_FILE = "telco_clean.csv"
HF_CLEAN_FILE = "hf_clean.csv"
FEATURES_FILE = "features_telco.csv"
LABELS_FILE = "labels_telco.csv"

# Create timestamp-based folder structure for archiving
def get_timestamped_path(base_path: Path) -> Path:
    """Create timestamped folder structure: YYYY/MM/DD"""
    now = datetime.now()
    return base_path / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"

# Create all directories if they don't exist
def create_directory_structure():
    """Create the complete directory structure."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, TRANSFORMED_DATA_DIR,
        INGESTION_DIR, STAGING_DIR,
        RAW_STORAGE_DIR, RAW_ARCHIVE_DIR,
        VALIDATION_DIR,
        PREPARATION_DIR, CLEANED_DATA_DIR,
        TRANSFORMATION_DIR,
        FEATURE_STORE_DIR,
        VERSIONING_DIR,
        MODEL_BUILDING_DIR,
        ORCHESTRATION_DIR,
        MODELS_DIR, LOGS_DIR, REPORTS_DIR, DOCS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directory structure
create_directory_structure()
