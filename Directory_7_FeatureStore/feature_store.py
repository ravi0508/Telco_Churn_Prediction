"""
Feature Store Module - Directory 7
End-to-End Data Management Pipeline for Machine Learning
Handles feature storage and metadata management
"""

import os
import sqlite3
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pickle

# Add parent directory to path for config import
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import BASE_DIR, FEATURE_STORE_DIR, LOG_FORMAT

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class FeatureStore:
    """Class to handle feature storage and metadata management."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the feature store.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path or (FEATURE_STORE_DIR / "feature_store.db")
        self.features_dir = FEATURE_STORE_DIR / "features"
        self.metadata_dir = FEATURE_STORE_DIR / "metadata"
        
        # Create directories
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create features table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        feature_count INTEGER,
                        row_count INTEGER,
                        file_path TEXT,
                        metadata_path TEXT,
                        UNIQUE(name, version)
                    )
                ''')
                
                # Create feature_metadata table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feature_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feature_id INTEGER,
                        column_name TEXT,
                        data_type TEXT,
                        null_count INTEGER,
                        unique_count INTEGER,
                        min_value REAL,
                        max_value REAL,
                        mean_value REAL,
                        FOREIGN KEY (feature_id) REFERENCES features (id)
                    )
                ''')
                
                conn.commit()
                logger.info("Feature store database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _generate_feature_metadata(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metadata for features and labels.
        
        Args:
            features: Features dataframe
            labels: Labels dataframe
            
        Returns:
            Dict[str, Any]: Feature metadata
        """
        try:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'features_shape': features.shape,
                'labels_shape': labels.shape,
                'features_columns': list(features.columns),
                'features_dtypes': features.dtypes.astype(str).to_dict(),
                'features_null_counts': features.isnull().sum().to_dict(),
                'features_unique_counts': features.nunique().to_dict(),
            }
            
            # Add statistical information for numeric columns
            numeric_features = features.select_dtypes(include=['number'])
            if not numeric_features.empty:
                metadata['features_stats'] = {
                    'min': numeric_features.min().to_dict(),
                    'max': numeric_features.max().to_dict(),
                    'mean': numeric_features.mean().to_dict(),
                    'std': numeric_features.std().to_dict()
                }
            
            # Add labels information
            if isinstance(labels, pd.DataFrame):
                if labels.shape[1] == 1:
                    labels_series = labels.iloc[:, 0]
                else:
                    labels_series = labels
            else:
                labels_series = labels
            
            if hasattr(labels_series, 'value_counts'):
                metadata['labels_distribution'] = labels_series.value_counts().to_dict()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to generate feature metadata: {e}")
            return {}
    
    def store_features(self, features: pd.DataFrame, labels: pd.DataFrame, 
                      name: str = "churn_features", version: str = None, 
                      description: str = None) -> Dict[str, Any]:
        """
        Store features and labels in the feature store (main method for pipeline integration).
        
        Args:
            features: Features dataframe
            labels: Labels dataframe
            name: Feature set name
            version: Feature set version
            description: Feature set description
            
        Returns:
            Dict[str, Any]: Storage results
        """
        try:
            logger.info(f"Storing features in feature store: {name}")
            
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate metadata
            metadata = self._generate_feature_metadata(features, labels)
            
            # Save features and labels to files
            features_file = self.features_dir / f"{name}_features_v{version}.csv"
            labels_file = self.features_dir / f"{name}_labels_v{version}.csv"
            metadata_file = self.metadata_dir / f"{name}_metadata_v{version}.json"
            
            # Save data
            features.to_csv(features_file, index=False)
            
            # Handle labels saving
            if isinstance(labels, pd.DataFrame):
                labels.to_csv(labels_file, index=False)
            else:
                pd.DataFrame({'labels': labels}).to_csv(labels_file, index=False)
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO features 
                    (name, version, description, feature_count, row_count, file_path, metadata_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    name, version, description or f"Feature set {name} version {version}",
                    features.shape[1], features.shape[0], str(features_file), str(metadata_file)
                ))
                
                feature_id = cursor.lastrowid
                
                # Store column metadata
                for column in features.columns:
                    col_data = features[column]
                    cursor.execute('''
                        INSERT INTO feature_metadata 
                        (feature_id, column_name, data_type, null_count, unique_count, min_value, max_value, mean_value)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        feature_id, column,
                        str(col_data.dtype),
                        col_data.isnull().sum(),
                        col_data.nunique(),
                        float(col_data.min()) if pd.api.types.is_numeric_dtype(col_data) else None,
                        float(col_data.max()) if pd.api.types.is_numeric_dtype(col_data) else None,
                        float(col_data.mean()) if pd.api.types.is_numeric_dtype(col_data) else None
                    ))
                
                conn.commit()
            
            result = {
                'status': 'success',
                'name': name,
                'version': version,
                'features_file': str(features_file),
                'labels_file': str(labels_file),
                'metadata_file': str(metadata_file),
                'feature_count': features.shape[1],
                'row_count': features.shape[0],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Features stored successfully: {name} v{version}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def load_features(self, name: str, version: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load features and labels from the feature store.
        
        Args:
            name: Feature set name
            version: Feature set version (latest if None)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and labels
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if version:
                    query = "SELECT * FROM features WHERE name = ? AND version = ?"
                    params = (name, version)
                else:
                    query = "SELECT * FROM features WHERE name = ? ORDER BY created_at DESC LIMIT 1"
                    params = (name,)
                
                result = pd.read_sql_query(query, conn, params=params)
                
                if result.empty:
                    raise ValueError(f"No features found for {name} version {version}")
                
                feature_record = result.iloc[0]
                
                # Load features and labels
                features = pd.read_csv(feature_record['file_path'])
                labels_file = feature_record['file_path'].replace('_features_', '_labels_')
                labels = pd.read_csv(labels_file)
                
                logger.info(f"Features loaded: {name} v{feature_record['version']}")
                return features, labels
                
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            raise
    
    def list_features(self) -> pd.DataFrame:
        """
        List all feature sets in the store.
        
        Returns:
            pd.DataFrame: Feature sets information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM features ORDER BY created_at DESC"
                result = pd.read_sql_query(query, conn)
                return result
                
        except Exception as e:
            logger.error(f"Failed to list features: {e}")
            return pd.DataFrame()

def main():
    """Main function to demonstrate feature store."""
    try:
        feature_store = FeatureStore()
        
        # Example usage
        print("Feature Store initialized successfully!")
        print(f"Database path: {feature_store.db_path}")
        print(f"Features directory: {feature_store.features_dir}")
        
        # List existing features
        features_list = feature_store.list_features()
        print(f"Number of feature sets: {len(features_list)}")
        
        if not features_list.empty:
            print("Existing feature sets:")
            for _, row in features_list.iterrows():
                print(f"  - {row['name']} v{row['version']} ({row['feature_count']} features, {row['row_count']} rows)")
        
    except Exception as e:
        logger.error(f"Feature store demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
