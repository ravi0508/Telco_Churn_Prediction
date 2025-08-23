"""
Raw Data Storage Module - Directory 3
End-to-End Data Management Pipeline for Machine Learning
Handles raw data storage with efficient folder structure and organization
"""

import os
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Add parent directory to path for config import
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import BASE_DIR, RAW_DATA_DIR, LOG_FORMAT

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class RawDataStorage:
    """Class to handle raw data storage operations."""
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the raw data storage system.
        
        Args:
            storage_path: Path to the raw data storage directory
        """
        self.storage_path = Path(storage_path) if storage_path else RAW_DATA_DIR
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create organized folder structure
        self._create_folder_structure()
        
    def _create_folder_structure(self):
        """Create organized folder structure for raw data storage."""
        try:
            # Create folders by source
            sources = ['kaggle', 'huggingface', 'external']
            
            for source in sources:
                source_dir = self.storage_path / source
                source_dir.mkdir(exist_ok=True)
                
                # Create subfolders by date for partitioning
                today = datetime.now().strftime('%Y-%m-%d')
                daily_dir = source_dir / today
                daily_dir.mkdir(exist_ok=True)
                
            logger.info(f"Folder structure created in: {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to create folder structure: {e}")
            raise
    
    def store_raw_data(self, data: pd.DataFrame, source: str, data_type: str, 
                      filename: str = None) -> Dict[str, Any]:
        """
        Store raw data in organized folder structure.
        
        Args:
            data: Raw data DataFrame
            source: Data source (kaggle, huggingface, external)
            data_type: Type of data (telco, customer, transaction, etc.)
            filename: Custom filename (optional)
            
        Returns:
            Dict[str, Any]: Storage result information
        """
        try:
            # Create source directory if not exists
            source_dir = self.storage_path / source
            source_dir.mkdir(exist_ok=True)
            
            # Create date-based partition
            today = datetime.now().strftime('%Y-%m-%d')
            daily_dir = source_dir / today
            daily_dir.mkdir(exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{data_type}_raw_{timestamp}.csv"
            
            # Ensure .csv extension
            if not filename.endswith('.csv'):
                filename = f"{filename}.csv"
            
            # Full file path
            file_path = daily_dir / filename
            
            # Store the data
            data.to_csv(file_path, index=False)
            
            # Create metadata
            metadata = {
                'source': source,
                'data_type': data_type,
                'filename': filename,
                'file_path': str(file_path),
                'storage_timestamp': datetime.now().isoformat(),
                'data_shape': data.shape,
                'columns': list(data.columns),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
            # Save metadata
            metadata_file = file_path.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Raw data stored: {file_path}")
            logger.info(f"Data shape: {data.shape}, Size: {metadata['file_size_mb']:.2f}MB")
            
            return {
                'status': 'success',
                'file_path': str(file_path),
                'metadata_path': str(metadata_file),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to store raw data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def list_stored_data(self, source: str = None) -> List[Dict[str, Any]]:
        """
        List all stored raw data files.
        
        Args:
            source: Filter by source (optional)
            
        Returns:
            List[Dict[str, Any]]: List of stored data information
        """
        try:
            stored_files = []
            
            # Determine which sources to check
            if source:
                sources_to_check = [source]
            else:
                sources_to_check = [d.name for d in self.storage_path.iterdir() if d.is_dir()]
            
            for src in sources_to_check:
                source_dir = self.storage_path / src
                if not source_dir.exists():
                    continue
                
                # Find all CSV files recursively
                for csv_file in source_dir.rglob('*.csv'):
                    metadata_file = csv_file.with_suffix('.json')
                    
                    if metadata_file.exists():
                        # Load metadata
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        stored_files.append(metadata)
                    else:
                        # Create basic info if no metadata
                        stored_files.append({
                            'source': src,
                            'filename': csv_file.name,
                            'file_path': str(csv_file),
                            'file_size_mb': csv_file.stat().st_size / (1024 * 1024)
                        })
            
            logger.info(f"Found {len(stored_files)} stored data files")
            return stored_files
            
        except Exception as e:
            logger.error(f"Failed to list stored data: {e}")
            return []
    
    def load_raw_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load raw data from storage.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Optional[pd.DataFrame]: Loaded data or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            data = pd.read_csv(file_path)
            logger.info(f"Loaded data from: {file_path}, Shape: {data.shape}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            return None
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old raw data files to save storage space.
        
        Args:
            days_to_keep: Number of days to keep files
            
        Returns:
            Dict[str, Any]: Cleanup results
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            files_removed = 0
            space_freed = 0
            
            for file_path in self.storage_path.rglob('*.csv'):
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_date < cutoff_date:
                    file_size = file_path.stat().st_size
                    
                    # Remove CSV and associated metadata
                    file_path.unlink()
                    metadata_file = file_path.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    files_removed += 1
                    space_freed += file_size
            
            space_freed_mb = space_freed / (1024 * 1024)
            
            logger.info(f"Cleanup completed: {files_removed} files removed, {space_freed_mb:.2f}MB freed")
            
            return {
                'status': 'success',
                'files_removed': files_removed,
                'space_freed_mb': space_freed_mb
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """Main function to demonstrate raw data storage."""
    try:
        storage = RawDataStorage()
        
        print("Raw Data Storage System initialized successfully!")
        print(f"Storage path: {storage.storage_path}")
        
        # List stored data
        stored_files = storage.list_stored_data()
        print(f"Number of stored files: {len(stored_files)}")
        
        if stored_files:
            print("Stored files:")
            for file_info in stored_files[:5]:  # Show first 5
                print(f"  - {file_info.get('filename', 'Unknown')} ({file_info.get('source', 'Unknown')})")
        
        # Show folder structure
        print(f"\nFolder structure:")
        for item in storage.storage_path.rglob('*'):
            if item.is_dir():
                print(f"  📁 {item.relative_to(storage.storage_path)}")
        
    except Exception as e:
        logger.error(f"Raw data storage demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
