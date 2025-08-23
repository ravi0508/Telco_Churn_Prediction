"""
Data Versioning Module - Directory 8
End-to-End Data Management Pipeline for Machine Learning
Handles data versioning using Git for reproducibility
"""

import os
import subprocess
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import shutil

# Add parent directory to path for config import
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import BASE_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, FEATURE_STORE_DIR, LOG_FORMAT

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataVersioning:
    """Class to handle data versioning operations."""
    
    def __init__(self, repo_path: str = None):
        """
        Initialize the data versioning system.
        
        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = repo_path or BASE_DIR
        self.version_metadata = {}
        
    def initialize_git_repo(self):
        """Initialize Git repository if not exists."""
        try:
            git_dir = self.repo_path / ".git"
            if not git_dir.exists():
                subprocess.run(["git", "init"], cwd=self.repo_path, check=True)
                logger.info("Git repository initialized")
            else:
                logger.info("Git repository already exists")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Git repository: {e}")
            raise
    
    def add_files_to_git(self, file_patterns: List[str] = None):
        """
        Add files to Git staging area.
        
        Args:
            file_patterns: List of file patterns to add
        """
        try:
            if file_patterns is None:
                file_patterns = ["*.py", "*.csv", "*.json", "README.md"]
            
            for pattern in file_patterns:
                try:
                    subprocess.run(["git", "add", pattern], cwd=self.repo_path, check=True)
                except subprocess.CalledProcessError:
                    # Pattern might not match any files, continue
                    pass
            
            logger.info(f"Files added to Git staging area: {file_patterns}")
            
        except Exception as e:
            logger.error(f"Failed to add files to Git: {e}")
            raise
    
    def commit_changes(self, message: str = None):
        """
        Commit changes to Git repository.
        
        Args:
            message: Commit message
        """
        try:
            if message is None:
                message = f"Data versioning commit - {datetime.now().isoformat()}"
            
            # Configure git user if not set
            try:
                subprocess.run(["git", "config", "user.name", "Pipeline User"], 
                             cwd=self.repo_path, check=True)
                subprocess.run(["git", "config", "user.email", "pipeline@example.com"], 
                             cwd=self.repo_path, check=True)
            except:
                pass
            
            subprocess.run(["git", "commit", "-m", message], cwd=self.repo_path, check=True)
            logger.info(f"Changes committed to Git: {message}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git commit failed (might be no changes): {e}")
    
    def create_version_tag(self, tag_name: str, message: str = None):
        """
        Create a version tag in Git.
        
        Args:
            tag_name: Name of the tag
            message: Tag message
        """
        try:
            if message is None:
                message = f"Version {tag_name} - {datetime.now().isoformat()}"
            
            subprocess.run(["git", "tag", "-a", tag_name, "-m", message], 
                         cwd=self.repo_path, check=True)
            logger.info(f"Version tag created: {tag_name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create version tag: {e}")
            raise
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "version_metadata.json"):
        """
        Save version metadata to file.
        
        Args:
            metadata: Metadata dictionary
            filename: Metadata filename
        """
        try:
            metadata_path = self.repo_path / filename
            
            # Add timestamp
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['repo_path'] = str(self.repo_path)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Version metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def version_data(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Version the data with Git (main method for pipeline integration).
        
        Args:
            features: Features dataframe
            labels: Labels dataframe
            
        Returns:
            Dict[str, Any]: Version information
        """
        try:
            logger.info("Starting data versioning process...")
            
            # Initialize Git repository
            self.initialize_git_repo()
            
            # Create version metadata
            version_info = {
                'features_shape': features.shape,
                'labels_shape': labels.shape,
                'features_columns': list(features.columns),
                'timestamp': datetime.now().isoformat(),
                'data_hash': hash(str(features.values.tobytes()) + str(labels.values.tobytes())) % (10**8)
            }
            
            # Save metadata
            self.save_metadata(version_info)
            
            # Add files to Git
            self.add_files_to_git([
                "*.py", 
                "*.csv", 
                "*.json", 
                "README.md",
                "requirements.txt",
                "version_metadata.json"
            ])
            
            # Commit changes
            commit_message = f"Data version - Features: {features.shape}, Labels: {labels.shape}"
            self.commit_changes(commit_message)
            
            # Create version tag
            version_tag = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.create_version_tag(version_tag, f"Data version {version_tag}")
            
            version_info['version_tag'] = version_tag
            version_info['commit_message'] = commit_message
            
            logger.info(f"Data versioning completed: {version_tag}")
            return version_info
            
        except Exception as e:
            logger.error(f"Data versioning failed: {e}")
            # Return basic info even if versioning fails
            return {
                'features_shape': features.shape,
                'labels_shape': labels.shape,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get version history from Git.
        
        Returns:
            List[Dict[str, Any]]: List of version information
        """
        try:
            # Get Git log
            result = subprocess.run(
                ["git", "log", "--oneline", "--decorate"], 
                cwd=self.repo_path, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            history = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    commit_hash = parts[0]
                    commit_message = parts[1] if len(parts) > 1 else ""
                    
                    history.append({
                        'commit_hash': commit_hash,
                        'commit_message': commit_message
                    })
            
            return history
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get version history: {e}")
            return []

def main():
    """Main function to demonstrate data versioning."""
    try:
        versioning = DataVersioning()
        
        # Example usage
        print("Data Versioning System initialized successfully!")
        print(f"Repository path: {versioning.repo_path}")
        
        # Get version history if available
        history = versioning.get_version_history()
        if history:
            print(f"Number of versions: {len(history)}")
            print("Latest versions:")
            for i, version in enumerate(history[:3]):
                print(f"  {i+1}. {version['commit_hash']}: {version['commit_message']}")
        else:
            print("No version history found")
            
    except Exception as e:
        logger.error(f"Data versioning demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
