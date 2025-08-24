"""
Data Preparation Module - Directory 5
End-to-End Data Management Pipeline for Machine Learning
Handles data cleaning, preprocessing, and preparation for machine learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Add parent directory to path for config import
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import RAW_DATA_DIR, CLEANED_DATA_DIR, DATABASE_PATH, REPORTS_DIR, LOG_FORMAT

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataPreparation:
    """Class to handle data preparation and cleaning."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputers = {}
        self.encoders = {}
        
    def load_kaggle_data(self) -> pd.DataFrame:
        """
        Load the primary Kaggle dataset for processing.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            kaggle_file = RAW_DATA_DIR / "kaggle_churn.csv"
            if not kaggle_file.exists():
                raise FileNotFoundError(f"Kaggle dataset not found at {kaggle_file}")
            
            df = pd.read_csv(kaggle_file)
            logger.info(f"Kaggle dataset loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Kaggle data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        try:
            df_clean = df.copy()
            
            # Check for missing values
            missing_summary = df_clean.isnull().sum()
            logger.info(f"Missing values summary:\n{missing_summary[missing_summary > 0]}")
            
            # Handle numeric columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    # Use median imputation for numeric columns
                    imputer = SimpleImputer(strategy='median')
                    df_clean[col] = imputer.fit_transform(df_clean[[col]])
                    self.imputers[col] = imputer
                    logger.info(f"Imputed missing values in {col} using median")
            
            # Handle categorical columns
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    # Use mode imputation for categorical columns
                    mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_value, inplace=True)
                    logger.info(f"Imputed missing values in {col} using mode: {mode_value}")
            
            logger.info("Missing values handled successfully")
            return df_clean
            
        except Exception as e:
            logger.error(f"Failed to handle missing values: {e}")
            raise
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records from the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        try:
            initial_rows = len(df)
            df_clean = df.drop_duplicates()
            final_rows = len(df_clean)
            
            duplicates_removed = initial_rows - final_rows
            logger.info(f"Removed {duplicates_removed} duplicate rows. "
                       f"Dataset reduced from {initial_rows} to {final_rows} rows")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Failed to remove duplicates: {e}")
            raise
    
    def clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize data types.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with cleaned data types
        """
        try:
            df_clean = df.copy()
            
            # Common data cleaning for Telco dataset
            # Handle TotalCharges column (often stored as string with spaces)
            if 'TotalCharges' in df_clean.columns:
                # Replace empty strings with NaN
                df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
                # Convert to numeric
                df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
                logger.info("Cleaned TotalCharges column and converted to numeric")
            
            # Handle SeniorCitizen column (convert to categorical)
            if 'SeniorCitizen' in df_clean.columns:
                df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
                logger.info("Converted SeniorCitizen to categorical")
            
            # Standardize Yes/No columns
            yes_no_columns = []
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    unique_vals = df_clean[col].unique()
                    if set(unique_vals).issubset({'Yes', 'No', 'yes', 'no'}):
                        df_clean[col] = df_clean[col].str.title()  # Standardize to Yes/No
                        yes_no_columns.append(col)
            
            if yes_no_columns:
                logger.info(f"Standardized Yes/No columns: {yes_no_columns}")
            
            logger.info("Data types cleaned successfully")
            return df_clean
            
        except Exception as e:
            logger.error(f"Failed to clean data types: {e}")
            raise
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Args:
            df: Input dataframe
            method: Method to handle outliers ('iqr', 'zscore', 'cap')
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        try:
            df_clean = df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            outliers_summary = {}
            
            for col in numeric_cols:
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_count = ((df_clean[col] < lower_bound) | 
                                    (df_clean[col] > upper_bound)).sum()
                    
                    # Cap outliers instead of removing them
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    
                    outliers_summary[col] = {
                        'outliers_found': outliers_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
            
            logger.info(f"Outliers handled using {method} method")
            logger.info(f"Outliers summary: {outliers_summary}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Failed to handle outliers: {e}")
            raise
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for machine learning.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        try:
            df_encoded = df.copy()
            
            # Drop identifier columns that shouldn't be encoded
            identifier_cols = ['customerID']  # Add other ID columns if they exist
            columns_to_drop = [col for col in identifier_cols if col in df_encoded.columns]
            if columns_to_drop:
                df_encoded = df_encoded.drop(columns=columns_to_drop)
                logger.info(f"Dropped identifier columns: {columns_to_drop}")
            
            categorical_cols = df_encoded.select_dtypes(include=['object']).columns
            
            # Binary encoding for Yes/No columns
            binary_cols = []
            for col in categorical_cols:
                unique_vals = set(df_encoded[col].unique())
                if unique_vals.issubset({'Yes', 'No'}):
                    df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
                    binary_cols.append(col)
            
            logger.info(f"Binary encoded columns: {binary_cols}")
            
            # One-hot encoding for other categorical columns (excluding high cardinality)
            remaining_categorical = [col for col in categorical_cols if col not in binary_cols]
            
            # Filter out high cardinality categorical columns (more than 50 unique values)
            high_cardinality_cols = []
            low_cardinality_cols = []
            
            for col in remaining_categorical:
                unique_count = df_encoded[col].nunique()
                if unique_count > 50:  # Threshold to prevent feature explosion
                    high_cardinality_cols.append(col)
                    # For high cardinality columns, use label encoding instead
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                else:
                    low_cardinality_cols.append(col)
            
            if high_cardinality_cols:
                logger.info(f"Label encoded high cardinality columns: {high_cardinality_cols}")
            
            if low_cardinality_cols:
                # Use get_dummies for one-hot encoding only for low cardinality columns
                df_encoded = pd.get_dummies(df_encoded, columns=low_cardinality_cols, drop_first=True)
                logger.info(f"One-hot encoded columns: {low_cardinality_cols}")
            
            logger.info(f"Final dataset shape after encoding: {df_encoded.shape}")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Failed to encode categorical variables: {e}")
            raise
    
    def normalize_features(self, df: pd.DataFrame, target_column: str = 'Churn') -> pd.DataFrame:
        """
        Normalize numeric features.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column to exclude from normalization
            
        Returns:
            pd.DataFrame: Dataframe with normalized features
        """
        try:
            df_normalized = df.copy()
            
            # Get numeric columns excluding the target
            numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
            if target_column in numeric_cols:
                numeric_cols = numeric_cols.drop(target_column)
            
            # Standardize numeric features
            if len(numeric_cols) > 0:
                df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols])
                logger.info(f"Normalized columns: {list(numeric_cols)}")
            
            return df_normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize features: {e}")
            raise
    
    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Fully prepared dataset
        """
        try:
            logger.info("Starting data preparation pipeline...")
            
            # Step 1: Clean data types
            df_clean = self.clean_data_types(df)
            
            # Step 2: Handle missing values
            df_clean = self.handle_missing_values(df_clean)
            
            # Step 3: Remove duplicates
            df_clean = self.remove_duplicates(df_clean)
            
            # Step 4: Handle outliers
            df_clean = self.handle_outliers(df_clean)
            
            # Step 5: Encode categorical variables
            df_encoded = self.encode_categorical_variables(df_clean)
            
            # Step 6: Normalize features (optional - can be done during model training)
            # df_normalized = self.normalize_features(df_encoded)
            
            logger.info("Data preparation pipeline completed successfully")
            logger.info(f"Final dataset shape: {df_encoded.shape}")
            
            return df_encoded
            
        except Exception as e:
            logger.error(f"Data preparation pipeline failed: {e}")
            raise
    
    def save_to_database(self, df: pd.DataFrame, table_name: str = "prepared_data") -> str:
        """
        Save prepared data to SQLite database with column limit handling.
        
        Args:
            df: Prepared dataframe
            table_name: Name of the table to create
            
        Returns:
            str: Path to the database file
        """
        try:
            # Check SQLite column limit (2000 columns max)
            sqlite_column_limit = 2000
            
            # Create database connection
            conn = sqlite3.connect(DATABASE_PATH)
            
            if len(df.columns) > sqlite_column_limit:
                logger.warning(f"Dataset has {len(df.columns)} columns, exceeding SQLite limit of {sqlite_column_limit}")
                logger.info("Saving only metadata to database, full data saved to CSV only")
                
                # Check if metadata table exists and get its schema
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata';")
                table_exists = cursor.fetchone() is not None
                
                if table_exists:
                    # Get existing columns
                    cursor.execute("PRAGMA table_info(metadata);")
                    existing_columns = [row[1] for row in cursor.fetchall()]
                    
                    # Use compatible metadata structure
                    if 'columns_sample' in existing_columns:
                        metadata = {
                            'table_name': [table_name],
                            'creation_timestamp': [datetime.now().isoformat()],
                            'row_count': [len(df)],
                            'column_count': [len(df.columns)],
                            'columns_sample': [','.join(df.columns.tolist()[:100])],
                            'data_location': ['CSV_FILE_ONLY'],
                            'reason': ['TOO_MANY_COLUMNS_FOR_SQLITE']
                        }
                    else:
                        # Use original schema
                        metadata = {
                            'table_name': [table_name],
                            'creation_timestamp': [datetime.now().isoformat()],
                            'row_count': [len(df)],
                            'column_count': [len(df.columns)],
                            'columns': [','.join(df.columns.tolist()[:100])]  # Truncated to fit
                        }
                else:
                    # Create new table with extended schema
                    metadata = {
                        'table_name': [table_name],
                        'creation_timestamp': [datetime.now().isoformat()],
                        'row_count': [len(df)],
                        'column_count': [len(df.columns)],
                        'columns_sample': [','.join(df.columns.tolist()[:100])],
                        'data_location': ['CSV_FILE_ONLY'],
                        'reason': ['TOO_MANY_COLUMNS_FOR_SQLITE']
                    }
                
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_sql('metadata', conn, if_exists='append', index=False)
                
                conn.close()
                
                logger.info(f"Metadata saved to database: {DATABASE_PATH}")
                logger.info(f"Full dataset must be accessed via CSV files due to column limit")
                return str(DATABASE_PATH)
            
            else:
                # Normal database saving for datasets within column limit
                # Save dataframe to database
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                
                # Create metadata table with compatible schema
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata';")
                table_exists = cursor.fetchone() is not None
                
                if table_exists:
                    # Get existing columns
                    cursor.execute("PRAGMA table_info(metadata);")
                    existing_columns = [row[1] for row in cursor.fetchall()]
                    
                    if 'columns_sample' in existing_columns:
                        metadata = {
                            'table_name': [table_name],
                            'creation_timestamp': [datetime.now().isoformat()],
                            'row_count': [len(df)],
                            'column_count': [len(df.columns)],
                            'columns_sample': [','.join(df.columns.tolist())],
                            'data_location': ['SQLITE_DATABASE'],
                            'reason': ['NORMAL_STORAGE']
                        }
                    else:
                        metadata = {
                            'table_name': [table_name],
                            'creation_timestamp': [datetime.now().isoformat()],
                            'row_count': [len(df)],
                            'column_count': [len(df.columns)],
                            'columns': [','.join(df.columns.tolist())]
                        }
                else:
                    metadata = {
                        'table_name': [table_name],
                        'creation_timestamp': [datetime.now().isoformat()],
                        'row_count': [len(df)],
                        'column_count': [len(df.columns)],
                        'columns': [','.join(df.columns.tolist())]
                    }
                
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_sql('metadata', conn, if_exists='append', index=False)
                
                conn.close()
                
                logger.info(f"Data saved to database: {DATABASE_PATH}, table: {table_name}")
                return str(DATABASE_PATH)
                
        except Exception as e:
            logger.error(f"Failed to save data to database: {e}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save prepared data to CSV file.
        
        Args:
            df: Prepared dataframe
            filename: Custom filename (optional)
            
        Returns:
            str: Path to the saved CSV file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"prepared_churn_data_{timestamp}.csv"
            
            file_path = CLEANED_DATA_DIR / filename
            df.to_csv(file_path, index=False)
            
            logger.info(f"Prepared data saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save data to CSV: {e}")
            raise
    
    def generate_preparation_report(self, original_df: pd.DataFrame, 
                                  prepared_df: pd.DataFrame) -> str:
        """
        Generate a data preparation report.
        
        Args:
            original_df: Original dataset
            prepared_df: Prepared dataset
            
        Returns:
            str: Path to the preparation report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"data_preparation_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                
                # Summary comparison
                summary_data = [
                    ['Metric', 'Original', 'Prepared'],
                    ['Number of Rows', len(original_df), len(prepared_df)],
                    ['Number of Columns', len(original_df.columns), len(prepared_df.columns)],
                    ['Missing Values', original_df.isnull().sum().sum(), prepared_df.isnull().sum().sum()],
                    ['Duplicate Rows', original_df.duplicated().sum(), prepared_df.duplicated().sum()],
                    ['Memory Usage (bytes)', original_df.memory_usage(deep=True).sum(), 
                     prepared_df.memory_usage(deep=True).sum()]
                ]
                
                summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Column comparison
                original_info = pd.DataFrame({
                    'Column': original_df.columns,
                    'Original_Type': original_df.dtypes.astype(str),
                    'Original_Nulls': original_df.isnull().sum().values
                })
                
                prepared_info = pd.DataFrame({
                    'Column': prepared_df.columns,
                    'Prepared_Type': prepared_df.dtypes.astype(str),
                    'Prepared_Nulls': prepared_df.isnull().sum().values
                })
                
                # Merge column information
                column_comparison = pd.merge(original_info, prepared_info, 
                                           on='Column', how='outer')
                column_comparison.to_excel(writer, sheet_name='Column_Comparison', index=False)
            
            logger.info(f"Data preparation report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate preparation report: {e}")
            raise

    def clean_datasets(self, telco_data: pd.DataFrame, hf_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean both datasets (main method for pipeline integration).
        
        Args:
            telco_data: Kaggle telco dataset
            hf_data: Hugging Face dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Cleaned datasets
        """
        try:
            logger.info("Starting dataset cleaning process...")
            
            # Clean Telco dataset
            logger.info("Cleaning Telco dataset...")
            telco_clean = self.prepare_dataset(telco_data)
            
            # Clean Hugging Face dataset
            logger.info("Cleaning Hugging Face dataset...")
            hf_clean = self.prepare_dataset(hf_data)
            
            # Save cleaned datasets
            telco_path = self.save_to_csv(telco_clean, "telco_clean.csv")
            hf_path = self.save_to_csv(hf_clean, "hf_clean.csv")
            
            logger.info(f"Cleaned datasets saved:")
            logger.info(f"  Telco: {telco_path}")
            logger.info(f"  HF: {hf_path}")
            
            return telco_clean, hf_clean
            
        except Exception as e:
            logger.error(f"Dataset cleaning failed: {e}")
            raise

def main():
    """Main function to run data preparation."""
    try:
        preparator = DataPreparation()
        
        # Load the primary dataset (Kaggle)
        original_df = preparator.load_kaggle_data()
        
        # Prepare the data
        prepared_df = preparator.prepare_dataset(original_df)
        
        # Save prepared data
        csv_path = preparator.save_to_csv(prepared_df)
        db_path = preparator.save_to_database(prepared_df)
        
        # Generate report
        report_path = preparator.generate_preparation_report(original_df, prepared_df)
        
        print("Data Preparation Completed Successfully!")
        print(f"Prepared data saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  Database: {db_path}")
        print(f"  Report: {report_path}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
