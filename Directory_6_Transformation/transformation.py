"""
Data Transformation Module - Directory 6
End-to-End Data Management Pipeline for Machine Learning
Handles feature engineering and advanced data transformations
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Add parent directory to path for config import
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import RAW_DATA_DIR, CLEANED_DATA_DIR, DATABASE_PATH, FEATURE_STORE_DIR, LOG_FORMAT

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Add parent directory to path for config import
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import PROCESSED_DATA_DIR, DATABASE_PATH, FEATURE_STORE_DIR, REPORTS_DIR


# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataTransformation:
    """Class to handle feature engineering and data transformations."""
    
    def __init__(self):
        self.feature_metadata = {}
        self.transformers = {}
        
    def load_prepared_data(self) -> pd.DataFrame:
        """
        Load prepared data from database or CSV.
        
        Returns:
            pd.DataFrame: Prepared dataset
        """
        try:
            # Try loading from database first
            if DATABASE_PATH.exists():
                conn = sqlite3.connect(DATABASE_PATH)
                df = pd.read_sql_query("SELECT * FROM prepared_data", conn)
                conn.close()
                logger.info(f"Data loaded from database. Shape: {df.shape}")
            else:
                # Fallback to CSV
                csv_files = list(CLEANED_DATA_DIR.glob("prepared_churn_data_*.csv"))
                if csv_files:
                    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                    df = pd.read_csv(latest_file)
                    logger.info(f"Data loaded from CSV: {latest_file}. Shape: {df.shape}")
                else:
                    raise FileNotFoundError("No prepared data found")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load prepared data: {e}")
            raise
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with tenure features
        """
        try:
            df_transformed = df.copy()
            
            if 'tenure' in df_transformed.columns:
                # Tenure categories
                df_transformed['tenure_group'] = pd.cut(
                    df_transformed['tenure'], 
                    bins=[0, 12, 24, 48, 72, float('inf')],
                    labels=['0-1_year', '1-2_years', '2-4_years', '4-6_years', '6+_years']
                )
                
                # Tenure in years
                df_transformed['tenure_years'] = df_transformed['tenure'] / 12
                
                # Tenure squared (non-linear relationship)
                df_transformed['tenure_squared'] = df_transformed['tenure'] ** 2
                
                # Log tenure (handle zero values)
                df_transformed['tenure_log'] = np.log1p(df_transformed['tenure'])
                
                # One-hot encode tenure groups
                tenure_dummies = pd.get_dummies(df_transformed['tenure_group'], prefix='tenure')
                df_transformed = pd.concat([df_transformed, tenure_dummies], axis=1)
                df_transformed.drop('tenure_group', axis=1, inplace=True)
                
                self.feature_metadata['tenure_features'] = {
                    'created_features': ['tenure_years', 'tenure_squared', 'tenure_log'] + list(tenure_dummies.columns),
                    'description': 'Tenure-based engineered features including categories and transformations'
                }
                
                logger.info("Tenure features created successfully")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Failed to create tenure features: {e}")
            raise
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with financial features
        """
        try:
            df_transformed = df.copy()
            
            # Calculate financial ratios and aggregations
            if 'MonthlyCharges' in df_transformed.columns and 'TotalCharges' in df_transformed.columns:
                
                # Average monthly charges
                df_transformed['avg_monthly_charges'] = (
                    df_transformed['TotalCharges'] / df_transformed['tenure']
                ).replace([np.inf, -np.inf], 0)
                
                # Charges ratio
                df_transformed['total_to_monthly_ratio'] = (
                    df_transformed['TotalCharges'] / df_transformed['MonthlyCharges']
                ).replace([np.inf, -np.inf], 0)
                
                # Monthly charges categories
                df_transformed['monthly_charges_category'] = pd.cut(
                    df_transformed['MonthlyCharges'],
                    bins=[0, 35, 65, 95, float('inf')],
                    labels=['Low', 'Medium', 'High', 'Very_High']
                )
                
                # Total charges categories
                df_transformed['total_charges_category'] = pd.cut(
                    df_transformed['TotalCharges'],
                    bins=[0, 1000, 3000, 6000, float('inf')],
                    labels=['Low', 'Medium', 'High', 'Very_High']
                )
                
                # One-hot encode charge categories
                monthly_dummies = pd.get_dummies(df_transformed['monthly_charges_category'], prefix='monthly_charges')
                total_dummies = pd.get_dummies(df_transformed['total_charges_category'], prefix='total_charges')
                
                df_transformed = pd.concat([df_transformed, monthly_dummies, total_dummies], axis=1)
                df_transformed.drop(['monthly_charges_category', 'total_charges_category'], axis=1, inplace=True)
                
                self.feature_metadata['financial_features'] = {
                    'created_features': ['avg_monthly_charges', 'total_to_monthly_ratio'] + 
                                      list(monthly_dummies.columns) + list(total_dummies.columns),
                    'description': 'Financial features including ratios and charge categories'
                }
                
                logger.info("Financial features created successfully")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Failed to create financial features: {e}")
            raise
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create service-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with service features
        """
        try:
            df_transformed = df.copy()
            
            # Service columns (binary services)
            service_columns = [col for col in df_transformed.columns if any(
                service in col.lower() for service in 
                ['internet', 'phone', 'online', 'backup', 'protection', 'support', 'streaming']
            )]
            
            if service_columns:
                # Count total services
                df_transformed['total_services'] = df_transformed[service_columns].sum(axis=1)
                
                # Service density (services per dollar)
                if 'MonthlyCharges' in df_transformed.columns:
                    df_transformed['service_density'] = (
                        df_transformed['total_services'] / df_transformed['MonthlyCharges']
                    ).replace([np.inf, -np.inf], 0)
                
                # Premium services indicator
                premium_services = [col for col in service_columns if any(
                    premium in col.lower() for premium in ['streaming', 'backup', 'protection', 'support']
                )]
                
                if premium_services:
                    df_transformed['has_premium_services'] = (
                        df_transformed[premium_services].sum(axis=1) > 0
                    ).astype(int)
                
                self.feature_metadata['service_features'] = {
                    'created_features': ['total_services', 'service_density', 'has_premium_services'],
                    'description': 'Service-based features including service counts and premium indicators',
                    'service_columns_used': service_columns
                }
                
                logger.info("Service features created successfully")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Failed to create service features: {e}")
            raise
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        try:
            df_transformed = df.copy()
            
            # Tenure and charges interactions
            if all(col in df_transformed.columns for col in ['tenure', 'MonthlyCharges']):
                df_transformed['tenure_monthly_interaction'] = (
                    df_transformed['tenure'] * df_transformed['MonthlyCharges']
                )
            
            # Contract and payment method interactions
            contract_cols = [col for col in df_transformed.columns if 'contract' in col.lower()]
            payment_cols = [col for col in df_transformed.columns if 'payment' in col.lower()]
            
            if contract_cols and payment_cols:
                # Create interaction between contract type and payment method
                for contract_col in contract_cols:
                    for payment_col in payment_cols:
                        interaction_name = f"{contract_col}_{payment_col}_interaction"
                        df_transformed[interaction_name] = (
                            df_transformed[contract_col] * df_transformed[payment_col]
                        )
            
            # Senior citizen and service interactions
            if 'SeniorCitizen' in df_transformed.columns and 'total_services' in df_transformed.columns:
                df_transformed['senior_service_interaction'] = (
                    df_transformed['SeniorCitizen'] * df_transformed['total_services']
                )
            
            interaction_features = [col for col in df_transformed.columns if 'interaction' in col]
            
            self.feature_metadata['interaction_features'] = {
                'created_features': interaction_features,
                'description': 'Interaction features between key variables'
            }
            
            logger.info(f"Created {len(interaction_features)} interaction features")
            return df_transformed
            
        except Exception as e:
            logger.error(f"Failed to create interaction features: {e}")
            raise
    
    def apply_feature_scaling(self, df: pd.DataFrame, target_column: str = 'Churn', 
                            method: str = 'standard') -> pd.DataFrame:
        """
        Apply feature scaling to numeric features.
        
        Args:
            df: Input dataframe
            target_column: Target column to exclude from scaling
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        try:
            df_scaled = df.copy()
            
            # Get numeric columns excluding target
            numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
            if target_column in numeric_cols:
                numeric_cols = numeric_cols.drop(target_column)
            
            # Apply scaling
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            if len(numeric_cols) > 0:
                df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
                self.transformers[f'{method}_scaler'] = scaler
                
                logger.info(f"Applied {method} scaling to {len(numeric_cols)} numeric features")
            
            return df_scaled
            
        except Exception as e:
            logger.error(f"Failed to apply feature scaling: {e}")
            raise
    
    def select_features(self, df: pd.DataFrame, target_column: str = 'Churn', 
                       k: int = 20) -> pd.DataFrame:
        """
        Select top k features using statistical tests.
        
        Args:
            df: Input dataframe
            target_column: Target column name
            k: Number of features to select
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        try:
            if target_column not in df.columns:
                logger.warning(f"Target column {target_column} not found. Skipping feature selection.")
                return df
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle non-numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_cols]
            
            if len(X_numeric.columns) < k:
                k = len(X_numeric.columns)
            
            # Apply feature selection
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X_numeric, y)
            
            # Get selected feature names
            selected_features = X_numeric.columns[selector.get_support()]
            
            # Create final dataframe with selected features and target
            df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
            df_selected[target_column] = y
            
            self.feature_metadata['feature_selection'] = {
                'method': 'SelectKBest with f_classif',
                'selected_features': list(selected_features),
                'num_selected': len(selected_features),
                'feature_scores': dict(zip(selected_features, selector.scores_[selector.get_support()]))
            }
            
            logger.info(f"Selected {len(selected_features)} features using SelectKBest")
            return df_selected
            
        except Exception as e:
            logger.error(f"Failed to select features: {e}")
            raise
    
    def transform_dataset(self, apply_scaling: bool = True, 
                         apply_feature_selection: bool = False) -> pd.DataFrame:
        """
        Complete data transformation pipeline.
        
        Args:
            apply_scaling: Whether to apply feature scaling
            apply_feature_selection: Whether to apply feature selection
            
        Returns:
            pd.DataFrame: Fully transformed dataset
        """
        try:
            logger.info("Starting data transformation pipeline...")
            
            # Load prepared data
            df = self.load_prepared_data()
            
            # Feature engineering steps
            df_transformed = self.create_tenure_features(df)
            df_transformed = self.create_financial_features(df_transformed)
            df_transformed = self.create_service_features(df_transformed)
            df_transformed = self.create_interaction_features(df_transformed)
            
            # Feature scaling
            if apply_scaling:
                df_transformed = self.apply_feature_scaling(df_transformed)
            
            # Feature selection
            if apply_feature_selection:
                df_transformed = self.select_features(df_transformed)
            
            logger.info("Data transformation pipeline completed successfully")
            logger.info(f"Final transformed dataset shape: {df_transformed.shape}")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"Data transformation pipeline failed: {e}")
            raise
    
    def save_transformed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save transformed data to CSV.
        
        Args:
            df: Transformed dataframe
            filename: Custom filename (optional)
            
        Returns:
            str: Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"transformed_churn_data_{timestamp}.csv"
            
            file_path = CLEANED_DATA_DIR / filename
            df.to_csv(file_path, index=False)
            
            logger.info(f"Transformed data saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save transformed data: {e}")
            raise
    
    def save_feature_metadata(self) -> str:
        """
        Save feature metadata to JSON file.
        
        Returns:
            str: Path to metadata file
        """
        try:
            import json
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_path = FEATURE_STORE_DIR / f"feature_metadata_{timestamp}.json"
            
            # Add transformation timestamp
            self.feature_metadata['transformation_timestamp'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(self.feature_metadata, f, indent=2, default=str)
            
            logger.info(f"Feature metadata saved to: {metadata_path}")
            return str(metadata_path)
            
        except Exception as e:
            logger.error(f"Failed to save feature metadata: {e}")
            raise

    def transform_datasets(self, telco_clean: pd.DataFrame, hf_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform both cleaned datasets (main method for pipeline integration).
        
        Args:
            telco_clean: Cleaned Telco dataset
            hf_clean: Cleaned Hugging Face dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and labels
        """
        try:
            logger.info("Starting dataset transformation process...")
            
            # For now, use the Telco dataset as primary for transformation
            # Combine both datasets or use telco as main source
            primary_df = telco_clean.copy()
            
            # Apply transformation pipeline
            transformed_df = self.create_tenure_features(primary_df)
            transformed_df = self.create_financial_features(transformed_df)
            transformed_df = self.create_service_features(transformed_df)
            transformed_df = self.create_interaction_features(transformed_df)
            
            # Separate features and labels
            if 'Churn' in transformed_df.columns:
                target_col = 'Churn'
            elif 'churn' in transformed_df.columns:
                target_col = 'churn'
            else:
                # Create a dummy target if not present
                target_col = 'Churn'
                transformed_df[target_col] = 0
                logger.warning("No churn column found, created dummy target")
            
            # Split features and labels
            features = transformed_df.drop(columns=[target_col])
            labels = transformed_df[target_col]
            
            # Save transformed data
            features_path = self.save_transformed_data(features, "features_telco.csv")
            labels_path = self.save_transformed_data(pd.DataFrame(labels), "labels_telco.csv")
            
            logger.info(f"Transformation completed:")
            logger.info(f"  Features shape: {features.shape}")
            logger.info(f"  Labels shape: {labels.shape}")
            logger.info(f"  Features saved: {features_path}")
            logger.info(f"  Labels saved: {labels_path}")
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Dataset transformation failed: {e}")
            raise

def main():
    """Main function to run data transformation."""
    try:
        transformer = DataTransformation()
        
        # Transform the dataset
        transformed_df = transformer.transform_dataset(
            apply_scaling=True,
            apply_feature_selection=False
        )
        
        # Save transformed data and metadata
        data_path = transformer.save_transformed_data(transformed_df)
        metadata_path = transformer.save_feature_metadata()
        
        print("Data Transformation Completed Successfully!")
        print(f"Transformed data saved to: {data_path}")
        print(f"Feature metadata saved to: {metadata_path}")
        print(f"Final dataset shape: {transformed_df.shape}")
        
    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        raise

if __name__ == "__main__":
    main()
