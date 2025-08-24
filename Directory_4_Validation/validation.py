"""
Data Validation Module - Directory 4
End-to-End Data Management Pipeline for Machine Learning
Performs comprehensive data quality checks and generates validation reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for config import
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import RAW_DATA_DIR, REPORTS_DIR, LOG_FORMAT

# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataValidator:
    """Class to perform comprehensive data validation."""
    
    def __init__(self):
        self.validation_results = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def check_missing_values(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """
        Check for missing values in the dataset.
        
        Args:
            df: Input dataframe
            source: Data source name
            
        Returns:
            Dict: Missing values analysis
        """
        try:
            missing_info = {
                'source': source,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values_count': df.isnull().sum().to_dict(),
                'missing_values_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'columns_with_missing': df.columns[df.isnull().any()].tolist(),
                'rows_with_missing': df.isnull().any(axis=1).sum(),
                'percentage_rows_with_missing': (df.isnull().any(axis=1).sum() / len(df)) * 100
            }
            
            logger.info(f"Missing values check completed for {source}")
            return missing_info
            
        except Exception as e:
            logger.error(f"Failed to check missing values: {e}")
            raise
    
    def check_data_types(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """
        Check data types and type consistency.
        
        Args:
            df: Input dataframe
            source: Data source name
            
        Returns:
            Dict: Data type analysis
        """
        try:
            type_info = {
                'source': source,
                'column_types': df.dtypes.astype(str).to_dict(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
                'mixed_type_columns': []
            }
            
            # Check for mixed types in object columns
            for col in type_info['categorical_columns']:
                sample_types = df[col].dropna().apply(type).value_counts()
                if len(sample_types) > 1:
                    type_info['mixed_type_columns'].append({
                        'column': col,
                        'types_found': sample_types.to_dict()
                    })
            
            logger.info(f"Data type check completed for {source}")
            return type_info
            
        except Exception as e:
            logger.error(f"Failed to check data types: {e}")
            raise
    
    def check_duplicates(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """
        Check for duplicate records.
        
        Args:
            df: Input dataframe
            source: Data source name
            
        Returns:
            Dict: Duplicate analysis
        """
        try:
            duplicate_info = {
                'source': source,
                'total_duplicates': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                'unique_rows': len(df.drop_duplicates()),
                'columns_with_all_duplicates': []
            }
            
            # Check for columns with all duplicate values
            for col in df.columns:
                if df[col].nunique() == 1:
                    duplicate_info['columns_with_all_duplicates'].append(col)
            
            logger.info(f"Duplicate check completed for {source}")
            return duplicate_info
            
        except Exception as e:
            logger.error(f"Failed to check duplicates: {e}")
            raise
    
    def check_data_ranges(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """
        Check data ranges and identify potential outliers.
        
        Args:
            df: Input dataframe
            source: Data source name
            
        Returns:
            Dict: Data range analysis
        """
        try:
            range_info = {
                'source': source,
                'numeric_ranges': {},
                'negative_values': {},
                'zero_values': {},
                'outliers': {}
            }
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                
                range_info['numeric_ranges'][col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'median': float(col_data.median())
                }
                
                # Check for negative values
                negative_count = (col_data < 0).sum()
                range_info['negative_values'][col] = {
                    'count': int(negative_count),
                    'percentage': float((negative_count / len(col_data)) * 100)
                }
                
                # Check for zero values
                zero_count = (col_data == 0).sum()
                range_info['zero_values'][col] = {
                    'count': int(zero_count),
                    'percentage': float((zero_count / len(col_data)) * 100)
                }
                
                # Check for outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                range_info['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': float((len(outliers) / len(col_data)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            
            logger.info(f"Data range check completed for {source}")
            return range_info
            
        except Exception as e:
            logger.error(f"Failed to check data ranges: {e}")
            raise
    
    def check_categorical_values(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """
        Check categorical value distributions and consistency.
        
        Args:
            df: Input dataframe
            source: Data source name
            
        Returns:
            Dict: Categorical analysis
        """
        try:
            categorical_info = {
                'source': source,
                'categorical_distributions': {},
                'high_cardinality_columns': [],
                'low_cardinality_columns': []
            }
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                unique_count = df[col].nunique()
                
                categorical_info['categorical_distributions'][col] = {
                    'unique_values': int(unique_count),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head().to_dict()
                }
                
                # Classify as high or low cardinality
                if unique_count > len(df) * 0.5:  # More than 50% unique values
                    categorical_info['high_cardinality_columns'].append(col)
                elif unique_count < 10:  # Less than 10 unique values
                    categorical_info['low_cardinality_columns'].append(col)
            
            logger.info(f"Categorical values check completed for {source}")
            return categorical_info
            
        except Exception as e:
            logger.error(f"Failed to check categorical values: {e}")
            raise
    
    def validate_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive validation on a dataset.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Dict: Complete validation results
        """
        try:
            source = Path(file_path).stem
            df = self.load_data(file_path)
            
            validation_results = {
                'source': source,
                'file_path': file_path,
                'validation_timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
            }
            
            # Perform all validation checks
            validation_results['missing_values'] = self.check_missing_values(df, source)
            validation_results['data_types'] = self.check_data_types(df, source)
            validation_results['duplicates'] = self.check_duplicates(df, source)
            validation_results['data_ranges'] = self.check_data_ranges(df, source)
            validation_results['categorical_values'] = self.check_categorical_values(df, source)
            
            # Summary of issues
            validation_results['summary'] = self._generate_summary(validation_results)
            
            logger.info(f"Dataset validation completed for {source}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate dataset: {e}")
            raise
    
    def validate_datasets(self, telco_data: pd.DataFrame, hf_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate multiple datasets passed as DataFrames.
        
        Args:
            telco_data: Telco customer DataFrame
            hf_data: Hugging Face DataFrame
            
        Returns:
            Dict: Combined validation results for both datasets
        """
        try:
            logger.info("Starting validation of multiple datasets")
            
            combined_results = {
                'validation_timestamp': datetime.now().isoformat(),
                'datasets_validated': ['telco', 'hf'],
                'telco_validation': {},
                'hf_validation': {},
                'combined_summary': {}
            }
            
            # Validate telco dataset
            combined_results['telco_validation'] = self._validate_dataframe(telco_data, 'telco')
            
            # Validate HF dataset
            combined_results['hf_validation'] = self._validate_dataframe(hf_data, 'hf')
            
            # Generate combined summary
            combined_results['combined_summary'] = self._generate_combined_summary(
                combined_results['telco_validation'], 
                combined_results['hf_validation']
            )
            
            logger.info("Multi-dataset validation completed successfully")
            return combined_results
            
        except Exception as e:
            logger.error(f"Failed to validate datasets: {e}")
            raise
    
    def _validate_dataframe(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """Validate a single DataFrame."""
        validation_results = {
            'source': source,
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        }
        
        # Perform all validation checks
        validation_results['missing_values'] = self.check_missing_values(df, source)
        validation_results['data_types'] = self.check_data_types(df, source)
        validation_results['duplicates'] = self.check_duplicates(df, source)
        validation_results['data_ranges'] = self.check_data_ranges(df, source)
        validation_results['categorical_values'] = self.check_categorical_values(df, source)
        
        # Summary of issues
        validation_results['summary'] = self._generate_summary(validation_results)
        
        return validation_results
    
    def _generate_combined_summary(self, telco_results: Dict, hf_results: Dict) -> Dict[str, Any]:
        """Generate a combined summary from multiple dataset validations."""
        combined_summary = {
            'total_datasets': 2,
            'total_records': telco_results['dataset_info']['shape'][0] + hf_results['dataset_info']['shape'][0],
            'total_features': telco_results['dataset_info']['shape'][1] + hf_results['dataset_info']['shape'][1],
            'overall_issues': telco_results['summary']['total_issues'] + hf_results['summary']['total_issues'],
            'dataset_summaries': {
                'telco': telco_results['summary'],
                'hf': hf_results['summary']
            },
            'recommendations': []
        }
        
        # Add combined recommendations
        if combined_summary['overall_issues'] > 0:
            combined_summary['recommendations'].append("Review individual dataset issues")
            combined_summary['recommendations'].append("Address missing values and duplicates before modeling")
        else:
            combined_summary['recommendations'].append("Datasets appear to be in good quality for modeling")
        
        return combined_summary
    
    def _generate_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of validation issues."""
        summary = {
            'total_issues': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for missing values
        missing_cols = validation_results['missing_values']['columns_with_missing']
        if missing_cols:
            summary['warnings'].append(f"Missing values found in {len(missing_cols)} columns")
            summary['recommendations'].append("Consider imputation or removal of missing values")
        
        # Check for duplicates
        duplicate_count = validation_results['duplicates']['total_duplicates']
        if duplicate_count > 0:
            summary['warnings'].append(f"{duplicate_count} duplicate rows found")
            summary['recommendations'].append("Remove duplicate rows before analysis")
        
        # Check for high cardinality columns
        high_card_cols = validation_results['categorical_values']['high_cardinality_columns']
        if high_card_cols:
            summary['warnings'].append(f"High cardinality columns found: {high_card_cols}")
            summary['recommendations'].append("Consider feature engineering for high cardinality columns")
        
        # Check for outliers
        outlier_info = validation_results['data_ranges']['outliers']
        high_outlier_cols = [col for col, info in outlier_info.items() if info['percentage'] > 5]
        if high_outlier_cols:
            summary['warnings'].append(f"High percentage of outliers in: {high_outlier_cols}")
            summary['recommendations'].append("Investigate and handle outliers appropriately")
        
        summary['total_issues'] = len(summary['critical_issues']) + len(summary['warnings'])
        
        return summary
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            str: Path to the generated report
        """
        try:
            source = validation_results['source']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON report
            json_report_path = REPORTS_DIR / f"{source}_validation_report_{timestamp}.json"
            with open(json_report_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            # Excel report
            excel_report_path = REPORTS_DIR / f"{source}_validation_report_{timestamp}.xlsx"
            with pd.ExcelWriter(excel_report_path, engine='openpyxl') as writer:
                
                # Summary sheet
                summary_data = []
                summary = validation_results['summary']
                summary_data.append(['Total Issues', summary['total_issues']])
                summary_data.append(['Critical Issues', len(summary['critical_issues'])])
                summary_data.append(['Warnings', len(summary['warnings'])])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Missing values sheet
                missing_data = validation_results['missing_values']['missing_values_count']
                missing_df = pd.DataFrame(list(missing_data.items()), 
                                        columns=['Column', 'Missing_Count'])
                missing_df['Missing_Percentage'] = [
                    validation_results['missing_values']['missing_values_percentage'][col] 
                    for col in missing_df['Column']
                ]
                missing_df.to_excel(writer, sheet_name='Missing_Values', index=False)
                
                # Data types sheet
                types_data = validation_results['data_types']['column_types']
                types_df = pd.DataFrame(list(types_data.items()), 
                                      columns=['Column', 'Data_Type'])
                types_df.to_excel(writer, sheet_name='Data_Types', index=False)
                
                # Outliers sheet
                outliers_data = []
                for col, info in validation_results['data_ranges']['outliers'].items():
                    outliers_data.append([col, info['count'], info['percentage']])
                
                outliers_df = pd.DataFrame(outliers_data, 
                                         columns=['Column', 'Outlier_Count', 'Outlier_Percentage'])
                outliers_df.to_excel(writer, sheet_name='Outliers', index=False)
            
            logger.info(f"Validation report generated: {excel_report_path}")
            return str(excel_report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            raise
    
    def validate_all_datasets(self) -> Dict[str, str]:
        """
        Validate all datasets in the raw data directory.
        
        Returns:
            Dict[str, str]: Dictionary with dataset names and report paths
        """
        try:
            reports = {}
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            
            if not csv_files:
                logger.warning("No CSV files found in raw data directory")
                return reports
            
            for csv_file in csv_files:
                logger.info(f"Validating dataset: {csv_file}")
                validation_results = self.validate_dataset(str(csv_file))
                report_path = self.generate_validation_report(validation_results)
                reports[csv_file.stem] = report_path
            
            logger.info("All datasets validated successfully")
            return reports
            
        except Exception as e:
            logger.error(f"Failed to validate all datasets: {e}")
            raise

def main():
    """Main function to run data validation."""
    try:
        validator = DataValidator()
        reports = validator.validate_all_datasets()
        
        print("Data Validation Completed Successfully!")
        print("Generated reports:")
        for dataset, report_path in reports.items():
            print(f"  {dataset}: {report_path}")
            
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise

if __name__ == "__main__":
    main()
