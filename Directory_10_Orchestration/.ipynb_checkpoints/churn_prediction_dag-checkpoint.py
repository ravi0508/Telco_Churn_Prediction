"""
Airflow DAG for Churn Prediction Pipeline
This DAG orchestrates the complete end-to-end data management and ML pipeline
"""

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project source directory to Python path
project_root = Path('/home/jupyter/DMML/churn_prediction_pipeline')
sys.path.append(str(project_root))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

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

# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create the DAG
dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction data pipeline',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'churn', 'data-pipeline']
)

def task_data_ingestion(**context):
    """Task function for data ingestion."""
    try:
        ingestion = DataIngestion()
        results = ingestion.ingest_all_data()
        
        # Log results for downstream tasks
        context['task_instance'].xcom_push(key='ingestion_results', value=results)
        
        print(f"Data ingestion completed. Files: {results}")
        return "Data ingestion successful"
        
    except Exception as e:
        print(f"Data ingestion failed: {e}")
        raise

def task_data_validation(**context):
    """Task function for data validation."""
    try:
        validator = DataValidator()
        reports = validator.validate_all_datasets()
        
        # Log results
        context['task_instance'].xcom_push(key='validation_reports', value=reports)
        
        print(f"Data validation completed. Reports: {reports}")
        return "Data validation successful"
        
    except Exception as e:
        print(f"Data validation failed: {e}")
        raise

def prepare_data(**context):
    """Data preparation task."""
    try:
        preparator = DataPreparation()
        # Get data from previous task
        ti = context['ti']
        telco_data = ti.xcom_pull(task_ids='ingest_data', key='telco_data')
        hf_data = ti.xcom_pull(task_ids='ingest_data', key='hf_data')
        
        # Prepare data
        telco_clean, hf_clean = preparator.clean_datasets(telco_data, hf_data)
        
        # Push to XCom for next task
        ti.xcom_push(key='telco_clean', value=telco_clean)
        ti.xcom_push(key='hf_clean', value=hf_clean)
        
        print("Data preparation completed successfully")
        
    except Exception as e:
        print(f"Data preparation failed: {e}")
        raise

def task_data_transformation(**context):
    """Task function for data transformation."""
    try:
        transformer = DataTransformation()
        
        # Transform dataset
        transformed_df = transformer.transform_dataset(
            apply_scaling=True,
            apply_feature_selection=False
        )
        
        # Save transformed data and metadata
        data_path = transformer.save_transformed_data(transformed_df)
        metadata_path = transformer.save_feature_metadata()
        
        # Log results
        results = {
            'data_path': data_path,
            'metadata_path': metadata_path,
            'shape': transformed_df.shape
        }
        context['task_instance'].xcom_push(key='transformation_results', value=results)
        
        print(f"Data transformation completed. Results: {results}")
        return "Data transformation successful"
        
    except Exception as e:
        print(f"Data transformation failed: {e}")
        raise

def task_feature_store(**context):
    """Task function for feature store operations."""
    try:
        feature_store = FeatureStore()
        
        # Populate feature store
        status = feature_store.load_and_populate_from_transformed_data()
        
        # Get statistics
        stats = feature_store.get_feature_statistics()
        
        # Export catalog
        catalog_path = feature_store.export_feature_catalog()
        
        # Log results
        results = {
            'status': status,
            'statistics': stats,
            'catalog_path': catalog_path
        }
        context['task_instance'].xcom_push(key='feature_store_results', value=results)
        
        print(f"Feature store completed. Results: {results}")
        return "Feature store successful"
        
    except Exception as e:
        print(f"Feature store failed: {e}")
        raise

def task_data_versioning(**context):
    """Task function for data versioning."""
    try:
        dvc = DataVersioning()
        status = dvc.setup_data_versioning()
        
        # Get version history
        history = dvc.get_version_history()
        
        # Log results
        results = {
            'status': status,
            'version_count': len(history)
        }
        context['task_instance'].xcom_push(key='versioning_results', value=results)
        
        print(f"Data versioning completed. Results: {results}")
        return "Data versioning successful"
        
    except Exception as e:
        print(f"Data versioning failed: {e}")
        raise

def task_model_building(**context):
    """Task function for model building."""
    try:
        model_builder = ModelBuilder()
        results = model_builder.build_and_evaluate_models()
        
        # Extract key information for logging
        summary_results = {
            'best_model_name': results['best_model']['name'],
            'best_model_f1': results['best_model']['metrics']['f1_score'],
            'best_model_accuracy': results['best_model']['metrics']['accuracy'],
            'report_path': results['report_path'],
            'data_shape': results['data_shape'],
            'feature_count': results['feature_count']
        }
        
        context['task_instance'].xcom_push(key='model_results', value=summary_results)
        
        print(f"Model building completed. Best model: {summary_results['best_model_name']} "
              f"(F1: {summary_results['best_model_f1']:.4f})")
        return "Model building successful"
        
    except Exception as e:
        print(f"Model building failed: {e}")
        raise

def task_pipeline_summary(**context):
    """Task function to generate pipeline summary."""
    try:
        # Get results from all previous tasks
        ingestion_results = context['task_instance'].xcom_pull(
            task_ids='data_ingestion', key='ingestion_results')
        validation_results = context['task_instance'].xcom_pull(
            task_ids='data_validation', key='validation_reports')
        preparation_results = context['task_instance'].xcom_pull(
            task_ids='data_preparation', key='preparation_results')
        transformation_results = context['task_instance'].xcom_pull(
            task_ids='data_transformation', key='transformation_results')
        feature_store_results = context['task_instance'].xcom_pull(
            task_ids='feature_store', key='feature_store_results')
        versioning_results = context['task_instance'].xcom_pull(
            task_ids='data_versioning', key='versioning_results')
        model_results = context['task_instance'].xcom_pull(
            task_ids='model_building', key='model_results')
        
        # Create summary
        summary = {
            'pipeline_execution_date': context['ds'],
            'pipeline_status': 'SUCCESS',
            'ingestion': ingestion_results,
            'validation': validation_results,
            'preparation': preparation_results,
            'transformation': transformation_results,
            'feature_store': feature_store_results,
            'versioning': versioning_results,
            'model_building': model_results
        }
        
        # Save summary to file
        import json
        from src.config import REPORTS_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = REPORTS_DIR / f"pipeline_summary_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Pipeline Summary:")
        print(f"  Execution Date: {summary['pipeline_execution_date']}")
        print(f"  Status: {summary['pipeline_status']}")
        if model_results:
            print(f"  Best Model: {model_results['best_model_name']}")
            print(f"  Model F1 Score: {model_results['best_model_f1']:.4f}")
            print(f"  Final Dataset Shape: {model_results['data_shape']}")
        print(f"  Summary saved to: {summary_path}")
        
        return "Pipeline summary completed"
        
    except Exception as e:
        print(f"Pipeline summary failed: {e}")
        raise

# Define tasks
data_ingestion_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=task_data_ingestion,
    dag=dag,
    doc_md="""
    ## Data Ingestion Task
    Downloads data from Kaggle and Hugging Face APIs.
    - Kaggle: Telco Customer Churn dataset
    - Hugging Face: Churn prediction dataset
    """
)

data_validation_task = PythonOperator(
    task_id='data_validation',
    python_callable=task_data_validation,
    dag=dag,
    doc_md="""
    ## Data Validation Task
    Performs comprehensive data quality checks including:
    - Missing values analysis
    - Data type validation
    - Duplicate detection
    - Outlier identification
    """
)

data_preparation_task = PythonOperator(
    task_id='data_preparation',
    python_callable=task_data_preparation,
    dag=dag,
    doc_md="""
    ## Data Preparation Task
    Cleans and preprocesses the raw data:
    - Handle missing values
    - Remove duplicates
    - Clean data types
    - Encode categorical variables
    """
)

data_transformation_task = PythonOperator(
    task_id='data_transformation',
    python_callable=task_data_transformation,
    dag=dag,
    doc_md="""
    ## Data Transformation Task
    Performs feature engineering:
    - Create tenure-based features
    - Generate financial features
    - Build service interaction features
    - Apply feature scaling
    """
)

feature_store_task = PythonOperator(
    task_id='feature_store',
    python_callable=task_feature_store,
    dag=dag,
    doc_md="""
    ## Feature Store Task
    Manages engineered features:
    - Store features in database
    - Maintain feature metadata
    - Version feature definitions
    - Export feature catalog
    """
)

data_versioning_task = PythonOperator(
    task_id='data_versioning',
    python_callable=task_data_versioning,
    dag=dag,
    doc_md="""
    ## Data Versioning Task
    Implements data version control:
    - Initialize Git/DVC repositories
    - Version all data artifacts
    - Track data lineage
    - Generate version reports
    """
)

model_building_task = PythonOperator(
    task_id='model_building',
    python_callable=task_model_building,
    dag=dag,
    doc_md="""
    ## Model Building Task
    Trains and evaluates ML models:
    - Train multiple algorithms
    - Perform cross-validation
    - Select best model
    - Generate evaluation reports
    """
)

pipeline_summary_task = PythonOperator(
    task_id='pipeline_summary',
    python_callable=task_pipeline_summary,
    dag=dag,
    doc_md="""
    ## Pipeline Summary Task
    Generates comprehensive pipeline execution summary:
    - Collect results from all tasks
    - Create execution report
    - Log final metrics
    """
)

# Install requirements task (runs before the pipeline)
install_requirements_task = BashOperator(
    task_id='install_requirements',
    bash_command=f'cd {project_root} && pip install -r requirements.txt',
    dag=dag,
    doc_md="""
    ## Install Requirements Task
    Installs all required Python packages for the pipeline.
    """
)

# Define task dependencies
install_requirements_task >> data_ingestion_task
data_ingestion_task >> data_validation_task
data_validation_task >> data_preparation_task
data_preparation_task >> data_transformation_task
data_transformation_task >> [feature_store_task, data_versioning_task]
[feature_store_task, data_versioning_task] >> model_building_task
model_building_task >> pipeline_summary_task

# Add task documentation
dag.doc_md = """
# Churn Prediction Pipeline

This DAG implements a complete end-to-end data management pipeline for customer churn prediction.

## Pipeline Stages:

1. **Data Ingestion**: Download data from Kaggle and Hugging Face APIs
2. **Data Validation**: Comprehensive data quality checks and reporting
3. **Data Preparation**: Data cleaning, preprocessing, and basic transformations
4. **Data Transformation**: Advanced feature engineering and transformations
5. **Feature Store**: Centralized feature management and versioning
6. **Data Versioning**: Version control for datasets and reproducibility
7. **Model Building**: Train, evaluate, and select best ML models
8. **Pipeline Summary**: Generate comprehensive execution reports

## Business Context:
Customer churn prediction for a telecommunications company using telco customer data.
The pipeline addresses addressable churn scenarios where intervention could prevent customer loss.

## Key Features:
- Automated data quality monitoring
- Comprehensive feature engineering
- Model comparison and selection
- Complete data lineage tracking
- Reproducible ML workflows

## Monitoring:
- All tasks include comprehensive logging
- Pipeline generates detailed reports at each stage
- XCom is used for inter-task communication
- Failed tasks include detailed error information
"""
