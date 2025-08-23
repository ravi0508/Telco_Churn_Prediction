"""
Model Building Module
This module handles machine learning model training, evaluation, and saving
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Add parent directory to path for config import
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from config import MODELS_DIR, CLEANED_DATA_DIR, REPORTS_DIR, RANDOM_STATE, LOG_FORMAT


# Setup logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelBuilder:
    """Class to handle machine learning model building and evaluation."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_transformed_data(self) -> pd.DataFrame:
        """
        Load transformed data for model training.
        
        Returns:
            pd.DataFrame: Transformed dataset
        """
        try:
            # Look for transformed data files
            csv_files = list(CLEANED_DATA_DIR.glob("transformed_churn_data_*.csv"))
            
            if not csv_files:
                csv_files = list(CLEANED_DATA_DIR.glob("prepared_churn_data_*.csv"))
            
            if not csv_files:
                raise FileNotFoundError("No transformed or prepared data files found")
            
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            logger.info(f"Data loaded from: {latest_file}. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load transformed data: {e}")
            raise
    
    def prepare_features_target(self, df: pd.DataFrame, target_column: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variables.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        try:
            # Handle missing target column
            if target_column not in df.columns:
                # Look for variations of churn column
                possible_targets = [col for col in df.columns if 'churn' in col.lower()]
                if possible_targets:
                    target_column = possible_targets[0]
                    logger.info(f"Using {target_column} as target column")
                else:
                    raise ValueError(f"Target column {target_column} not found in dataset")
            
            # Separate features and target
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Remove non-numeric columns that might cause issues
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                logger.info(f"Removing non-numeric columns: {list(non_numeric_cols)}")
                X = X.select_dtypes(include=[np.number])
            
            # Handle missing values in features
            if X.isnull().sum().sum() > 0:
                logger.warning("Found missing values in features, filling with median")
                X = X.fillna(X.median())
            
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            logger.info(f"Target distribution:\n{y.value_counts()}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare features and target: {e}")
            raise
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
            )
            
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Train target distribution:\n{y_train.value_counts()}")
            logger.info(f"Test target distribution:\n{y_test.value_counts()}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise
    
    def define_models(self) -> Dict[str, Any]:
        """
        Define machine learning models to train.
        
        Returns:
            Dict: Dictionary of model instances
        """
        try:
            models = {
                'logistic_regression': LogisticRegression(
                    random_state=RANDOM_STATE, 
                    max_iter=1000
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE
                ),
                'svm': SVC(
                    random_state=RANDOM_STATE,
                    probability=True
                )
            }
            
            logger.info(f"Defined {len(models)} models for training")
            return models
            
        except Exception as e:
            logger.error(f"Failed to define models: {e}")
            raise
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all defined models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dict: Trained models
        """
        try:
            models = self.define_models()
            trained_models = {}
            
            # Scale features for algorithms that need it
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Use scaled features for SVM and Logistic Regression
                if name in ['svm', 'logistic_regression']:
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                
                trained_models[name] = model
                logger.info(f"Completed training {name}")
            
            self.models = trained_models
            logger.info("All models trained successfully")
            
            return trained_models
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict: Evaluation results for all models
        """
        try:
            if not self.models:
                raise ValueError("No trained models found. Train models first.")
            
            results = {}
            X_test_scaled = self.scaler.transform(X_test)
            
            for name, model in self.models.items():
                logger.info(f"Evaluating {name}...")
                
                # Use scaled features for appropriate models
                if name in ['svm', 'logistic_regression']:
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                logger.info(f"Completed evaluation for {name}")
            
            self.results = results
            
            # Find best model based on F1 score
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
            self.best_model = {
                'name': best_model_name,
                'model': self.models[best_model_name],
                'metrics': results[best_model_name]
            }
            
            logger.info(f"Best model: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate models: {e}")
            raise
    
    def cross_validate_best_model(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the best model.
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Dict: Cross-validation results
        """
        try:
            if not self.best_model:
                raise ValueError("No best model found. Evaluate models first.")
            
            model = self.best_model['model']
            model_name = self.best_model['name']
            
            # Use appropriate data based on model type
            if model_name in ['svm', 'logistic_regression']:
                X_for_cv = self.scaler.fit_transform(X)
            else:
                X_for_cv = X
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_for_cv, y, cv=cv, scoring='f1')
            
            cv_results = {
                'cv_mean_f1': cv_scores.mean(),
                'cv_std_f1': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"Cross-validation completed for {model_name}")
            logger.info(f"CV F1 Score: {cv_results['cv_mean_f1']:.4f} (+/- {cv_results['cv_std_f1']:.4f})")
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Failed to perform cross-validation: {e}")
            raise
    
    def save_models(self) -> Dict[str, str]:
        """
        Save trained models to disk.
        
        Returns:
            Dict: Mapping of model names to file paths
        """
        try:
            if not self.models:
                raise ValueError("No trained models found")
            
            saved_models = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for name, model in self.models.items():
                model_filename = f"{name}_model_{timestamp}.joblib"
                model_path = MODELS_DIR / model_filename
                
                joblib.dump(model, model_path)
                saved_models[name] = str(model_path)
                
                logger.info(f"Saved {name} model to {model_path}")
            
            # Save scaler
            scaler_path = MODELS_DIR / f"scaler_{timestamp}.joblib"
            joblib.dump(self.scaler, scaler_path)
            saved_models['scaler'] = str(scaler_path)
            
            # Save best model separately
            if self.best_model:
                best_model_path = MODELS_DIR / f"best_model_{timestamp}.joblib"
                joblib.dump(self.best_model['model'], best_model_path)
                saved_models['best_model'] = str(best_model_path)
            
            logger.info(f"Saved {len(saved_models)} model files")
            return saved_models
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def generate_model_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """
        Generate comprehensive model evaluation report.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            str: Path to the generated report
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"model_evaluation_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                
                # Model comparison sheet
                if self.results:
                    results_df = pd.DataFrame(self.results).T
                    results_df.to_excel(writer, sheet_name='Model_Comparison')
                
                # Best model details
                if self.best_model:
                    best_model_info = pd.DataFrame([{
                        'Best_Model': self.best_model['name'],
                        'Accuracy': self.best_model['metrics']['accuracy'],
                        'Precision': self.best_model['metrics']['precision'],
                        'Recall': self.best_model['metrics']['recall'],
                        'F1_Score': self.best_model['metrics']['f1_score'],
                        'ROC_AUC': self.best_model['metrics']['roc_auc']
                    }])
                    best_model_info.to_excel(writer, sheet_name='Best_Model', index=False)
                
                # Feature importance (if available)
                if self.best_model and hasattr(self.best_model['model'], 'feature_importances_'):
                    feature_names = X_test.columns
                    importance_data = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': self.best_model['model'].feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    importance_data.to_excel(writer, sheet_name='Feature_Importance', index=False)
            
            logger.info(f"Model report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate model report: {e}")
            raise
    
    def track_with_mlflow(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Track model experiments with MLflow.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping experiment tracking")
            return
        
        try:
            mlflow.set_experiment("churn_prediction")
            
            for name, model in self.models.items():
                with mlflow.start_run(run_name=name):
                    # Log parameters
                    mlflow.log_params(model.get_params())
                    
                    # Log metrics
                    if name in self.results:
                        metrics = self.results[name]
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(metric_name, value)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, name)
                    
                    logger.info(f"Logged {name} to MLflow")
            
            logger.info("MLflow tracking completed")
            
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {e}")
    
    def build_and_evaluate_models(self, features: pd.DataFrame = None, labels: pd.DataFrame = None) -> Tuple[str, Dict[str, Any]]:
        """
        Complete model building and evaluation pipeline.
        
        Args:
            features: Features dataframe (optional, will load from file if not provided)
            labels: Labels dataframe (optional, will load from file if not provided)
        
        Returns:
            Tuple[str, Dict]: Best model name and complete results including models, metrics, and file paths
        """
        try:
            logger.info("Starting model building pipeline...")
            
            # Use provided data or load from files
            if features is not None and labels is not None:
                # Combine features and labels
                if isinstance(labels, pd.DataFrame):
                    if labels.shape[1] == 1:
                        y = labels.iloc[:, 0]
                    else:
                        y = labels
                else:
                    y = labels
                
                X = features
                logger.info(f"Using provided features: {X.shape}, labels: {y.shape if hasattr(y, 'shape') else len(y)}")
            else:
                # Load data from files
                df = self.load_transformed_data()
                X, y = self.prepare_features_target(df)
                logger.info("Loaded data from files")
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Train models
            trained_models = self.train_models(X_train, y_train)
            
            # Evaluate models
            results = self.evaluate_models(X_test, y_test)
            
            # Cross-validate best model
            cv_results = self.cross_validate_best_model(X, y)
            
            # Save models
            saved_models = self.save_models()
            
            # Generate report
            report_path = self.generate_model_report(X_test, y_test)
            
            # Track with MLflow
            self.track_with_mlflow(X_train, y_train)
            
            # Prepare return values for main.py
            best_model_name = self.best_model['name'] if self.best_model else 'Unknown'
            best_model_metrics = self.best_model['metrics'] if self.best_model else {}
            
            final_results = {
                'models': trained_models,
                'evaluation_results': results,
                'cross_validation': cv_results,
                'best_model': self.best_model,
                'saved_model_paths': saved_models,
                'report_path': report_path,
                'data_shape': X.shape if features is not None else df.shape,
                'feature_count': len(X.columns)
            }
            
            logger.info("Model building pipeline completed successfully")
            return best_model_name, best_model_metrics
            
        except Exception as e:
            logger.error(f"Model building pipeline failed: {e}")
            # Return default values in case of error
            return 'Error', {'error': str(e)}
    
    def build_and_evaluate_models_detailed(self) -> Dict[str, Any]:
        """
        Complete model building and evaluation pipeline (legacy method with full results).
        
        Returns:
            Dict: Complete results including models, metrics, and file paths
        """
        best_model_name, best_model_metrics = self.build_and_evaluate_models()
        
        return {
            'best_model_name': best_model_name,
            'best_model_metrics': best_model_metrics,
            'status': 'success' if best_model_name != 'Error' else 'error'
        }

def main():
    """Main function to run model building."""
    try:
        model_builder = ModelBuilder()
        results = model_builder.build_and_evaluate_models()
        
        print("Model Building Completed Successfully!")
        print(f"\nBest Model: {results['best_model']['name']}")
        print("Best Model Metrics:")
        for metric, value in results['best_model']['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nModels saved to: {MODELS_DIR}")
        print(f"Report generated: {results['report_path']}")
        
        print(f"\nDataset Info:")
        print(f"  Shape: {results['data_shape']}")
        print(f"  Features: {results['feature_count']}")
        
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise

if __name__ == "__main__":
    main()
