# End-to-End Data Management Pipeline for Machine Learning
## Customer Churn Prediction

**Course**: Data Management for Machine Learning  
**Assignment**: Assignment I (20% Weightage)  
**Submission Date**: 24 Aug 2025 11:50 PM  
**Title**: End-to-End Data Management Pipeline for Machine Learning  

## 📋 Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Business Context](#business-context)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Directory Structure](#directory-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage Instructions](#usage-instructions)
7. [Pipeline Components](#pipeline-components)
8. [Expected Outputs](#expected-outputs)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Challenges & Solutions](#challenges--solutions)

## 🎯 Problem Formulation

### Business Problem
Customer churn occurs when existing customers stop using a company's services, leading to revenue decline and increased acquisition costs. Research from PWC indicates that "Financial institutions will lose 24% of revenue in the next 3-5 years, mainly due to customer churn to new fintech companies."

### Key Business Objectives
- **Primary**: Build an automated pipeline to predict customer churn
- **Secondary**: Implement robust data management practices for ML workflows
- **Tertiary**: Ensure pipeline scalability and reproducibility

### Data Sources
1. **Kaggle API**: Telco customer churn dataset
2. **Hugging Face API**: Additional customer behavior data
3. **Attributes**:
   - Customer demographics (gender, age, tenure)
   - Service information (phone, internet, streaming)
   - Contract details (type, billing, payment method)
   - Usage patterns and charges

### Expected Pipeline Outputs
- **Clean datasets** for exploratory data analysis (EDA)
- **Transformed features** ready for machine learning
- **Deployable model** to predict customer churn with confidence scores
- **Automated reports** on data quality and model performance

## 🏢 Business Context

Customer churn poses significant challenges:
- **Revenue Impact**: Direct loss of recurring revenue
- **Cost Multiplication**: High customer acquisition costs (5-25x retention cost)
- **Competitive Risk**: Lost customers often move to competitors
- **Indirect Effects**: Churn can influence other loyal customers

**Solution**: Proactive churn prediction enables:
- Early intervention strategies
- Targeted retention campaigns  
- Resource optimization
- Improved customer lifetime value

## 🏗 Pipeline Architecture

The pipeline follows a **modular architecture** with clear separation of concerns:

```
Data Sources → Ingestion → Storage → Validation → Preparation → 
Transformation → Feature Store → Versioning → Model Building → Orchestration
```

### Design Principles
- **Modularity**: Each stage is independent and reusable
- **Scalability**: Designed to handle increasing data volumes
- **Reliability**: Comprehensive error handling and logging
- **Reproducibility**: Version control for data and models
- **Monitoring**: Built-in logging and alerting mechanisms

## 📁 Directory Structure

```
churn_prediction_pipeline/
├── README.md                           # This documentation
├── requirements.txt                    # Python dependencies
├── config.py                          # Configuration settings
├── main.py                            # Main pipeline execution
├── kaggle.json                        # Kaggle API credentials
├── hugging_face.json                  # Hugging Face API credentials
│
├── Directory_2_Ingestion/             # Task 2: Data Ingestion
│   └── ingestion.py                   # API data fetching scripts
│
├── Directory_3_RawStorage/            # Task 3: Raw Data Storage
│                                      # Data lake folder structure
│
├── Directory_4_Validation/            # Task 4: Data Validation
│   └── validation.py                 # Data quality checks
│
├── Directory_5_Preparation/           # Task 5: Data Preparation
│   ├── preparation.py                # Data cleaning & preprocessing
│   └── cleaned/                      # Cleaned datasets storage
│
├── Directory_6_Transformation/        # Task 6: Data Transformation
│   └── transformation.py             # Feature engineering
│
├── Directory_7_FeatureStore/         # Task 7: Feature Store
│   ├── feature_store.py              # Feature management system
│   ├── feature_store.db              # SQLite feature database
│   ├── features/                     # Feature storage
│   └── metadata/                     # Feature metadata
│
├── Directory_8_Versioning/           # Task 8: Data Versioning
│   └── versioning.py                # Git-based versioning
│
├── Directory_9_ModelBuilding/        # Task 9: Model Building
│   └── model_building.py            # ML model training
│
├── Directory_10_Orchestration/       # Task 10: Pipeline Orchestration
│   └── churn_prediction_dag.py      # Airflow DAG
│
├── data/                             # Data storage
│   ├── raw/                         # Raw ingested data
│   ├── raw_data/                    # Additional raw storage
│   └── transformed/                 # Transformed datasets
│
├── logs/                            # Pipeline execution logs
├── models/                          # Trained model storage
└── reports/                         # Generated reports
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- SQLite3
- Internet connection (for API access)

### Setup Instructions

1. **Clone/Extract the Pipeline**
   ```bash
   cd churn_prediction_pipeline/
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Credentials**
   - Add your Kaggle API key to `kaggle.json`
   - Add your Hugging Face token to `hugging_face.json`

4. **Verify Installation**
   ```bash
   python3 config.py  # Should display configuration
   ```

## 🚀 Usage Instructions

### Option 1: Full Pipeline Execution
```bash
python3 main.py
```

### Option 2: Individual Module Testing
```bash
# Test data ingestion
python3 Directory_2_Ingestion/ingestion.py

# Test data validation
python3 Directory_4_Validation/validation.py

# Test data preparation
python3 Directory_5_Preparation/preparation.py

# Continue with other modules...
```

### Option 3: Orchestrated Execution (Advanced)
```bash
# Using Apache Airflow (requires Airflow setup)
airflow dags trigger churn_prediction_pipeline
```

## 🔧 Pipeline Components

### Directory_2_Ingestion (Task 2)
**Purpose**: Automated data collection from multiple sources
- **Features**:
  - Kaggle API integration
  - Hugging Face API integration
  - Error handling and retry logic
  - Comprehensive logging
- **Output**: Raw datasets in `data/raw/`

### Directory_3_RawStorage (Task 3)
**Purpose**: Organized raw data storage
- **Features**:
  - Partitioned folder structure
  - Timestamp-based organization
  - Source-based categorization

### Directory_4_Validation (Task 4)
**Purpose**: Data quality assurance
- **Features**:
  - Missing value detection
  - Data type validation
  - Anomaly identification
  - Quality report generation
- **Output**: Data quality reports in `reports/`

### Directory_5_Preparation (Task 5)
**Purpose**: Data cleaning and preprocessing
- **Features**:
  - Missing value imputation
  - Outlier handling
  - Data standardization
  - EDA visualizations
- **Output**: Clean datasets in `Directory_5_Preparation/cleaned/`

### Directory_6_Transformation (Task 6)
**Purpose**: Feature engineering and transformation
- **Features**:
  - Feature creation and aggregation
  - Categorical encoding
  - Numerical scaling
  - Feature selection
- **Output**: Transformed features and labels

### Directory_7_FeatureStore (Task 7)
**Purpose**: Centralized feature management
- **Features**:
  - SQLite-based storage
  - Feature metadata tracking
  - Version control for features
  - API for feature retrieval
- **Output**: Engineered features in `feature_store.db`

### Directory_8_Versioning (Task 8)
**Purpose**: Data and model versioning
- **Features**:
  - Git-based version control
  - Automated tagging
  - Change tracking
  - Reproducibility assurance
- **Output**: Version history and metadata

### Directory_9_ModelBuilding (Task 9)
**Purpose**: Machine learning model development
- **Features**:
  - Multiple algorithm experimentation
  - Hyperparameter tuning
  - Model evaluation and comparison
  - Model persistence
- **Output**: Trained models in `models/` directory

### Directory_10_Orchestration (Task 10)
**Purpose**: Pipeline automation and monitoring
- **Features**:
  - Apache Airflow DAG
  - Task dependency management
  - Failure handling
  - Monitoring and alerting
- **Output**: Automated pipeline execution

## 📊 Expected Outputs

### Data Products
1. **Raw Datasets**: Original data from APIs
2. **Clean Datasets**: Processed and validated data
3. **Feature Sets**: Engineered features ready for ML
4. **Trained Models**: Optimized churn prediction models

### Reports & Documentation
1. **Data Quality Reports**: Validation results and statistics
2. **EDA Reports**: Exploratory analysis with visualizations
3. **Model Performance Reports**: Accuracy, precision, recall, F1-score
4. **Pipeline Logs**: Execution history and debugging information

### Artifacts
1. **Model Files**: Serialized ML models (.pkl, .joblib)
2. **Feature Metadata**: Descriptions and lineage
3. **Version Tags**: Data and model versions
4. **Configuration Files**: Pipeline settings and parameters

## 📈 Evaluation Metrics

### Business Metrics
- **Churn Prediction Accuracy**: Target >85%
- **False Positive Rate**: <10% (avoid unnecessary interventions)
- **Model Recall**: >80% (catch actual churners)
- **Pipeline Execution Time**: <30 minutes for full run

### Technical Metrics
- **Data Quality Score**: >95% clean records
- **Feature Coverage**: >90% non-null values
- **Model Performance**: F1-score >0.75
- **Pipeline Reliability**: >99% successful runs

### Operational Metrics
- **Time to Detection**: <24 hours for data issues
- **Recovery Time**: <5 minutes for failed tasks
- **Resource Utilization**: <4GB RAM, <50% CPU
- **Storage Efficiency**: <1GB total storage

## 🚧 Challenges & Solutions

### Challenge 1: Data Collection
**Issue**: Inconsistent data sources and API limitations
**Solution**: 
- Implemented retry mechanisms with exponential backoff
- Added data source validation and fallback options
- Created synthetic data generation for testing

### Challenge 2: Data Quality
**Issue**: Missing values, duplicates, and inconsistent formats
**Solution**:
- Comprehensive validation framework
- Automated data cleaning pipelines
- Statistical imputation methods for missing values

### Challenge 3: Feature Engineering
**Issue**: High cardinality categorical variables
**Solution**:
- Implemented target encoding for high-cardinality features
- Used dimensionality reduction techniques
- Created feature importance ranking

### Challenge 4: Model Performance
**Issue**: Class imbalance in churn data (typically 70-30 split)
**Solution**:
- Applied SMOTE for synthetic sample generation
- Used stratified sampling for train/test splits
- Implemented cost-sensitive learning algorithms

### Challenge 5: Pipeline Orchestration
**Issue**: Complex task dependencies and error propagation
**Solution**:
- Designed robust DAG with proper dependencies
- Implemented graceful error handling and rollback
- Added comprehensive logging and monitoring

### Challenge 6: Reproducibility
**Issue**: Ensuring consistent results across runs
**Solution**:
- Version control for data, code, and models
- Fixed random seeds for all stochastic processes
- Containerization for environment consistency

## 🔄 Continuous Improvement

### Monitoring & Maintenance
- **Data Drift Detection**: Monitor for changes in data distribution
- **Model Performance Tracking**: Regular evaluation on new data
- **Pipeline Health Checks**: Automated testing and validation
- **Resource Optimization**: Performance tuning and scaling

### Future Enhancements
- **Real-time Predictions**: Streaming data processing
- **Advanced ML Models**: Deep learning and ensemble methods
- **Cloud Deployment**: Scalable cloud infrastructure
- **A/B Testing**: Model comparison and validation frameworks

---

## 📝 License & Contact

This pipeline is developed for educational purposes as part of the Data Management for Machine Learning course.

**Note**: This implementation demonstrates industry best practices for ML data pipelines while maintaining academic integrity and learning objectives.
