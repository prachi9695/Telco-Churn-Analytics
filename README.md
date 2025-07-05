# Telco Customer Churn Analytics

## Overview

This project implements advanced analytics and machine learning techniques to predict customer churn in the telecommunications industry. By analyzing customer behavior patterns and service usage data, we develop predictive models to help telecom companies identify at-risk customers and implement effective retention strategies.

## Features

- Comprehensive data analysis and visualization
- Multiple machine learning models for churn prediction:
  - Logistic Regression
  - Decision Trees
  - Random Forests
- Feature importance analysis
- Model performance comparison
- Customer segmentation insights
- Retention strategy recommendations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/prachi9695/Telco-Churn-Analytics-2024.git
cd Telco-Churn-Analytics-2024
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:

```python
from src.preprocessing.data_processor import DataProcessor

processor = DataProcessor()
processed_data = processor.preprocess_data('data/raw/telco_data.csv')
```

2. Model Training:

```python
from src.models.model_trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_model(processed_data, model_type='random_forest')
```

3. Prediction:

```python
predictions = model.predict(new_customer_data)
```

## Model Performance

Our models are evaluated using various metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Detailed performance metrics and comparisons are available in the notebooks directory.

## Acknowledgments

- Dataset provided by [source]
