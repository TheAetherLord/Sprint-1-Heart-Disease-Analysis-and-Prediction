# Heart Disease Prediction and Analysis

## Overview
This project implements a machine learning solution for heart disease prediction using various health indicators. The system analyzes patient data including age, cholesterol levels, blood pressure, and other vital statistics to predict the likelihood of heart disease.

## Features
- Comprehensive data exploration and visualization
- Multiple machine learning models (Logistic Regression and Random Forest)
- Interactive visualizations using Plotly
- Feature importance analysis
- Cross-validation for model evaluation
- Prediction functionality for new patient data

## Technical Requirements
- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - scikit-learn
  - imbalanced-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn
```

## Dataset Description
The dataset includes the following features:
- `age`: Patient's age in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trtbps`: Resting blood pressure (in mmHg)
- `chol`: Cholesterol levels in mg/dL
- `fbs`: Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalachh`: Maximum heart rate achieved
- `exng`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slp`: Slope of peak exercise ST segment
- `caa`: Number of major vessels colored by fluoroscopy (0-3)
- `thall`: Thalassemia (0-3)
- `output`: Target variable (1 = heart disease, 0 = no heart disease)

## Usage

### Data Preprocessing
```python
from preprocessing import preprocess_data

# Load and preprocess the data
X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = preprocess_data(df)
```

### Model Training and Evaluation
```python
from models import train_and_evaluate_models

# Train and evaluate models
lr_model, rf_model, feature_importance = train_and_evaluate_models(
    X_train_scaled, 
    X_test_scaled, 
    y_train, 
    y_test, 
    X
)
```

### Making Predictions
```python
from prediction import predict_heart_attack_risk

# Predict for new patients
prediction_results = predict_heart_attack_risk(new_patient_data, rf_model, scaler)
probabilities = prediction_results['probabilities']
predictions = prediction_results['predictions']
```

## Model Performance
The project includes two models:

### Logistic Regression
- Average accuracy: 85%
- Cross-validation score: 0.8180 (±0.0997)
- Strong performance in both precision and recall

### Random Forest
- Average accuracy: 84%
- Cross-validation score: 0.8058 (±0.0200)
- More stable performance across different metrics
- Provides feature importance insights

## Visualizations
The project includes several interactive visualizations:
- Feature correlation heatmap
- Distribution plots for numeric features
- Count plots for categorical features
- ROC curves for model comparison
- Feature importance plots

## Analysis Features
1. **Data Exploration**
   - Missing value analysis
   - Feature distribution analysis
   - Correlation analysis

2. **Statistical Analysis**
   - Gender distribution in heart disease cases
   - Age-related patterns
   - Feature importance ranking

3. **Interactive Visualizations**
   - KDE plots for numeric features
   - Categorical feature distributions
   - Model performance comparisons

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Akash Mukherjee

## Acknowledgments
- Dataset provided by UCI Machine Learning Repository
- Special thanks to the scikit-learn and Plotly development teams

## Contact
For any queries regarding this project, please open an issue in the GitHub repository.# Heart Disease Prediction and Analysis

## Overview
This project implements a machine learning solution for heart disease prediction using various health indicators. The system analyzes patient data including age, cholesterol levels, blood pressure, and other vital statistics to predict the likelihood of heart disease.

## Features
- Comprehensive data exploration and visualization
- Multiple machine learning models (Logistic Regression and Random Forest)
- Interactive visualizations using Plotly
- Feature importance analysis
- Cross-validation for model evaluation
- Prediction functionality for new patient data

## Technical Requirements
- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - scikit-learn
  - imbalanced-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn
```

## Dataset Description
The dataset includes the following features:
- `age`: Patient's age in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trtbps`: Resting blood pressure (in mmHg)
- `chol`: Cholesterol levels in mg/dL
- `fbs`: Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalachh`: Maximum heart rate achieved
- `exng`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slp`: Slope of peak exercise ST segment
- `caa`: Number of major vessels colored by fluoroscopy (0-3)
- `thall`: Thalassemia (0-3)
- `output`: Target variable (1 = heart disease, 0 = no heart disease)

## Usage

### Data Preprocessing
```python
from preprocessing import preprocess_data

# Load and preprocess the data
X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = preprocess_data(df)
```

### Model Training and Evaluation
```python
from models import train_and_evaluate_models

# Train and evaluate models
lr_model, rf_model, feature_importance = train_and_evaluate_models(
    X_train_scaled, 
    X_test_scaled, 
    y_train, 
    y_test, 
    X
)
```

### Making Predictions
```python
from prediction import predict_heart_attack_risk

# Predict for new patients
prediction_results = predict_heart_attack_risk(new_patient_data, rf_model, scaler)
probabilities = prediction_results['probabilities']
predictions = prediction_results['predictions']
```

## Model Performance
The project includes two models:

### Logistic Regression
- Average accuracy: 85%
- Cross-validation score: 0.8180 (±0.0997)
- Strong performance in both precision and recall

### Random Forest
- Average accuracy: 84%
- Cross-validation score: 0.8058 (±0.0200)
- More stable performance across different metrics
- Provides feature importance insights

## Visualizations
The project includes several interactive visualizations:
- Feature correlation heatmap
- Distribution plots for numeric features
- Count plots for categorical features
- ROC curves for model comparison
- Feature importance plots

## Analysis Features
1. **Data Exploration**
   - Missing value analysis
   - Feature distribution analysis
   - Correlation analysis

2. **Statistical Analysis**
   - Gender distribution in heart disease cases
   - Age-related patterns
   - Feature importance ranking

3. **Interactive Visualizations**
   - KDE plots for numeric features
   - Categorical feature distributions
   - Model performance comparisons

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Author
Akash Mukherjee

## Acknowledgments
- Dataset taken from Kaggle.
- Special thanks to Thakur Khushbu Deepaksinh my IT trainer.
- Anudip Foundation for providing this amazing opportunity.
