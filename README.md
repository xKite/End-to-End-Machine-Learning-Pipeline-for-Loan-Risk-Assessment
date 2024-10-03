# Ho Kwok Leong

# Project Title: Script-Based-End-to-End-Machine-Learning-Pipeline-for-Loan-Risk-Prediction

## Project Description
This project analyzes a loan dataset to predict the risk category of borrowers based on various financial and personal attributes. By leveraging exploratory data analysis (EDA) and machine learning techniques, the project develops a pipeline for data preprocessing, feature engineering, and model training. The primary goal is to classify loan applicants into 'low-risk' and 'high-risk' categories, enabling financial institutions to make informed lending decisions.

## Prerequisites and Installation
To run this project, ensure you have the following prerequisites installed:

- **Python Version**: Python 3.x (preferred version: 3.8 or above)
- **Required Libraries**: All necessary libraries are listed in the `requirements.txt` file.

### Installation Steps
Follow these steps to set up the project environment:

1. **Clone the Repository**:
   [git clone here](https://github.com/xKite/End-to-End-Machine-Learning-Pipeline-for-Loan-Risk-Assessment)
   
2. **Install Required Packages: Use pip to install the necessary libraries.**
`pip install -r requirements.txt`

# Instructions for Executing the Pipeline

To execute the end-to-end machine learning pipeline, follow these steps:

1. **Modify the Configuration**: 
   - Open the `config.yaml` file in a text editor to adjust any parameters necessary for your specific use case. 
   - Key parameters to modify might include:
     - Data source paths
     - Model hyperparameters (e.g., learning rate, number of trees)
     - Feature selection criteria

2. **Run the Main Script**: 
   - Execute the main pipeline script to initiate the workflow. This script will automatically handle data loading, processing, training, and evaluation.
`python main.py`
3. **Check Output**:
   - After the script completes, check the output files in the specified output directory (as defined in the config.yaml), which may include model files, logs, and performance metrics.

## Overview of Key Findings from EDA
The exploratory data analysis revealed several important patterns and insights that guided the design of the preprocessing pipeline:

- **Data Quality:** The dataset contained missing values in categorical and numerical features, which were addressed using appropriate imputation strategies to ensure data integrity.

- **Categorical Imbalances:** Analysis of the 'Risk_Category' showed an imbalance between 'low-risk' and 'high-risk' categories, highlighting the need for careful evaluation during model training to avoid bias.

- **Correlation Insights:** Spearman correlation analysis indicated strong relationships between features such as 'Credit_Score' and 'Risk_Category', suggesting these features are critical for risk classification.

- **Outlier Detection:** Several numerical features exhibited outliers, particularly in 'Loan_Amount' and 'Annual_Income', which were visualized through box plots. Handling these outliers is crucial for model performance.

- **Feature Importance:** Certain features, including 'Debt-to-Income_Ratio' and 'Number_of_Open_Accounts', significantly influence risk classification, underscoring the importance of selecting relevant features for the predictive model.

## Feature Handling Description
The dataset underwent a comprehensive cleaning and preprocessing process, which included:

- **Absolute Value Conversion:** Numerical features such as 'Loan_Amount' and 'Credit_Score' were converted to absolute values to eliminate negative entries that could skew analysis.

- **Missing Value Imputation:**
  - Categorical features ('Risk_Category' and 'Loan_Purpose') were filled with their respective modes.
  - Numerical features were filled with the median to preserve the distribution.

- **Encoding Categorical Features:** The 'Loan_Purpose' categorical variable was one-hot encoded to convert it into a format suitable for model training.

- **Outlier Treatment:** Identified outliers using the Interquartile Range (IQR) method and visualized them through box plots to assess their impact on the dataset.

## Feature Selection
The features selected for model training included:

- **Categorical:** Loan_Purpose
- **Numerical:** Loan_Amount, Credit_Score, Annual_Income, Employment_Length, Debt-to-Income_Ratio, Number_of_Open_Accounts, Number_of_Past_Due_Payments

## Summary of Feature Handling
| Feature                       | Handling Technique           | Comments                              |
|-------------------------------|------------------------------|---------------------------------------|
| Loan_Amount                   | Absolute value conversion     | Removed negative values               |
| Credit_Score                  | Absolute value conversion     | Removed negative values               |
| Annual_Income                 | Median imputation             | Addressed missing values              |
| Employment_Length              | Median imputation             | Addressed missing values              |
| Debt-to-Income_Ratio         | Median imputation             | Addressed missing values              |
| Number_of_Open_Accounts       | Median imputation             | Addressed missing values              |
| Number_of_Past_Due_Payments   | Median imputation             | Addressed missing values              |
| Risk_Category                 | Mode imputation               | Addressed missing values, also replaced labels |
| Loan_Purpose                  | One-hot encoding              | Prepared categorical variable for model training |


## Description of Logical Steps/Flow of the Pipeline
The machine learning pipeline consists of several logical steps, each serving a specific purpose in the overall process:

### 1. Data Loading
- Load the dataset from a specified location. This data should include features and labels for supervised learning tasks.

### 2. Data Cleaning
- Handle missing values using `SimpleImputer`. You can choose strategies like mean, median, or mode depending on the nature of the data.

### 3. Preprocessing
- Use `ColumnTransformer` to apply different transformations to numerical and categorical features, including scaling and encoding.

### 4. Oversampling
- Implement SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset if it is imbalanced.

### 5. Model Training
- Train various classifiers (Logistic Regression, KNN, Decision Tree) using the training dataset. Each model is encapsulated in a `Pipeline` that includes preprocessing.

### 6. Hyperparameter Tuning
- Use `GridSearchCV` to perform hyperparameter optimization, searching for the best parameter combinations based on cross-validated performance.

### 7. Model Evaluation
- Evaluate each model's performance on the validation set using metrics like accuracy, precision, recall, F1 score, and ROC AUC.

### 8. Prediction
- Use the best-performing model to make predictions on new data.

## Explanation of Model Choices

### Logistic Regression
- **Reason for Choice:** Simple and effective for binary classification tasks. Provides probabilities and interpretable coefficients.
- **Use Case:** Ideal for linearly separable data.

### K-Nearest Neighbors (KNN)
- **Reason for Choice:** Non-parametric and easy to understand. Works well with a smaller number of dimensions.
- **Use Case:** Effective for instances where similar cases can be clustered together.

### Decision Tree
- **Reason for Choice:** Provides an intuitive representation of decision-making processes. Can capture non-linear relationships well.
- **Use Case:** Useful for datasets with complex interactions between features.

## Evaluation of Models
Each model's performance is assessed using multiple metrics:

- **Accuracy:** Proportion of correctly predicted instances out of all instances.
- **Precision:** Proportion of true positive results in all positive predictions. Useful when the cost of false positives is high.
- **Recall:** Proportion of true positive results in all actual positive instances. Important when the cost of false negatives is high.
- **F1 Score:** Harmonic mean of precision and recall. Balances both metrics.
- **ROC AUC:** Area Under the Receiver Operating Characteristic Curve. Indicates how well the model separates classes.

## Performance Summary
After evaluating the models, summarize the results in a table for clarity.

| Model                           | Metric                  | Training Score  | Validation Score |
|---------------------------------|-------------------------|-----------------|------------------|
| **Tuned Logistic Regression**   | Accuracy                | 0.99100         | 0.99000          |
|                                 | Precision               | 0.99124         | 0.98127          |
|                                 | Recall                  | 0.97753         | 0.98127          |
|                                 | F1 Score                | 0.98433         | 0.98127          |
|                                 | ROC AUC                 | 0.99789         | 0.99213          |
| **Tuned K-Nearest Neighbors**   | Accuracy                | 1.00000         | 0.98500          |
|                                 | Precision               | 1.00000         | 0.98837          |
|                                 | Recall                  | 1.00000         | 0.95506          |
|                                 | F1 Score                | 1.00000         | 0.97143          |
|                                 | ROC AUC                 | 1.00000         | 0.99922          |
| **Tuned Decision Tree**         | Accuracy                | 0.99963         | 0.99900          |
|                                 | Precision               | 0.99957         | 1.00000          |
|                                 | Recall                  | 0.99914         | 0.99625          |
|                                 | F1 Score                | 0.99935         | 0.99812          |
|                                 | ROC AUC                 | 1.00000         | 0.99999          |


## Considerations for Deployment
When deploying the models, consider the following:

- **Scalability:** Ensure the model can handle increased loads and larger datasets. Evaluate if you need to implement batch processing or streaming.

- **Real-time Performance:** If the model will serve predictions in real-time, ensure it can provide quick responses under expected operational loads.

- **Integration with Other Systems:** Ensure the model can easily integrate with existing systems and databases for seamless operation.

- **Monitoring and Maintenance:** Establish a monitoring system to track model performance over time and a plan for retraining as new data becomes available.

- **Documentation:** Maintain comprehensive documentation of the model's functionality, usage instructions, and any dependencies to facilitate future maintenance and updates.
