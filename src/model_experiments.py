from typing import Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

class ModelTraining:
    """
    A class used to train and evaluate machine learning models on predicting loan risk data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """
    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def run_baseline(self, df):
        """
        Train and evaluate a logistic regression model on the provided dataset.
        Args:
        -----
        df: pandas.DataFrame - The cleaned data for model training.
        """
        # Define features and target
        X = df.drop('Risk_Category', axis=1)
        y = df['Risk_Category']

        # Split the data into training (80%) and test-validation (20%) sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

        # Split the test-validation set (20%) into validation (10%) and test (10%) sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Create the Logistic Regression model
        logistic_model = LogisticRegression(random_state=42)

        # Create a pipeline combining preprocessing and the classifier
        logistic_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', logistic_model)
        ])

        # Define the parameter grid for GridSearchCV
        param_grid = [
            {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['liblinear'],
                'classifier__penalty': ['l1']
            },
            {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['lbfgs'],
                'classifier__penalty': ['l2']
            }
        ]

        # Set up GridSearchCV
        grid_search = GridSearchCV(logistic_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

        # Fit GridSearchCV to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_model = grid_search.best_estimator_

        # Predict on the validation set using the best model
        y_val_pred = best_model.predict(X_val)
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

        # Calculate metrics for the validation set
        self.calculate_metrics(y_val, y_val_pred, y_val_pred_proba, "Logistic Regression")

    def run_knn(self, df):
        """
        Train and evaluate a KNN classifier on the provided dataset.
        Args:
        -----
        df: pandas.DataFrame - The cleaned data for model training.
        """
        # Define features and target
        X = df.drop('Risk_Category', axis=1)
        y = df['Risk_Category']

        # Split the data into training (80%) and test-validation (20%) sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

        # Split the test-validation set (20%) into validation (10%) and test (10%) sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Create the KNN model
        knn_model = KNeighborsClassifier()

        # Create a pipeline combining preprocessing and the classifier
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', knn_model)
        ])

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

        # Fit GridSearchCV to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_model = grid_search.best_estimator_

        # Predict on the validation set using the best model
        y_val_pred = best_model.predict(X_val)
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

        # Calculate metrics for the validation set
        self.calculate_metrics(y_val, y_val_pred, y_val_pred_proba, "KNN")

    def run_decision_tree(self, df):
        """
        Train and evaluate a Decision Tree classifier on the provided dataset.
        Args:
        -----
        df: pandas.DataFrame - The cleaned data for model training.
        """
        # Define features and target
        X = df.drop('Risk_Category', axis=1)
        y = df['Risk_Category']

        # Split the data into training (80%) and test-validation (20%) sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

        # Split the test-validation set (20%) into validation (10%) and test (10%) sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Create the Decision Tree model
        decision_tree_model = DecisionTreeClassifier(random_state=42)

        # Create a pipeline combining preprocessing and the classifier
        dt_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', decision_tree_model)
        ])

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'classifier__max_depth': [None, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(dt_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

        # Fit GridSearchCV to the training data
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_model = grid_search.best_estimator_

        # Predict on the validation set using the best model
        y_val_pred = best_model.predict(X_val)
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

        # Calculate metrics for the validation set
        self.calculate_metrics(y_val, y_val_pred, y_val_pred_proba, "Decision Tree")

    def calculate_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """
        Calculate and display evaluation metrics for the given model.
        Args:
        -----
        y_true: true labels for the validation set
        y_pred: predicted labels for the validation set
        y_pred_proba: predicted probabilities for the validation set
        model_name: name of the model being evaluated
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label='high-risk')
        recall = recall_score(y_true, y_pred, average='binary', pos_label='high-risk')
        f1 = f1_score(y_true, y_pred, average='binary', pos_label='high-risk')
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        print(f"Tuned {model_name} Validation Metrics:")
        print(f'Validation Accuracy: {accuracy:.2f}')
        print(f'Validation Precision: {precision:.2f}')
        print(f'Validation Recall: {recall:.2f}')
        print(f'Validation F1 Score: {f1:.2f}')
        print(f'Validation ROC AUC: {roc_auc:.2f}')

        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, model_name)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot confusion matrix for the given model's predictions.
        Args:
        -----
        y_true: true labels for the validation set
        y_pred: predicted labels for the validation set
        model_name: name of the model being evaluated
        """
        cm = confusion_matrix(y_true, y_pred, labels=['high-risk', 'low-risk'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['high-risk', 'low-risk'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()