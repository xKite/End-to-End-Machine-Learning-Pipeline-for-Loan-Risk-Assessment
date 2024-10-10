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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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

        # List to store metrics for each model
        self.metrics_summary = {"Logistic Regression": [], "KNN": [], "Decision Tree": []}

        # Initialize the label encoder
        self.label_encoder = LabelEncoder()
        
    def run_model(self, df, model, param_grid, model_name):
        """
        Train and evaluate a model using GridSearchCV and store metrics.
        Args:
        df: pandas.DataFrame - The cleaned data for model training.
        model: sklearn model - The model to be trained.
        param_grid: dict - Parameter grid for GridSearchCV.
        model_name: str - Name of the model (for metrics tracking).
        """
        # Define features and target
        X = df.drop('Risk_Category', axis=1)
        y = df['Risk_Category']

        # Encode target variable
        y = self.label_encoder.fit_transform(y)

        # Split the data into training (80%) and test-validation (20%) sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

        # Split the test-validation set (20%) into validation (10%) and test (10%) sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Create pipeline
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])

        # GridSearchCV
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Best model
        best_model = grid_search.best_estimator_

        # Predict and calculate metrics
        y_val_pred = best_model.predict(X_val)
        y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

        self.calculate_metrics(y_val, y_val_pred, y_val_pred_proba, model_name)

    def run_baseline(self, df):
        """
        Train and evaluate a logistic regression model on the provided dataset.
        Args:
        -----
        df: pandas.DataFrame - The cleaned data for model training.
        """                
        # Create the Logistic Regression model
        logistic_model = LogisticRegression(random_state=42, max_iter=3000)
        
        # Define the parameter grid for GridSearchCV
        param_grid = [
            {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['liblinear'],
                'classifier__penalty': ['l1']
            },
            {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__solver': ['saga'],
                'classifier__penalty': ['l2']
            }
        ]
        self.run_model(df, logistic_model, param_grid, "Logistic Regression")

    def run_knn(self, df):
        """
        Train and evaluate a KNN classifier on the provided dataset.
        Args:
        -----
        df: pandas.DataFrame - The cleaned data for model training.
        """        
        # Create the KNN model
        knn_model = KNeighborsClassifier()

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }
        self.run_model(df, knn_model, param_grid, "KNN")       

    def run_decision_tree(self, df):
        """
        Train and evaluate a Decision Tree classifier on the provided dataset.
        Args:
        -----
        df: pandas.DataFrame - The cleaned data for model training.
        """        
        # Create the Decision Tree model
        decision_tree_model = DecisionTreeClassifier(random_state=42)

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'classifier__max_depth': [None, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        }
        self.run_model(df, decision_tree_model, param_grid, "Decision Tree")

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
        # Check the shape of y_pred_proba
        if y_pred_proba.ndim == 1:
            # If it's 1-dimensional, assume it's the probability for the positive class
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            # If it's 2-dimensional, access the positive class probabilities
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            raise ValueError("Invalid shape for y_pred_proba. It must be 1D or 2D with shape (n_samples, n_classes).")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
        f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)        

        # Append metrics to summary for this model
        self.metrics_summary[model_name].append({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC AUC': roc_auc
        })

        # Display calculated metrics
        self.display_metrics_summary(y_true, y_pred, model_name)

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
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['high-risk', 'low-risk'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()

    def display_metrics_summary(self, y_true=None, y_pred=None, model_name=None):
        """
        Display a summary of all metrics for each model in a structured format.
        """
        print("\n==== Model Metrics Summary ====")
        for model_name, metrics in self.metrics_summary.items():
            print(f"\n{model_name}:")
            for idx, metric_dict in enumerate(metrics, start=1):
                print(f"  Experiment {idx}:")
                for metric_name, metric_value in metric_dict.items():
                    print(f"    {metric_name}: {metric_value:.4f}")
        print("\n===============================")