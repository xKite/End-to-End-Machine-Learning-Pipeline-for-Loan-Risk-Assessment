# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: A tuple containing the training, validation, and test features and target variables.
        """
        logging.info("Starting data splitting.")
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config["val_test_size"], random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["val_size"], random_state=42
        )
        logging.info("Data split into train, validation, test sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("regressor", model)]
            )
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )

        return pipelines, metrics
        
    # Train models using GridSearchCV
    def train_model(pipeline: Pipeline, param_grid: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a model using GridSearchCV and returns the best estimator.
    
        Args:
        -----
        pipeline : Pipeline
            Pipeline object containing the preprocessor and classifier.
        param_grid : dict
            Parameter grid for GridSearchCV.
        X_train : pd.DataFrame
            Training feature set.
        y_train : pd.Series
            Training target labels.

        Returns:
        --------
        best_model : estimator
            Best model found by GridSearchCV.
        """
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    # Evaluate model performance
    def evaluate_model(self, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Evaluates the model's performance on both training and validation sets.

        Args:
        -----
        model : Pipeline
            Trained model pipeline.
        X_train : pd.DataFrame
            Training feature set.
        y_train : pd.Series
            Training target labels.
        X_val : pd.DataFrame
            Validation feature set.
        y_val : pd.Series
            Validation target labels.
        """
        # Training evaluation
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
        print("Training Metrics:")
        print_metrics(y_train, y_train_pred, y_train_pred_proba)

        # Validation evaluation
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        print("Validation Metrics:")
        print_metrics(y_val, y_val_pred, y_val_pred_proba)

    # Print performance metrics
    def print_metrics(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series = None) -> None:
        """
        Prints various evaluation metrics for classification models.

        Args:
        -----
        y_true : pd.Series
            True target labels.
        y_pred : pd.Series
            Predicted target labels.
        y_pred_proba : pd.Series
            Predicted probabilities of the positive class (optional for models that don't support it).
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='high-risk')
        recall = recall_score(y_true, y_pred, pos_label='high-risk')
        f1 = f1_score(y_true, y_pred, pos_label='high-risk')

        print(f'Accuracy: {accuracy:.5f}')
        print(f'Precision: {precision:.5f}')
        print(f'Recall: {recall:.5f}')
        print(f'F1 Score: {f1:.5f}')

        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            print(f'ROC AUC: {roc_auc:.5f}')

    # Main function to run the training and evaluation
    def main(config: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Main function that orchestrates the model training and evaluation process.

        Args:
        -----
        config : dict
            Configuration dictionary for preprocessing.
        X_train : pd.DataFrame
            Training feature set.
        y_train : pd.Series
            Training target labels.
        X_val : pd.DataFrame
            Validation feature set.
        y_val : pd.Series
            Validation target labels.
        """      
        # Logistic Regression Model
        logistic_model = LogisticRegression(random_state=42)
        logistic_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', logistic_model)])
        logistic_param_grid = [
            {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__solver': ['liblinear'], 'classifier__penalty': ['l1']},
            {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__solver': ['lbfgs'], 'classifier__penalty': ['l2']}
        ]
        print("Training Logistic Regression...")
        best_logistic = train_model(logistic_pipeline, logistic_param_grid, X_train, y_train)
        evaluate_model(best_logistic, X_train, y_train, X_val, y_val)

        # K-Nearest Neighbors Model
        knn_model = KNeighborsClassifier()
        knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', knn_model)])
        knn_param_grid = {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance']}
        print("\nTraining KNN...")
        best_knn = train_model(knn_pipeline, knn_param_grid, X_train, y_train)
        evaluate_model(best_knn, X_train, y_train, X_val, y_val)

        # Decision Tree Model
        decision_tree_model = DecisionTreeClassifier(random_state=42)
        dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', decision_tree_model)])
        dt_param_grid = {
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__criterion': ['gini', 'entropy']
        }
        print("\nTraining Decision Tree...")
        best_dt = train_model(dt_pipeline, dt_param_grid, X_train, y_train)
        evaluate_model(best_dt, X_train, y_train, X_val, y_val)