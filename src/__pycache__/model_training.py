import logging
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)

class ModelTraining:
    """
    A class to handle model training, hyperparameter tuning, and evaluation for loan risk prediction.
    
    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming features.
    """
    
    def __init__(self, config: Dict[str, Any], preprocessor: Any):
        self.config = config
        self.preprocessor = preprocessor
        self.smote = SMOTE(random_state=self.config["training"]["random_state"])

    def split_data(self, df: pd.DataFrame) -> Tuple:
        """
        Splits the cleaned data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The cleaned DataFrame.

        Returns:
        --------
        Tuple: Split data into training, validation, and test sets.
        """            
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        logging.info(f"Columns in DataFrame after cleaning: {df.columns.tolist()}")
    
        target_column = 'Risk_Category'  # Assuming this is the target column
    
        if target_column not in df.columns:
            logging.error(f"Target column '{target_column}' not found in DataFrame.")
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    
        X = df.drop(columns=[target_column])
        y = df[target_column]
    
        logging.info(f"y unique values: {y.unique()}")
        logging.info(f"Columns in X after dropping target: {X.columns.tolist()}")
    
        # Split the data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config["training"]["test_size"], random_state=self.config["training"]["random_state"]
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.config["training"]["val_size"], random_state=self.config["training"]["random_state"]
        )
    
        logging.info(f"Columns for preprocessing: {X_train.columns.tolist()}")
    
        # Preprocessing step
        try:
            X_train_preprocessed = self.preprocessor.fit_transform(X_train)
            # Convert back to DataFrame and set appropriate column names
            X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=self.preprocessor.get_feature_names_out())
            logging.info(f"Columns after preprocessing: {X_train_preprocessed.columns.tolist()}")
        except ValueError as e:
            logging.error(f"Preprocessing error: {e}")
            raise
    
        X_val_preprocessed = self.preprocessor.transform(X_val)
        X_test_preprocessed = self.preprocessor.transform(X_test)
    
        # Convert validation and test sets back to DataFrames
        X_val_preprocessed = pd.DataFrame(X_val_preprocessed, columns=self.preprocessor.get_feature_names_out())
        X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=self.preprocessor.get_feature_names_out())
    
        # Apply SMOTE
        smote = SMOTE(random_state=self.config["training"]["random_state"])
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
    
        logging.info("Data split into training, validation, and test sets.")
        return X_train_resampled, X_val_preprocessed, X_test_preprocessed, y_train_resampled, y_val, y_test

    def train_and_evaluate_baseline_models(self, X_train, y_train, X_val, y_val):
        """
        Trains baseline models with default hyperparameters and evaluates them.

        Args:
        -----
        X_train, y_train, X_val, y_val: Data splits for training and validation.

        Returns:
        --------
        Dict: Models and their evaluation metrics.
        """
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.config["training"]["random_state"]),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=self.config["training"]["random_state"])
        }

        baseline_models = {}
        baseline_metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
            pipeline.fit(X_train, y_train)

            # Predictions on the validation set
            y_val_pred = pipeline.predict(X_val)
            y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

            # Calculate evaluation metrics
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, pos_label='high-risk')
            val_recall = recall_score(y_val, y_val_pred, pos_label='high-risk')
            val_f1 = f1_score(y_val, y_val_pred, pos_label='high-risk')
            val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

            baseline_models[model_name] = pipeline
            baseline_metrics[model_name] = {
                'Accuracy': val_accuracy,
                'Precision': val_precision,
                'Recall': val_recall,
                'F1 Score': val_f1,
                'ROC AUC': val_roc_auc
            }

            logging.info(f"Baseline {model_name} Validation Metrics: {baseline_metrics[model_name]}")

        return baseline_models, baseline_metrics

    def train_and_evaluate_tuned_models(self, X_train, y_train, X_val, y_val):
        """
        Trains models using hyperparameter tuning and evaluates them.

        Args:
        -----
        X_train, y_train, X_val, y_val: Data splits for training and validation.

        Returns:
        --------
        Dict: Tuned models and their evaluation metrics.
        """
        param_grids = {
            'Logistic Regression': {
                'classifier__C': self.config['logistic_regression']['C_values'],
                'classifier__solver': self.config['logistic_regression']['solver'],
                'classifier__penalty': self.config['logistic_regression']['penalties']
            },
            'K-Nearest Neighbors': {
                'classifier__n_neighbors': self.config['knn']['n_neighbors'],
                'classifier__weights': self.config['knn']['weights'],
                'classifier__metric': self.config['knn']['metrics']
            },
            'Decision Tree': {
                'classifier__max_depth': self.config['decision_tree']['max_depth'],
                'classifier__min_samples_split': self.config['decision_tree']['min_samples_split'],
                'classifier__min_samples_leaf': self.config['decision_tree']['min_samples_leaf'],
                'classifier__criterion': self.config['decision_tree']['criterion']
            }
        }

        models = {
            'Logistic Regression': LogisticRegression(random_state=self.config["training"]["random_state"]),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=self.config["training"]["random_state"])
        }

        tuned_models = {}
        tuned_metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])

            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grids[model_name],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Predictions on the validation set
            y_val_pred = best_model.predict(X_val)
            y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]

            # Calculate evaluation metrics
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, pos_label='high-risk')
            val_recall = recall_score(y_val, y_val_pred, pos_label='high-risk')
            val_f1 = f1_score(y_val, y_val_pred, pos_label='high-risk')
            val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)

            tuned_models[model_name] = best_model
            tuned_metrics[model_name] = {
                'Accuracy': val_accuracy,
                'Precision': val_precision,
                'Recall': val_recall,
                'F1 Score': val_f1,
                'ROC AUC': val_roc_auc
            }

            logging.info(f"Tuned {model_name} Validation Metrics: {tuned_metrics[model_name]}")

        return tuned_models, tuned_metrics

    def evaluate_final_model(self, model, X_test, y_test, model_name: str):
        """
        Evaluates the best model on the test set.

        Args:
        -----
        model: The trained model.
        X_test, y_test: Test data.
        model_name (str): Name of the model being evaluated.

        Returns:
        --------
        Dict: Evaluation metrics on the test set.
        """
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, pos_label='high-risk')
        test_recall = recall_score(y_test, y_test_pred, pos_label='high-risk')
        test_f1 = f1_score(y_test, y_test_pred, pos_label='high-risk')
        test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)

        final_metrics = {
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1,
            'ROC AUC': test_roc_auc
        }

        logging.info(f"Final {model_name} Test Metrics: {final_metrics}")
        return final_metrics
