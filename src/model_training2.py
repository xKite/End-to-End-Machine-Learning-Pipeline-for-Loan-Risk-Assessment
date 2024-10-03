import logging
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

class ModelTraining:
    """
    A class used to handle model training, evaluation, and hyperparameter tuning.
    
    Attributes:
    -----------
    config : dict
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessor pipeline used for transforming the dataset.
    """
    
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
        -----
        df : pd.DataFrame
            The cleaned DataFrame.

        Returns:
        --------
        tuple
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logging.info("Splitting data into train, validation, and test sets.")
        
        X = df.drop(columns=[self.config['columns']['target_column']])
        y = df[self.config['columns']['target_column']]
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, 
                                                            test_size=self.config['training']['test_size'], 
                                                            random_state=self.config['training']['random_state'])
        
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                                        test_size=0.5, 
                                                        random_state=self.config['training']['random_state'])
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(self, X_train, y_train, X_val, y_val):
        """
        Train and evaluate baseline models using default hyperparameters.

        Args:
        -----
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Training labels.
        X_val : pd.DataFrame
            Validation data.
        y_val : pd.Series
            Validation labels.

        Returns:
        --------
        tuple
            Dictionary of trained baseline models and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")

        baseline_models = {}
        baseline_metrics = {}

        # Logistic Regression
        logistic_model = LogisticRegression(random_state=42)
        logistic_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', logistic_model)
        ])
        logistic_pipeline.fit(X_train, y_train)
        baseline_models['Logistic Regression'] = logistic_pipeline
        baseline_metrics['Logistic Regression'] = self.evaluate_model(logistic_pipeline, X_val, y_val, "Logistic Regression")

        # K-Nearest Neighbors
        knn_model = KNeighborsClassifier()
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', knn_model)
        ])
        knn_pipeline.fit(X_train, y_train)
        baseline_models['KNN'] = knn_pipeline
        baseline_metrics['KNN'] = self.evaluate_model(knn_pipeline, X_val, y_val, "KNN")

        # Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', dt_model)
        ])
        dt_pipeline.fit(X_train, y_train)
        baseline_models['Decision Tree'] = dt_pipeline
        baseline_metrics['Decision Tree'] = self.evaluate_model(dt_pipeline, X_val, y_val, "Decision Tree")

        return baseline_models, baseline_metrics

    def train_and_evaluate_tuned_models(self, X_train, y_train, X_val, y_val):
        """
        Train and evaluate models with hyperparameter tuning.

        Args:
        -----
        X_train : pd.DataFrame
            Training data.
        y_train : pd.Series
            Training labels.
        X_val : pd.DataFrame
            Validation data.
        y_val : pd.Series
            Validation labels.

        Returns:
        --------
        tuple
            Dictionary of trained tuned models and their evaluation metrics.
        """
        logging.info("Training and evaluating tuned models.")
        
        tuned_models = {}
        tuned_metrics = {}

        # Logistic Regression
        logistic_model = LogisticRegression(random_state=42)
        logistic_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', logistic_model)
        ])
        logistic_param_grid = self.config['logistic_regression']
        logistic_grid = GridSearchCV(logistic_pipeline, param_grid=logistic_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        logistic_grid.fit(X_train, y_train)
        tuned_models['Logistic Regression'] = logistic_grid.best_estimator_
        tuned_metrics['Logistic Regression'] = self.evaluate_model(logistic_grid.best_estimator_, X_val, y_val, "Logistic Regression (Tuned)")

        # K-Nearest Neighbors
        knn_model = KNeighborsClassifier()
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', knn_model)
        ])
        knn_param_grid = self.config['knn']
        knn_grid = GridSearchCV(knn_pipeline, param_grid=knn_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        tuned_models['KNN'] = knn_grid.best_estimator_
        tuned_metrics['KNN'] = self.evaluate_model(knn_grid.best_estimator_, X_val, y_val, "KNN (Tuned)")

        # Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', dt_model)
        ])
        dt_param_grid = self.config['decision_tree']
        dt_grid = GridSearchCV(dt_pipeline, param_grid=dt_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        dt_grid.fit(X_train, y_train)
        tuned_models['Decision Tree'] = dt_grid.best_estimator_
        tuned_metrics['Decision Tree'] = self.evaluate_model(dt_grid.best_estimator_, X_val, y_val, "Decision Tree (Tuned)")

        return tuned_models, tuned_metrics

    def evaluate_model(self, model, X, y, model_name):
        """
        Evaluate a model on the given dataset and log the results.

        Args:
        -----
        model : sklearn estimator
            Trained model to evaluate.
        X : pd.DataFrame
            Features for evaluation.
        y : pd.Series
            True labels.
        model_name : str
            The name of the model being evaluated.

        Returns:
        --------
        dict
            Dictionary of evaluation metrics.
        """
        logging.info(f"Evaluating {model_name}.")
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, pos_label='high-risk')
        recall = recall_score(y, y_pred, pos_label='high-risk')
        f1 = f1_score(y, y_pred, pos_label='high-risk')
        roc_auc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None
        
        logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
        logging.info(f"{model_name} Precision: {precision:.4f}")
        logging.info(f"{model_name} Recall: {recall:.4f}")
        logging.info(f"{model_name} F1 Score: {f1:.4f}")
        if roc_auc:
            logging.info(f"{model_name} ROC AUC: {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=['low-risk', 'high-risk']).plot()
        plt.title(f"Confusion Matrix for {model_name}")
        plt.show()

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc if roc_auc else "N/A"
        }

    def evaluate_final_model(self, model, X_test, y_test, model_name):
        """
        Evaluate the final model on the test set.

        Args:
        -----
        model : sklearn estimator
            The final trained model.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            True test labels.
        model_name : str
            The name of the model being evaluated.

        Returns:
        --------
        dict
            Dictionary of evaluation metrics for the test set.
        """
        logging.info(f"Evaluating final model {model_name} on the test set.")
        return self.evaluate_model(model, X_test, y_test, model_name)
