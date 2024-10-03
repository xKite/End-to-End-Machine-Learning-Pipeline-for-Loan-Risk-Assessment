import logging
from typing import Any, Dict
import pandas as pd
import numpy as np
import unittest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)

class DataPreparation:
    """
    A class used to clean and preprocess predicting loan risk data.
    
    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """
        
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = self._create_preprocessor()

        try:
            logging.info("Starting data cleaning.")
            df = pd.read_csv(self.config["data"]["dataset_path"], index_col=self.config["data"].get("index_column", 0))
            logging.info(f"Rows before cleaning: {len(df)}")

        # Proceed with cleaning or other operations here...
            
        except FileNotFoundError:
            logging.error(f"File not found: {self.config['data']['dataset_path']}")
            df = None

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            df = None

        # Continue with the rest of the initialization steps if df is successfully loaded
        if df is not None:
            self.df = df
        else:
            logging.error("Data frame is not loaded. Exiting.")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.
    
        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.
    
        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        logging.info("Starting data cleaning.")
        logging.info(f"Rows before cleaning: {len(df)}")
        
        # Drop duplicates
        df = self._drop_duplicates(df)
        
        # Clean Risk_Category column
        df = self._clean_risk_category(df)
        
        # Convert columns to absolute values
        df = self._convert_columns_to_absolute(df)
        
        # Fill missing values
        df = self._fill_missing_values(df)
        
        # Detect and handle outliers
        df = self._detect_outliers(df)
        
        # Feature engineering: create interaction terms
        df = self._create_interaction_terms(df)
        
        # Handle missing values and outliers for newly created features
        df = self._fill_missing_values(df)
        df = self._detect_outliers(df)
        
        logging.info(f"Rows after cleaning: {len(df)}")
        logging.info("Data cleaning completed.")

        logging.info("Starting full pipeline processing.")
        pipeline = self.build_pipeline()
        df_cleaned = pipeline.fit_transform(df)
        logging.info("Pipeline processing completed.")
        
        return df
    
    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops duplicate rows from the DataFrame.
        """
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        logging.info(f"Dropped {before - after} duplicates.")
        return df
    
    def _clean_risk_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the Risk_Category column by replacing specific values.
        """
        df['Risk_Category'] = df['Risk_Category'].replace({
            'low-ris': 'low-risk', 
            'high-ris': 'high-risk'
        })
        return df
    
    def _convert_columns_to_absolute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts specified numerical columns to their absolute values.
        """
        columns_to_absolute = [
            'Loan_Amount', 'Credit_Score', 'Annual_Income',
            'Employment_Length', 'Debt-to-Income_Ratio',
            'Number_of_Open_Accounts', 'Number_of_Past_Due_Payments'
        ]
        df[columns_to_absolute] = df[columns_to_absolute].abs()
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values for both categorical and numerical columns.
        """
        logging.info("Filling missing values.")
        df.fillna({
            'Risk_Category': df['Risk_Category'].mode()[0],
            'Loan_Purpose': df['Loan_Purpose'].mode()[0],
            'Loan_Amount': df['Loan_Amount'].median(),
            'Credit_Score': df['Credit_Score'].median(),
            'Annual_Income': df['Annual_Income'].median(),
            'Debt-to-Income_Ratio': df['Debt-to-Income_Ratio'].median(),
            'Number_of_Past_Due_Payments': df['Number_of_Past_Due_Payments'].median()
        }, inplace=True)
        logging.info("Missing values filled.")
        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using the Interquartile Range (IQR) method.
        
        Args:
        -----
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        --------
        pd.DataFrame: DataFrame with outliers handled.
        """
        logging.info("Detecting and handling outliers.")
        numerical_features = self.config["columns"]["numerical_features"]
    
        for col in numerical_features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        logging.info("Outliers handled.")
        return df

    def _create_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction terms between numerical features and polynomial features.
        
        Args:
        -----
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        --------
        pd.DataFrame: DataFrame with interaction terms.
        """
        logging.info("Creating interaction terms and polynomial features.")
        
        # Ensure the required columns are present
        required_columns = ['Annual_Income', 'Loan_Amount', 'Debt-to-Income_Ratio']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Required columns {missing_columns} are missing from the DataFrame.")

        # Ensure the required columns are present
        required_columns = ['Annual_Income', 'Loan_Amount', 'Debt-to-Income_Ratio']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is missing from the DataFrame.")
        
        # Create interaction terms
        df['Income_to_Loan_Ratio'] = df['Annual_Income'] / df['Loan_Amount']
        df['Debt_to_Income'] = df['Debt-to-Income_Ratio'] / df['Annual_Income']
    
        # Polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[required_columns])
    
        # Creating DataFrame for polynomial features
        poly_feature_names = poly.get_feature_names_out(required_columns)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        
        # Avoiding duplication of columns by checking for conflicts
        duplicated_columns = set(poly_df.columns).intersection(df.columns)
        if duplicated_columns:
            logging.warning(f"Duplicated columns found: {duplicated_columns}. These columns will be replaced.")
            df.drop(duplicated_columns, axis=1, inplace=True)
    
        # Combine the original DataFrame with polynomial features
        df = pd.concat([df, poly_df], axis=1)
        
        logging.info("Interaction terms and polynomial features created.")
        return df

    def build_pipeline(self) -> Pipeline:
        """
        Builds the full pipeline that integrates feature engineering and preprocessing.
        """
        pipeline = Pipeline(steps=[
            ('preprocessing', self.preprocessor),
            ('feature_engineering', FeatureEngineering()),
            # If more steps are required (like model training), they can be added here
        ])
        
        return pipeline

    
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_features = self.config["columns"]["numerical_features"]
        nominal_features = ["Loan_Purpose"]

        # Define individual transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outliers', OutlierHandler(numerical_features)),
            ('scaler', StandardScaler())
        ])
        nominal_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(
                 categories=self.config["columns"]["loan_purpose_categories"],
                 handle_unknown="ignore"))            
            ])                    
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["columns"]["numerical_features"]),
                ("nom", nominal_transformer, self.config["columns"]["nominal_features"]),                                
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor

# Custom Outlier Detection Transformer
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    A custom transformer for handling outliers in a pipeline.
    """
    def __init__(self, method='cap', factor=1.5):
        """
        Initialize the OutlierHandler with a method for handling outliers and a factor for IQR.
        
        Args:
        method (str): The method to handle outliers. Options are 'remove' or 'cap'. Default is 'cap'.
        factor (float): The IQR multiplier for detecting outliers. Default is 1.5.
        """
        self.method = method
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Detects and handles outliers in the dataset based on the selected method.
        
        Args:
        X (pd.DataFrame): The input DataFrame.
        
        Returns:
        pd.DataFrame: DataFrame with outliers handled.
        """
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            
            if self.method == 'cap':
                X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
                X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
            elif self.method == 'remove':
                X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        
        return X

# Custom Feature Engineering Transformer
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        logging.info("Creating interaction terms and polynomial features.")
        
        # Ensure necessary columns are present
        if 'Annual_Income' not in X_copy.columns or 'Loan_Amount' not in X_copy.columns:
            raise ValueError("Required columns 'Annual_Income' or 'Loan_Amount' are missing.")
        
        if 'Debt-to-Income_Ratio' not in X_copy.columns:
            raise ValueError("Required column 'Debt-to-Income_Ratio' is missing.")
        
        # Create interaction features
        X_copy['Income_to_Loan_Ratio'] = X_copy['Annual_Income'] / X_copy['Loan_Amount']
        X_copy['Debt_to_Income'] = X_copy['Debt-to-Income_Ratio'] / X_copy['Annual_Income']

        # Polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(X_copy[['Annual_Income', 'Loan_Amount', 'Debt-to-Income_Ratio']])
        
        # Creating DataFrame for polynomial features
        poly_feature_names = poly.get_feature_names_out(['Annual_Income', 'Loan_Amount', 'Debt-to-Income_Ratio'])
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X_copy.index)
        
        # Combine the original DataFrame with polynomial features
        X_copy = pd.concat([X_copy, poly_df], axis=1)
        
        return X_copy

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.config = {'data': {'dataset_path': 'path/to/data.csv'}}
        self.data_preparation = DataPreparation(self.config)

    def test_drop_duplicates(self):
        df = pd.DataFrame({'A': [1, 2, 2], 'B': [3, 4, 4]})
        cleaned_df = self.data_preparation._drop_duplicates(df)
        self.assertEqual(len(cleaned_df), 2)

    def test_empty_dataframe(self):
        with self.assertRaises(ValueError):
            self.data_preparation.clean_data(pd.DataFrame())

if __name__ == '__main__':
    unittest.main()
