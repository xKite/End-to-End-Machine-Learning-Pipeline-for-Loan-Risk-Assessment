import logging
from typing import Any, Dict
import pandas as pd
import numpy as np
import unittest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)

class OutlierHandler:
    def __init__(self, columns_to_check=None, iqr_factor=1.5):
        # If no columns are provided, initialize with default columns
        if columns_to_check is None:
            self.columns_to_check = ['Loan_Amount', 'Credit_Score', 'Annual_Income', 
                                     'Employment_Length', 'Debt-to-Income_Ratio', 
                                     'Number_of_Open_Accounts', 'Number_of_Past_Due_Payments']
        else:
            self.columns_to_check = columns_to_check
        self.iqr_factor = iqr_factor

    def fit(self, X, y=None):
        # If X is a DataFrame, just return self
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        # Convert to DataFrame if X is a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_to_check)

        # Log the available columns
        print("Columns in the DataFrame:", X.columns.tolist())
                
        for col in self.columns_to_check:
            # Ensure the column exists before trying to process it
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                # Outlier handling logic: filtering data
                X = X[(X[col] >= (Q1 - 1.5 * IQR)) & (X[col] <= (Q3 + 1.5 * IQR))]
            else:
                print(f"Column '{col}' not found in DataFrame.")
        
        return X

    def fit_transform(self, X, y=None):
        """
        Calls the fit method followed by the transform method. Accepts X and an optional y argument.
        """
        self.fit(X, y)  # Fit logic can be called, but you likely don't need to use 'y'
        return self.transform(X)

class DataPreparation:
    """
    A class used to clean and preprocess predicting loan risk data.
    
    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.        
    Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
    """
        
    def __init__(self, config: Dict[str, Any]):                    
        self.config = config
        self.preprocessor = self._create_preprocessor()
        
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

        # Create the Income_to_Loan_Ratio feature
        df = self._create_income_to_loan_ratio(df)
        
        # Detect and handle outliers
        df = self._detect_outliers(df)

        # Fill missing values again after outlier removal
        df = self._fill_missing_values(df)
                        
        # Feature engineering: create interaction terms
        df = self._create_interaction_terms(df)

        # Handle missing values and outliers for newly created features
        df = self._fill_missing_values(df)
        df = self._detect_outliers(df)
        
        logging.info(f"Rows after cleaning: {len(df)}")
        logging.info("Data cleaning completed.")
        
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
        # Automatically identify all numerical columns (including newly created features)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Use SimpleImputer for numerical columns (filling with median)
        imputer = SimpleImputer(strategy='median')
        df[num_cols] = imputer.fit_transform(df[num_cols])
        
        # Fill missing values for categorical columns (filling with mode)
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
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
        if 'Income_to_Loan_Ratio' not in df.columns:
            raise ValueError("The 'Income_to_Loan_Ratio' column is missing.")
            
        logging.info("Detecting and handling outliers.")
        numerical_features = self.config["columns"]["numerical_features"]    
        outlier_handler = OutlierHandler(numerical_features)
        df[numerical_features] = outlier_handler.fit_transform(df[numerical_features])
        logging.info("Outliers handled.")
        return df

    def _create_income_to_loan_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the Income_to_Loan_Ratio feature by dividing Annual_Income by Loan_Amount.
        """
        logging.info("Creating Income_to_Loan_Ratio feature.")
        
        # Avoid division by zero by checking for zeros in Loan_Amount
        if 'Loan_Amount' in df.columns and 'Annual_Income' in df.columns:
            # Handle potential Loan_Amount values that are zero or very small to avoid division errors
            df['Loan_Amount'] = df['Loan_Amount'].replace(0, 1)  # Replace zero Loan_Amount to avoid division by zero
            df['Income_to_Loan_Ratio'] = df['Annual_Income'] / df['Loan_Amount']
            logging.info("Income_to_Loan_Ratio feature created.")
        else:
            raise KeyError("Either 'Loan_Amount' or 'Annual_Income' column is missing.")
    
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
        if 'Annual_Income' not in df.columns or 'Loan_Amount' not in df.columns:
            raise ValueError("Required columns 'Annual_Income' or 'Loan_Amount' are missing from the DataFrame.")
        
        if 'Debt-to-Income_Ratio' not in df.columns:
            raise ValueError("Required column 'Debt-to-Income_Ratio' is missing from the DataFrame.")
                                
        df['Debt_to_Income'] = df['Debt-to-Income_Ratio'] / df['Annual_Income']

        # Polynomial features
        cols = self.config["columns"]["numerical_features"]
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[cols])

        # Creating DataFrame for polynomial features
        poly_feature_names = poly.get_feature_names_out(cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

        # Combine the original DataFrame with polynomial features
        df = pd.concat([df, poly_df], axis=1)

        # Avoiding duplication of columns by checking for conflicts
        duplicated_columns = set(poly_df.columns).intersection(df.columns)
        if duplicated_columns:
            logging.warning(f"Duplicated columns found: {duplicated_columns}. These columns will be replaced.")
            df = df.loc[:, ~df.columns.duplicated()]

        # drop unrelated columns
        df.drop(columns=['Annual_Income^2', 'Loan_Amount^2', 'Debt-to-Income_Ratio^2'],  errors='ignore', inplace=True)
                
        logging.info("Interaction terms and polynomial features created.")
        return df

    
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        # Specify which columns are numerical and which are categorical
        numerical_features = self.config["columns"]["numerical_features" ]
        nominal_features = self.config["columns"]["nominal_features"]
        
        # Create transformers for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly_features', PolynomialFeatures(degree=2))
            ])
        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ("onehot", 
             OneHotEncoder(
                 categories=[self.config["columns"]["loan_purpose_categories"]],
                 handle_unknown="ignore"))  # Ignore unknown categories during encoding
        ])
        
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["columns"]["numerical_features"]),
                ("nom", nominal_transformer, self.config["columns"]["nominal_features"]),
                # ("ord", ordinal_transformer, self.config["columns"]["ordinal_features"]),
                # ("pass", "passthrough", self.config["columns"]["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.config = {
            'data': {'dataset_path': 'path/to/data.csv'},
            'columns': {
                'numerical_features': ['Loan_Amount', 'Credit_Score', 'Annual_Income', 'Debt-to-Income_Ratio'],
                'nominal_features': ['Loan_Purpose'],
                'loan_purpose_categories': ['business', 'home improvement', 'medical expenses', 'debt consolidation', 'car'],
            },
        }

        # Create a mock DataFrame for testing
        self.df = pd.DataFrame({
            'Risk_Category': ['low-risk', 'high-risk'],
            'Loan_Amount': [10000, 20000, 15000],
            'Credit_Score': [700, np.nan, 650],
            'Loan_Purpose': ['business', 'home improvement', 'medical expenses', 'debt consolidation', 'car'],
            'Annual_Income': [50000, 60000, np.nan],
            'Employment_Length': [5, 10, 2],
            'Debt-to-Income_Ratio': [0.2, 0.3, 0.25],
            'Number_of_Open_Accounts': [2, 3, 1],
            'Number_of_Past_Due_Payments': [0, 1, 0]
        })

    def test_clean_data(self):
        dp = DataPreparation(self.config)
        cleaned_df = dp.clean_data(self.df)
        
        # Assertions to verify the data cleaning
        self.assertEqual(len(cleaned_df), len(self.df))
        self.assertTrue('Income_to_Loan_Ratio' in cleaned_df.columns)

if __name__ == "__main__":
    unittest.main()