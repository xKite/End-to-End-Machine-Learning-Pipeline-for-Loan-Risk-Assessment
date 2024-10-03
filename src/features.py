import logging
from typing import Any, Dict
import pandas as pd
import numpy as np
import unittest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
    Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
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
            df = None  # Initialize df to None in case of failure

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            df = None  # Initialize df to None in case of other errors

        # Continue with the rest of the initialization steps if df is successfully loaded
        if df is not None:
            self.df = df  # Assign the dataframe to the class attribute for future use
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
        
        # Create the Income_to_Loan_Ratio feature
        df = self._create_income_to_loan_ratio(df)
        
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
        if 'Income_to_Loan_Ratio' not in df.columns:
            raise ValueError("The 'Income_to_Loan_Ratio' column is missing.")
            
        logging.info("Detecting and handling outliers.")
        numerical_features = self.config["columns"]["numerical_features"]
    
        for col in numerical_features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Option 1: Remove outliers
            # df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Option 2: Cap outliers (to keep rows intact)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        logging.info("Outliers handled.")
        return df

    def _create_income_to_loan_ratio(df):
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
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[['Annual_Income', 'Loan_Amount', 'Debt-to-Income_Ratio']])

        # Creating DataFrame for polynomial features
        poly_feature_names = poly.get_feature_names_out(['Annual_Income', 'Loan_Amount', 'Debt-to-Income_Ratio'])
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

    
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        # Specify which columns are numerical and which are categorical
        numerical_features = self.config["columns"]["numerical_features"]
        categorical_features = self.config["columns"]["nominal_features"]
        
        # Create transformers for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outliers', OutlierHandler(numerical_features)),
            ('scaler', StandardScaler())
            ])
        nominal_transformer = Pipeline(steps=[
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
        self.config = {'data': {'dataset_path': 'path/to/data.csv'}}
        self.data_preparation = DataPreparation(self.config)

    def test_drop_duplicates(self):
        df = pd.DataFrame({'A': [1, 2, 2], 'B': [3, 4, 4]})
        cleaned_df = self.data_preparation._drop_duplicates(df)
        self.assertEqual(len(cleaned_df), 2)  # Ensure duplicates are dropped

    def test_empty_dataframe(self):
        with self.assertRaises(ValueError):
            self.data_preparation.clean_data(pd.DataFrame())

if __name__ == '__main__':
    unittest.main()