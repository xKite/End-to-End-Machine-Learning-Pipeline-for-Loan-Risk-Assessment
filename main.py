# Standard library imports
import logging
import os
import argparse

# Third-party imports
import pandas as pd
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

# Configure logging to output to both console and a log file
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, DEBUG, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for the log messages
    handlers=[
        logging.FileHandler("model_training.log"),  # Log messages to a file
        logging.StreamHandler()  # Log messages to the console
    ]
)

@ignore_warnings(category=Warning)
def main():
    """
    Main function to load data, perform data preparation, and handle errors.
    """
    # Command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Run machine learning model training pipeline.")
    parser.add_argument("--model", type=str, default="all", help="Specify which model to run: 'Logistic Regression', 'knn', 'decision_tree', or 'all'")
    args = parser.parse_args()
    
    # Configuration file path
    config = {
        "data": {
            "dataset_path": "./data/data.csv",
            "index_column": None  # Optional: If you have a specific index column
        },
        "columns": {
            "numerical_features": [
                "Loan_Amount", "Credit_Score", "Annual_Income",
                "Employment_Length", "Debt-to-Income_Ratio",
                "Number_of_Open_Accounts", "Number_of_Past_Due_Payments"
            ],
            "nominal_features": ["Loan_Purpose"],
            "loan_purpose_categories": ['business', 'home improvement', 'medical expenses', 'debt consolidation', 'car'
            ]
        }
    }

    dataset_path = config["data"]["dataset_path"]

    # Check if the dataset file exists
    if not os.path.exists(dataset_path):
        logging.error(f"The file at {dataset_path} does not exist.")
        return

    try:
        # Load CSV file into a DataFrame
        logging.info(f"Loading data from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Check if 'Unnamed: 0' exists and drop it
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            logging.info("Dropped 'Unnamed: 0' column.")        
        
        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError("The loaded DataFrame is empty. Please check the dataset.")

        # Initialize and run data preparation
        data_prep = DataPreparation(config)
        cleaned_df = data_prep.clean_data(df)
        logging.info("Data preparation completed successfully.")
        logging.info(f"Columns after cleaning: {df.columns}")

        # Initialize the model training class
        model_trainer = ModelTraining(config, data_prep.preprocessor)

        # Model experimentation based on user input
        if args.model == 'all' or args.model == 'Logistic Regression':
            logging.info("Starting logistic model evaluation.")
            model_trainer.run_baseline(cleaned_df)
            
        if args.model == 'all' or args.model == 'knn':
            logging.info("Starting KNN model evaluation.")
            model_trainer.run_knn(cleaned_df)

        if args.model == 'all' or args.model == 'decision_tree':
            logging.info("Starting Decision Tree model evaluation.")
            model_trainer.run_decision_tree(cleaned_df)

        logging.info("Model evaluation completed successfully.")

        # Run baseline model
        # logging.info("Starting baseline model evaluation.")
        # model_trainer.run_baseline(cleaned_df)

        # Run KNN classifier
        # logging.info("Starting KNN model evaluation.")
        # model_trainer.run_knn(cleaned_df)

        # Run Decision Tree classifier
        # logging.info("Starting Decision Tree model evaluation.")
        # model_trainer.run_decision_tree(cleaned_df)

        # logging.info("Model evaluation completed successfully.")

    except FileNotFoundError:
        logging.error(f"File not found: {dataset_path}")
    except pd.errors.EmptyDataError:
        logging.error(f"The file at {dataset_path} is empty or corrupt.")
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()