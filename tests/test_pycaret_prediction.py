import pandas as pd
from pycaret.time_series import *
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_prepare_data(file_path):
    """
    Load and prepare the data for time series modeling
    """
    try:
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert Date column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Sort index to ensure chronological order
        df = df.sort_index()
        
        # Ensure monthly frequency and forward fill any missing values
        df = df.asfreq('M', method='ffill')
        
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def train_predict_model(data):
    """
    Train a time series model and make predictions
    """
    try:
        logging.info("Initializing model setup")
        s = setup(
            data=data,
            target='cad_oas',
            fh=1,
            fold=3,
            seasonal_period=12,
            fold_strategy='expanding',
            transform_target=None,
            session_id=123
        )
        
        # Compare all available models
        logging.info("Training and comparing models")
        best_model = compare_models(sort='MAPE')
        
        # Get model performance metrics
        logging.info("Getting model performance metrics")
        model_metrics = pull()
        
        # Finalize the best model
        logging.info("Finalizing best model")
        final_model = finalize_model(best_model)
        
        return final_model, model_metrics
    except Exception as e:
        logging.error(f"Error in model training/prediction: {str(e)}")
        raise

def analyze_model_performance(model, data):
    """
    Analyze model performance metrics
    
    Args:
        model: Trained model from PyCaret
        data: Training data DataFrame
    """
    try:
        # Get performance metrics
        metrics = pull()
        print("\nModel Performance Metrics:")
        print(metrics)
        
        # Get exogenous variables for prediction
        exog_data = data.drop('cad_oas', axis=1).iloc[-1:].copy()
        
        # Make predictions
        logging.info("Making predictions")
        predictions = predict_model(model, X=exog_data)
        
        # Save predictions
        csv_path = 'csv_outputs/cad_oas_predictions.csv'
        predictions.to_csv(csv_path)
        logging.info(f"Predictions saved to {os.path.abspath(csv_path)}")
        
        # Get last actual and predicted values
        last_actual = data['cad_oas'].iloc[-1]
        # Get the prediction column (should be the last column)
        pred_col = predictions.columns[-1]
        next_pred = predictions[pred_col].iloc[-1]
        
        print("\nPrediction Results:")
        print(f"Last actual value: {last_actual:.4f}")
        print(f"Predicted next value: {next_pred:.4f}")
        
    except Exception as e:
        logging.error(f"Error analyzing model performance: {str(e)}")
        raise

def main():
    try:
        # File path
        file_path = 'c:/Users/Eddy/Documents/auto_ml/csv_outputs/monthly_oas_pycaret.csv'
        
        # Load and prepare data
        data = load_and_prepare_data(file_path)
        
        # Train model and get predictions
        model, metrics = train_predict_model(data)
        
        # Analyze model performance
        analyze_model_performance(model, data)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
