# PyCaret Time Series Prediction Development Changelog

## Project Overview
- **Goal**: Create a predictive model using PyCaret's time series module to forecast the one-period-ahead value of `cad_oas`
- **Data Source**: `monthly_oas_pycaret.csv`
- **Primary Target**: `cad_oas` column (univariate with exogenous variables)
- **Development Period**: January 2025

## Version History

### Current Version: v1.0.0_AdaBoost_CDS_DT_Univariate_with_Exog_Monthly_CAD_OAS_12M_Season
- **Version Name Breakdown**:
  - v1.0.0: Initial version
  - AdaBoost: Best performing model type
  - CDS_DT: Conditional Deseasonalization and Detrending
  - Univariate_with_Exog: Single target variable with exogenous predictors
  - Monthly: Data frequency
  - CAD_OAS: Target variable
  - 12M_Season: 12-month seasonality period

- **Version Description**:
  - Model: AdaBoost with Conditional Deseasonalization and Detrending
  - Type: Univariate with Exogenous Variables
  - Target: CAD OAS
  - Frequency: Monthly
  - Seasonality: 12 months
  - Features: All available as exogenous variables
  - Cross-validation: Expanding window
  - Forecast Horizon: 1 month ahead

## Development Journey

### Initial Setup and Implementation

1. **File Creation**
   - Created `test_pycaret_prediction.py` in the tests directory
   - Implemented basic structure with data loading and model training functions

2. **Initial Data Preparation**
   - Implemented `load_and_prepare_data` function
   - Key steps:
     - Load CSV data
     - Convert 'Date' column to datetime
     - Set date as index
     - Sort chronologically
     - Set monthly frequency

### Challenges and Solutions

1. **Data Frequency Issue**
   - **Problem**: Initial implementation didn't properly handle monthly frequency
   - **Solution**: Added `asfreq('M', method='ffill')` to ensure consistent monthly frequency
   - **Learning**: Time series data requires explicit frequency setting for proper modeling

2. **PyCaret Setup Parameters**
   - **Initial Error**: Incorrect parameters in setup function
   - **Fixed Parameters**:
     ```python
     setup(
         data=data,
         target='cad_oas',
         fh=1,
         fold=3,
         seasonal_period=12,
         fold_strategy='expanding',
         transform_target=None,
         session_id=123
     )
     ```
   - **Key Learnings**:
     - Removed unsupported parameters like 'train_size' and 'numeric_features'
     - Seasonal_period=12 for monthly data
     - Expanding fold strategy is better for time series data

3. **Prediction Access Error**
   - **Problem**: Initially tried to access predictions with incorrect column name
   - **Error Message**: `KeyError: 'Prediction'`
   - **Solution**: 
     - Used dynamic column access: `predictions.columns[-1]`
     - PyCaret's prediction column name varies based on the model
   - **Learning**: Always check actual column names in prediction output

4. **Error Handling and Logging**
   - **Initial Issue**: Lack of proper error handling and debugging information
   - **Solution**:
     - Added comprehensive try/except blocks
     - Implemented logging with meaningful messages
     ```python
     logging.basicConfig(
         level=logging.INFO,
         format='%(asctime)s - %(levelname)s - %(message)s'
     )
     ```

### Model Performance Analysis

1. **Best Performing Model**: AdaBoost with Conditional Deseasonalize & Detrending
   - MAPE: 2.43%
   - MAE: 2.5009
   - RMSE: 2.5009
   - Training Time: 0.0433 seconds

2. **Model Comparison Results**:
   - AdaBoost performed best (MASE: 0.2917)
   - Followed by Bayesian Ridge (MASE: 0.3594)
   - Traditional methods like ARIMA showed moderate performance (MASE: 0.5247)
   - Croston performed worst (MASE: 2.4669)

3. **Prediction Results**:
   - Last actual value: 98.5117
   - Predicted next value: 101.9598

### Version Control and Organization

1. **Version Naming Convention**:
   ```
   v[Major].[Minor].[Patch]_[Model]_[Preprocessing]_[Frequency]_[Target]_[Season]
   ```
   - Major: Significant architecture changes
   - Minor: Feature additions/improvements
   - Patch: Bug fixes and minor improvements
   - Model: Primary model type
   - Preprocessing: Key preprocessing steps
   - Frequency: Data frequency
   - Target: Target variable
   - Season: Seasonal configuration

2. **Version Documentation**:
   - Each version includes:
     - Detailed description
     - Model configuration
     - Performance metrics
     - Key changes from previous version
     - Known limitations

3. **Output Organization**:
   - Version-specific directories:
     ```
     outputs/pycaret_model/[version_name]/
     ├── model.joblib
     ├── predictions.csv
     ├── model_metrics.csv
     ├── feature_importance.csv
     └── visualizations/
         ├── actual_vs_predicted.png
         └── model_comparison.png
     ```

### Code Organization and Structure

1. **Main Components**:
   ```python
   def load_and_prepare_data(file_path):
       # Data loading and preparation
   
   def train_predict_model(data):
       # Model training and prediction
   
   def main():
       # Orchestration and execution
   ```

2. **File Structure**:
   - Main script: `test_pycaret_prediction.py`
   - Output predictions: `cad_oas_predictions.csv`
   - Input data: `monthly_oas_pycaret.csv`
   - Configuration: `config/model_config.yaml`

## Proposed Next Steps

1. **Model Improvements**:
   - Implement feature selection to identify most important predictors
   - Add cross-validation metrics reporting
   - Experiment with different seasonal periods
   - Try ensemble methods combining top performers

2. **Code Enhancements**:
   - Add model persistence (saving and loading)
   - Implement performance visualization
   - Add parameter tuning for the best model
   - Create a configuration file for model parameters

3. **Documentation and Testing**:
   - Add docstring documentation
   - Create unit tests
   - Add input data validation
   - Document model assumptions and limitations

4. **Production Readiness**:
   - Add model versioning
   - Implement monitoring for prediction accuracy
   - Create automated retraining pipeline
   - Add model interpretation tools

## Key Learnings

1. **Time Series Specific**:
   - Proper frequency handling is crucial
   - Expanding window cross-validation is preferred
   - Seasonal patterns need explicit handling

2. **PyCaret Usage**:
   - Setup parameters must match time series requirements
   - Model comparison provides valuable insights
   - Prediction column names are model-dependent

3. **Best Practices**:
   - Always implement proper error handling
   - Add comprehensive logging
   - Use dynamic column access for predictions
   - Save model artifacts and predictions
   - Implement clear versioning strategy

## Dependencies and Environment

- Python 3.11
- PyCaret (time series module)
- pandas
- numpy
- logging
- yaml (for configuration)
- joblib (for model persistence)

## References

- PyCaret Time Series Documentation
- Time Series Cross-Validation Best Practices
- Error Handling in Python Best Practices
- Semantic Versioning 2.0.0
