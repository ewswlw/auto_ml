"""
Test file to verify PyCaret time series functionality with a simple trading strategy.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pycaret.time_series import TSForecastingExperiment
import plotly.graph_objects as go

def create_dummy_data(periods=500, base_price=100):
    """Create dummy stock price data with trend, seasonality, and noise."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    
    # Components
    trend = np.linspace(0, 50, periods)
    noise = np.random.normal(0, 5, periods)
    seasonal = 10 * np.sin(np.linspace(0, 8*np.pi, periods))
    
    # Combine components
    prices = base_price + trend + noise + seasonal
    return pd.Series(prices, index=dates, name='Close')

def generate_signals(predictions, threshold=0.01):
    """Generate trading signals based on predicted price movements."""
    signals = pd.DataFrame(index=predictions.index)
    signals['Predicted'] = predictions['y_pred']
    signals['Returns'] = signals['Predicted'].pct_change()
    signals['Signal'] = 0
    signals.loc[signals['Returns'] > threshold, 'Signal'] = 1  # Buy signal
    signals.loc[signals['Returns'] < -threshold, 'Signal'] = -1  # Sell signal
    return signals

def plot_results(y, predictions, signals):
    """Plot the actual prices, predictions, and trading signals."""
    fig = go.Figure()

    # Convert datetime index to strings for plotly
    y_index = [str(idx) for idx in y.index]
    pred_index = [str(idx) for idx in predictions.index]

    # Add actual prices
    fig.add_trace(go.Scatter(
        x=y_index,
        y=y.values,
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=pred_index,
        y=predictions['y_pred'],
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    ))

    # Add buy/sell signals
    buy_signals = signals[signals['Signal'] == 1]
    sell_signals = signals[signals['Signal'] == -1]

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=[str(idx) for idx in buy_signals.index],
            y=buy_signals['Predicted'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))

    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=[str(idx) for idx in sell_signals.index],
            y=sell_signals['Predicted'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))

    fig.update_layout(
        title='Stock Price Forecasting with Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    return fig

def main():
    """Main function to test PyCaret time series functionality."""
    print("Creating dummy data...")
    y = create_dummy_data()
    
    print("Initializing TSForecastingExperiment...")
    exp = TSForecastingExperiment()
    
    print("Setting up the experiment...")
    exp.setup(
        data=y,
        fh=30,  # Forecast horizon of 30 days
        fold=3,  # Number of cross-validation folds
        session_id=123,
        verbose=True
    )
    
    print("Available models:")
    print(exp.models())
    
    print("\nComparing models...")
    best_models = exp.compare_models(n_select=3)
    
    print("\nBest model details:")
    print(best_models[0])
    
    print("\nGenerating predictions...")
    predictions = exp.predict_model(best_models[0])
    
    print("\nGenerating trading signals...")
    signals = generate_signals(predictions)
    
    print("\nCreating plot...")
    fig = plot_results(y, predictions, signals)
    
    print("\nSaving plot...")
    fig.write_html("trading_strategy_results.html")
    
    print("\nTrading statistics:")
    print(f"Number of Buy Signals: {len(signals[signals['Signal'] == 1])}")
    print(f"Number of Sell Signals: {len(signals[signals['Signal'] == -1])}")
    
    return exp, predictions, signals

if __name__ == "__main__":
    main()
