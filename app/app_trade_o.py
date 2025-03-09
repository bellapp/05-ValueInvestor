import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import time  # Add this import at the top of the file
from polygon import RESTClient
import os
from langchain_community.llms import Ollama
import io

# Add this near the top of your file
# You'll need to get an API key from https://polygon.io/
POLYGON_API_KEY = "nDUjoFrR87jrZ5RwLM3fbKs6Vn8F584Q"  # Replace with your API key or use environment variable

# Define top 10 market stocks (by market cap)
TOP_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corporation',
    'META': 'Meta Platforms Inc.',
    'BRK-B': 'Berkshire Hathaway Inc.',
    'TSM': 'Taiwan Semiconductor Manufacturing',
    'V': 'Visa Inc.',
    'JPM': 'JPMorgan Chase & Co.'
}

# Define top 10 cryptocurrencies (by market cap)
TOP_CRYPTOS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'XRP-USD': 'XRP',
    'ADA-USD': 'Cardano',
    'SOL-USD': 'Solana',
    'DOGE-USD': 'Dogecoin',
    'DOT-USD': 'Polkadot',
    'SHIB-USD': 'Shiba Inu',
    'LTC-USD': 'Litecoin'
}

### Utility functions for Bollinger Bands strategy

def generate_bollinger_bands(df, period="D", window=20):
    """
    Generate Bollinger Bands and other technical indicators from stock data.
    """
    df = df.copy()

    # Reset index if possible
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        has_date_index = True
    else:
        has_date_index = False

    close_series = df["Close"].astype(float)
    ma20 = close_series.rolling(window=window).mean()
    df["MA20"] = ma20
    rolling_std = close_series.rolling(window=window).std()
    df["Upper_Band"] = ma20 + (2 * rolling_std)
    df["Lower_Band"] = ma20 - (2 * rolling_std)

    # Ensure numeric types
    for col in ["MA20", "Upper_Band", "Lower_Band"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if has_date_index:
        df = df.set_index("Date")

    return df

def backtest_bollinger_strategy(df, initial_capital=10000):
    """
    Backtest Bollinger Bands trading strategy with an initial capital.
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        has_date_index = True
    else:
        has_date_index = False

    df['Signal'] = 'HOLD'
    df['Position'] = 0
    df['Shares'] = 0
    df['Portfolio_Value'] = 0
    df['Cash'] = initial_capital

    # Generate signals based on Close crossing above/below the bands.
    # Use .squeeze() to ensure the data is 1D.
    close_series = df['Close'].squeeze()
    lower_series = df['Lower_Band'].squeeze()
    upper_series = df['Upper_Band'].squeeze()
    cond_buy = pd.Series(close_series.to_numpy() < lower_series.to_numpy(), index=df.index)
    cond_sell = pd.Series(close_series.to_numpy() > upper_series.to_numpy(), index=df.index)
    df.loc[cond_buy, 'Signal'] = 'BUY'
    df.loc[cond_sell, 'Signal'] = 'SELL'

    looking_to_buy = True
    shares_held = 0
    cash = initial_capital

    for i in range(len(df)):
        current_price = float(df['Close'].iloc[i])
        current_signal = df['Signal'].iloc[i]

        if looking_to_buy and current_signal == 'BUY':
            shares_to_buy = cash // current_price
            if shares_to_buy > 0:
                shares_held = shares_to_buy
                cash -= shares_to_buy * current_price
                df.iloc[i, df.columns.get_loc('Position')] = 1
                df.iloc[i, df.columns.get_loc('Shares')] = shares_held
                looking_to_buy = False
        elif not looking_to_buy and current_signal == 'SELL':
            if shares_held > 0:
                cash += shares_held * current_price
                df.iloc[i, df.columns.get_loc('Position')] = -1
                df.iloc[i, df.columns.get_loc('Shares')] = -shares_held
                shares_held = 0
                looking_to_buy = True

        df.iloc[i, df.columns.get_loc('Portfolio_Value')] = cash + (shares_held * current_price)
        df.iloc[i, df.columns.get_loc('Cash')] = cash

    if has_date_index:
        df = df.set_index("Date")

    return df

def plot_strategy_results(df, company_name):
    """
    Plot backtest results including portfolio value and trade signals.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])

    ax1.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.6)
    ax1.plot(df.index, df['MA20'], label='20-day MA', color='gray', alpha=0.6)
    ax1.plot(df.index, df['Upper_Band'], label='Upper Band', color='red', linestyle='--', alpha=0.4)
    ax1.plot(df.index, df['Lower_Band'], label='Lower Band', color='green', linestyle='--', alpha=0.4)

    buy_signals = df[df['Position'] > 0]
    sell_signals = df[df['Position'] < 0]

    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell')

    ax1.set_title(f"{company_name} - Bollinger Bands Strategy Backtest")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(df.index, df['Portfolio_Value'], label='Portfolio Value', color='purple')
    ax2.fill_between(df.index, df['Portfolio_Value'], alpha=0.3, color='purple')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    return fig

### Forecasting functions

def predict_arima_model(train_df, test_df, forecast_period=15, order=(3,1,3)):
    """
    Fit an ARIMA model on the training data (Close) and forecast for (len(test_df)+forecast_period) steps.
    Returns:
      - test_pred: forecasted values corresponding to the test period
      - future_pred: forecasted values corresponding to the future (forecast period)
      - test_dates: dates from test_df index
      - future_dates: a generated date range for the forecast period
      - model_fit: the ARIMA model fit for display purposes
    """
    steps = len(test_df) + forecast_period
    model = ARIMA(train_df["Close"], order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=steps)

    test_pred = predictions.iloc[:len(test_df)]
    future_pred = predictions.iloc[len(test_df):]
    test_dates = test_df.index
    future_dates = pd.date_range(start=train_df.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B')

    # Set indices for both test and future predictions
    test_pred.index = test_dates
    future_pred.index = future_dates

    return test_pred, future_pred, test_dates, future_dates, model_fit

def predict_sarimax_model(train_df, test_df, forecast_period=15, order=(1, 1, 3)):
    """
    Fit a SARIMAX model (using Open, High, Low, Volume as exogenous predictors) on the training data and forecast
    for (len(test_df)+forecast_period) steps. Returns similar outputs as the ARIMA forecasting function.
    """
    exog_columns = ['Open', 'High', 'Low', 'Volume']

    # Use the original index for training exogenous values so they align with train_df["Close"]
    train_exog = train_df[exog_columns].astype(float)

    # Rebuild test exogenous DataFrame from its underlying data to guarantee column consistency
    test_exog = pd.DataFrame(test_df[exog_columns].values, columns=exog_columns)

    # Generate future exogenous predictors for the forecast period.
    future_exog = np.empty((forecast_period, len(exog_columns)))
    for i, col in enumerate(exog_columns):
        if col == 'Volume':
            future_exog[:, i] = np.median(train_df[col].tail(30))
        else:
            # Convert last_value to scalar float explicitly
            last_value = float(train_df[col].iloc[-1])
            historical_volatility = train_df[col].pct_change().std()
            random_walk = np.random.normal(0, historical_volatility, forecast_period)
            # Use simple percentage changes instead of cumulative sum
            future_exog[:, i] = last_value * (1 + np.cumsum(random_walk))
    
    # Keep model configuration improvements
    model = SARIMAX(train_df["Close"], exog=train_exog, order=order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                    trend='n')
    model_fit = model.fit(disp=False)

    # Forecast using the combined exogenous predictors.
    predictions = model_fit.forecast(steps=len(test_exog) + forecast_period, exog=pd.concat([test_exog, pd.DataFrame(future_exog, columns=exog_columns)]))
    test_pred = predictions.iloc[:len(test_exog)]
    future_pred = predictions.iloc[len(test_exog):]

    test_dates = test_df.index

    # IMPORTANT: future forecast dates now start from the last day in the test set (your end date)
    future_dates = pd.date_range(start=test_df.index[-1] + pd.Timedelta(days=1),
                                 periods=forecast_period, freq='B')

    # Set indices for both test and future predictions
    test_pred.index = test_dates
    future_pred.index = future_dates

    return test_pred, future_pred, test_dates, future_dates, model_fit

def predict_prophet_model(train_df, test_df, forecast_period=15):
    """
    Fit a more sophisticated Prophet model on the training data and forecast for (test + forecast) periods.
    Uses additional regressors and optimized seasonality parameters.
    """
    # Prepare data in Prophet format
    temp = train_df.reset_index()
    # Flatten hierarchical column names (if any) and standardize
    temp.columns = [col[0] if isinstance(col, tuple) else str(col).strip() for col in temp.columns]
    # Use the actual index column name if available; default to 'index'
    col_name = train_df.index.name if train_df.index.name is not None else 'index'
    
    # Determine which price column to use (check for 'Close' then 'Adj Close')
    if 'Close' in temp.columns:
        price_col = 'Close'
    elif 'Adj Close' in temp.columns:
        price_col = 'Adj Close'
    else:
        raise KeyError("No valid closing price column found in training data: " + str(temp.columns))
    
    # Create Prophet-formatted dataframes
    prophet_train = temp[[col_name, price_col]].rename(columns={col_name: 'ds', price_col: 'y'})
    prophet_train['y'] = pd.to_numeric(prophet_train['y'], errors='coerce')
    prophet_train = prophet_train.dropna(subset=['y'])
    
    # Handle empty data case
    if len(prophet_train) < 2:
        raise ValueError("Insufficient training data for Prophet model")
    
    # Initialize Prophet with improved parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10
    )
    
    # Add additional regressors if available
    additional_regressors = ['Open', 'High', 'Low', 'Volume']
    for regressor in additional_regressors:
        if regressor in temp.columns and regressor != price_col:
            # Add the regressor to Prophet
            model.add_regressor(regressor)
            # Add the regressor data to the training dataframe
            prophet_train[regressor] = pd.to_numeric(temp[regressor], errors='coerce')
    
    # Fit the model
    model.fit(prophet_train)
    
    # Create future dataframe for test period
    prophet_test = pd.DataFrame({'ds': test_df.reset_index()[col_name]})
    
    # Add regressor values for test period
    for regressor in additional_regressors:
        if regressor in model.extra_regressors:
            prophet_test[regressor] = pd.to_numeric(test_df[regressor].values, errors='coerce')
    
    # Create future dataframe for forecast period
    future_dates = pd.date_range(
        start=test_df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_period,
        freq='B'
    )
    prophet_future = pd.DataFrame({'ds': future_dates})
    
    # Generate future regressor values using historical patterns
    for regressor in additional_regressors:
        if regressor in model.extra_regressors:
            if regressor == 'Volume':
                # Use median volume for future predictions
                prophet_future[regressor] = np.median(train_df[regressor].tail(30))
            else:
                # Use price-based projection for other regressors
                last_value = float(train_df[regressor].iloc[-1])
                historical_volatility = train_df[regressor].pct_change().std()
                random_walk = np.random.normal(0, historical_volatility, forecast_period)
                prophet_future[regressor] = last_value * (1 + np.cumsum(random_walk))
    
    # Combine test and future periods for prediction
    combined_future = pd.concat([prophet_test, prophet_future])
    forecast = model.predict(combined_future)
    
    # Split forecast into test and future periods
    test_forecast = forecast.iloc[:len(test_df)]
    future_forecast = forecast.iloc[len(test_df):]
    
    # Extract predictions and confidence intervals
    test_pred = pd.Series(test_forecast['yhat'].values, index=test_df.index)
    future_pred = pd.Series(future_forecast['yhat'].values, index=future_dates)
    
    # Also store confidence intervals
    test_pred.name = 'yhat'
    future_pred.name = 'yhat'
    
    # Return in format compatible with existing code
    return test_pred, future_pred, test_df.index, future_dates, model, forecast

def predict_llm_model(train_df, test_df, forecast_period=15, model_name="llama3.2:latest", temperature=0):
    """
    Use an LLM to predict stock prices based on historical data.
    Returns predictions in the same format as other forecasting functions.
    """
    # Initialize the LLM
    try:
        llm = Ollama(model=model_name, temperature=temperature)
    except Exception as e:
        raise ValueError(f"Failed to initialize Ollama with model {model_name}: {str(e)}")
    
    # Prepare input data for the LLM - last 90 days of training data to avoid context limits
    input_data = train_df.tail(90).copy()
    
    # Format the data as a simple CSV string with date and price
    df_for_llm = input_data["Close"].reset_index()
    df_for_llm.columns = ["Date", "Price"]
    timeseries_csv = df_for_llm.to_csv(index=False)
    
    # Determine the start date for predictions (first day after training ends)
    start_predicted_date = test_df.index[0].strftime('%Y-%m-%d')
    
    # Total days to predict (test period + forecast period)
    total_days_to_predict = len(test_df) + forecast_period
    
    # Get predictions from LLM
    try:
        csv_output = predict_timeseries(
            timeseries=timeseries_csv,
            days_to_predict=total_days_to_predict,
            start_predicted_date=start_predicted_date,
            llm=llm
        )
    except Exception as e:
        st.error(f"LLM prediction failed: {str(e)}")
        # Return empty data with proper structure
        return pd.Series(), pd.Series(), test_df.index, pd.date_range(
            start=test_df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_period,
            freq='B'
        ), None, None
    
    # Parse the predictions
    try:
        predictions_df = pd.read_csv(io.StringIO(csv_output))
        predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
        predictions_df = predictions_df.set_index("Date")
        
        # Split into test and forecast predictions
        test_pred = pd.Series(predictions_df["Price"].values[:len(test_df)], index=test_df.index)
        
        # Future dates start after the last test date
        future_dates = pd.date_range(
            start=test_df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_period,
            freq='B'
        )
        
        # Extract future predictions (may be shorter than forecast_period)
        available_future_preds = predictions_df["Price"].values[len(test_df):len(test_df)+forecast_period]
        
        # Handle case where LLM didn't provide enough predictions
        if len(available_future_preds) < forecast_period:
            # Pad with NaN or extrapolate
            padded_preds = np.full(forecast_period, np.nan)
            padded_preds[:len(available_future_preds)] = available_future_preds
            future_pred = pd.Series(padded_preds, index=future_dates)
        else:
            future_pred = pd.Series(available_future_preds, index=future_dates)
        
        return test_pred, future_pred, test_df.index, future_dates, None, None
        
    except Exception as e:
        st.error(f"Failed to parse LLM predictions: {str(e)}")
        # Return empty data with proper structure
        return pd.Series(), pd.Series(), test_df.index, pd.date_range(
            start=test_df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_period,
            freq='B'
        ), None, None

def predict_timeseries(timeseries, days_to_predict, start_predicted_date, llm, temperature=0):
    """
    Predicts future stock prices based on an input time series using LLM.
    Enhanced with extremely robust parsing for different LLM output formats.
    """
    output = llm.invoke(
        f"""You are a stock price prediction assistant.
Based on this sequence of stock closing prices: {timeseries},
predict the closing prices on the next {days_to_predict} days starting from {start_predicted_date}.
Instructions:
- IMPORTANT: Your output MUST be in CSV format with no explanations
- The first line must be exactly 'Date,Price'
- Each line after must be in format: YYYY-MM-DD,NUMBER
- Ensure all dates are VALID calendar dates
- Only include {days_to_predict} days of predictions (no more)
- Start from {start_predicted_date} and continue sequentially
- DO NOT include any text, thinking, explanations, or markdown
- Your entire response must be ONLY the CSV data
"""
    )

    # Display the raw output for debugging
    st.expander("Raw LLM Output (Debug)", expanded=False).code(output, language="text")
    
    # Check if the output is already in CSV-like format without a header
    # First, clean up any thinking sections or preamble
    clean_output = []
    in_thinking_section = False
    for line in output.strip().split('\n'):
        if "<think>" in line:
            in_thinking_section = True
            continue
        if "</think>" in line:
            in_thinking_section = False
            continue
        if in_thinking_section:
            continue
        if line.strip():  # Skip empty lines
            clean_output.append(line.strip())
    
    # If first line is in format YYYY-MM-DD,NUMBER, the output is already CSV format
    # but missing the header
    first_line_is_data = False
    if clean_output and ',' in clean_output[0]:
        parts = clean_output[0].split(',')
        if len(parts) == 2 and is_date_format(parts[0]) and is_numeric(parts[1]):
            first_line_is_data = True
    
    if first_line_is_data:
        # Add header and use the clean output directly
        csv_lines = ["Date,Price"] + clean_output
        csv_output = '\n'.join(csv_lines)
        st.success("Successfully parsed LLM prediction data")
        st.expander("Processed LLM Predictions", expanded=True).code(csv_output, language="csv")
        return csv_output
    
    # If we didn't detect valid CSV format, try our regular parsing approaches
    csv_lines = ["Date,Price"]  # Start with header
    valid_data_rows = 0
    
    # Look for any lines that match YYYY-MM-DD,NUMBER pattern
    for line in clean_output:
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                date_part = parts[0].strip()
                price_part = parts[1].strip()
                
                # Validate date format and check if it's a valid date
                if is_date_format(date_part):
                    try:
                        # Try to parse the date to validate it
                        parsed_date = datetime.strptime(date_part, '%Y-%m-%d')
                        # Check if the price is a valid number
                        if is_numeric(price_part):
                            csv_lines.append(f"{date_part},{price_part}")
                            valid_data_rows += 1
                    except ValueError:
                        # Skip invalid dates
                        continue
    
    # If we found valid data rows, use them
    if valid_data_rows > 0:
        csv_output = '\n'.join(csv_lines)
        st.success(f"Extracted {valid_data_rows} valid predictions from LLM output")
        st.expander("Processed LLM Predictions", expanded=True).code(csv_output, language="csv")
        return csv_output
    
    # If we still don't have valid data, generate fallback values
    st.warning("Could not extract valid CSV data from LLM output. Using generated values.")
    
    # Generate business days using pandas for fallback
    import pandas as pd
    base_date = datetime.strptime(start_predicted_date, '%Y-%m-%d')
    business_days = pd.date_range(
        start=base_date,
        periods=days_to_predict,
        freq='B'  # Business day frequency
    )
    
    # Get last price from input data for baseline
    last_price = float(timeseries.split('\n')[-1].split(',')[1]) if ',' in timeseries.split('\n')[-1] else 100.0
    
    # Generate dates with small random changes in price (trending upward)
    import random
    current_price = last_price
    for date in business_days:
        # Small random change (-1% to +1.5%)
        change = random.uniform(-0.01, 0.015) * current_price
        current_price += change
        csv_lines.append(f"{date.strftime('%Y-%m-%d')},{current_price:.2f}")
    
    # Ensure we only have the requested number of predictions
    if len(csv_lines) > days_to_predict + 1:  # +1 for header
        csv_lines = csv_lines[:days_to_predict + 1]
    
    # Join to a CSV string
    csv_output = '\n'.join(csv_lines)
    st.expander("Fallback Predictions", expanded=True).code(csv_output, language="csv")
    
    return csv_output

def is_date_format(text):
    """Check if a string looks like a date in format YYYY-MM-DD"""
    import re
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', text))

def is_numeric(text):
    """Check if a string can be converted to a number"""
    try:
        float(text)
        return True
    except ValueError:
        return False

### Main Streamlit App

def main():
    st.title("Stock Trading Strategy Backtester")

    # Sidebar selections
    st.sidebar.header("Trading Parameters")
    asset_type = st.sidebar.radio("Select Asset Type", options=["Stocks", "Crypto"])
    asset_dict = TOP_STOCKS if asset_type == "Stocks" else TOP_CRYPTOS

    selected_asset = st.sidebar.selectbox(
        "Select Asset",
        options=list(asset_dict.keys()),
        format_func=lambda x: f"{x} - {asset_dict[x]}"
    )

    today = datetime.now()
    default_start = today - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=today)

    use_forecast = st.sidebar.checkbox("Apply Forecast", value=False)
    # When forecast is enabled, allow choosing model(s) and setting test/forecast period lengths.
    if use_forecast:
        forecast_model = st.sidebar.selectbox(
            "Forecast Model",
            options=["ARIMA", "SARIMAX", "Prophet", "LLM"]
        )
        
        # Move LLM model selection here, right after forecast model choice
        # Only show if LLM is selected
        if forecast_model == "LLM":
            llm_model = st.sidebar.selectbox(
                "LLM Model",
                options=["llama3.2:latest", "deepseek-r1:8b", "qwen2.5:3b", "deepseek-r1:1.5b"],
                index=0
            )
        
        test_period = st.sidebar.number_input("Test Period (days)", min_value=1, value=30, step=1)
        forecast_period = st.sidebar.number_input("Forecast Period (days)", min_value=1, value=15, step=1)
        
        st.sidebar.info(f"Training: from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, Testing: last {test_period} days, Forecast: next {forecast_period} days")

    initial_investment = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=1000,
        value=10000,
        step=1000
    )

    rolling_window = st.sidebar.number_input(
        "Rolling Window (days)",
        min_value=5,
        value=20,
        step=1
    )

    strategy = st.sidebar.selectbox(
        "Select Strategy",
        options=["Bollinger Bands"],
        disabled=True
    )

    if st.sidebar.button("Run Backtest"):
        # Download data from user-specified start_date to end_date (or extended_start_date if available)
        max_retries = 3
        retry_delay = 10  # seconds
        df = None
        
        # Get API key from environment or use default
        api_key = os.environ.get('POLYGON_API_KEY', POLYGON_API_KEY)
        
        # Check if API key is provided
        if not api_key or api_key == "YOUR_POLYGON_API_KEY":
            st.error("Please provide a Polygon.io API key.")
            st.info("Get an API key at: https://polygon.io/dashboard/signup")
            return
        
        # Initialize Polygon client
        client = RESTClient(api_key)
        
        # Always use the user-selected start_date
        actual_start_date = start_date
        
        for attempt in range(max_retries):
            try:
                with st.spinner(f"Downloading data (attempt {attempt + 1}/{max_retries})..."):
                    # Format dates for Polygon API
                    start_date_str = actual_start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    
                    # For stocks
                    if asset_type == "Stocks":
                        # Get daily data
                        aggs = client.get_aggs(
                            ticker=selected_asset,
                            multiplier=1,
                            timespan="day",
                            from_=start_date_str,
                            to=end_date_str
                        )
                        
                        # Convert to DataFrame
                        data = pd.DataFrame([{
                            'Date': pd.to_datetime(item.timestamp, unit='ms'),
                            'Open': item.open,
                            'High': item.high,
                            'Low': item.low,
                            'Close': item.close,
                            'Volume': item.volume
                        } for item in aggs])
                        
                    # For crypto
                    else:
                        # Get crypto ticker in Polygon format (remove -USD suffix)
                        crypto_ticker = 'X:' + selected_asset.split('-')[0] + 'USD'
                        
                        # Get daily data
                        aggs = client.get_aggs(
                            ticker=crypto_ticker,
                            multiplier=1,
                            timespan="day",
                            from_=start_date_str,
                            to=end_date_str
                        )
                        
                        # Convert to DataFrame
                        data = pd.DataFrame([{
                            'Date': pd.to_datetime(item.timestamp, unit='ms'),
                            'Open': item.open,
                            'High': item.high,
                            'Low': item.low,
                            'Close': item.close,
                            'Volume': item.volume
                        } for item in aggs])
                    
                    # Set index to Date
                    if not data.empty:
                        data = data.set_index('Date')
                        df = data.sort_index()
                    
                    # If df is empty, data wasn't found in the range
                    if df is None or df.empty:
                        raise ValueError(f"No data available for {selected_asset} in the selected date range")
                    
                    break
            except Exception as e:
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        if df is None or df.empty:
            st.error("Failed to download data after multiple attempts. Please try again later.")
            return

        # If forecasting is enabled, perform a train/test split.
        if use_forecast:
            if len(df) < test_period + 10:
                st.error("Not enough data for the specified test period.")
                return
            # Split the data: training uses all rows except the last 'test_period' rows.
            training_df = df.iloc[:-test_period]
            test_df = df.iloc[-test_period:]

            st.subheader("Forecast Model Evaluation")

            # Change the conditional blocks for each model
            if forecast_model == "ARIMA":
                (arima_test_pred, arima_future_pred, arima_test_dates,
                arima_future_dates, arima_model_fit) = predict_arima_model(training_df, test_df, forecast_period=forecast_period)

                # Set indices for both test and future predictions
                arima_test_pred.index = arima_test_dates
                arima_future_pred.index = arima_future_dates

                # Ensure the arrays are 1D if needed
                actual_arr = np.asarray(test_df["Close"]).ravel()
                arima_pred_arr = np.asarray(arima_test_pred).ravel()

                mape_arima = np.mean(np.abs((actual_arr - arima_pred_arr) / actual_arr)) * 100
                rmse_arima = np.sqrt(np.mean((actual_arr - arima_pred_arr) ** 2))
                corr_arima = (np.corrcoef(actual_arr, arima_pred_arr)[0, 1] if len(actual_arr) > 1 else np.nan)

                st.markdown("#### ARIMA Model")
                st.text(arima_model_fit.summary())
                st.write(f"MAPE: {mape_arima:.2f}%, RMSE: ${rmse_arima:.2f}, Correlation: {corr_arima:.2f}")

                # Combine test predictions and future forecast into one continuous series
                combined_pred = pd.concat([arima_test_pred, arima_future_pred])

                # Create the combined plot
                fig_combined, ax_combined = plt.subplots(figsize=(10, 6))

                # Plot actual prices over the test period for context
                ax_combined.plot(test_df.index, test_df["Close"], label="Actual Price", color="blue", alpha=0.5)

                # Plot the predictions over the test period and the future forecast together
                ax_combined.plot(combined_pred.index, combined_pred, label="Predictions + Forecast",
                                marker="o", linestyle="-", color="green")
                ax_combined.set_title("ARIMA: Combined Test Predictions and Future Forecast")
                ax_combined.set_xlabel("Date")
                ax_combined.set_ylabel("Close Price")
                ax_combined.legend()
                ax_combined.grid(True)
                st.pyplot(fig_combined)
            

            if forecast_model == "SARIMAX":
                # Obtain ARIMAX predictions and forecast using your custom function.
                (arimax_test_pred, arimax_future_pred, arimax_test_dates, arimax_future_dates,
                  arimax_model_fit) = predict_sarimax_model(training_df, test_df, forecast_period=forecast_period)

                # Calculate evaluation metrics on the test period predictions.
                actual_arr = np.asarray(test_df["Close"]).ravel()
                arimax_pred_arr = np.asarray(arimax_test_pred).ravel()

                mape_arimax = np.mean(np.abs((actual_arr - arimax_pred_arr) / actual_arr)) * 100
                rmse_arimax = np.sqrt(np.mean((actual_arr - arimax_pred_arr) ** 2))
                corr_arimax = (np.corrcoef(actual_arr, arimax_pred_arr)[0,1] if len(actual_arr) > 1 else np.nan)

                st.markdown("#### ARIMAX Model")
                st.text(arimax_model_fit.summary())
                st.write(f"MAPE: {mape_arimax:.2f}%, RMSE: ${rmse_arimax:.2f}, Correlation: {corr_arimax:.2f}")

                # Assign the proper index (dates) to the future predictions.
                arimax_future_pred.index = arimax_future_dates

                # Combine test predictions and future forecasts into one continuous series.
                arimax_combined_pred = pd.concat([arimax_test_pred, arimax_future_pred])

                # Create a combined plot.
                fig_arimax, ax_arimax = plt.subplots(figsize=(10, 6))

                # Plot the actual closing prices (for context) over the test period.
                ax_arimax.plot(test_df.index, test_df["Close"], label="Actual", color="blue", alpha=0.5)

                # Plot the combined ARIMAX predictions (test + forecast).
                ax_arimax.plot(arimax_combined_pred.index, arimax_combined_pred,
                            label="ARIMAX Prediction + Forecast", marker="o", linestyle="-", color="purple")
                ax_arimax.set_title("ARIMAX: Combined Test Predictions and Future Forecast")
                ax_arimax.set_xlabel("Date")
                ax_arimax.set_ylabel("Close Price")
                ax_arimax.legend()
                ax_arimax.grid(True)

                st.pyplot(fig_arimax)

            # For Prophet, use the training data only.
            df_for_backtest = training_df.copy()

            if forecast_model == "Prophet":
                (prophet_test_pred, prophet_future_pred, 
                 prophet_test_dates, prophet_future_dates, 
                 prophet_model, prophet_forecast) = predict_prophet_model(training_df, test_df, forecast_period)
                
                # Calculate metrics
                actual_arr = np.asarray(test_df["Close"]).ravel()
                prophet_pred_arr = np.asarray(prophet_test_pred).ravel()
                
                mape_prophet = np.mean(np.abs((actual_arr - prophet_pred_arr) / actual_arr)) * 100
                rmse_prophet = np.sqrt(np.mean((actual_arr - prophet_pred_arr) ** 2))
                mae_prophet = np.mean(np.abs(actual_arr - prophet_pred_arr))
                corr_prophet = np.corrcoef(actual_arr, prophet_pred_arr)[0, 1] if len(actual_arr) > 1 else np.nan
                
                # Calculate R2 score if scikit-learn is available
                try:
                    from sklearn.metrics import r2_score
                    r2_prophet = r2_score(actual_arr, prophet_pred_arr)
                    r2_display = f", R²: {r2_prophet:.4f}"
                except ImportError:
                    r2_display = ""
                
                st.markdown("#### Prophet Model")
                st.write(f"MAPE: {mape_prophet:.2f}%, RMSE: ${rmse_prophet:.2f}, MAE: ${mae_prophet:.2f}, Correlation: {corr_prophet:.2f}{r2_display}")
                
                # Plot components
                fig_components = prophet_model.plot_components(prophet_forecast)
                st.pyplot(fig_components)
                
                # Enhanced combined plot with confidence intervals
                fig_prophet, ax_prophet = plt.subplots(figsize=(12, 6))
                
                # Plot actual values
                ax_prophet.plot(test_df.index, test_df["Close"], label="Actual", color="blue", alpha=0.7)
                
                # Plot training data if available
                if len(training_df) > 0:
                    ax_prophet.plot(training_df.index[-30:], training_df["Close"].iloc[-30:], 
                                    label="Training Data (last 30 days)", color="gray", alpha=0.5)
                
                # Plot test predictions
                ax_prophet.plot(prophet_test_dates, prophet_test_pred, 
                               label="Test Predictions", color="green", linestyle="--")
                
                # Plot future predictions
                ax_prophet.plot(prophet_future_dates, prophet_future_pred, 
                               label="Future Forecast", color="orange", marker="o")
                
                # Add confidence intervals if available
                if 'yhat_lower' in prophet_forecast.columns and 'yhat_upper' in prophet_forecast.columns:
                    test_lower = prophet_forecast['yhat_lower'].iloc[:len(test_df)].values
                    test_upper = prophet_forecast['yhat_upper'].iloc[:len(test_df)].values
                    future_lower = prophet_forecast['yhat_lower'].iloc[len(test_df):].values
                    future_upper = prophet_forecast['yhat_upper'].iloc[len(test_df):].values
                    
                    # Test period confidence interval
                    ax_prophet.fill_between(prophet_test_dates, test_lower, test_upper,
                                           color="green", alpha=0.2, label="95% Confidence Interval (Test)")
                    
                    # Future period confidence interval
                    ax_prophet.fill_between(prophet_future_dates, future_lower, future_upper,
                                           color="orange", alpha=0.2, label="95% Confidence Interval (Forecast)")
                
                ax_prophet.set_title("Prophet: Enhanced Model Forecast")
                ax_prophet.set_xlabel("Date")
                ax_prophet.set_ylabel("Price")
                ax_prophet.legend(loc="best")
                ax_prophet.grid(True)
                st.pyplot(fig_prophet)

            if forecast_model == "LLM":
                with st.spinner("Running LLM forecasting (this may take a minute)..."):
                    (llm_test_pred, llm_future_pred, 
                     llm_test_dates, llm_future_dates, _, _) = predict_llm_model(
                         training_df, test_df, forecast_period, model_name=llm_model
                    )
                
                # Calculate metrics
                if len(llm_test_pred) > 0:
                    actual_arr = np.asarray(test_df["Close"]).ravel()
                    llm_pred_arr = np.asarray(llm_test_pred).ravel()
                    
                    mape_llm = np.mean(np.abs((actual_arr - llm_pred_arr) / actual_arr)) * 100
                    rmse_llm = np.sqrt(np.mean((actual_arr - llm_pred_arr) ** 2))
                    mae_llm = np.mean(np.abs(actual_arr - llm_pred_arr))
                    corr_llm = np.corrcoef(actual_arr, llm_pred_arr)[0, 1] if len(actual_arr) > 1 else np.nan
                    
                    # Calculate R2 score if scikit-learn is available
                    try:
                        from sklearn.metrics import r2_score
                        r2_llm = r2_score(actual_arr, llm_pred_arr)
                        r2_display = f", R²: {r2_llm:.4f}"
                    except ImportError:
                        r2_display = ""
                    
                    st.markdown("#### LLM Forecast Model")
                    st.write(f"MAPE: {mape_llm:.2f}%, RMSE: ${rmse_llm:.2f}, MAE: ${mae_llm:.2f}, Correlation: {corr_llm:.2f}{r2_display}")
                    
                    # Create plot
                    fig_llm, ax_llm = plt.subplots(figsize=(12, 6))
                    
                    # Plot actual values
                    ax_llm.plot(test_df.index, test_df["Close"], label="Actual", color="blue", alpha=0.7)
                    
                    # Plot training data if available
                    if len(training_df) > 0:
                        ax_llm.plot(training_df.index[-30:], training_df["Close"].iloc[-30:], 
                                    label="Training Data (last 30 days)", color="gray", alpha=0.5)
                    
                    # Plot test predictions
                    ax_llm.plot(llm_test_dates, llm_test_pred, 
                               label="LLM Test Predictions", color="green", linestyle="--")
                    
                    # Plot future predictions
                    ax_llm.plot(llm_future_dates, llm_future_pred, 
                               label="LLM Future Forecast", color="orange", marker="o")
                    
                    ax_llm.set_title(f"LLM Forecast using {llm_model}")
                    ax_llm.set_xlabel("Date")
                    ax_llm.set_ylabel("Price")
                    ax_llm.legend(loc="best")
                    ax_llm.grid(True)
                    st.pyplot(fig_llm)
                    
                    # Add a note about LLM forecasting
                    st.info("LLM forecasting uses language models to predict trends based on historical patterns. Results may vary based on model and market conditions.")
        else:
            df_for_backtest = df.copy()

        # Generate Bollinger Bands and run the Bollinger-based backtest.
        df_for_backtest = generate_bollinger_bands(df_for_backtest, window=rolling_window)
        with st.spinner("Running backtest..."):
            df_backtest = backtest_bollinger_strategy(df_for_backtest, initial_investment)

        st.header("Backtest Results")
        total_return = ((df_backtest["Portfolio_Value"].iloc[-1] - initial_investment) /
                        initial_investment * 100)
        buy_hold_return = ((float(df_backtest["Close"].iloc[-1]) - float(df_for_backtest["Close"].iloc[0])) /
                           float(df_for_backtest["Close"].iloc[0])) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Portfolio Value", f"${df_backtest['Portfolio_Value'].iloc[-1]:,.2f}")
        with col2:
            st.metric("Strategy Return", f"{total_return:.2f}%")
        with col3:
            st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")

        trades = df_backtest[df_backtest["Position"] != 0]
        st.subheader("Trading Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            win_rate = (len(trades[trades["Portfolio_Value"] > trades["Portfolio_Value"].shift(1)]) / len(trades))*100 if len(trades) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.2f}%")

        st.pyplot(plot_strategy_results(df_backtest, asset_dict[selected_asset]))

if __name__ == "__main__":
    main()