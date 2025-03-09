# ValueInvestor - Stock Trading Strategy Application

## Overview
This repository contains a comprehensive stock trading strategy application that enables users to analyze, backtest, and forecast stock and cryptocurrency performance. The application utilizes various time series forecasting models and technical indicators to provide insights for investment decisions.

## Features

### Trading Strategies
- **Bollinger Bands Strategy**: Implementation of the popular Bollinger Bands technical indicator for identifying overbought and oversold conditions in the market.

### Forecasting Models
- **ARIMA**: Autoregressive Integrated Moving Average model for time series forecasting
- **SARIMAX**: Seasonal ARIMA with exogenous variables for more complex time series with seasonal patterns
- **Prophet**: Facebook's Prophet model for forecasting with daily seasonality and holiday effects
- **LLM-based Forecasting**: Innovative approach using Large Language Models (llama3.2, deepseek, qwen2.5) for time series predictions

### Interactive Web Interface
- Built with Streamlit for a user-friendly experience
- Customizable parameters for trading strategies
- Visualization tools for strategy performance and forecasting results
- Support for both stocks and cryptocurrencies

## Technologies Used
- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Plotly**: Data visualization
- **Statsmodels**: Statistical modeling and time series analysis
- **Prophet**: Time series forecasting library
- **Polygon.io API**: Market data provider
- **Langchain & Ollama**: Integration with LLM models

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Polygon.io API key (register at https://polygon.io/dashboard/signup)

### Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ValueInvestor.git
   cd ValueInvestor
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Polygon.io API key:
   - Option 1: Edit the `app_trading.py` file to include your API key
   - Option 2: Set an environment variable: `export POLYGON_API_KEY="your_api_key_here"`

### Running the Application
Launch the Streamlit application:
```
streamlit run app/app_trading.py
```

## Usage Guide

1. **Select Asset Type**: Choose between Stocks or Cryptocurrencies
2. **Choose Asset**: Select from top market cap assets
3. **Set Date Range**: Define the historical data period for analysis
4. **Configure Strategy Parameters**:
   - Initial investment amount
   - Rolling window for indicators
   - Enable forecast option if desired
5. **Run Backtest**: Calculate and visualize strategy performance
6. **Analyze Results**: Review performance metrics and charts

## Project Structure
- `app/app_trading.py`: Main application file with Streamlit interface and trading logic
- `notebook_Trading.ipynb`: Jupyter notebook with development process and analysis
- `requirements.txt`: List of Python dependencies
- `data/`: Directory containing sample data

## License
This project is available for educational and research purposes.

## Disclaimer
This application is for educational purposes only and should not be considered financial advice. Always conduct your own research before making investment decisions. 