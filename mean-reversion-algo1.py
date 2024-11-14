import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

# Load the historical data
file_path = '~/Quant_Fin/EURUSD_15min/EURUSD_15m_BID_01012010_to_31122016.csv'
data = pd.read_csv(file_path)
data['Time'] = pd.to_datetime(data['Time'])

# Check for missing values
missing_values = data.isnull().sum()
if missing_values.sum() > 0:
    print(f"Warning: There are {missing_values.sum()} missing values in the data.")

# Calculate the 20-period moving average and standard deviation for Bollinger Bands
data['MA20'] = data['Close'].rolling(window=20).mean()
data['STD20'] = data['Close'].rolling(window=20).std()

# Calculate upper and lower Bollinger Bands
data['Upper_Band'] = data['MA20'] + (2 * data['STD20'])
data['Lower_Band'] = data['MA20'] - (2 * data['STD20'])

# Perform Augmented Dickey-Fuller test to check for stationarity
adf_result = adfuller(data['Close'])
if adf_result[1] > 0.05:
    print("The EURUSD time series is not stationary. Mean reversion may not be appropriate.")
    
# Define the mean reversion trading strategy
def mean_reversion_trader(data, initial_capital=100000, trading_cost=0.005, risk_threshold=0.15):
    position = 0
    cash = initial_capital
    portfolio_value = initial_capital
    trade_log = []

    for i in range(20, len(data)):
        current_price = data.loc[i, 'Close']

        # Enter long position if the price closes below the lower band
        if current_price < data.loc[i, 'Lower_Band'] and position == 0:
            entry_price = current_price * (1 + trading_cost)
            position = 1
            cash -= entry_price
            trade_log.append(('Buy', data.loc[i, 'Time'], entry_price))

        # Enter short position if the price closes above the upper band
        elif current_price > data.loc[i, 'Upper_Band'] and position == 0:
            entry_price = current_price * (1 - trading_cost)
            position = -1
            cash += entry_price
            trade_log.append(('Sell', data.loc[i, 'Time'], entry_price))

        # Exit long position when the price reverts to the moving average
        elif position == 1 and current_price >= data.loc[i, 'MA20']:
            exit_price = current_price * (1 - trading_cost)
            cash += exit_price
            position = 0
            trade_log.append(('Sell', data.loc[i, 'Time'], exit_price))

        # Exit short position when the price reverts to the moving average
        elif position == -1 and current_price <= data.loc[i, 'MA20']:
            exit_price = current_price * (1 + trading_cost)
            cash -= exit_price
            position = 0
            trade_log.append(('Buy', data.loc[i, 'Time'], exit_price))

        # Calculate portfolio value (MTM)
        if position == 1:
            portfolio_value = cash + current_price
        elif position == -1:
            portfolio_value = cash - current_price
        else:
            portfolio_value = cash

        # Check MTM risk threshold
        if portfolio_value < (1 - risk_threshold) * initial_capital:
            print(f"Warning: MTM value dropped below {risk_threshold * 100}% of capital at {data.loc[i, 'Time']}")
            break

    return trade_log, portfolio_value

# Run the mean reversion trading strategy
trade_log, final_portfolio_value = mean_reversion_trader(data)

# Print the trade log and final portfolio value
print(trade_log)
print(f"Final Portfolio Value: {final_portfolio_value}")

# Implement volatility forecasting and switch to directional trading if mean reversion is unlikely
def volatility_forecasting(data, window=20):
    """
    Forecast volatility using a simple moving average approach.
    Returns a boolean indicating whether mean reversion is likely or not.
    """
    data['Volatility'] = data['Close'].rolling(window=window).std()
    volatility_threshold = data['Volatility'].mean() + 2 * data['Volatility'].std()
    
    if data['Volatility'].iloc[-1] > volatility_threshold:
        return False
    else:
        return True

is_mean_reversion_likely = volatility_forecasting(data)

if is_mean_reversion_likely:
    print("Mean reversion is likely. Executing mean reversion trading strategy.")
    trade_log, final_portfolio_value = mean_reversion_trader(data)
else:
    print("Mean reversion is unlikely. Switching to directional trading strategy.")
    # Implement directional trading strategy here
    pass

print(trade_log)
print(f"Final Portfolio Value: {final_portfolio_value}")
