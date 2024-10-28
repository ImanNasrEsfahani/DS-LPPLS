# Import necessary libraries
import os
from bidi import get_display
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import arabic_reshaper
from scipy.optimize import minimize

# Set global variables
max_window_size = 400
Min_window_size = 20
ssw = 1

# Define a helper function for Arabic text display
def _(text):
    return get_display(arabic_reshaper.reshape(u'%s' % str(text)))

# Function to check if the CSV file exists and download if not
def get_data(currency, start_date, end_date, folder='data'):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Define the filename based on currency and date range
    filename = f"{folder}/{currency.replace('-', '_')}_{start_date}_{end_date}.csv"

    # Check if the file already exists
    if os.path.isfile(filename):
        print(f"Loading data from {filename}...")
        data = pd.read_csv(filename, header=0, parse_dates=[2], skiprows=[1])
        data = data.dropna()
        data['Close'] = data['Close'].astype(float)
        data['Date'] = pd.to_datetime(data['Price'])
        data.set_index('Date', inplace=True)
        print(data.head())
    else:
        print(f"Downloading data for {currency} from Yahoo Finance...")
        data = yf.download(currency, start=start_date, end=end_date)
        data.to_csv(filename)
        print(f"Data saved to {filename}.")

    # Calculate daily returns
    data['Return'] = data['Close'].pct_change()
    
    return data

# Define the LPPLS function (based on source equation 9)
def lppls(t, tc, m, w, a, b, c1, c2):
    dt = np.abs(tc - t) + 1e-8
    return a + np.power(dt, m) * (
        b + ((c1 * np.cos(w * np.log(dt))) + (c2 * np.sin(w * np.log(dt))))
    )

# Define the objective function (sum of squared residuals)
def objective_function(params, t, log_prices):
    tc, m, w, A, B, C1, C2 = params
    model_prices = lppls(t, tc, m, w, A, B, C1, C2)  
    residuals = log_prices - model_prices
    return np.sum(residuals**2)

# Function to fit LPPLS model
def fit_lppls(data, t):
    """
    Fits the LPPLS model to a given time series of log prices.

    Args:
        data: values.
        t: interval time series

    Returns:
        tuple: Tuple containing the optimized parameters and the sum of squared errors (SSE).
    """
    # Define initial guess for optimization
    initial_guess = [t[-1], 0.5, 10, np.mean(data["Close"]), -1, 1, 1] 

    # Define bounds for optimization
    bounds = [(t[0], t[-1]), (0.01, 0.99), (2, 25), (None, None), (None, None), (None, None), (None, None)]
    
    # Optimize LPPLS parameters
    result = minimize(objective_function, 
                      initial_guess,
                      args=(t, data["log_prices"]),
                      bounds=bounds, 
                      method='L-BFGS-B')
    
    return result.x, result.fun

# Function to perform rolling window LPPLS
def rolling_window_lppls(data, max_window=400, min_window=120, ssw=1):
    """
    Performs DS-LPPLS model calibration using rolling windows.

    Args:
      data: prices of the asset.
      max_window: Maximum size of the rolling window (in data points).
      min_window: Minimum size of the rolling window (in data points).
      ssw: Step shrinking window (reduction in window size after each iteration).

    Returns:
      A list of dictionaries, where each dictionary contains the results
      (optimized parameters and goodness-of-fit metrics) for a specific rolling window.
    """
    data["log_prices"] = np.log(data["Close"]) 
    print(data, " data")

    results = []
    n = len(data)
    print(n, " n records in data")

    for i in range(max_window, n):
        print(i, " I : in for 1")

        for window_size in range(max_window, min_window - 1, -ssw):
            start = i - window_size
            end = i

            t = np.arange(start, end)
            params, sse = fit_lppls(data[start:end], t)

            results.append({
               'start': start,
                'end': end,
                'window_size': window_size,
                'params': params,
               'sse': sse
            })

    print(results, " result of rolling_window_lppls")
    return results

# Function to calculate confidence indicators
def confidence_indicator(results, data, window):
    print("confident_indicator")
    print()

    for result in results:
        params = result['params']  # Get the fitted LPPLS parameters
        print(params, " params params params params params")

        # Filtering Condition 1 from Table 1 (Source [1])
        condition_1_met = (
            0.01 <= params['beta'] <= 0.99 and  # Beta range [0.01, 0.99]
            2 <= params['omega'] <= 25 and    # Omega range [2, 3]
            result['start'] <= params['tc'] <= result['end']  # tc within window
        )

        # Filtering Condition 2 from Table 1 (Source [1]) 
        number_oscillations = (params['omega'] / 2) * np.log(
            abs(params['tc'] - result['start']) / abs(result['end'] - result['start'])
        )

        damping = (params['beta'] * abs(params['B'])) / (params['omega'] * abs(params['C']))

        condition_2_met = (
            params['beta'] > 0.01 and 
            params['omega'] > 2 and
            params['tc'] > result['start'] and params['tc'] < result['end'] and
            number_oscillations >= 2.5 and     # From Table 1
            damping >= 1                      # From Table 1
        )

        result['filter_condition_1'] = condition_1_met
        result['filter_condition_2'] = condition_2_met

    filtered_results_1 = [result for result in results if result['filter_condition_1']]
    filtered_results_2 = [result for result in results if result['filter_condition_2']]

    plt.figure(figsize=(12, 6)) 
    plt.plot(data.index, data['Price'], label='Price Data') 

    for result in filtered_results_1:
        start, end = result['start'], result['end']
        params = result['params']
        print(params, " params params params params params")
        time_window = data.index[start:end] 

        # Calculate c1 and c2
        c1 = C * np.cos(phi)
        c2 = C * np.sin(phi)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Filtered LPPLS Fits Overlaid on Price Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    return 0

# Function to calculate double confidence indicator
def double_confidence_indicator(data, window):
    rolling_window = data['Return'].rolling(window=window)
    mean = rolling_window.mean()
    std = rolling_window.std()
    double_confidence = (mean / std) * np.sqrt(window) * (1 - (data['Return'] / data['Return'].max()))
    return double_confidence

# Function for second step analysis
def second_step(data):
    print("second_step")

    # Define parameters and constraints
    params_bounds = {
        'β': [0, 2],
        'ω': [1, 50],
        'tc': [None, None],  # To be set later
    }

    # Filter conditions
    filter_conditions = {
        'Condition 1': {
            'β': [0.01, 1.2],
            'ω': [2, 25],
            'tc': [None, None],  # To be set later
        },
        'Condition 2': {
            'β': [0.01, 0.99],
            'ω': [2, 25],
            'tc': [None, None],  # To be set later
        }
    }

    filtered_data = data['Close'].dropna()  # Remove NaN values

    t = np.arange(len(filtered_data))
    p = filtered_data.values

    # Calculate daily returns
    daily_returns = filtered_data.pct_change()

    # Sample data for oscillation and damping calculations
    time = np.linspace(0, 10, 100)
    amplitude = np.exp(-0.1 * time) * np.sin(2 * np.pi * time)
    
    # 1. Count oscillations
    def count_oscillations(data):
        peaks = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1
        return len(peaks)

    num_oscillations = count_oscillations(amplitude)

    # 2. Damping
    A0 = amplitude[0]  # Initial amplitude
    A = amplitude[-1]  # Final amplitude
    damping = A0 / A if A != 0 else 0

    # 3. Relative error
    true_value = 1.0  # True value
    estimated_value = 0.9  # Estimated value
    relative_error = abs(true_value - estimated_value) / abs(true_value) * 100

    # Display results
    print(f"Number of oscillations: {num_oscillations}")
    print(f"Damping: {damping}")
    print(f"Relative error: {relative_error:.2f}%")

    # Plot the graph
    plt.plot(time, amplitude)
    plt.title('Oscillation Graph')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

    return 0

# Function to plot various charts
def plot_charts(results, data):
    print("plot_charts")
    print()

    while True:
        print("\nSelect the chart you want to display:")
        print("1. Daily Return Chart")
        print("2. Confidence Indicator Chart")
        print("3. Double Confidence Indicator Chart")
        print("4. DS-LPPLS Chart")
        print("5. Crash Lock-in Chart")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            plt.figure(figsize=(12, 6))
            plt.plot(data['Return'])
            plt.title(_('Daily Return ') + currency)
            plt.xlabel(_('Date'))
            plt.ylabel(_('Return'))
            plt.show()

        elif choice == '2':
            confidence_indicator(results, data, 200)
            plt.figure(figsize=(12, 6))
            plt.plot(data['Confidence'])
            plt.title(_('Confidence Indicator ') + currency)
            plt.xlabel(_('Date'))
            plt.ylabel(_('Confidence Indicator'))
            plt.show()

        elif choice == '3':
            plt.figure(figsize=(12, 6))
            plt.plot(data['Double Confidence'])
            plt.title(_('Double Confidence Indicator ') + currency)
            plt.xlabel(_('Date'))
            plt.ylabel(_('Double Confidence Indicator'))
            plt.show()

        elif choice == '4':
            plt.figure(figsize=(12, 6))
            plt.plot(data['DS-LPPLS'])
            plt.title('DS-LPPLS ' + currency)
            plt.xlabel(_('Date'))
            plt.ylabel('DS-LPPLS')
            plt.show()

        elif choice == '5':
            # Implement Crash Lock-in Chart
            pass

        elif choice == '6':
            break

        else:
            print("Invalid choice. Please try again.")

# Main execution
if __name__ == "__main__":
    # Add your main execution code here
    pass
