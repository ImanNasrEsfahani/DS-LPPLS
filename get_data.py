import os
import re
import pandas as pd
import yfinance as yf

from settings import available_pairs

def get_data(currency, start_date, end_date, folder='data'):
    # Create the fo
    # lder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Define the filename based on currency and date range
    filename = f"{folder}/{currency.replace('-', '_')}_{start_date}_{end_date}.csv"

    # Check if the file already exists
    if os.path.isfile(filename):
        print(f"Loading data from {filename}...")
        if currency in available_pairs:
            data = pd.read_csv(filename, header=0, date_format='%Y-%m-%d', parse_dates=[2], skiprows=[1])
        else:
            data = pd.read_csv(filename, header=0, date_format='%Y-%m-%d')

        data = data.dropna()
        data['Close'] = data['Close'].astype(float)
        data['Date'] = pd.to_datetime(data['Price'])

        data.set_index('Date', inplace=True)
        print("Data Exist")

    else:
        print(f"Downloading data for {currency} from Yahoo Finance...")
        data = yf.download(currency, start=start_date, end=end_date)
        
        # Save to CSV
        data.to_csv(filename)
        data = pd.read_csv(filename, header=0, date_format='%Y-%m-%d', parse_dates=[2], skiprows=[1])
        data = data.dropna()
        data['Close'] = data['Close'].astype(float)
        data['Date'] = pd.to_datetime(data['Price'])

        data.set_index('Date', inplace=True)
        print(f"Data saved to {filename}.")

    return data


def list_files_with_prefix(path, prefix):
    # Check if the path exists
    if not os.path.exists(path):
        print(f"Error: The path '{path}' does not exist.")
        return []

    # Check if the path is a directory
    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a directory.")
        return []

    # Create a regex pattern to match files starting with the prefix followed by numbers
    pattern = re.compile(rf'^{re.escape(prefix)}_\d+\.joblib')

    # List to hold matching file names
    matching_files = []

    # Iterate over the files in the specified directory
    for filename in os.listdir(path):
        if pattern.match(filename):  # Check if the filename matches the pattern
            matching_files.append(filename)  # Add matching file to the list
    return sorted(matching_files)
