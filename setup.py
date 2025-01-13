import os
import pandas as pd
from get_data import get_data
from joblib import Parallel, delayed, dump, load
from run_model import resume_processing_model, rolling_window_lppls
from filter_model import running_filters, DS_LPPLS_Confidence, DS_LPPLS_Trust
from show_charts import confident_trust_chart, critical_time_chart, lppl_simulate_chart
from get_data import list_files_with_prefix
from settings import available_pairs

def get_user_input():
    print("What do you want to do?")
    print("1. Choose a currency pair or stock index")

    # Get currency selection
    currency = select_currency()

    # Check if the user selected 'Other' option
    if currency == "Other":
        custom_filename = input("Please enter your custom filename: ")
        print(f"You have chosen to use the custom filename: {custom_filename}")
        currency = custom_filename

    # Get date range
    start_date = input("Enter the start date (YYYY-MM-DD, default 2015-01-01): ") or "2015-01-01"
    # end_date = input("Enter the end date (YYYY-MM-DD, default today): ") or "2024-11-10"
    end_date = input("Enter the end date (YYYY-MM-DD, default today): ") or pd.to_datetime("today").strftime("%Y-%m-%d")

    # Ask for operation
    print("\nWhat do you want to do?")
    if currency in available_pairs:
        print("1. Download data file")
    print("2. Run or continue a model")
    print("3. Apply filters")
    print("4. Display data or chart")

    # Initialize operation_type
    operation_type = ""

    # Loop until a valid input is received
    while operation_type not in {'1', '2', '3', '4'}:
        operation_type = input("Select an operation number (1-4): ").strip()

        if operation_type not in {'1', '2', '3', '4'}:
            print("Invalid option. Please choose 1, 2, 3, or 4.")

    # Execute the corresponding function based on the user's choice
    if operation_type == '1' and currency in available_pairs:
        get_data(currency, start_date, end_date)
    elif operation_type == '2':
        handle_model_execution(currency, start_date, end_date)
    elif operation_type == '3':
        handle_filter_application(currency, start_date, end_date)
    elif operation_type == '4':
        handle_display_chart(currency, start_date, end_date)

def select_currency():
    print("Available currency pairs:")
    for index, pair in enumerate(available_pairs):
        print(f"{index + 1}. {pair}")

    # Get user selection
    while True:
        try:
            choice = int(input("Select a currency pair by number (or choose 'Other' to specify a filename): "))
            if 1 <= choice <= len(available_pairs):
                return available_pairs[choice - 1]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a valid number.")

# def select_currency():
#     print("\nAvailable currency pairs:")
#     for index, pair in enumerate(available_pairs):
#         print(f"{index + 1}. {pair}")
#     while True:
#         try:
#             choice = int(input("Select the currency by number (default 1 for BTC-USD): ") or "1")
#             if 1 <= choice <= len(available_pairs):
#                 return available_pairs[choice - 1]
#             else:
#                 print(f"Please select a number between 1 and {len(available_pairs)}.")
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")


def handle_model_execution(currency: str, start_date: str, end_date: str):
    time_scale = int(input("What was the previous max window size? (default 400): ") or 400)
    min_window_size = int(input("What was the previous min window size? (default 120): ") or 120)
    ssw = int(input("What was the ssw size? (default 1): ") or 1)
    max_win_tc = float(input("What bounds for the critical time will be limited? (default 0.05): ") or 0.05)
    print()
    print("Getting All scopes of the parameters")
    min_beta = float(input("What is your minimum of the Beta parameter for scope of searching? (default 0): ") or 0)
    max_beta = float(input("What is your maximum of the Beta parameter for scope of searching? (default 2): ") or 2)
    min_omet = float(
        input("What is your minimum of the Omega (Omet) parameter for scope of searching? (default 1): ") or 1)
    max_omet = float(input("What is your maximum of the Omega (Omet) for scope of searching? (default 50): ") or 50)
    print()

    # model_folder = f"models/{currency.replace('-', '_')}_{start_date}_{end_date}"
    filename = f"{currency.replace('-', '_')}_{start_date}_{end_date}"
    path = f"./models/{currency}_{start_date}_{end_date}/"

    data = get_data(currency, start_date, end_date)

    if len(list_files_with_prefix(path=path, prefix=filename)) > 0:
        while True:  # Start an infinite loop
            continue_model = input(f"Model for {currency} found. Continue? (y/n): ").strip().lower()

            if continue_model in ['y', 'n']:  # Check if the input is either 'y' or 'n'
                break  # Valid input, exit the loop
            else:
                print("Invalid input. Please enter 'y' for Yes or 'n' for No.")  # Prompt for valid input

        if continue_model == 'y':
            # Continue the model here with retrieved parameters
            print("resume")
            resume_processing_model(data=data, time_scale=time_scale, min_window=min_window_size,
                                    max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta, min_omet=min_omet,
                                    max_omet=max_omet, step_shrinking_window=ssw, path=path, filename=filename)
        else:
            # Start a new model with these parameters
            rolling_window_lppls(data=data, time_scale=time_scale, min_window=min_window_size,
                                 max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta, min_omet=min_omet,
                                 max_omet=max_omet, step_shrinking_window=ssw, path=path, filename=filename)
    else:
        print("No previous model found. Starting new model.")
        if not os.path.exists(path):
            os.makedirs(path)
        rolling_window_lppls(data=data, time_scale=time_scale, min_window=min_window_size,
                             max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta, min_omet=min_omet,
                             max_omet=max_omet, step_shrinking_window=ssw, path=path, filename=filename)
        # Start a new model


def handle_filter_application(currency: str, start_date: str, end_date: str):
    filename = f"{currency.replace('-', '_')}_{start_date}_{end_date}"
    path = f"./models/{currency}_{start_date}_{end_date}/"

    # filename = f"./models/{currency}_{start_date}_{end_date}/{currency.replace('-', '_')}_{start_date}_{end_date}.sav"
    # filtered_windows_file = f"./models/{currency}_{start_date}_{end_date}/{currency.replace('-', '_')}_{start_date}_{end_date}_filtered.sav"

    max_window_size = int(input("What was the previous max window size? (default 400): ") or 400)
    min_window_size = int(input("What was the previous min window size? (default 120): ") or 120)
    ssw = int(input("What was the ssw size? (default 1): ") or 1)
    boot_rep = int(input("How many reputation will be done? (default 100): ") or 100)
    clust_number = int(input("How many thread will calculate parallel? (default -1): ") or -1)

    print()
    print("Getting All scopes of the parameters")
    min_beta = float(
        input("What is your minimum of the Beta parameter for scope of filtering? (default 0.01): ") or 0.01)
    max_beta = float(
        input("What is your maximum of the Beta parameter for scope of filtering? (default 0.99): ") or 0.99)
    min_omet = float(
        input("What is your minimum of the Omega (Omet) parameter for scope of filtering? (default 2): ") or 2)
    max_omet = float(input("What is your maximum of the Omega (Omet) for scope of filtering? (default 25): ") or 25)
    print()

    if len(list_files_with_prefix(path=path, prefix=filename)) == 0:
        print("Model file doesn't exist")
        print("Please first create model file")
        handle_model_execution(currency, start_date, end_date)
    else:
        running_filters(max_window_size=max_window_size, min_window_size=min_window_size, ssw=ssw, boot_rep=boot_rep,
                        clust_number=clust_number,
                        min_beta=min_beta, max_beta=max_beta, min_omet=min_omet, max_omet=max_omet, path=path,
                        filename=filename)
        print("Now You can plot your filtered data")


def handle_display_chart(currency: str, start_date: str, end_date: str):
    # Check required files
    # model_windows_file = f"./models/{currency}_{start_date}_{end_date}/{currency.replace('-', '_')}_{start_date}_{end_date}.sav"
    filename = f"{currency.replace('-', '_')}_{start_date}_{end_date}"
    path = f"./models/{currency}_{start_date}_{end_date}/"
    filtered_windows_file = f"./models/{currency}_{start_date}_{end_date}/{currency.replace('-', '_')}_{start_date}_{end_date}_filtered.csv"

    chart_type = ""
    show_bubbles = 0

    # Loop for chart type selection
    while chart_type not in {'1', '2', '3'}:
        print("Choose a chart:")
        print("1. Confident and Trust chart")
        print("2. Stimulate Price chart")
        print("3. Critical Time chart")
        chart_type = input("Select chart type (1, 2 or 3): ").strip()

        if chart_type not in {'1', '2', '3'}:
            print("Invalid option. Please choose 1 or 2 or 3.")

    # Handle the selected chart type
    if chart_type == '1':
        # Loop for bubble type selection (only for Confident and Trust chart)
        while show_bubbles not in {1, 2, 3}:
            print("Choose a confident and trust line:")
            print("1. Show positive bubbles")
            print("2. Show negative bubbles")
            print("3. Show both positive and negative bubbles")

            try:
                show_bubbles = int(input("Select chart type (1 or 2 or 3): ").strip())
                if show_bubbles not in {1, 2, 3}:
                    print("Invalid option. Please choose 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Display Confident and Trust chart
        confident_trust_chart(currency=currency, start_date=start_date, end_date=end_date,
                              filtered_data_filename=filtered_windows_file, kind=show_bubbles)

    elif chart_type == '2':
        max_window_size = int(input("What was the previous max window size? (default 400): ") or 400)
        if len(list_files_with_prefix(path=path, prefix=filename)) == 0:
            print("Model file doesn't exist")
            print("Please first create model file")
            handle_model_execution(currency, start_date, end_date)

        # Display stimulate chart
        lppl_simulate_chart(currency=currency, start_date=start_date, end_date=end_date,
                            max_window_size=max_window_size, path=path, filename=filename)

    elif chart_type == '3':
        # Display Critical Time chart
        print("Critical Time chart")
        critical_time_chart(currency=currency, start_date=start_date, end_date=end_date,
                            filtered_windows_file=filtered_windows_file)


# Replace main() function
if __name__ == "__main__":
    get_user_input()
