from joblib import Parallel, delayed, dump, load
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from get_data import get_data, list_files_with_prefix


def confident_trust_chart(currency: str, start_date: str, end_date: str, filtered_data_filename: str, kind: int):
    """
    Plots Bitcoin price along with confidence and trust indicators.

    Arguments:
    - real_data_filename: str, path to the CSV file containing Bitcoin price data.
    - filtered_data_filename: str, path to the CSV file containing filtered LPPLS windows data.
    - kind: int, controls which lines to show (1 = confidence only, 2 = trust only, 3 = both).
    """

    # Load the filtered LPPLS windows data (confidence/trust)
    print(filtered_data_filename, ' filename')
    # ds_lppls_confidence_filtered_windows = load(filtered_data_filename)
    ds_lppls_confidence_filtered_windows = pd.read_csv(filtered_data_filename)
    ds_lppls_confidence_filtered_windows['t2_date'] = pd.to_datetime(ds_lppls_confidence_filtered_windows['t2_date'])
    ds_lppls_confidence_filtered_windows.set_index('t2_date', inplace=True)

    # Load real price data
    real_data = get_data(currency, start_date, end_date)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # Plot Bitcoin price on the left y-axis
    l1, = ax1.plot(real_data.index, real_data['Close'], 'b-', label='Price', linewidth=1)
    ax1.set_ylabel('Price', color=(64/255, 117/255, 204/255))
    ax1.tick_params(axis='y', labelcolor='b')

    # Set up secondary y-axis for confidence and trust
    ax2 = ax1.twinx()
    ax2.set_ylabel('Confidence and Trust Indicators')
    ax2.set_ylim(0, 1, auto=False)  # Set fixed limits before plotting

    lines = [l1]  # Start with price line
    labels = ['Price']

    # Plot positive and negative confidence lines (solid)
    if kind in [1, 3]:
        l2, = ax2.plot(ds_lppls_confidence_filtered_windows.index,
                       ds_lppls_confidence_filtered_windows['confident_positive_bubble'], color=(15/255,115/255,15/255),
                       label='Positive Confidence', linewidth=1)
        l3, = ax2.plot(ds_lppls_confidence_filtered_windows.index,
                       ds_lppls_confidence_filtered_windows["trust_positive_bubble"], color=(0,128/255,50/255), linestyle='--',
                       label='Positive Trust', linewidth=1)
        lines.extend([l2, l3])
        labels.extend(['Positive Confidence', 'Positive Trust'])

    # Plot positive and negative trust lines (dashed)
    if kind in [2, 3]:
        # Placeholder for trust data
        l4, = ax2.plot(ds_lppls_confidence_filtered_windows.index,
                       ds_lppls_confidence_filtered_windows["confident_negative_bubble"], color=(190/255,0,0),
                       label='Negative Confidence', linewidth=1)
        l5, = ax2.plot(ds_lppls_confidence_filtered_windows.index,
                       ds_lppls_confidence_filtered_windows["trust_negative_bubble"], color=(128/255,0,0), linestyle='--',
                       label='Negative Trust', linewidth=1)
        lines.extend([l4, l5])
        labels.extend(['Negative Confidence', 'Negative Trust'])

    # Combine legends from both axes into one legend in the top-left corner
    ax1.legend(lines, labels, loc='upper left')
    ax2.set_ylim(0, 1, auto=False)  # Confidence and Trust are between 0 and 1

    plt.title('Price with Confidence and Trust Indicators')
    plt.xlabel('Date')

    plt.grid(True)

    # Show the plot
    plt.show()

def critical_time_chart(currency, start_date, end_date, filtered_windows_file):

    # Load the filtered LPPLS windows data (confidence/trust)
    ds_lppls_confidence_filtered_windows = load(filtered_windows_file)
    ds_lppls_confidence_filtered_windows['t2_date'] = pd.to_datetime(ds_lppls_confidence_filtered_windows['t2_date'])
    ds_lppls_confidence_filtered_windows.set_index('t2_date', inplace=True)

    # Load real price data
    real_data = get_data(currency, start_date, end_date)

    print(ds_lppls_confidence_filtered_windows.columns)

    critical_data_frame = ds_lppls_confidence_filtered_windows["tc"]
    critical_data_frame = critical_data_frame.to_frame()
    critical_data_frame["tc_index"] = critical_data_frame["tc"] + list(range(0, len(ds_lppls_confidence_filtered_windows)))
    critical_times_reshaped = critical_data_frame['tc_index'].to_numpy().reshape(-1, 1)

    # Apply Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=50)  # You can adjust distance_threshold
    clusters = agg_clustering.fit_predict(critical_times_reshaped)

    critical_data_frame["group"] = clusters
    grouped = critical_data_frame.groupby('group')

    # Plot real price
    plt.figure(figsize=(12, 6))

    # Plot the real Bitcoin price line
    plt.plot(real_data.index, real_data['Close'], label="Real Price", color="orange", linewidth=2)

    for name, group in grouped:
        # print(f""" name {name} group len {len(group)} min {group["tc_index"].min()} max {group["tc_index"].max()}""")
        # Get the min and max indices for this group (cluster)
        min_index = group["tc_index"].min()
        max_index = group["tc_index"].max()

        # Convert tc_index back to real dates in the real data index
        t1_date = real_data.iloc[min_index].name
        t2_date = real_data.iloc[max_index].name

        # Fill between t1_date and t2_date under the price line
        plt.fill_between(real_data.loc[t1_date:t2_date].index,
                         real_data.loc[t1_date:t2_date]['Close'],
                         color='skyblue', alpha=0.4)


    # Add labels and title
    plt.title(f"Real vs Simulated Price ({currency}) for last window")
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Add legend
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    return 0


def lppl_simulate_chart(currency: str, start_date: str, end_date: str, max_window_size: int, path: str, filename: str):
    # Load the filtered LPPLS windows data (confidence/trust)
    # ds_lppls_windows = load(model_data_filename)
    file_list = list_files_with_prefix(path=path, prefix=filename)
    print(f""" {len(file_list)} files has been founded from model""")
    raw_data = []

    # for file in file_list:
    #     df = load(f"{path}{file}")
    #     raw_data.append(df)
    raw_data.append(load(file_list[-1]))
    ds_lppls_windows = pd.concat(raw_data, ignore_index=True)
    ds_lppls_windows['t1_date'] = pd.to_datetime(ds_lppls_windows['t1_date'])
    ds_lppls_windows['t2_date'] = pd.to_datetime(ds_lppls_windows['t2_date'])
    ds_lppls_windows.set_index('t2_date', inplace=True)

    # Load real price data
    real_data = get_data(currency, start_date, end_date)

    last_group = ds_lppls_windows[ds_lppls_windows['windows_count'] == ds_lppls_windows['windows_count'].max()]
    last_window = last_group[last_group['window_length'] == max_window_size]
    bet = last_window["bet"].values[0]
    ome = last_window["ome"].values[0]
    phi = last_window["phi"].values[0]
    A = last_window["A"].values[0]
    B = last_window["B"].values[0]
    C = last_window["C"].values[0]
    resids = last_window["resids"].values[0]

    # Extract t1 and t2 dates for simulation
    t1_date = pd.to_datetime(last_window["t1_date"].iloc[0])
    t2_date = pd.to_datetime(last_window.index[0])
    time_simulation = real_data.loc[t1_date:t2_date].index
    t_sim = np.arange(0, len(time_simulation))
    simulated_log_prices = simulate_lppls(t_sim, A, B, C, bet, ome, phi, len(t_sim), resids)

    # Convert log-prices back to normal prices using exponential
    simulated_prices = np.exp(simulated_log_prices)

    print(time_simulation, " time_simulation")

    # Plot
    plt.figure(figsize=(18, 8))
    plt.plot(real_data.index, real_data['Close'], label="Real Price", color="orange", linewidth=2)
    plt.plot(time_simulation, simulated_prices, label="Simulated Price", color="blue", linestyle='--', linewidth=2)

    # Add labels and title
    plt.title(f"Real vs Simulated Price ({currency}) for last window")
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Add legend
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    return 0

# LPPLS formula to simulate price
def simulate_lppls(t, A, B, C, beta, omega, phi, tc, resids):
    """
    Simulates prices using the LPPLS model.

    Parameters:
    - t: Time series (array of time points)
    - A, B, C: LPPLS parameters
    - beta, omega, phi: LPPLS parameters
    - tc: Critical time
    - resids: Bootstrapped residuals from the LPPLS model

    Returns:
    - Simulated price series (logarithmic)
    """
    delta_t = tc - t
    f_t = np.power(delta_t, beta)
    g_t = np.power(delta_t, beta) * np.cos(omega * np.log(delta_t) + phi)

    # Simulated log-prices (without residuals)
    log_prices = A + B * f_t + C * g_t

    # Add bootstrapped residuals (randomly sampled with replacement)
    boot_resids = np.random.choice(resids, size=len(t), replace=True)

    # Add residuals to log-prices
    log_prices_with_resids = log_prices + boot_resids

    return log_prices_with_resids
