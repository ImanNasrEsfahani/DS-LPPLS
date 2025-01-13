import os
from joblib import Parallel, delayed, dump, load
import pandas as pd
import numpy as np
from run_model import lppls
from run_model import lppl_simulate_boot, lppl_estimate_rob_3all
from get_data import list_files_with_prefix
import multiprocessing

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def DS_LPPLS_Confidence(lppls_calibrations: list, max_window_size, min_window_size, ssw, high_threshold=0.05):
    """
  Calculates the DS LPPLS Confidence indicator.

  Args:
      lppls_calibrations: A list of LPPLS calibration results.
      filtering_condition: A function that takes an LPPLS calibration result
          and returns True if it satisfies the filtering condition, False otherwise.

  Returns:
      True if the DS LPPLS Confidence is high, False otherwise.
  """
    num_valid_calibrations = 0

    results = pd.DataFrame(
        columns=["start", "end", "window_size", "price", "log_price", "tc", "m", "w", "A", "B", "C1", "C2", "sse",
                 "windows_code"])
    for index, calibration in lppls_calibrations.iterrows():

        if lppls_filter(tc=calibration.tc, m=calibration.m, w=calibration.w, a=calibration.A, b=calibration.B,
                        c1=calibration.C1, c2=calibration.C2, t1=max_window_size - calibration.window_size,
                        t2=max_window_size, filter_condition=1):
            num_valid_calibrations += 1
            results.loc[len(results)] = calibration

    if (num_valid_calibrations / len(lppls_calibrations) < high_threshold):
        return pd.DataFrame(
            columns=["start", "end", "window_size", "price", "log_price", "tc", "m", "w", "A", "B", "C1", "C2", "sse",
                     "window_code", "confident_factor"])

    results["confident_factor"] = num_valid_calibrations / len(lppls_calibrations)
    results = pd.DataFrame(results, dtype='object')  # All columns as object type

    return results


def DS_LPPLS_Trust(lppls_calibrations, max_window_size, min_window_size, ssw, high_threshold=0.10):
    """
  Calculates the DS LPPLS Trust indicator.

  Args:
    lppls_calibrations: The calibrated LPPLS structure.

  Returns:
      True if the DS LPPLS Trust is high, False otherwise.
  """
    results = pd.DataFrame(
        columns=["start", "end", "window_size", "price", "log_price", "tc", "m", "w", "A", "B", "C1", "C2", "sse",
                 "windows_code", "synthetic_price", "residuals"])
    for index, calibration in lppls_calibrations.iterrows():
        if lppls_filter(tc=calibration.tc, m=calibration.m, w=calibration.w, a=calibration.A, b=calibration.B,
                        c1=calibration.C1, c2=calibration.C2, t1=max_window_size - calibration.window_size,
                        t2=max_window_size, filter_condition=2):
            # Calibrate the LPPLS model on the synthetic price time series.
            synthetic_price = lppls(
                t=index + 1,
                tc=calibration["tc"],
                m=calibration["m"],
                w=calibration["w"],
                a=calibration["A"],
                b=calibration["B"],
                c1=calibration["C1"],
                c2=calibration["C2"]
            )
            calibration["synthetic_price"] = synthetic_price
            calibration["residuals"] = np.abs(calibration["log_price"] - synthetic_price)
            results.loc[len(results)] = calibration

        # pprint.pprint(results)
        if len(results["residuals"]) > 0:
            trust = np.median(results["residuals"] / results["log_price"])
        else:
            trust = 0

        if (trust > high_threshold):
            return pd.DataFrame(
                columns=["start", "end", "window_size", "price", "log_price", "tc", "m", "w", "A", "B", "C1", "C2",
                         "sse", "window_code", "confident_factor"])

        results["confident_factor"] = trust
    return results


# 6. Define the LPPLS filter conditions to check for valid fits
def lppls_filter(tc, m, w, a, b, c1, c2, t1, t2, filter_condition):
    """
    Applies LPPLS filter conditions to check for valid fits.
    Args:
        optimized_params (dict): Dictionary of optimized LPPLS parameters.
        filter_condition (int, opti
        onal): Filtering condition to apply (1 or 2). Defaults to 1.

    Returns:
        True if the fit passes the filter conditions, False otherwise.
    """

    # tc = optimized_params['tc']
    # m = optimized_params['m']
    # w = optimized_params['w']
    # A = optimized_params['A']
    # B = optimized_params['B']
    # C1 = optimized_params['C1']
    # C2 = optimized_params['C2']

    if filter_condition == 1:
        # Filtering Condition 1 from Table 1 in Source 1
        # if 0.01 <= m <= 1.2 and 2 <= w <= 25 and t <= tc <= t[-1]:
        if 0.01 <= m <= 1.2 and 2 <= w <= 25 and num_oscillations(w, tc, t1, t2) >= 2.5:
            return True
    elif filter_condition == 2:
        # Filtering Condition 2 from Table 1 in Source 1 (add other conditions as needed)
        # number_of_oscillations = w * 0.5 * np.log(np.abs(tc - t) / np.abs(t[-1] - t))
        # if 0.01 <= m <= 0.99 and 2 <= w <= 25 and t <= tc <= t[-1] and (2.5 <= number_of_oscillations):
        if 0.01 <= m <= 0.99 and 2 <= w <= 25 and num_oscillations(w, tc, t1, t2) >= 2.5:
            return True
    else:
        raise ValueError("Invalid filter_condition. Choose 1 or 2.")

    return False


def num_oscillations(w, tc, t1, t2):
    """
  Calculates the number of oscillations in the LPPLS model.

  Args:
    w: The angular log-frequency of the oscillations.
    tc: The critical time.
    t1: The start time of the fitting window.
    t2: The end time of the fitting window.

  Returns:
    The number of oscillations.
  """

    return (w / (2 * np.pi)) * np.log((tc - t1) / (tc - t2))


def calculate_oscillations(omega, tc, t1, t2):
    """
    Calculate the number of oscillations in the LPPLS model.

    Parameters:
    omega (float): Angular frequency
    tc (float): Critical time
    t1 (float): Start time of the time series
    t2 (float): End time of the time series

    Returns:
    float: Number of oscillations
    """
    # Calculate the phase at t1 and t2
    if tc == t2:
        return 0
    phase_t1 = abs((tc - t1) / (tc - t2))
    # print(phase_t1, " phase_t1 ", (tc - t1), " (tc - t1) " , (tc - t2), " (tc - t2) log", np.log(phase_t1))
    num_oscillations = (omega / 2) * np.log(phase_t1)
    return num_oscillations


def calculate_damping(beta, B, omega, c):
    """
    Calculate the damping factor for the LPPLS model.

    Parameters:
    beta (float): The beta parameter, representing the exponent of the power law growth
    B (float): The amplitude of the oscillations omega (float): The angular frequency of the oscillations
    c (float): The amplitude of the logarithmic price

    Returns:
    float: The damping factor
    """
    return (beta * abs(B)) / (omega * abs(c))


def calculate_relative_error(observed_prices, fitted_prices):
    """
    Calculate the relative error between observed and fitted prices.

    Parameters:
    observed_prices (np.array): Array of observed price values
    fitted_prices (np.array): Array of fitted price values from the LPPLS model

    Returns:
    np.array: Array of relative errors
    float: Mean relative error
    """
    return np.mean(np.abs(observed_prices - fitted_prices) / observed_prices)


# Initialize counters
counters_condition_1 = {
    "beta_condition": 0,
    "omega_condition": 0,
    "tc_condition": 0,
    "oscillations_condition": 0,
    "damping_condition": 0,
    "relative_error_condition": 0,
    "hazard_rate_condition": 0
}


# filter condition 1
def check_lppls_validity_filtering_condition_1(beta, omega, tc, t2, t1, B, C, observed_prices, fitted_prices):
    """
    Check the validity of LPPLS fit based on filtering condition 1 in Table 1.

    Parameters:
    beta (float): Power law exponent
    omega (float): Angular frequency
    tc (float): Critical time
    t2 (float): End time of the time series
    t1 (float): Start time of the time series
    B (float): Amplitude of the power law
    C (float): Amplitude of the log-periodic oscillations

    Returns:
    bool: True if the fit is valid, False otherwise
    """
    result: bool = True

    # Check beta condition
    if not 0.01 <= beta <= 1.2:
        counters_condition_1["beta_condition"] += 1
        result = False

    # Check omega condition
    if not 2 <= omega <= 25:
        counters_condition_1["omega_condition"] += 1
        result = False

    dt = t2 - t1
    # Check tc condition
    if not (t2 - (0.05 * dt)) <= tc <= (t2 + (0.1 * dt)):
        counters_condition_1["tc_condition"] += 1
        result = False

    # Check number of oscillations condition
    if not 2.5 <= calculate_oscillations(omega=omega, tc=tc, t1=t1, t2=t2):
        # print(f""" calculate_oscillations : {calculate_oscillations(omega=omega, tc=tc, t1=t1, t2=t2)}""")
        counters_condition_1["oscillations_condition"] += 1
        result = False

    # Check damping condition
    if not 0.5 <= calculate_damping(beta=beta, B=B, omega=omega, c=C):
        counters_condition_1["damping_condition"] += 1
        result = False

    # Check relative error condition
    if not 0 <= calculate_relative_error(observed_prices, fitted_prices) <= 0.05:
        counters_condition_1["relative_error_condition"] += 1
        result = False

    # Check hazard rate condition
    if not B * beta - C * np.sqrt(beta ** 2 + omega ** 2) > 0:
        counters_condition_1["hazard_rate_condition"] += 1
        result = False

    return result


# Initialize counters
counters_condition_2 = {
    "beta_condition": 0,
    "omega_condition": 0,
    "tc_condition": 0,
    "oscillations_condition": 0,
    "damping_condition": 0,
    "relative_error_condition": 0,
    "hazard_rate_condition": 0
}


# filter condition 2
def check_lppls_validity_filtering_condition_2(beta, omega, tc, t2, t1, B, C, observed_prices, fitted_prices):
    """
    Check the validity of LPPLS fit based on filtering condition 2 in Table 1.

    Parameters:
    beta (float): Power law exponent
    omega (float): Angular frequency
    tc (float): Critical time
    t2 (float): End time of the time series
    t1 (float): Start time of the time series
    B (float): Amplitude of the power law
    C (float): Amplitude of the log-periodic oscillations

    Returns:
    bool: True if the fit is valid, False otherwise
    """

    result: bool = True

    # Check beta condition
    if not (0.01 <= beta <= 0.99):
        counters_condition_2["beta_condition"] += 1
        result = False

    # Check ome condition
    if not (2 <= omega <= 25):
        counters_condition_2["omega_condition"] += 1
        result = False

    dt = t2 - t1
    # Check tc condition
    if not (t2 - (0.05 * dt)) <= tc <= (t2 + (0.1 * dt)):
        counters_condition_2["tc_condition"] += 1
        result = False

    # Check number of oscillations condition
    if not 2.5 <= calculate_oscillations(omega=omega, tc=tc, t1=t1, t2=t2):
        counters_condition_2["oscillations_condition"] += 1
        result = False

    # Check damping condition
    if not 0.8 <= calculate_damping(beta=beta, B=B, omega=omega, c=C):
        counters_condition_2["damping_condition"] += 1
        result = False

    # Check number of oscillations condition
    if not 0 <= calculate_relative_error(observed_prices, fitted_prices) <= 0.2:
        counters_condition_2["relative_error_condition"] += 1
        result = False

    # Check hazard rate condition
    if not B * beta - C * np.sqrt(beta ** 2 + omega ** 2) > 0:
        counters_condition_2["hazard_rate_condition"] += 1
        result = False

    return result


# --------------------------------

def process_group(group_name, rollings_windows_generated, path, filename, min_beta, max_beta, min_omet, max_omet, count,
                  columns, boot_rep):
    print(
        GREEN + f"""{group_name}th Window is selected from {count}, it has {len(rollings_windows_generated)} sub-records. Progress {round((group_name / count) * 100, 2)}%""" + RESET)

    mean_valid_windows_df = pd.DataFrame(columns=columns)
    valid_windows = pd.DataFrame(columns=columns)
    conditions1_pos_bub: float = 0
    conditions1_neg_bub: float = 0
    a2_pos_boot: float = 0
    a2_neg_boot: float = 0

    for _, one_window in rollings_windows_generated.iterrows():
        # print(f""" {index}/{len(rollings_windows_generated)} filtered""")

        if (check_lppls_validity_filtering_condition_1(beta=one_window["bet"], omega=one_window["ome"],
                                                       tc=one_window["tc_index"], t2=one_window["t2"],
                                                       t1=one_window["t1"], B=one_window["B"], C=one_window["C"],
                                                       observed_prices=one_window["observed_prices"],
                                                       fitted_prices=one_window["fitted_prices"])):
            # condition 1
            if one_window["B"] < 0: conditions1_pos_bub += 1
            if one_window["B"] > 0: conditions1_neg_bub += 1
            # Add this valid window to the DataFrame for further processing
            valid_windows.loc[len(valid_windows)] = one_window

        if (check_lppls_validity_filtering_condition_2(beta=one_window["bet"], omega=one_window["ome"],
                                                       tc=one_window["tc_index"], t2=one_window["t2"],
                                                       t1=one_window["t1"], B=one_window["B"], C=one_window["C"],
                                                       observed_prices=one_window["observed_prices"],
                                                       fitted_prices=one_window["fitted_prices"])):
            # condition 2
            a_pos_boot = np.zeros(boot_rep)
            a_neg_boot = np.zeros(boot_rep)

            for j in range(boot_rep):
                params = [one_window["bet"], one_window["ome"], one_window["phi"], one_window["A"], one_window["B"],
                          one_window["C"], one_window["tc"]]

                z_est = lppl_simulate_boot(len(one_window['resids']), params, one_window['resids'])
                z_boot_est = lppl_estimate_rob_3all(x=z_est, max_win_tc=0.2, min_beta=min_beta, max_beta=max_beta,
                                                    min_omet=min_omet, max_omet=max_omet)
                conditions2_boot_pos = 0
                conditions2_boot_neg = 0

                if 0 < z_boot_est['param_JLS']["bet"] < 1 and z_boot_est['param_JLS']["B"] < 0 and z_boot_est[
                    'crash_rate'] > 0 and z_boot_est['KPSS_res'] < 0.463:
                    conditions2_boot_neg = 1

                if 0 < z_boot_est['param_JLS']["bet"] < 1 and z_boot_est['param_JLS']["B"] > 0 and z_boot_est[
                    'crash_rate'] < 0 and z_boot_est['KPSS_res'] < 0.463:
                    conditions2_boot_pos = 1

                a_pos_boot[j] = conditions2_boot_pos
                a_neg_boot[j] = conditions2_boot_neg

            a2_pos_boot = np.sum(a_pos_boot) / boot_rep
            a2_neg_boot = np.sum(a_neg_boot) / boot_rep

    if not valid_windows.empty:
        # print(valid_windows, "valid_windows")
        mean_valid_windows = valid_windows.loc[:,
                             ["bet", "ome", "phi", "A", "B", "C", "tc", "window_length", "windows_count"]].mean(axis=0)
        mean_valid_windows['tc'] = round(mean_valid_windows['tc'])
        mean_valid_windows['confident_positive_bubble'] = conditions1_pos_bub / len(rollings_windows_generated)
        mean_valid_windows['confident_negative_bubble'] = conditions1_neg_bub / len(rollings_windows_generated)
        mean_valid_windows['trust_positive_bubble'] = a2_pos_boot
        mean_valid_windows['trust_negative_bubble'] = a2_neg_boot

        # Add additional information to the result
        mean_valid_windows['t2'] = rollings_windows_generated.iloc[0]['t2']
        mean_valid_windows['t2_date'] = rollings_windows_generated.iloc[0]['t2_date']
        mean_valid_windows['rel_err'] = rollings_windows_generated.iloc[0]["resids"][0]
        mean_valid_windows['observed_prices'] = rollings_windows_generated.iloc[0]["observed_prices"][0]
        mean_valid_windows['fitted_prices'] = rollings_windows_generated.iloc[0]["fitted_prices"][0]
        return mean_valid_windows.to_frame().T.reset_index(drop=True)

    return pd.DataFrame()  # Return an empty DataFrame if no valid windows


def running_filters(max_window_size: int, min_window_size: int, ssw: int, boot_rep: int, clust_number: float,
                    min_beta: float, max_beta: float, min_omet: float, max_omet: float, path: str, filename: str):
    file_list = list_files_with_prefix(path=path, prefix=filename)
    print(f""" {len(file_list)} files has been founded from model""")
    raw_data = []

    for file in file_list:
        df = load(f"{path}{file}")
        raw_data.append(df)

    all_rollings_windows_generated = pd.concat(raw_data, ignore_index=True)
    del raw_data
    all_grouped_rollings_windows_generated = all_rollings_windows_generated.groupby("windows_count")
    columns = ['bet', 'ome', 'phi', 'A', 'B', 'C', 'tc', 'window_length', 'windows_count', 't2_date',
               'confident_positive_bubble', 'confident_negative_bubble', 'trust_positive_bubble',
               'trust_negative_bubble', 'rel_err', 't2', 'observed_prices', 'fitted_prices']
    print()
    print(
        f"""The {len(all_grouped_rollings_windows_generated)} window has found from {len(all_rollings_windows_generated)} records""")

    # Use Joblib to parallelize the processing of groups
    num_cores = multiprocessing.cpu_count()
    results_df_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_group)(group_name, rollings_windows_generated, path, filename,
                               min_beta, max_beta, min_omet, max_omet, len(all_grouped_rollings_windows_generated),
                               columns, boot_rep)
        for group_name, rollings_windows_generated in all_grouped_rollings_windows_generated)

    # Combine results into a single DataFrame
    mean_valid_windows_df = pd.concat(results_df_list).reset_index(drop=True)
    dump(mean_valid_windows_df, f"{path}{filename}_filtered.joblib")
    mean_valid_windows_df.to_csv(f"{path}{filename}_filtered.csv")
    return mean_valid_windows_df