import warnings
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss
from joblib import Parallel, delayed, dump, load
from scipy.optimize import minimize
from get_data import list_files_with_prefix

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def get_initial_guess(x):
    return {
        'tc': len(x),  # tc is initially set to the last time point in the data (t[-1])
        'm': 0.5,  # m and β the exponent of the power law growth is initially 0.5
        'w': 10,  # w is initially 10
        'A': 1,  # A is initially 1 (previously the average of the logarithmic prices)
        'B': -1,  # B is initially -1
        'C1': 1,  # C1 is initially 1
        'C2': 1  # C2 is initially 1
    }


# 2- Define the LPPLS function (based on source equation 9)
def lppls(t, tc: float, m: float, w: float, a: float, b: float, c1: float, c2: float):
    with np.errstate(invalid='ignore', divide='ignore'):
        dt = tc - t
        return a + np.power(dt, m) * (
                b + ((c1 * np.cos(w * np.log(dt))) + (c2 * np.sin(w * np.log(dt))))
        )


# # 3. Define the objective function (sum of squared residuals)
def objective_function(tc, m, w, A, B, C1, C2, t, log_prices):
    model_prices = lppls(t, tc, m, w, A, B, C1, C2)
    residuals = log_prices - model_prices
    return np.sum((residuals) ** 2)


def resume_processing_model(data: pd.DataFrame, time_scale: int, min_window: int, step_shrinking_window: int,
                            max_win_tc: float, min_beta: float, max_beta: float, min_omet: float, max_omet: float,
                            path: str, filename: str):
    previous_model = load(f"{path}{list_files_with_prefix(path=path, prefix=filename)[-1]}")
    windows_count = previous_model.iloc[-1]["windows_count"] + 1
    remaining_data = data.iloc[previous_model.iloc[-1]["windows_count"]:]

    rolling_window_lppls(data=remaining_data, time_scale=time_scale, min_window=min_window,
                         step_shrinking_window=step_shrinking_window, max_win_tc=max_win_tc, min_beta=min_beta,
                         max_beta=max_beta, min_omet=min_omet, max_omet=max_omet, path=path, filename=filename,
                         windows_count=windows_count,
                         file_counter=len(list_files_with_prefix(path=path, prefix=filename)))
    return 0


# def rolling_window_lppls(data, max_window, min_window, ssw, path_file_name, windows_count=1):
def rolling_window_lppls(data: pd.DataFrame, time_scale: int, min_window: int, step_shrinking_window: int,
                         max_win_tc: float, min_beta: float, max_beta: float, min_omet: float, max_omet: float,
                         path: str, filename: str, windows_count=1, max_iteration_in_each_file=250, file_counter=0):
    # Initialize the DataFrame to store rolling results
    x = data["Close"].to_numpy()
    columns = ['bet', 'ome', 'phi', 'A', 'B', 'C', 'tc', 'window_length', 'rel_err', 'observed_prices', 'fitted_prices',
               'resids', 'KPSS_res', 'p_value', 'lags', 'crit', 'crash_rate', 'windows_count', 'tc_index', 't1', 't2',
               't1_date', 't2_date']

    roll_results = pd.DataFrame(columns=columns)
    print()

    # Rolling window calculation
    for t2 in range(time_scale, len(x)):
        print(
            GREEN + f""" Start of a new window progress ({round(((t2 - time_scale) / (len(x) - time_scale)) * 100, 2)} %) End Date: {data.iloc[t2].name} t2: {t2} from time_scale: {time_scale} of Total {len(x)}""" + RESET)
        t2_date = data.iloc[t2].name
        z = x[t2 - time_scale:t2]
        z = np.log(z)
        stepf = range(0, time_scale - min_window, step_shrinking_window)

        # Parallel processing across steps
        def process_step(z, i, data, t2, time_scale, t2_date, windows_count, max_win_tc):
            result = calculate_lppl(z=z, i=i, max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta,
                                    min_omet=min_omet, max_omet=max_omet)
            result["windows_count"] = windows_count
            result['t1'] = (t2 - time_scale) + i  # Ensure t1 is based on the larger index
            result['tc_index'] = result['tc'] + result['t1']
            result['t2'] = t2
            result['t1_date'] = data.iloc[(t2 - time_scale) + i].name
            result["t2_date"] = t2_date
            return result

        # Parallel processing across steps
        results = Parallel(n_jobs=-1)(
            delayed(process_step)(
                z, i, data, t2, time_scale, t2_date, windows_count, max_win_tc
            ) for i in stepf
        )

        if roll_results.empty:
            roll_results = pd.DataFrame(results, columns=columns)
        else:
            roll_results = pd.concat([roll_results, pd.DataFrame(results, columns=columns)])

        roll_results = roll_results.reset_index(drop=True)
        windows_count += 1

        # Save results every 250 iterations
        if windows_count % max_iteration_in_each_file == 0:
            file_counter += 1  # Increment file counter for a new file
            current_file_name = f"{path}{filename}_{file_counter:02d}.joblib"
            dump(roll_results, current_file_name)

            # Clear roll_results after dumping
            roll_results.drop(roll_results.index, inplace=True)

    print("Last file saving ...")
    file_counter += 1  # Increment file counter for a new file
    current_file_name = f"{path}{filename}_{file_counter:02d}.joblib"
    dump(roll_results, current_file_name)

    return pd.DataFrame(roll_results, columns=columns)


# Function for parallel processing of rolling windows
def calculate_lppl(z: list, i: int, max_win_tc, min_beta: float, max_beta: float, min_omet: float,
                   max_omet: float):
    z_window = z[i:]
    lppl_par = lppl_estimate_rob_3all(z_window, max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta,
                                      min_omet=min_omet, max_omet=max_omet)  # LPPL estimation

    return {
        'bet': lppl_par['param_JLS']["bet"],
        'ome': lppl_par['param_JLS']["ome"],
        'phi': lppl_par['param_JLS']["phi"],
        'A': lppl_par['param_JLS']["A"],
        'B': lppl_par['param_JLS']["B"],
        'C': lppl_par['param_JLS']["C"],
        'tc': round(lppl_par['param_JLS']["tc"]),
        'KPSS_res': lppl_par['KPSS_res'],
        'p_value': lppl_par['p_value'],
        'lags': lppl_par['lags'],
        'crit': lppl_par['crit'],
        'crash_rate': lppl_par['crash_rate'],
        'window_length': len(z_window),
        'rel_err': lppl_par["relative_error"],
        'observed_prices': lppl_par["observed_prices"],
        'fitted_prices': lppl_par["fitted_prices"],
        'resids': lppl_par["resids"]
    }
    # return [*lppl_par['param_JLS'], lppl_par['KPSS_res'], lppl_par['crash_rate'], len(z_window), num_osc, damping, lppl_par['relative_error'], lppl_par["resids"]]


def bounds(x, max_win_tc, min_beta, max_beta, min_omet, max_omet):
    # 'tc', 'm', 'w', 'A', 'B', 'C1', 'C2'
    return [
        ((len(x) - (max_win_tc * len(x))), (len(x) + (max_win_tc * len(x)))),
        # tc is limited to the time duration 0.2 dt(t2-t1)
        (min_beta, max_beta),  # m and β the exponent of the power law growth is limited to [0, 2]
        (min_omet, max_omet),  # w is limited to [1, 50]
        (None, None),  # A is the average of the logarithmic prices
        (None, None),  # B
        (None, None),  # C1
        (None, None),  # C2
    ]


def lppl_estimate_rob_3s(x, par1, par2, max_win_tc, min_beta, max_beta, min_omet, max_omet):
    # mean for the first and second step of minimize
    par_start = np.mean([par1, par2], axis=0)
    tc, m, w, A, B, C1, C2 = par_start
    initial_guess = {
        'tc': tc,  # tc is initially set to the last time point in the data (t[-1])
        'm': m,  # m and β the exponent of the power law growth is initially 0.5
        'w': w,  # w is initially 10
        'A': A,  # A is initially 1 (previously the average of the logarithmic prices)
        'B': B,  # B is initially -1
        'C1': C1,  # C1 is initially 1
        'C2': C2  # C2 is initially 1
    }
    params_to_optimize = ['tc', 'm', 'w', 'A', 'B', 'C1', 'C2']
    x0 = [initial_guess[param] for param in params_to_optimize]

    result = minimize(
        fun=lambda params, *args: objective_function(tc=params[0], m=params[1], w=params[2], A=params[3], B=params[4],
                                                     C1=params[5], C2=params[6], t=args[0], log_prices=args[1]),
        x0=np.array(x0),
        args=(np.arange(0, len(x)), x),
        bounds=bounds(x=x, max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta, min_omet=min_omet,
                      max_omet=max_omet),
        method='L-BFGS-B',
        options={
            'ftol': 1e-15,
            'gtol': 1e-15,
            'maxiter': 15000,
            'maxfun': 20000,
        })

    optimized_params = dict(zip(params_to_optimize, result.x))
    Y = np.log(x)

    et = Y - lppls(t=np.arange(0, len(x)), tc=optimized_params["tc"], m=optimized_params["m"], w=optimized_params["w"],
                   a=optimized_params["A"], b=optimized_params["B"], c1=optimized_params["C1"],
                   c2=optimized_params["C2"])

    # Perform KPSS test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The test statistic is outside of the range of p-values available")
        kpss_stat, p_value, lags, crit = kpss(et)

    optimized_params['kpss_stat'] = kpss_stat
    optimized_params['p_value'] = p_value
    optimized_params['lags'] = lags
    optimized_params['crit'] = crit
    return optimized_params


def lppl_estimate_rob_2s(x, par, max_win_tc, min_beta, max_beta, min_omet, max_omet):
    tc, m, w, A, B, C1, C2 = par
    initial_guess = {
        'tc': tc,  # tc is initially set to the last time point in the data (t[-1])
        'm': m,  # m and β the exponent of the power law growth is initially 0.5
        'w': w,  # w is initially 10
        'A': A,  # A is initially 1 (previously the average of the logarithmic prices)
        'B': B,  # B is initially -1
        'C1': C1,  # C1 is initially 1
        'C2': C2  # C2 is initially 1
    }
    params_to_optimize = ['tc', 'm', 'w', 'A', 'B', 'C1', 'C2']
    x0 = [initial_guess[param] for param in params_to_optimize]

    result = minimize(
        fun=lambda params, *args: objective_function(tc=params[0], m=params[1], w=params[2], A=params[3], B=params[4],
                                                     C1=params[5], C2=params[6], t=args[0], log_prices=args[1]),
        x0=np.array(x0),
        args=(np.arange(0, len(x)), np.flip(x)),
        bounds=bounds(x=x, max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta, min_omet=min_omet,
                      max_omet=max_omet),
        method='L-BFGS-B',
        options={
            'ftol': 1e-15,
            'gtol': 1e-15,
            'maxiter': 15000,
            'maxfun': 20000,
        })

    return result.x


# def callback(xk):
#     print("in lppl_estimate_rob_1s")
#     print(f"Current parameters: {xk}")
#     # print(f"Current objective value: {objective_function(*xk, t, log_prices)}")

def lppl_estimate_rob_1s(x, initial_guess, max_win_tc, min_beta, max_beta, min_omet, max_omet):
    params_to_optimize = ['tc', 'm', 'w', 'A', 'B', 'C1', 'C2']
    x0 = [initial_guess[param] for param in params_to_optimize]

    result = minimize(
        fun=lambda params, *args: objective_function(tc=params[0], m=params[1], w=params[2], A=params[3], B=params[4],
                                                     C1=params[5], C2=params[6], t=args[0], log_prices=args[1]),
        x0=np.array(x0),
        args=(np.arange(0, len(x)), x),
        bounds=bounds(x=x, max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta, min_omet=min_omet,
                      max_omet=max_omet),
        method='L-BFGS-B',
        # callback=callback,
        options={
            'ftol': 1e-15,
            'gtol': 1e-15,
            'maxiter': 15000,
            'maxfun': 20000,
        })

    return result.x


def lppl_estimate_rob_3all(x, max_win_tc, min_beta, max_beta, min_omet, max_omet):
    bb1 = lppl_estimate_rob_1s(x=x, initial_guess=get_initial_guess(x), max_win_tc=max_win_tc, min_beta=min_beta,
                               max_beta=max_beta, min_omet=min_omet, max_omet=max_omet)
    bb2 = lppl_estimate_rob_2s(x=x, par=bb1, max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta,
                               min_omet=min_omet, max_omet=max_omet)
    bb3 = lppl_estimate_rob_3s(x=x, par1=bb1, par2=bb2, max_win_tc=max_win_tc, min_beta=min_beta, max_beta=max_beta,
                               min_omet=min_omet, max_omet=max_omet)

    # bet, ome, A, B, C1, C2, tc
    C_param = bb3["C2"] / np.cos(np.arctan(bb3["C1"] / bb3["C2"]))
    # b = [- B x beta - |C| x sqrt(beta^2  + omega^2)] >=0.
    crash_rate = -bb3["B"] * bb3["m"] - np.abs(C_param) * np.sqrt(bb3["m"] ** 2 + bb3["w"] ** 2)
    # param_JLS = np.array([bb3[0], bb3[1], np.arccos(bb3[4] / C_param), bb3[2], bb3[3], C_param, bb3[6]])
    params_to_optimize = ["bet", "ome", "phi", "A", "B", "C", "tc"]
    param_JLS = dict(zip(params_to_optimize, np.array(
        [bb3["m"], bb3["w"], np.nan_to_num(bb3["B"] / C_param, nan=0).astype(int), bb3["A"], bb3["B"], C_param,
         bb3["tc"]])))

    y = x
    y_fit = lppls(t=np.arange(0, len(x)), tc=bb3["tc"], m=bb3["m"], w=bb3["w"], a=bb3["A"], b=bb3["B"], c1=bb3["C1"],
                  c2=bb3["C2"])

    relative_error = np.abs(np.mean((y - y_fit) / y_fit))

    results = {
        'param_JLS': param_JLS,
        'param_FS': bb3,
        'KPSS_res': bb3["kpss_stat"],
        'p_value': bb3["p_value"],
        'lags': bb3["lags"],
        'crit': bb3["crit"],
        'crash_rate': crash_rate,
        'relative_error': relative_error,
        'observed_prices': x,
        'fitted_prices': y_fit,
        'resids': y - y_fit
    }
    return results


def lppl_simulate_boot(T=500, true_parm=None, resids=None, bitcoin_close=None):
    if true_parm is None or resids is None:
        raise ValueError("true_parm and resids must be provided")

    bet, ome, phi, A, B, C, tc = true_parm
    tt_sim = np.arange(0, T)
    sdum = np.ones(T)
    f_t = (tc - tt_sim) ** bet
    dt = tc - tt_sim
    # Disable specific warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cos")
    g_t = ((tc - tt_sim) ** bet) * np.cos(ome * np.log(dt) + phi)
    g_t = np.nan_to_num(g_t, nan=0.0)

    # Simulate price series with bootstrapping
    boot_resids = np.random.choice(resids, size=T, replace=True)
    x = np.exp(A * sdum + B * f_t + C * g_t + boot_resids)

    return x
