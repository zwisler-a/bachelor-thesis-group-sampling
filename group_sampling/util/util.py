import logging
import os
from math import ceil
from typing import List
from datetime import datetime
import numpy as np
import pandas
from sklearn import metrics
from z3 import z3

from sampling.sampler import Sampler
from testing.tester import Tester
from util.progress_loop import progress_iterations


def to_full_representation(config: List[str], vm):
    print(type(config))
    return [(1 if opt in config else 0) for opt in vm.get_features()]


def flatten(t):
    return [item for sublist in t for item in sublist]


def abs(x):
    return z3.If(x >= 0, x, -x)


def print_regression_metricts(y_test, y_predicted):
    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    r2 = metrics.r2_score(y_test, y_predicted)

    print(f"The model performance for testing set - len = {len(y_predicted)}")
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2))


def get_regression_metricts(y_test, y_predicted):
    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    r2 = metrics.r2_score(y_test, y_predicted)
    mape = metrics.mean_absolute_percentage_error(y_test, y_predicted)
    return [mae, mse, r2, mape]


def get_samples_with_results(sampler: Sampler, tester: Tester, test_size, quiet=True):
    test = progress_iterations(sampler.get_sample, test_size, quiet)
    x_test = sampler.convert_to_full_representation(test)
    y_test = tester.get_result(test)
    return x_test, y_test


def get_samples_with_results_full(sampler: Sampler, tester: Tester, test_size, quiet=True):
    test = progress_iterations(sampler.get_sample, test_size, quiet)
    x_test = sampler.convert_to_full_representation(test)
    y_test = tester.get_result(test)
    return x_test, y_test, test


def make_df_representation(x, y):
    df = pandas.DataFrame(flatten(x))
    df['result'] = flatten(y)
    return df


def distribution_with_rest(total, group_size):
    part = max(ceil(total / group_size), 1)
    sizes = []
    for j in range(group_size):
        if j == group_size - 1:
            sizes.append(max(0, total))
        else:
            sizes.append(part if total > 0 else 0)
            total -= part
    return sizes


def expand_wrapper(func, params):
    return func(*params)


def setup_logger(console_level=None, debug_log=True, info_log=True, error_log=True):
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    log = logging.getLogger()  # init the root logger
    log.setLevel(logging.DEBUG)  # make sure we capture all levels for our test

    # make a formatter to use for the file logs, shortened version from yours...
    base_path = "../logs/"

    formatter = logging.Formatter("[%(asctime)s | %(name)-40s | %(lineno)4s | %(levelname)-5s] %(message)s")

    # first file handler:
    debug_handler = logging.FileHandler(base_path + "debug.log")  # create a 'log1.log' handler
    debug_handler.setLevel(logging.DEBUG)  # make sure all levels go to it
    debug_handler.setFormatter(formatter)  # use the above formatter
    if debug_log:
        log.addHandler(debug_handler)  # add the file handler to the root logger

    info_handler = logging.FileHandler(base_path + "info.log")  # create a 'log1.log' handler
    info_handler.setLevel(logging.INFO)  # make sure all levels go to it
    info_handler.setFormatter(formatter)  # use the above formatter
    if info_log:
        log.addHandler(info_handler)  # add the file handler to the root logger

    error_handler = logging.FileHandler(base_path + "error.log")  # create a 'log1.log' handler
    error_handler.setLevel(logging.ERROR)  # make sure all levels go to it
    error_handler.setFormatter(formatter)  # use the above formatter
    if info_log:
        log.addHandler(error_handler)  # add the file handler to the root logger

    # second file handler:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level if console_level is not None else logging.CRITICAL)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)


def make_test_folder():
    now = datetime.now()
    dt_string = now.strftime("%d_%m-%H_%M_%S")
    file_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.normpath(os.path.join(file_path, '../', 'result', dt_string))
    os.makedirs(folder)
    return folder
