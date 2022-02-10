import logging
import multiprocessing
import os
import time
from functools import partial
from multiprocessing import Pool

import pandas
import tqdm

from learning.grouped_learning import Berta
from learning.grouped_learning_with_influences import Soeren
from model.datasets import Datasets
from sampling.from_tests_sampler import FromTestsSamplingStrategy
from sampling.group_sampling.mutex_aware_group_sampling_strategy import IndependentFeatureGroupSamplingStrategy
from sampling.random_sampling.distance_sampling_strategy import DistanceSamplingStrategy
from sampling.sampler import Sampler
from testing.grouped_tester import GroupedTester
from testing.tester import Tester
from util.network_util import save_mutex_graph, generate_mutex_graph_squential
from util.util import get_regression_metricts, expand_wrapper, flatten, setup_logger, make_test_folder
from util.util import get_samples_with_results

log = logging.getLogger("Test_Case_2")
test_folder = make_test_folder()


def do_test_run(
        iteration,
        name, vm, x_test, y_test,
        grouped_tester,
        strategy, strategy_name,
        group_size, max_sample_size
):
    ctx_str = f"[{name}][It: {iteration}][Variant: {strategy_name}][Group Size: {group_size:02d}]"
    log.debug(f'{ctx_str} Do sample size run for , group_size:...')
    grouped_sampler = Sampler(strategy(vm, group_size=group_size, load_mutex=f"../result/tmp/{name}.json"))
    data = pandas.DataFrame(columns=["mae", "mse", "r2", "mape"])
    coefficients = pandas.DataFrame(columns=["intercept", "coefficients"])
    current_sample_size = 0
    while current_sample_size < max_sample_size:
        current_sample_size = current_sample_size + 1
        start = time.perf_counter()
        x, y = get_samples_with_results(grouped_sampler, grouped_tester, current_sample_size)

        if len(x) != current_sample_size:
            log.debug(
                f"{ctx_str} Could not find enough samples!"
                f" Discarding run with sample size {current_sample_size}"
            )
            continue
        grouped_sampler.reset()
        start_2 = time.perf_counter()
        learner = Soeren(vm, group_size)
        learner.fit(x, y)
        y_predicted = learner.predict(x_test)

        metrics = get_regression_metricts(y_test, y_predicted)
        log.info(
            f'{ctx_str}[Sample size: {current_sample_size}]'
            f'[Sampling: {(time.perf_counter() - start):0.4f}]'
            f'[Modeling: {(time.perf_counter() - start_2):0.4f}] : {metrics}')
        data = data.append([metrics])
        coefficients = coefficients.append([*flatten(learner.model.coef_), learner.model.intercept_])

    log.debug(f"{ctx_str} Saving results to file.")
    data.to_csv(f'{test_folder}/{name}_groupsize_{group_size}_method_{strategy_name}_{iteration}.csv')
    coefficients.to_csv(
        f'{test_folder}/{name}_groupsize_{group_size}_method_{strategy_name}_{iteration}_model.csv')


def generate_mutex_if_missing(name, vm):
    if not os.path.isfile(f"../result/tmp/{name}.json"):
        log.debug(f"[{name}] Could not find mutex graph for {name}.")
        save_mutex_graph(generate_mutex_graph_squential(vm), f"../result/tmp/{name}.json")


def generate_test_dataset(name, vm, test_strategy):
    tester = Tester(test_strategy)
    sample_size = 100
    if "syn" in name:
        log.info(f"[{name}] Use Diversity promotion for testing ... increasing sample size")
        sampler = Sampler(DistanceSamplingStrategy(vm))
    else:
        sampler = Sampler(FromTestsSamplingStrategy(vm, tester))

    log.debug(f"[{name}] Gathering test data ...")
    t0 = time.time()
    x_test, y_test = get_samples_with_results(sampler, tester, sample_size, False)
    log.debug(f"[{name}] Test data gathered in: {time.time() - t0}")
    return x_test, y_test


def generate_jobs(iteration_size, max_group_size, datasets, max_sample_size=50):
    try:
        iter(iteration_size)
        iterations = iteration_size
    except TypeError:
        iterations = range(iteration_size)
    try:
        iter(max_group_size)
        group_sizes = max_group_size
    except TypeError:
        group_sizes = range(2, max_group_size)

    test_suite = Datasets().generate_testsuite(datasets)
    strategies = [
        (IndependentFeatureGroupSamplingStrategy, "mutex"),
        # (HammingGroupSamplingStrategy, "hamming")
    ]
    jobs = []

    for name, vm, test_strategy in test_suite:
        x_test, y_test = generate_test_dataset(name, vm, test_strategy)
        generate_mutex_if_missing(name, vm)
        for iteration in iterations:
            for strategy, strategy_name in strategies:
                grouped_tester = GroupedTester(test_strategy)
                for group_size in group_sizes:
                    jobs.append((
                        iteration,
                        name, vm, x_test, y_test,
                        grouped_tester,
                        strategy, strategy_name,
                        group_size, max_sample_size
                    ))
    return jobs


setup_logger(None, True)
log.info("Generating jobs ...")

max_processes = multiprocessing.cpu_count()
pool = Pool(max_processes)
# jobs = [(20, 10, ['syn-50-pre']), (20, 20, ['syn-100-pre']), (10, 60, ['syn-1000-pre'])]
jobs = [
    # (20, range(5, 10), Datasets.REAL, 20),
    # (1, 30, ['syn-500-pre-simple'], 5),
    # (1, 60, ['syn-1000-pre-simple'], 5),
    # (range(0, 10), 20, ['syn-100-pre'], 10),
    # (range(0, 10), 30, ['syn-500-pre'], 10),
    (10, 6, Datasets.SYN_INTERACTIONS, 30),
]
execution_params = flatten(pool.map(partial(expand_wrapper, generate_jobs), jobs))
# random.shuffle(execution_params)
log.info(f"Starting test runs with {max_processes} processes!")
with Pool(max_processes - 3) as p:
    max_ = len(execution_params)
    log.info(f"Scheduled {max_} tests to run!")
    with tqdm.tqdm(total=max_) as pbar:
        for i, _ in enumerate(p.imap_unordered(partial(expand_wrapper, do_test_run), execution_params)):
            pbar.update()
