import multiprocessing
import time
from multiprocessing import Pool

import numpy as np
import pandas
from sklearn import linear_model

from sampling.from_tests_sampler import FromTestsSamplingStrategy
from sampling.random_sampling.better_random_sampling_strategy import BetterRandomSamplingStrategy
from sampling.random_sampling.distance_sampling_strategy import DistanceSamplingStrategy
from sampling.sampler import Sampler
from model.datasets import Datasets
from testing.tester import Tester
from util.util import get_regression_metricts, flatten
from util.util import get_samples_with_results


def execution(it):
    # for name, vm, test_strategy in Datasets().generate_testsuite(Datasets.REAL):
    test_suite = ['syn-100-pre']
    for name, vm, test_strategy in Datasets().generate_testsuite(test_suite):
        print(f'Running Dataset {name}')

        tester = Tester(test_strategy)
        # sampler = Sampler(FromTestsSamplingStrategy(vm, tester))
        # sampler = Sampler(BetterRandomSamplingStrategy(vm))
        sampler = Sampler(DistanceSamplingStrategy(vm))

        x_test, y_test = get_samples_with_results(sampler, tester, 100)
        sampler.reset()

        abort_size = 20
        sample_size = 0
        data = pandas.DataFrame()
        coefficients = pandas.DataFrame(columns=["intercept", "coefficients"])
        while sample_size < abort_size:
            sample_size = sample_size + 1

            x, y = get_samples_with_results(sampler, tester, sample_size)
            sampler.reset()

            clf = linear_model.LinearRegression()
            clf.fit(x, y)
            print("", clf.intercept_)
            y_predicted = clf.predict(x_test)

            metrics = get_regression_metricts(y_test, y_predicted)
            print(f'{sample_size} : {metrics}')
            data = data.append([metrics])
            coefficients = coefficients.append(
                [*pandas.DataFrame(clf.coef_, columns=['C'])['C'], clf.intercept_])

        data.to_csv(f'../result/test_case_4/{name}_distance_{it}.csv')
        coefficients.to_csv(f'../result/test_case_4/{name}_distance_{it}_model.csv')


max_processes = multiprocessing.cpu_count() // 2
with Pool(max_processes) as p:
    p.map(execution, range(1, 21))
