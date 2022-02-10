from sklearn import linear_model

from graphing.regression_graphs import plot_linear_regression
from sampling.random_sampling.better_random_sampling_strategy import BetterRandomSamplingStrategy
from sampling.sampler import Sampler
from model.datasets import Datasets
from testing.tester import Tester
from util.progress_loop import progress_iterations

for name, vm, test_strategy in Datasets().generate_testsuite(Datasets.SYN_SMALL):
    print(f'Running Dataset {name}')

    sample_size = 100
    test_size = 20

    sampler = Sampler(BetterRandomSamplingStrategy(vm))
    tester = Tester(test_strategy)

    samples = progress_iterations(sampler.get_sample, sample_size)
    x = sampler.convert_to_full_representation(samples)
    y = tester.get_result(samples)

    sampler.reset()

    test = progress_iterations(sampler.get_sample, test_size, quiet=True)
    x_test = sampler.convert_to_full_representation(test)
    y_test = tester.get_result(test)

    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    y_predicted = clf.predict(x_test)

    plot_linear_regression([x for x, idx in enumerate(x_test)], y_test, y_predicted, name)
