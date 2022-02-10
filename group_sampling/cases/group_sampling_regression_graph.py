import os

from graphing.regression_graphs import plot_linear_regression
from learning.grouped_legacy import Gustav
from model import dimacs_model_reader
from sampling.group_sampling.hamming_group_sampling_strategy import HammingGroupSamplingStrategy
from sampling.random_sampling.better_random_sampling_strategy import BetterRandomSamplingStrategy
from sampling.sampler import Sampler
from testing.cvs_testing_strategy import CsvTestingStrategy
from testing.grouped_tester import GroupedTester
from testing.tester import Tester
from util.progress_loop import progress_iterations

featureModelFile = os.getcwd() + "/../resources/PostgreSQL_pervolution_bin.dimacs"
resultFile = os.getcwd() + "/../resources/PostgreSQL_pervolution_bin_measurements.csv"
vm = dimacs_model_reader.read_file(featureModelFile)
test_strategy = CsvTestingStrategy(resultFile)

sample_size = 10
test_size = 100
group_size = 2

grouped_sampler = Sampler(HammingGroupSamplingStrategy(vm, group_size=group_size))
grouped_samples = progress_iterations(grouped_sampler.get_sample, sample_size // group_size)
random_sampler = Sampler(BetterRandomSamplingStrategy(vm))
grouped_tester = GroupedTester(test_strategy)
tester = Tester(test_strategy)

x_grouped = grouped_sampler.convert_to_full_representation(grouped_samples)
y_grouped = grouped_tester.get_result(grouped_samples)

test = progress_iterations(random_sampler.get_sample, test_size, quiet=True)
x_test = random_sampler.convert_to_full_representation(test)
y_test = tester.get_result(test)

g = Gustav(vm, group_size)
g.fit(x_grouped, y_grouped)
model = g.combine_models()

y_predicted = model.predict(x_test)

print([x for x, idx in enumerate(x_test)], len(x_test))
print(y_test, len(y_test))
print(y_predicted, len(y_predicted))

plot_linear_regression([x for x, idx in enumerate(x_test)], y_test, y_predicted)
