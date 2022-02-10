import os

from graphing.sampling_graphs import create_sample_distribution_graph_groupings
from model import dimacs_model_reader
from sampling.group_sampling.hamming_group_sampling_strategy import HammingGroupSamplingStrategy
from sampling.sampler import Sampler
from util.progress_loop import progress_iterations

featureModelFile = os.getcwd() + "/../resources/PostgreSQL_pervolution_bin.dimacs"
resultFile = os.getcwd() + "/../resources/PostgreSQL_pervolution_bin_measurements.csv"
vm = dimacs_model_reader.read_file(featureModelFile)

sample_size = 50
group_size = 4

for i in range(1, 16):
    group_size = i
    grouped_sampler = Sampler(HammingGroupSamplingStrategy(vm, group_size=group_size))
    grouped_samples = progress_iterations(grouped_sampler.get_sample, sample_size // group_size)
    create_sample_distribution_graph_groupings(grouped_samples, vm)


