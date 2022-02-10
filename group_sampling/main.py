import json

import numpy as np
from networkx.readwrite import json_graph

from learning.grouped_legacy import Gustav
from model.datasets import Datasets
from testing.synthetic_testing_strategy import SyntheticTestingStrategy
from util.network_util import generate_mutex_graph
from model.synth_data_factory import SyntheticDataBuilder


def main():
    # vm, testing_strategy = Datasets().get_dataset('berkley')
    sb = SyntheticDataBuilder(
        parameters=50,
        influential_parameters=1,
        mutex_groups=[4]*2,
        mutex_required=[True]*2,
        constrains=2,
        interactions=4
    )
    sb.print_overview()
    vm = sb.get_vm()
    test: SyntheticTestingStrategy = sb.get_testing_strategy()
    test.save_to_file('./resources/syn-50-2-int-test.json')
    with open('./resources/syn-50-2-int.dimacs', 'w') as f:
        f.write(vm.to_dimacs())
        f.flush()
        f.close()

    # mg = generate_mutex_graph(vm)
    #
    # data = (json_graph.node_link_data(mg))
    # with open('resources/syn-500-mutex_graph.json', 'w') as f:
    #     f.write(json.dumps(data))
    #     f.flush()
    #     f.close()

    sample_size = 2
    group_size = 4
    # sampler = Sampler(HammingGroupSamplingStrategy(vm, group_size=8))
    # sampler = Sampler(MutexDistributionGroupSamplingStrategy(vm, group_size=group_size))
    # tester = GroupedTester(testing_strategy)
    # x, y, samples = get_samples_with_results_full(sampler, tester, sample_size)

    g = Gustav(vm, group_size)
    # g.fit(x, y)
    # model = g.combine_models()

    # print(tester.get_result(samples))
    # create_sample_distribution_graph(samples, vm)
    # create_sample_distribution_graph_groupings(samples, vm)
    # featureModelFile = os.getcwd() + "/resources/berkely.dimacs"
    # resultFile = os.getcwd() + "/resources/berkley_measurements.xml"
    # featureModelFile = os.getcwd() + "/resources/PostgreSQL_pervolution_bin.dimacs"
    # resultFile = os.getcwd() + "/resources/PostgreSQL_pervolution_bin_measurements.csv"
    # vm = dimacs_model_reader.read_file(featureModelFile)
    # test_strategy = CsvTestingStrategy(resultFile)


if __name__ == '__main__':
    main()
