import logging
import os
from abc import abstractmethod
from typing import List, Tuple, Iterator

import numpy as np

from model.variability_model import VariabilityModel
from model import dimacs_model_reader
from testing.cvs_testing_strategy import CsvTestingStrategy
from testing.synthetic_testing_strategy import SyntheticTestingStrategy
from testing.testing_strategy import TestingStrategy
from testing.xml_testing_strategy import XmlTestingStrategy
from model.synth_data_factory import SyntheticDataBuilder

dir_path = os.path.dirname(os.path.realpath(__file__))


class Datasets:
    REAL = ["apache", "postgre", "berkley", "javagc"]
    SYN_INTERACTIONS = ["syn-10-int-pre", "syn-10-2-int-pre", "syn-50-int-pre", "syn-50-2-int-pre"]
    SYN_SMALL = ["syn-20", 'syn-50', "syn-100"]
    SYN_BIG = ["syn-1000", "syn-5000"]
    SYN_PRE = ["syn-50-pre", "syn-100-pre", "syn-500-pre", "syn-1000-pre"]
    ALL = ["apache", "postgre", "berkley", "syn-20", "syn-100", "syn-1000"]

    def __init__(self):
        self.sets = {
            'apache': ApacheDataset(),
            'postgre': PostgreSQLDataset(),
            'berkley': BerkleyDBDataset(),
            'javagc': JavaGCDataset(),
            'syn-20': SynDataset(20, 2, 3, 1, 2),
            'syn-20-simple': SynDataset(20, 0, 0, 2, 2),
            'syn-50': SynDataset(50, 2, 5, 1, 2),
            'syn-500': SynDataset(500, 4, 5, 1, 2),
            'syn-100': SynDataset(100, 4, 5, 1, 2),
            'syn-1000': SynDataset(1000, 20, 50, 2, 1),
            'syn-50-pre': Syn50PreDataset(),
            'syn-100-pre': Syn100PreDataset(),
            'syn-500-pre': Syn500PreDataset(),
            'syn-10-int-pre': Syn10IntPreDataset(),
            'syn-10-2-int-pre': Syn10Int2PreDataset(),
            'syn-50-int-pre': Syn50IntPreDataset(),
            'syn-50-2-int-pre': Syn50Int2PreDataset(),
            'syn-500-pre-simple': Syn500PreDataset(),
            'syn-1000-pre': Syn1000PreDataset(),
            'syn-1000-pre-simple': Syn1000PreDataset(),
            'syn-5000': SynDataset(5000, 30, 50, 13, 4),
        }

    def get_dataset(self, dataset: str) -> Tuple[VariabilityModel, TestingStrategy]:
        ds = self.sets.get(dataset)
        if ds is None:
            raise Exception('Unknown dataset' + dataset)
        return ds.get_vm(), ds.get_testing_start()

    def generate_testsuite(self, ds: List[str]) -> Iterator[Tuple[str, VariabilityModel, TestingStrategy]]:
        return [(name, self.get_dataset(name)[0], self.get_dataset(name)[1]) for name in ds]


class Dataset:
    @abstractmethod
    def get_vm(self):
        pass

    @abstractmethod
    def get_testing_start(self):
        pass


class SynDataset(Dataset):
    def __init__(self, params, mutexes, constrains, influential, interactions):
        self.builder = SyntheticDataBuilder(
            parameters=params,
            influential_parameters=influential,
            mutex_groups=np.random.randint(2, 3 + mutexes, mutexes) if mutexes > 0 else [],
            mutex_required=[True] * mutexes,
            constrains=constrains,
            interactions=interactions
        )

    def get_vm(self):
        return self.builder.get_vm()

    def get_testing_start(self):
        return self.builder.get_testing_strategy()

    def overview(self):
        self.builder.print_overview()


class Syn10IntPreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-10-int.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-10-int-test.json")


class Syn10Int2PreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-10-2-int.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-10-2-int-test.json")


class Syn50IntPreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-50-int.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-50-int-test.json")


class Syn50Int2PreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-50-2-int.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-50-2-int-test.json")


class Syn50PreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-50.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-50-test.json")


class Syn100PreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-100.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-100-test.json")


class Syn500PreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-500.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-500-test.json")


class Syn500PreSimpleDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-500-simple.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-500-simple-test.json")


class Syn1000PreDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-1000.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-1000-test.json")


class Syn1000PreSimpleDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/syn-1000-simple.dimacs")

    def get_testing_start(self):
        return SyntheticTestingStrategy.from_file(dir_path + "/../resources/syn-1000-simple-test.json")


class PostgreSQLDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/PostgreSQL_pervolution_bin.dimacs")

    def get_testing_start(self):
        return CsvTestingStrategy(dir_path + "/../resources/PostgreSQL_pervolution_bin_measurements.csv",
                                  result_key='performance', result_keys={'revision', 'performance', 'cpu'})


class BerkleyDBDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/berkely.dimacs")

    def get_testing_start(self):
        return XmlTestingStrategy(dir_path + "/../resources/berkley_measurements.xml")


class ApacheDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/apache_pervolution_energy_bin.dimacs")

    def get_testing_start(self):
        return CsvTestingStrategy(dir_path + "/../resources/apache_pervolution_energy_bin_measurements.csv",
                                  result_key='performance',
                                  result_keys={'performance', 'cpu', 'benchmark-energy', 'fixed-energy', 'revision'})


class JavaGCDataset(Dataset):

    def get_vm(self):
        return dimacs_model_reader.read_file(dir_path + "/../resources/javagc.dimacs")

    def get_testing_start(self):
        return CsvTestingStrategy(dir_path + "/../resources/javagc_measurements.csv", result_key="performance",
                                  result_keys={"performance"})
