import random
from typing import List

import numpy as np

from model.cnf_variability_model import CnfVariabilityModel
from model.variability_model import VariabilityModel
from testing.synthetic_testing_strategy import SyntheticTestingStrategy
from testing.testing_strategy import TestingStrategy
from util.util import flatten


class SyntheticDataBuilder:

    def __init__(self, **kwargs):
        #random.seed(kwargs.get('seed', 491595484-10))
        #np.random.seed(kwargs.get('seed', 491595484-10))

        self.parameters = kwargs.get('parameters', 100)
        self.influential_parameters = kwargs.get('influential_parameters', 5)
        self.constrains = kwargs.get('constrains', 150)
        self.constrains_complexity = kwargs.get('constrain_complexity', 4)
        self.mutex_groups = kwargs.get('mutex_groups', [4])
        self.mutex_required = kwargs.get('mutex_required', [True])
        self.interactions = kwargs.get('interactions', 1)

        self.mutexes = []
        self.feature_map = {}
        self.constrain_clauses = []
        self.influential_features = []
        self.interacting_features = []
        self.weights = []
        self.vm = self.generate_vm()

    def generate_mutex_group(self, start, size):
        mutex_parent = start
        mutex_clauses = []
        mutex_all_or_nothing = [-mutex_parent]
        for i in range(mutex_parent + 1, mutex_parent + size + 1):
            mutex_clauses.append([mutex_parent, -i])
            mutex_all_or_nothing.append(i)
            for j in range(i + 1, mutex_parent + size + 1):
                mutex_clauses.append([-i, -j])
        mutex_clauses.append(mutex_all_or_nothing)
        self.mutexes.append([i for i in range(mutex_parent, mutex_parent + size + 1)])
        return mutex_clauses

    def generate_vm(self) -> VariabilityModel:
        index_list = [i for i in range(1, self.parameters + 1)]
        clauses = []

        # Generate Mutex Groups
        mutex_size_total = 0
        for mutex_size in self.mutex_groups:
            clauses = clauses + self.generate_mutex_group(
                len(index_list) - mutex_size_total - mutex_size,
                mutex_size
            )
            mutex_size_total += mutex_size + 1

        features_not_in_mutex = set(index_list) - set(flatten(self.mutexes))

        # Random Constrains
        for i in range(0, self.constrains):
            complexity = random.randint(1, self.constrains_complexity)
            constrain = random.sample(features_not_in_mutex, complexity)
            constrain = [c if i == 0 else -c for i, c in enumerate(constrain)]
            self.constrain_clauses.append(constrain)
            clauses.append(constrain)

        # Make rest of features "root" features
        for feature in features_not_in_mutex:
            clauses.append([1, -feature])

        clauses.append([1])

        # Make mutex parents "root" feature
        for idx, mutex in enumerate(self.mutexes):
            clauses.append([1, -mutex[0]])
            if self.mutex_required[idx] if len(self.mutex_required) > idx else False:
                clauses.append([mutex[0]])

        # Pick influential features
        self.influential_features = random.sample(index_list, self.influential_parameters)

        self.interacting_features = [random.sample(set(index_list) - {1}, 2) for i in range(self.interactions)]

        # Generate weightings
        self.weights = [
            random.randint(-5, 5)
            if param not in self.influential_features else
            random.randint(100, 150)
            for param in index_list
        ]

        self.feature_map = {i: f'Parameter_{i - 1}' if i != 1 else f'root' for i in index_list}

        vm = CnfVariabilityModel()
        vm.name = 'Synthetic Data'
        vm.clauses = clauses
        vm.feature_map = self.feature_map

        return vm

    def get_testing_strategy(self) -> TestingStrategy:
        return SyntheticTestingStrategy(
            self.weights,
            {v: k for k, v in self.feature_map.items()},
            self.interacting_features
        )

    def print_overview(self):
        print('Synthetic data generation:')
        print(f' Generated {self.parameters} Parameters')
        print(f' {len(self.mutexes)} mutex groups')
        for mutex in self.mutexes:
            print(f'  - {mutex}')
        print(f' {self.constrains} Constrains')
        for c in self.constrain_clauses:
            print(f'  - {c}')
        print(f' Influential features: {self.influential_features}')
        print(f' Weightings: {self.weights}')
        print(f' Interacting features:')
        for c in self.interacting_features:
            print(f'  - {c}')

    def get_vm(self):
        return self.vm
