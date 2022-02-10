import json
from functools import reduce
from typing import List, Dict

from testing.testing_strategy import TestingStrategy


class SyntheticTestingStrategy(TestingStrategy):

    def get_random_configuration(self):
        raise Exception('Not possible to create random config from synthetic data tester ...')

    def __init__(self, weights: List[int], index_map: Dict[str, int], interactions: List[List[int]]):
        self.weights = weights
        self.index_map = index_map
        self.interactions = interactions

    def get_result(self, configuration: List[str]):
        performance = 200
        indices = [self.index_map.get(c) for c in configuration]
        interaction_bonus = 0
        for interaction in self.interactions:
            if set(interaction).issubset(set(indices)):
                interaction_bonus += 100
        if len(indices) == 0:
            return performance + reduce(lambda a, b: a + b, self.weights) / len(self.weights)
        return performance + reduce(lambda a, b: a + b, [self.weights[i - 1] * 1 for i in indices]) + interaction_bonus

    def save_to_file(self, path_to_file):
        data = {
            'weights': self.weights,
            'index_map': self.index_map,
            'interactions': self.interactions
        }
        with open(path_to_file, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def from_file(path_to_file):
        with open(path_to_file, 'r') as f:
            data: Dict = json.load(f)
            return SyntheticTestingStrategy(
                data.get('weights'),
                data.get('index_map'),
                data.get('interactions')
            )
