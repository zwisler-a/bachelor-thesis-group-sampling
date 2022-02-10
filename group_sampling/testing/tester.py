from typing import List

from testing.testing_strategy import TestingStrategy


class Tester:
    def __init__(self, strategy: TestingStrategy):
        self.strategy = strategy

    def get_result(self, configurations: List[List[str]]):
        return [self.strategy.get_result([co for co in c if co != "root"]) for c in configurations]

    def get_random_configuration(self):
        return self.strategy.get_random_configuration()
