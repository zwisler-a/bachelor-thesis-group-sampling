from abc import abstractmethod
from typing import List


class TestingStrategy:

    @abstractmethod
    def get_result(self, configuration: List[str]):
        pass

    @abstractmethod
    def get_random_configuration(self):
        pass
