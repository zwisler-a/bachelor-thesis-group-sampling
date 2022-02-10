import math
from abc import abstractmethod
from typing import List

from model.configuration_option import ConfigurationOption


class VariabilityModel:
    name = ""

    @abstractmethod
    def create_z3_constrains(self, shuffle=False):
        pass

    @abstractmethod
    def get_features(self) -> List[str]:
        pass

    @abstractmethod
    def to_dimacs(self):
        pass

    def __str__(self):
        return f'VariabilityModel[name={self.name}]'
