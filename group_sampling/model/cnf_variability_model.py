import random
from abc import abstractmethod
from typing import List, Dict

from z3 import z3

from model.variability_model import VariabilityModel


class CnfVariabilityModel(VariabilityModel):
    clauses: List[List[int]] = []
    feature_map: Dict[int, str] = {}

    @abstractmethod
    def create_z3_constrains(self, shuffle=False):
        bool_clauses = []
        if shuffle:
            for clause in self.clauses:
                random.shuffle(clause)
            random.shuffle(self.clauses)

        for clause in self.clauses:
            bool_clauses.append(z3.Or(
                [
                    z3.Bool(self.feature_map[var])
                    if var > 0 else
                    z3.Not(z3.Bool(self.feature_map[var * -1]))
                    for var in clause
                ]
            ))

        return z3.And(bool_clauses)

    def to_dimacs(self):
        dimacs = ''
        for key, feature in self.feature_map.items():
            dimacs += f'c {key} {feature} \n'
        for clause in self.clauses:
            for v in clause:
                dimacs += f'{v} '
            dimacs += f'0\n'
        return dimacs

    @abstractmethod
    def get_features(self) -> List[str]:
        return [self.feature_map[key] for key in self.feature_map]

    def __str__(self):
        return f'VariabilityModel[name={self.name}, feature_map{self.feature_map}, clauses={self.clauses}]'
