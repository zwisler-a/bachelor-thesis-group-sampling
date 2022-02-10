import logging

import numpy as np
import z3
from z3 import ModelRef, And, Not, is_true

from model.variability_model import VariabilityModel
from sampling.sampling_strategy import SamplingStrategy


class DistanceSamplingStrategy(SamplingStrategy):
    '''
    Distance-Based Sampling of Software Configuration Spaces
    https://www.se.cs.uni-saarland.de/publications/docs/KaGrSi+19.pdf
    '''
    def __init__(self, vm: VariabilityModel, **opts):
        super().__init__(vm, **opts)
        self.solver = None
        self.solutions = []
        self.vm = vm
        self.reset()

    def transform_model(self, model: ModelRef):
        m = []
        for variable in model:
            if is_true(model[variable]):
                m.append(variable.name())
        return m

    def reset(self):
        self.solver = z3.Optimize()
        self.solver.add(self.vm.create_z3_constrains(True))
        for solution in self.solutions:
            self.solver.add(solution)

    def get_sample(self):
        self.solver.push()

        distance = np.random.randint(self.opt.get('min', 0), self.opt.get('max', len(self.vm.get_features())))
        logging.debug(f"Distance: {distance}")
        distance_from_origin_func = z3.Sum([
            z3.If(z3.Bool(var), 1, 0)
            for var in self.vm.get_features()
        ])
        self.solver.add_soft(distance_from_origin_func == distance)

        if self.solver.check() == z3.sat:
            model = self.solver.model()
            self.solver.pop()
            self.solver.add([Not(And([v() == model[v] for v in model]))])
            self.solver.push()
            return self.transform_model(model)
        else:
            print(f"Can't find another solution ...")
            return None
