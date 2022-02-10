import z3
from z3 import Solver, ModelRef, And, Not, is_true

from model.variability_model import VariabilityModel
from sampling.sampling_strategy import SamplingStrategy


class BetterRandomSamplingStrategy(SamplingStrategy):

    def __init__(self, vm: VariabilityModel, **opts):
        super().__init__(vm, **opts)
        self.solver = None
        self.solutions = []
        self.vm = vm
        self.feature_count = opts.get('feature_count')

    def transform_model(self, model: ModelRef):
        m = []
        for variable in model:
            if is_true(model[variable]):
                m.append(variable.name())
        return m

    def reset(self):
        self.solutions = []

    def get_sample(self):
        self.solver = Solver()
        self.solver.add(self.vm.create_z3_constrains(True))
        for solution in self.solutions:
            self.solver.add(solution)

        if self.feature_count is not None:
            self.solver.add(
                z3.Sum([z3.If(z3.Bool(feature), 1, 0) for feature in self.vm.get_features()]) == self.feature_count
            )

        if self.solver.check() == z3.sat:
            model = self.solver.model()
            self.solutions.append([Not(And([v() == model[v] for v in model]))])
            return self.transform_model(model)
        else:
            print(f"Can't find another solution ...")
            return None
