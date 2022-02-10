import z3
from z3 import Solver, ModelRef, And, Not, is_true

from model.variability_model import VariabilityModel
from sampling.group_sampling.group_sampling_strategy import GroupSamplingStrategy


class NotReallyGroupSamplingStrategy(GroupSamplingStrategy):

    def __init__(self, vm: VariabilityModel, **opts):
        super().__init__(vm, **opts)
        self.solutions = []
        self.solver = Solver()
        self.solver.add(vm.create_z3_constrains())

    def transform_model(self, model: ModelRef):
        m = []
        for variable in model:
            if is_true(model[variable]):
                m.append(variable.name())
        return m

    def get_sample_internal(self):
        self.solver = Solver()
        self.solver.add(self.vm.create_z3_constrains(True))
        for solution in self.solutions:
            self.solver.add(solution)
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            self.solutions.append([Not(And([v() == model[v] for v in model]))])
            return self.transform_model(model)
        else:
            print(f"Can't find another solution ...")
            return None

    def get_sample(self):
        group = []
        for i in range(0, self.group_size):
            group.append(self.get_sample_internal())
        return group
