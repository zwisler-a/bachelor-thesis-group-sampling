import z3
from z3 import Solver, ModelRef, And, Not, is_true

from model.variability_model import VariabilityModel
from sampling.sampling_strategy import SamplingStrategy


class BasicRandomSamplingStrategy(SamplingStrategy):

    def __init__(self, vm: VariabilityModel, **opts):
        super().__init__(vm, **opts)
        self.solver = Solver()
        self.vm = vm
        self.solver.add(vm.create_z3_constrains())

    def transform_model(self, model: ModelRef):
        m = []
        for variable in model:
            if is_true(model[variable]):
                m.append(variable.name())
        return m

    def reset(self):
        self.solver.add(self.vm.create_z3_constrains())

    def get_sample(self):
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            # solution = self.get_solution_model(model)
            # self.solver.add(solution)
            self.solver.add(Not(And([v() == model[v] for v in model])))
            return self.transform_model(model)
        else:
            print(f"Can't find another solution ...")
            return None
