from z3 import ModelRef, z3

from model.variability_model import VariabilityModel
from sampling.sampling_strategy import SamplingStrategy


class GroupSamplingStrategy(SamplingStrategy):

    def __init__(self, vm: VariabilityModel, **opts):
        super().__init__(vm, **opts)
        self.group_size = opts['group_size']

    def get_sample(self):
        pass

    def transform_model(self, model: ModelRef):
        m = []
        for variable in model:
            if z3.is_true(model[variable]):
                m.append(variable.name())
        return m

    def update(self, **opts):
        self.group_size = opts['group_size']

    def convert_to_full_representation(self, samples):
        return [
            [
                [(1 if opt in sample else 0) for opt in self.vm.get_features()]
                for sample in group
            ]
            for group in samples
        ]
