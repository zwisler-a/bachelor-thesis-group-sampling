from abc import abstractmethod

from model.variability_model import VariabilityModel


class SamplingStrategy:
    def __init__(self, vm: VariabilityModel, **opt):
        self.vm = vm
        self.opt = opt

    @abstractmethod
    def get_sample(self):
        pass

    def reset(self):
        pass

    def update(self, **opts):
        pass

    def convert_to_full_representation(self, samples):
        return [[(1 if opt in sample else 0) for opt in self.vm.get_features()] for sample in samples]
