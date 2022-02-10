from model.variability_model import VariabilityModel
from sampling.sampling_strategy import SamplingStrategy
from testing.tester import Tester


class FromTestsSamplingStrategy(SamplingStrategy):

    def __init__(self, vm: VariabilityModel, tester: Tester, **opts):
        self.tester = tester
        self.vm = vm

    def get_sample(self):
        return self.tester.get_random_configuration()
