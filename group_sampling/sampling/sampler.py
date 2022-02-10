from sampling.sampling_strategy import SamplingStrategy


class Sampler:
    def __init__(self, strategy: SamplingStrategy):
        self.strategy = strategy

    def get_sample(self):
        return self.strategy.get_sample()

    def reset(self):
        self.strategy.reset()

    def convert_to_full_representation(self, samples):
        return self.strategy.convert_to_full_representation(samples)