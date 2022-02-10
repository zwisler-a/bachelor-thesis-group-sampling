from typing import List

import numpy as np
import pandas

from sampling.group_sampling.mutex_aware_group_sampling_strategy import IndependentFeatureGroupSamplingStrategy
from testing.tester import Tester
from util.util import flatten


class GroupedTesterSigns(Tester):
    '''
    Group Sampling with optimization introduced by saltelli.
    (
        Does not work with binary options.
        Sign flipping would result in simply changing
        the group the feature is assigned to
    )
    '''
    def __init__(self, strategy, sampling_strategy: IndependentFeatureGroupSamplingStrategy):
        super().__init__(strategy)
        self.independent_features = sampling_strategy.independent_features

    def get_result(self, groups: List[List[List[str]]]):
        signs = pandas.DataFrame(
            flatten([[np.random.randint(2, size=len(self.independent_features)) for j in i] for i in groups])
        )
        if_groups = pandas.DataFrame(
            flatten([
                [
                    [1 if feature in group else 0 for feature in self.independent_features]
                    for group in grouping
                ]
                for grouping in groups
            ])
        )

        flipped_if_groups = if_groups.where(signs == 0, (signs + if_groups) % 2)

        flipped_features = [
            [
                [
                    feature for feature_idx, feature in enumerate(self.independent_features) if
                    flipped_if_groups.iloc[group_idx + grouping_idx, feature_idx]
                ]
                for group_idx, group in enumerate(grouping)
            ]
            for grouping_idx, grouping in enumerate(groups)
        ]

        flipped_groups = [
            [
                (set(c) - set(self.independent_features)).union(flipped_features[co_idx][c_idx])
                for c_idx, c in enumerate(configurations)
            ]
            for co_idx, configurations in enumerate(groups)
        ]

        return [
            [
                self.strategy.get_result([co for co in c if co != "root"])
                for c in configurations
            ]
            for configurations in groups
            #for configurations in flipped_groups
        ]
