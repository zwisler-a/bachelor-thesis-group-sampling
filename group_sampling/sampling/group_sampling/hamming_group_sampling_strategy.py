import logging
import time

import numpy as np
import z3
from z3 import And, Not

from model.variability_model import VariabilityModel
from sampling.group_sampling.group_sampling_strategy import GroupSamplingStrategy
from util.util import flatten


class HammingGroupSamplingStrategy(GroupSamplingStrategy):

    def __init__(self, vm: VariabilityModel, **opts):
        self.log = logging.getLogger(self.__class__.__name__)
        self.last_group = None
        super().__init__(vm, **opts)
        self.all_first_groupings = []
        self.solutions = []

        self.solver = z3.Optimize()
        self.solver.set("timeout", 60000)
        self.solver.add(self.vm.create_z3_constrains(shuffle=True))
        self.solver.push()

    def reset(self):
        """
        Helper function to reset the sampler.
        """
        self.solver = z3.Optimize()
        self.solver.add(self.vm.create_z3_constrains(shuffle=True))
        self.solver.push()

    def _optimize_random_distance(self, i, groups):
        if i == 1:
            self.last_group = (groups[0])
        if i == 0 and self.last_group is not None:
            group_distance_funcs = z3.Sum([
                z3.If(z3.Bool(var.name()), 0, 1)
                if z3.is_true(self.last_group[var]) else 0
                for var in self.last_group
            ])
            max_features = len(self.vm.get_features()) // self.group_size
            distance = np.random.randint(0, max_features)
            self.solver.add_soft(group_distance_funcs == distance)

    def _set_preferred_group_size(self):
        # Set group sizes
        feature_per_group = len(self.vm.get_features()) // self.group_size
        sum_features = z3.Sum([
            z3.If(z3.Bool(feature), 1, 0)
            for feature in self.vm.get_features()
        ])
        # We set the group size as a soft constraint with a high weighting
        # This is done because z3 sometimes tries too hard to find a solution
        # fitting the constraint and the hamming distance and takes way too
        # long to return a result. If the soft constraint is set, it, in theory,
        # can break the constraint which lets it better compute an optimal
        # solution for the hamming distance. Breaking the constraint very rarely happens.
        self.solver.add_soft(sum_features == feature_per_group, weight=10)

    def _add_hamming_cost_function(self, groups):
        # If no other groups are created, we can't optimize the distance
        if len(groups) == 0:
            return

        distance_per_feature_per_group = [
            [
                z3.If(z3.And(z3.Bool(var.name()), z3.is_true(group[var])), 0, 1)
                for var in group
            ]
            for group in groups
        ]
        distance_all_features_all_groups = flatten(distance_per_feature_per_group)
        hamming_cost_function = z3.Sum(distance_all_features_all_groups)
        self.solver.maximize(hamming_cost_function)

    def get_sample(self):
        groups = []
        self.log.debug(f"Get sample with groups size {self.group_size}")
        # We want to keep track, how long the generation of a group takes
        t0_grouping = time.time()

        # Save the current state of the constraints in the solver
        self.solver.push()
        for i in range(0, self.group_size):
            self.log.debug(f"Searching for group {i}")
            # Keep track of when the sampling for the group started
            t0_group = time.time()
            self._set_preferred_group_size()

            self._optimize_random_distance(i, groups)

            self._add_hamming_cost_function(groups)

            if self.solver.check() == z3.sat:
                self.log.debug(f" Group generation {time.time() - t0_group}s")
                # We retrieve the solution Z3 found
                model = self.solver.model()
                # Restore the constraint state of the solver before group generation
                self.solver.pop()
                # Add found model to the constraints to avoid getting the same model again
                self.solver.add([Not(And([v() == model[v] for v in model]))])
                # Save the current state of the solver so that the found model is kept as
                # constraint for the next run
                self.solver.push()
                groups.append(model)
            else:
                raise Exception('No more samples can be found!')
        self.log.debug(f"Grouping generation: {time.time() - t0_grouping}s")
        return [self.transform_model(m) for m in groups]
