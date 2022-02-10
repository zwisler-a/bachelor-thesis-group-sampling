import logging
import time
from math import ceil

import numpy as np
import z3
from z3 import And, Not

from model.variability_model import VariabilityModel
from sampling.group_sampling.group_sampling_strategy import GroupSamplingStrategy
from util.network_util import find_components, load_mutex_graph, \
    generate_mutex_graph_squential, find_optional_features
from util.util import flatten, distribution_with_rest


class IndependentFeatureGroupSamplingStrategy(GroupSamplingStrategy):

    def __init__(self, vm: VariabilityModel, **opts):
        super().__init__(vm, **opts)
        self.log = logging.getLogger(self.__class__.__name__)
        self.last_group = None

        if opts.get('load_mutex'):
            self.log.debug(f'Loading mutex graph: {opts.get("load_mutex")}')
            self.mutex_graph = load_mutex_graph(opts.get('load_mutex'))
        else:
            self.log.debug(f'Generating mutex graph')
            self.mutex_graph = generate_mutex_graph_squential(vm)

        self.mutex_components = find_components(self.mutex_graph)
        self.optionals = find_optional_features(self.vm)
        self.independent_features = self.optionals - set(flatten(self.mutex_components))

        self.solver = z3.Optimize()
        self._add_constrains()
        self.solver.set("timeout", 60000)

        self.log.debug(f'Feature model Information: ------')
        self.log.debug(f'Mutexes: {self.mutex_components}')
        self.log.debug(f'Optional features: {self.optionals}')
        self.log.debug(f'Independent features: {self.independent_features} \n')

    def reset(self):
        """
        Helper function to reset the sampler
        """
        self.solver = z3.Optimize()
        self._add_constrains()
        self.solver.push()

    def _add_constrains(self):
        """
        Add the constraints of the feature model to the solver
        """
        self.solver.add(self.vm.create_z3_constrains(shuffle=True))

    def _independent_features_per_group(self, i):
        """
        Adds the amount of independent features as a constraint
        """
        # We want all independent features to be in a group.
        # This is not always possible with groups of the same size
        # We create a distribution of parameters of equal size with the last group
        # acting as overflow for features which do not fit in another group anymore
        group_sizes = distribution_with_rest(
            len(self.independent_features),
            self.group_size
        )

        sum_independent_features = z3.Sum([
            z3.If(z3.Bool(var), 1, 0)
            for var in self.independent_features
        ])
        # We set the amount of independent features as soft constraint to avoid
        # long runtimes of z3 when optimizing for a cost function
        self.solver.add_soft(sum_independent_features == group_sizes[i], weight=10)

    def _pick_random_mutex_feature(self):
        for idx, mutex in enumerate(self.mutex_components):
            mutex_feature = np.random.choice(mutex)
            self.solver.add(z3.Bool(mutex_feature))

    def _optimize_random_distance(self, i, groups):
        if i == 1:
            self.last_group = (groups[0])
        if i == 0 and self.last_group is not None:
            group_distance_funcs = z3.Sum([
                z3.If(z3.Bool(var.name()), 0, 1)
                if z3.is_true(self.last_group[var]) else 0
                for var in self.last_group
            ])
            max_features = ceil(len(self.independent_features) / self.group_size)

            distance = np.random.randint(0, max_features)
            self.solver.add_soft(group_distance_funcs == distance)

    def _no_overlap_in_independent_features(self, groups):
        if len(groups) != 0:
            distances = [[
                z3.If(z3.Bool(var.name()), 1, 0)
                if var.name() in self.independent_features and z3.is_true(group[var])
                else 0
                for var in group
            ] for group in groups]
            distance_func = z3.Sum(flatten(distances))
            self.solver.add(distance_func == 0)

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

            self._independent_features_per_group(i)

            self._pick_random_mutex_feature()

            self._no_overlap_in_independent_features(groups)

            self._optimize_random_distance(i, groups)

            if self.solver.check() == z3.sat:
                self.log.debug(f"Group generation: {time.time() - t0_group}s")

                model = self.solver.model()
                self.solver.pop()
                self.solver.add([Not(And([v() == model[v] for v in model]))])
                self.solver.push()
                groups.append(model)
            else:
                raise Exception('No more samples can be found!')

        self.log.debug(f"Grouping generation: {time.time() - t0_grouping}s")
        return [self.transform_model(m) for m in groups]
