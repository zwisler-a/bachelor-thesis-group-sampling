import itertools
import logging
import time

import networkx as nx
from z3 import z3

def generate_mutually_exclusive_feature_graph(vm):
    logging.debug('Start generating graph')
    # We want to know how long it takes
    t0 = time.time()

    mutex_graph = nx.Graph()
    result = []
    solver = z3.Solver()
    solver.add(vm.create_z3_constrains())

    # Iterate over each possible combination of two features
    for i, j in itertools.combinations(vm.get_features(), 2):
        # Save the current solver constraints
        solver.push()
        # Add a constraint where both are enabled
        constraint_i = z3.And([
            z3.Bool(i), z3.Bool(j)
        ])
        solver.add(constraint_i)

        if solver.check() == z3.unsat:
            # If it is not satisfiable, the 
            # featues are mutually exclusive
            result.append([i, j])
        solver.pop()

    for edge in result:
        i, j = edge
        mutex_graph.add_edge(i, j)

    logging.debug(f'Time {time.time() - t0}')
    return mutex_graph