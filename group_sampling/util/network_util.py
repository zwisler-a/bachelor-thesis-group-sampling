import itertools
import json
import logging
import multiprocessing
import time
from functools import partial

import networkx as nx
from networkx.readwrite import json_graph
from z3 import z3


def find_optional_features(vm):
    optionals = []
    constrains = vm.create_z3_constrains()
    for feature in vm.get_features():
        solver = z3.Solver()
        solver.add(constrains)

        solver.add(z3.Not(z3.Bool(feature)))

        if solver.check() == z3.sat:
            optionals.append(feature)

    return set(optionals)


def color_mutex_graph(g):
    groups = {}
    for feature, color in nx.coloring.greedy_color(g).items():
        if groups.get(color, None) is None:
            groups[color] = []
        groups[color].append(feature)
    return [val for val in groups.values()]


def load_mutex_graph(file):
    with open(file, 'r') as f:
        return json_graph.node_link_graph(json.load(f))


def save_mutex_graph(graph, file):
    with open(file, 'w') as f:
        data = json_graph.node_link_data(graph)
        json.dump(data, f)


def generate_mutex_graph(vm):
    logging.debug("Start generating mutex graph in parallel")
    t0 = time.time()

    mutex_graph = nx.Graph()

    pool = multiprocessing.Pool()
    result = pool.starmap(
        partial(check_if_mutex, vm=vm),
        itertools.combinations(vm.get_features(), 2)
    )
    result = filter(lambda r: r is not None, result)
    for edge in result:
        i, j = edge
        mutex_graph.add_edge(i, j)
    logging.debug(f'Time {time.time() - t0}')
    return mutex_graph


def generate_mutex_graph_squential(vm):
    logging.debug('Start generating mutex graph sequentially')
    t0 = time.time()

    mutex_graph = nx.Graph()
    result = []
    solver = z3.Solver()
    solver.add(vm.create_z3_constrains())
    for i, j in itertools.combinations(vm.get_features(), 2):
        constraint_i = z3.And([
            z3.Bool(i), z3.Bool(j)
        ])
        solver.push()
        solver.add(constraint_i)
        if solver.check() == z3.unsat:
            result.append([i, j])
        solver.pop()

    for edge in result:
        i, j = edge
        mutex_graph.add_edge(i, j)
    logging.debug(f'Time {time.time() - t0}')
    return mutex_graph


def find_components(graph):
    components = []
    for clique in nx.connected_components(graph):
        components.append(list(clique))
    return components


def find_cliques(graph):
    components = []
    for clique in nx.connected_components(graph):
        components.append(list(clique))
    return components


def check_if_mutex(i, j, vm):
    solver = z3.Solver()
    solver.add(vm.create_z3_constrains())
    constraint_i = z3.And([
        z3.Bool(i), z3.Bool(j)
    ])
    solver.add(constraint_i)

    if solver.check() == z3.unsat:
        return i, j
