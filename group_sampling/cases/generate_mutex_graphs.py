import json
import logging
import time

from networkx.readwrite import json_graph

from model.datasets import Datasets
from util.network_util import generate_mutex_graph, generate_mutex_graph_squential

logging.getLogger().setLevel(logging.DEBUG)

for name, vm, test_strategy in Datasets().generate_testsuite(Datasets.REAL + Datasets.SYN_PRE):
    logging.info(f"Generate graph for {name}")
    t0 = time.time()
    mg = generate_mutex_graph_squential(vm)

    data = (json_graph.node_link_data(mg))
    with open(f'../result/tmp/{name}.json', 'w') as f:
        f.write(json.dumps(data))
        f.flush()
        f.close()
    logging.info(f"Done with {name}!")
