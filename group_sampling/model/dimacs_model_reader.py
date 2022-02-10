from model.cnf_variability_model import CnfVariabilityModel
from model.variability_model import VariabilityModel


def read_file(path_to_file) -> VariabilityModel:
    '''
    https://github.com/smba/pcosa/blob/main/pycosa/modeling.py
    :param path_to_file:
    :return:
    '''
    dimacs = list()
    dimacs.append(list())
    with open(path_to_file) as mfile:
        lines = list(mfile)

        # parse names of features from DIMACS comments (lines starting with c)
        feature_lines = list(filter(lambda s: s.startswith("c"), lines))
        index_map = dict(
            map(
                lambda l: (int(l.split(" ")[1]), l.split(" ")[2].replace("\n", "")),
                feature_lines,
            )
        )

        index_map = {idx: index_map[idx] for idx in index_map}

        feature_map = {index_map[v]: v for v in index_map}
        # remove comments
        lines = list(filter(lambda s: not s.startswith("c"), lines))

        for line in lines:
            tokens = line.split()
            if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                for tok in tokens:
                    lit = int(tok)
                    if lit == 0:
                        dimacs.append(list())
                    else:
                        dimacs[-1].append(lit)
        assert len(dimacs[-1]) == 0
        dimacs.pop()

    vm = CnfVariabilityModel()
    vm.clauses = dimacs
    vm.feature_map = index_map
    # print('dimacs')
    # print(dimacs)
    # print('index_map')
    # print(index_map)
    # print('feature_map')
    # print(feature_map)
    return vm
