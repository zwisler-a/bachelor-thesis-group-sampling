from z3 import z3

def find_true_optional_features(vm):
    optionals = []
    constrains = vm.create_z3_constrains()
    for feature in vm.get_features():
        solver = z3.Solver()
        solver.add(constrains)

        # Add constraint to disable a feature
        solver.add(z3.Not(z3.Bool(feature)))

        if solver.check() == z3.sat:
            # If it is satisfiable, it is 
            # a true optional feature
            optionals.append(feature)

    return optionals