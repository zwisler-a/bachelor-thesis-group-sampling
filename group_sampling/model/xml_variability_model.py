import math
from typing import List

from z3 import And, Bools, Implies, Or, Not

from model.configuration_option import ConfigurationOption
from model.variability_model import VariabilityModel


class XmlVariabilityModel(VariabilityModel):


    name = ""
    binaryOptions: List[ConfigurationOption] = []
    numericOptions: List[ConfigurationOption] = []
    booleanConstraints = []
    nonBooleanConstrains = []

    def get_binary_configuration(self, name):
        return [opt for opt in self.binaryOptions if opt.name == name]

    def get_numeric_configuration(self, name):
        return [opt for opt in self.numericOptions if opt.name == name]

    def get_configuration_combinations(self):
        return math.factorial(len(self.binaryOptions) + len(self.numericOptions))

    def get_z3_bool(self, opt: ConfigurationOption):
        return Bools(opt.name)[0]

    def __init__(self, name):
        self.name = name

    def transform_boolean_constrain_variable(self, name: str):
        if name[0] == "!":
            var_name = name[1:]
            return Not(self.get_z3_bool(self.get_binary_configuration(var_name)[0]))
        else:
            return self.get_z3_bool(self.get_binary_configuration(name)[0])

    # Todo
    def transform_boolean_constrain(self, constrain: str):
        token = constrain.split(" ")
        if len(token) != 3:
            raise Exception("Constrain must be simple for now!")
        if token[1] == "|":
            return Or([self.transform_boolean_constrain_variable(token[0]),
                       self.transform_boolean_constrain_variable(token[2])])
        elif token[1] == "&":
            return And([self.transform_boolean_constrain_variable(token[0]),
                        self.transform_boolean_constrain_variable(token[2])])
        else:
            raise Exception("Nope")

    def create_z3_constrains(self):
        constrains = []
        for opt in self.binaryOptions:
            if opt.parent is not None and opt.parent != "":
                parent = self.get_binary_configuration(opt.parent)[0]  # TODO
                constrains.append(Implies(self.get_z3_bool(opt), self.get_z3_bool(parent)))
            for excludedOpt in opt.excludedOptions:
                constrains.append(
                    Implies(self.get_z3_bool(opt),
                            Not(self.get_z3_bool(self.get_binary_configuration(excludedOpt)[0]))))

        for constrain in self.booleanConstraints:
            constrains.append(self.transform_boolean_constrain(constrain))

        constrains.append(Or([self.get_z3_bool(opt) for opt in self.binaryOptions]))

        return And(constrains)

    def get_features(self) -> List[str]:
        pass

    def to_dimacs(self):
        pass

    def __str__(self):
        return f'VariabilityModel[name={self.name}, binaryOptions={self.binaryOptions}]'
