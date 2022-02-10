import logging
import xml.etree.ElementTree as ET

from model.configuration_option import ConfigurationOption
from model.variability_model import VariabilityModel


def read_file(path_to_file) -> VariabilityModel:
    tree = ET.parse(path_to_file)
    root = tree.getroot()
    vm = VariabilityModel(root.attrib.get("name"))
    for child in root:
        parse_node(child, vm)

    return vm


def parse_node(node: ET.Element, vm: VariabilityModel):
    known_tags = {
        "binaryOptions": parse_binary_options,
        "configurationOption": parse_configuration_option,
        "numericOptions": parse_numeric_options,
        "booleanConstraints": parse_boolean_constrains_node,
        "nonBooleanConstraints": noop,
        "impliedOptions": parse_options_node,
        "excludedOptions": parse_options_node,
        "constraint": parse_constraint_node,
    }
    parser = known_tags.get(node.tag)
    if parser:
        return parser(node, vm)
    else:
        logging.warning(f'Can\'t parse node {node.tag}')


def parse_boolean_constrains_node(node: ET.Element, vm: VariabilityModel):
    constraints = []
    for constraint in node:
        constraints.append(parse_node(constraint, vm))
    vm.booleanConstraints = constraints


def parse_constraint_node(node: ET.Element, vm: VariabilityModel):
    return node.text


def parse_binary_options(node: ET.Element, vm: VariabilityModel):
    binary_options = []
    for child in node:
        binary_options.append(parse_node(child, vm))
    vm.binaryOptions = binary_options


def parse_numeric_options(node: ET.Element, vm: VariabilityModel):
    numeric_options = []
    for child in node:
        numeric_options.append(parse_node(child, vm))
    vm.numericOptions = numeric_options


def noop(node: ET.Element, vm: VariabilityModel):
    logging.debug("Noop")


def parse_options_node(node: ET.Element, vm: VariabilityModel):
    opts = []
    for opt in node:
        opts.append(opt.text.strip())
    return opts


def parse_configuration_option(node: ET.Element, vm: VariabilityModel):
    opt = ConfigurationOption()
    opt.name = next(node.iter("name")).text.strip()
    opt.outputString = next(node.iter("outputString")).text.strip()
    opt.prefix = next(node.iter("prefix")).text.strip()
    opt.postfix = next(node.iter("postfix")).text.strip()
    opt.parent = next(node.iter("parent")).text.strip()
    opt.impliedOptions = parse_node(next(node.iter("impliedOptions")), vm)
    opt.excludedOptions = parse_node(next(node.iter("excludedOptions")), vm)
    opt.optional = next(node.iter("optional")).text == "True"
    return opt
