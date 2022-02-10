import random
from typing import List
import xml.etree.ElementTree as ET

from testing.testing_strategy import TestingStrategy


class XmlTestingStrategy(TestingStrategy):

    def __init__(self, path_to_file, result_idx=1):
        self.result_idx = result_idx
        self.root = ET.parse(path_to_file).getroot()

    def get_result(self, configuration: List[str]):
        configuration_set = set(configuration)
        for result in self.root:
            config: str = result[0].text
            measurement = result[self.result_idx].text

            config_list = [config.strip() for config in config.split(",") if config != '']

            config_set = set(config_list)
            if config_set == configuration_set:
                return float(measurement)

        print(f'Cant find measurement for configuration {configuration_set}')

    def get_random_configuration(self):
        config_idx = random.randint(0, len(self.root))
        config = self.root[config_idx][0].text
        config_list = [config.strip() for config in config.split(",") if config != '']
        config_list.append('root')
        config_set = set(config_list)
        return list(config_set)
