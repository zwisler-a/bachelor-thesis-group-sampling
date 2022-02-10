from typing import List

import pandas

from testing.testing_strategy import TestingStrategy


class CsvTestingStrategy(TestingStrategy):

    def __init__(self, path_to_file, result_key='performance', result_keys=None):
        self.result_keys = result_keys
        self.df: pandas.DataFrame = pandas.read_csv(path_to_file, delimiter=';')
        self.key = result_key

    def get_result(self, configuration: List[str]):
        configuration_set = set(configuration + ['root']).intersection(self.df.columns)
        contained = [f'{c} == 1' for c in configuration_set]
        missing_set = set(self.df.columns) - configuration_set - self.result_keys
        missing = [f'{c} == 0' for c in missing_set]
        query = ' & '.join(contained + missing)
        try:
            result = self.df.query(query)
            return result.iloc[0][self.key]
        except Exception as i:
            print(f'Cant find measurement for configuration {configuration_set}, {i}')
            return None

    def get_random_configuration(self):
        df = self.df
        row = df.sample().iloc[0]
        sample = []
        for feature in set(self.df.columns) - self.result_keys:
            if row[feature] == 1:
                sample.append(feature)
        return sample
