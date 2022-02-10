from typing import List

import pandas
from sklearn import linear_model

from model.variability_model import VariabilityModel


class Gustav:
    '''
    Deprecated
    '''

    def __init__(self, vm: VariabilityModel, group_size):
        self.training_influences = pandas.DataFrame()
        self.training_weights = pandas.DataFrame()
        self.training_intercepts = pandas.Series()
        self.group_size = group_size
        self.vm = vm

    def fit(self, x: List[List[List[int]]], y: List[List[int]]):
        '''
        - for each grouping:
            - create virtual parameter distribution to train linear reg.
            - fit linear reg. model with distribution and y data
            - for each group:
                - add row of features with corresponding lin.reg coef to a df
                - add intercept to a df
            - take mean of all coefs of features in a grouping
            - take mean of all intercepts in a grouping
            - > save in class state

            Distribution  Result  Features      Group coefs
            1 0 0         10      1 0 1 1 0 1   10   None 10   10   None 10
            0 1 0         11      1 1 1 0 0 1   11   11   11   None None 11
            0 0 1         12      1 0 0 1 1 1   12   None None 12   12   12
            Grouping Table row---------------------------------------------
                                                11   11   10,5 11   12   11
                                                ---------------------------
                                                                    12
        :param x:
        :param y:
        :return:
        '''
        grouping_influences = pandas.DataFrame()
        grouping_weights = pandas.DataFrame()
        grouping_intercepts = pandas.DataFrame()
        for idx, groupings in enumerate(x):
            # [[1,0,0],
            #  [0,1,0],
            #  [0,0,1]]
            distribution = [[1 if y == x else 0 for y in range(0, self.group_size)] for x in range(0, self.group_size)]
            # clf = linear_model.Lasso()
            clf = linear_model.LinearRegression()
            clf.fit(distribution, y[idx])

            group_result = pandas.DataFrame()
            group_weights = pandas.DataFrame()
            group_intercept = pandas.DataFrame()
            for key, dist in enumerate(distribution):
                group = groupings[key]
                influence = clf.coef_[key]
                group_intercept = group_intercept.append([clf.intercept_])
                row = pandas.DataFrame([[influence if feature == 1 else None for feature in group]])
                group_weights = group_weights.append([group])
                group_result = group_result.append(row)

            grouping_influences = grouping_influences.append(group_result, ignore_index=True)
            grouping_intercepts = grouping_intercepts.append(group_intercept, ignore_index=True)
            grouping_weights = grouping_weights.append(group_weights, ignore_index=True)

        self.training_influences = grouping_influences
        self.training_weights = grouping_weights
        self.training_intercepts = grouping_intercepts

    def combine_models(self):
        # mean_coef_simple = self.calc_coef_simple()
        mean_coef = self.calc_coef()

        model = linear_model.LinearRegression()
        model.coef_ = mean_coef.to_numpy()
        model.intercept_ = self.training_intercepts.mean().get(0)
        return model

    def calc_coef_simple(self):
        return self.training_influences.mean().fillna(value=0)

    def calc_coef(self):
        coefs = pandas.DataFrame()
        columns = self.training_influences.columns
        # return self.training_influences.mean().fillna(value=0)
        for i in range(0, len(self.vm.get_features())):
            mean_influences = self.training_influences.mean()
            max_idx = mean_influences.abs().idxmax()
            if max_idx != max_idx:
                max_idx = list(mean_influences.to_dict().keys())[0]
            max_influence = mean_influences[max_idx]
            coefs[max_idx] = [max_influence]
            print(max_idx)
            self.training_influences = self.training_influences.where(
                self.training_influences[max_idx] != self.training_influences[max_idx],
                self.training_influences - max_influence)
            if len(self.vm.get_features()) > i:
                self.training_influences = self.training_influences.drop(columns[max_idx], axis=1)

        return coefs.mean().fillna(value=0)
