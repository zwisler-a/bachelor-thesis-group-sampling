import logging
import time
import warnings
from typing import List

import numpy as np
import pandas
from numba import jit
from sklearn import linear_model

from model.variability_model import VariabilityModel
from util.util import flatten

warnings.simplefilter(action='ignore', category=FutureWarning)


class GroupSamplingModel:
    '''
        Distribution  Result  Features      Feature Influence/Sensitivity
        1 0 0         10      1 0 1 1 0 1   10   None 10   10   None 10
        0 1 0         11      1 1 1 0 0 1   11   11   11   None None 11
        0 0 1         12      1 0 0 1 1 1   12   None None 12   12   12
        1 0 0         12      1 1 0 0 1 1   12   12   None None 12   12
        [...]
        ---------------------------------------------------------------
                              Average:      11   11   10,5 11   12   11
                                            ---------------------------
                     Most influential:                          12
    '''

    def __init__(self, vm: VariabilityModel, group_size):
        self.log = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.group_size = group_size
        self.vm = vm
        self.df_shift = 0

    def get_group_sensitivity_values(self, x: List[List[List[int]]], y: List[List[int]]):
        grouping_influences = pandas.DataFrame()
        # Iterate over each grouping which got measured
        for idx, groupings in enumerate(x):
            # We create an identity matrix to fit our
            # linear regression on.
            distribution = np.identity(self.group_size)
            group_result = pandas.DataFrame()
            # translate the normal distribution to the actual features groupings
            # and create the feature sensitivity dataframe
            for key, dist in enumerate(distribution):
                group = groupings[key]
                influence = y[idx][key]
                expanded = [[influence if feature == 1 else None for feature in group]]
                group_result = group_result.append(expanded)

            grouping_influences = grouping_influences.append(
                group_result.astype("float64"),
                ignore_index=True
            )

        groupings = pandas.DataFrame(flatten(x))
        grouping_influences[0] = np.nan
        groupings[0] = np.nan
        grouping_influences = self.shift_to_mean(grouping_influences)
        return grouping_influences, groupings

    def perform_stepwise_analysis(self, grouping_influences, groupings):
        feature_sensitivity = pandas.DataFrame(columns=groupings.columns)
        # Iterate over each feature to determine the influence
        for i in range(0, len(groupings.columns)):
            sensitivity, index = self.get_outlier(grouping_influences)
            if sensitivity is not None:
                self.log.debug(
                    f"{i + 1}th influential parameter "
                    f"idx: {index} with {sensitivity}"
                )
                # Add the influence we determine for the i-th feature
                # to the result dataframe
                feature_sensitivity.loc[0, index] = sensitivity
                # Remove the influence of the i-th feature from each
                # measurement taken
                grouping_influences = pandas.DataFrame(
                    remove_influence(
                        grouping_influences.to_numpy(),
                        groupings.to_numpy(),
                        index, sensitivity
                    )
                )
                # Remove the feature from the table
                grouping_influences.loc[:, index] = np.NaN
        return feature_sensitivity

    def fit(self, x: List[List[List[int]]], y: List[List[int]]):
        self.log.debug(f"Fitting data")
        t0 = time.time()
        grouping_influences, groupings = self.get_group_sensitivity_values(x, y)
        self.log.debug(f"Determining group sensitivity took {time.time() - t0}.")

        t1 = time.time()
        sensitivity_values = self.perform_stepwise_analysis(
            grouping_influences,
            groupings
        )
        self.log.debug(f"Stepwise analysis took {time.time() - t1}.")

        intercept = sensitivity_values.mean().mean()
        sensitivity_values = sensitivity_values - intercept
        sensitivity_values = sensitivity_values.fillna(0)

        self.model = linear_model.LinearRegression()
        self.model.intercept_ = self.df_shift
        self.model.coef_ = sensitivity_values.to_numpy()
        self.log.info(f"Fitting data took {time.time() - t0}.")

    def shift_to_mean(self, df):
        self.df_shift = df.mean().mean()
        return df - self.df_shift

    def get_outlier(self, grouping_influences):
        '''
        Get the outlier value by shifting the average influence
        value of each feature by the average of all feature averages
        and looking for the biggest absolute value.
        '''
        mean_influences = grouping_influences.mean()
        mean_adjusted_distribution = (mean_influences - mean_influences.mean())
        max_idx = mean_adjusted_distribution.abs().idxmax()
        if max_idx != max_idx:
            return None, []
        max_influence = mean_influences[max_idx]
        return max_influence, max_idx

    def predict(self, x: List[List[int]]):
        return self.model.predict(x)


@jit(nopython=True)  # Speed up function by compiling to native code
def remove_influence(influences, groupings, index, sub):
    """
    Substracts the value "sub" from each feature, which shares the
    same group as the feature "index"
    :param influences: Actual influence values
    :param groupings: Configurations
    :param index: Feature to remove
    :param sub: Influence to remove
    :return:
    """
    for row in range(len(influences)):
        for col in range(len(influences[row])):
            if groupings[row][index] == 1:
                influences[row][col] = influences[row][col] - sub
    return influences
