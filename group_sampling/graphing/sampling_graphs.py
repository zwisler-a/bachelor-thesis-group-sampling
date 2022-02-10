from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
import matplotlib as mpl

import seaborn as sns
from model.variability_model import VariabilityModel


def create_sample_distribution_graph(samples: List[List[str]], vm: VariabilityModel):
    data = []
    for sample in samples:
        data.append([(1 if opt in sample else 0) for opt in vm.get_features()])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks(np.arange(0, len(vm.get_features())))
    ax.set_xticklabels(vm.get_features())
    plt.xticks(rotation=90, fontsize=8)

    plt.yticks(range(0, len(samples)))

    bounds = [0., 0.5, 1.]
    cmap = mpl.colors.ListedColormap(['w', 'k'])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(data, interpolation='none', aspect='auto', origin="lower", cmap=cmap, norm=norm)
    plt.tight_layout()
    plt.show()


def create_grouping_influence_graph(x, y, group_size):
    '''
    Display the measurements of each group in a bar plot
    :param x: Configurations/Groups
    :param y: Measurements
    :param group_size: Group size
    '''
    labels = [f'G{i}' for i in range(len(x))]

    # the label locations
    width = 0.75 / group_size  # the width of the bars

    x = np.arange(len(labels))
    fig, ax = plt.subplots()

    for group_idx in range(group_size):
        group = [grouping[group_idx] for grouping in y]
        rect = ax.bar(x + (width * group_idx), group, width, label=f'G{group_idx}')
        ax.bar_label(rect, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    fig.tight_layout()

    plt.show()


def create_sample_distribution_graph_groupings(samples: List[List[List[str]]], vm: VariabilityModel, marked=[]):
    data = []
    y_tick_labels = []
    for sample in samples:
        k = 0
        for group in sample:
            y_tick_labels.append(k)
            k += 1
            data.append([
                0.5 if opt in marked else
                (1 if opt in group else 0)
                for opt in vm.get_features()])

    y_size = len(samples) * len(samples[0])

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.yticks(range(0, y_size), y_tick_labels)
    ax.set_xticks(np.arange(0, len(vm.get_features())))
    ax.set_xticklabels(vm.get_features())
    for mark in marked:
        ax.get_xticklabels()[mark].set_color("red")
    plt.xticks(rotation=90, fontsize=8)

    ax.set_yticks(np.arange(-.5, y_size, len(samples[0])), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    bounds = [0., 0.5, 1.]
    cmap = mpl.colors.ListedColormap(['w', 'red', 'k'])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(data, interpolation='none', aspect='auto', origin="lower", cmap=cmap, norm=norm)
    plt.tight_layout()
    plt.show()



def correlation_graph(csv_file):
    df = pd.read_csv(csv_file, delimiter=";")
    corr = df.corr(method='spearman')

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)

    fig.suptitle('Correlation matrix of features', fontsize=15)
    ax.text(0.77, 0.2, '', fontsize=13, ha='center', va='center',
            transform=ax.transAxes, color='grey', alpha=0.5)

    fig.tight_layout()
    plt.show()
