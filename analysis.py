"""
Copyright (C) 2018 Shane Steinert-Threlkeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
from __future__ import division, print_function
from collections import defaultdict
import itertools as it
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
mpl.use('TkAgg')

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

import util
import verbs


COLORS = ['xkcd:forest green', 'xkcd:blue green',
          'xkcd:light orange', 'xkcd:peach']


def experiment_analysis(path, verbs, trials=range(60), plots=True,
                        confusion=True):
    """Prints statistical tests and makes plots for experiment one.

    Args:
        path: where the trials in CSV are
        plots: whether to make plots or not
    """

    threshold = 0.93
    # read the data in
    data = util.read_trials_from_csv(path, trials)
    # FILTER OUT TRIALS WHERE RNN DID NOT LEARN
    remove_bad_trials(data, threshold=threshold)
    # get convergence points per quantifier
    convergence_points = get_convergence_points(data, verbs, threshold)
    # TODO: no convergence points for this experiment? just final?
    # TODO: mean over last N=20 training steps?
    final_n = 5
    final_points = {verb: [
        np.mean(data[trial][verb.__name__ + '_accuracy'].values[-final_n:])
        for trial in data] for verb in verbs}

    if confusion:
        conf_mats = defaultdict(dict)
        all_dict = defaultdict(float)
        for stat in ['tp', 'tn', 'fp', 'fn']:
            for verb in verbs:
                name = verb.__name__
                conf_mats[name][stat] = np.mean(
                    [data[trial][name + '_' + stat].values[-1]
                     for trial in trials])
                print([data[trial][name + '_' + stat].values[-1] for trial in
                       trials])
            all_dict[stat] = sum([conf_mats[key][stat]
                                  for key in conf_mats])
        conf_mats['all'] = all_dict
        print(conf_mats)

    if plots:
        """
        # TODO: refactor this into its own method
        reshaped = pd.DataFrame()
        for trial in data:
            for verb in verbs:
                new_data = pd.DataFrame(
                    {'steps': data[trial]['global_step'],
                     'verb': verb.__name__,
                     'accuracy': smooth_data(
                         data[trial][verb.__name__ + '_accuracy'],
                         smooth_weight=0.7),
                     'trial': trial})
                reshaped = reshaped.append(new_data)
        sns.tsplot(reshaped, time='steps', value='accuracy',
                   condition='verb', unit='trial', err_style='unit_traces',
                   estimator=np.median)
        plt.ylim((0.8, 0.96))
        # plt.ylim((0.93, 0.96))
        # plt.xlim((10000, 11200))
        plt.show()
        """

        # make plots
        make_boxplots(convergence_points, verbs)
        make_boxplots(final_points, verbs)
        # make_barplots(convergence_points, verbs)

    gs = gridspec.GridSpec(2, 3)

    ax_acc = plt.subplot(gs[:, :-1])

    make_plot(data, verbs, ylim=(0.8, 0.96), threshold=None,
              inset={'zoom': 3.25,
                     'xlim': (9000, 11200),
                     'ylim': (0.93, 0.9525)},
              ax=ax_acc)

    pairs = list(it.combinations(verbs, 2))
    final_data = {}
    for pair in pairs:
        print('{} vs. {}'.format(pair[0].__name__, pair[1].__name__))
        print(stats.ttest_rel(final_points[pair[0]],
                              final_points[pair[1]]))
        print(stats.ttest_rel(convergence_points[pair[0]],
                              convergence_points[pair[1]]))
        pair_name = '{} - {}'.format(pair[0].__name__, pair[1].__name__)
        final_data[pair_name] = (
            np.array(final_points[pair[0]]) -
            np.array(final_points[pair[1]]))

    ax_dists1 = plt.subplot(gs[0, -1])
    for pair in pairs:
        pair_name = '{} - {}'.format(pair[0].__name__, pair[1].__name__)
        if pair[0].__name__ == 'Know':
            sns.distplot(final_data[pair_name], rug=True,
                         label=pair_name,
                         ax=ax_dists1)
    plt.legend()

    ax_dists2 = plt.subplot(gs[1, -1])
    for pair in pairs:
        pair_name = '{} - {}'.format(pair[0].__name__, pair[1].__name__)
        if pair[0].__name__ == 'BeCertain':
            sns.distplot(final_data[pair_name], rug=True,
                         label=pair_name,
                         ax=ax_dists2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    sns.barplot(data=pd.DataFrame(final_data))
    plt.show()


def remove_bad_trials(data, threshold=0.95):
    """Remove 'bad' trials from a data set.  A trial is bad if the total
    accuracy never converged to a value close to 1.  The bad trials are
    deleted from data, but nothing is returned.
    """
    accuracies = [data[key]['total_accuracy'].values for key in data]
    forward_accs = [forward_means(accs) for accs in accuracies]
    threshold_pos = [first_above_threshold(accs, threshold)
                     for accs in forward_accs]
    # a trial is bad if the forward mean never hit 0.99
    bad_trials = [idx for idx, thresh in enumerate(threshold_pos)
                  if thresh is None]
    print('Number of bad trials: {}'.format(len(bad_trials)))
    for trial in bad_trials:
        del data[trial]


def get_convergence_points(data, verbs, threshold):
    """Get convergence points by quantifier for the data.

    Args:
        data: a dictionary, intended to be made by util.read_trials_from_csv
        quants: list of quantifier names

    Returns:
        a dictionary, with keys the quantifier names, and values the list of
        the step at which accuracy on that quantifier converged on each trial.
    """
    convergence_points = {q: [] for q in verbs}
    for trial in data.keys():
        for verb in verbs:
            convergence_points[verb].append(
                data[trial]['global_step'][
                    convergence_point(
                        data[trial][verb.__name__ + '_accuracy'].values,
                        threshold)])
    return convergence_points


def diff(ls1, ls2):
    """List difference function.

    Args:
        ls1: first list
        ls2: second list

    Returns:
        pointwise difference ls1 - ls2
    """
    assert len(ls1) == len(ls2)
    return [ls1[i] - ls2[i] for i in range(len(ls1))]


def forward_means(arr, window_size=100):
    """Get the forward means of a list. The forward mean at index i is
    the sum of all the elements from i until i+window_size, divided
    by the number of such elements. If there are not window_size elements
    after index i, the forward mean is the mean of all elements from i
    until the end of the list.

    Args:
        arr: the list to get means of
        window_size: the size of the forward window for the mean

    Returns:
        a list, of same length as arr, with the forward means
    """
    return [(sum(arr[idx:min(idx+window_size, len(arr))])
             / min(window_size, len(arr)-idx))
            for idx in range(len(arr))]


def first_above_threshold(arr, threshold):
    """Return the point at which a list value is above a threshold.

    Args:
        arr: the list
        threshold: the threshold

    Returns:
        the first i such that arr[i] > threshold, or None if there is not one
    """
    means = forward_means(arr)
    for idx in range(len(arr)):
        if arr[idx] > threshold and means[idx] > threshold:
        # if means[idx] > threshold:
            return idx
    return None


def convergence_point(arr, threshold=0.95):
    """Get the point at which a list converges above a threshold.

    Args:
        arr: the list
        threshold: the threshold

    Returns:
        the first i such that forward_means(arr)[i] is above threshold
    """
    return first_above_threshold(arr, threshold)


def get_max_steps(data):
    """Gets the longest `global_step` column from a data set.

    Args:
        data: a dictionary, whose values are pandas.DataFrame, which have a
        column named `global_step`

    Returns:
        the values for the longest `global_step` column in data
    """
    max_val = None
    max_len = 0
    for key in data.keys():
        new_len = len(data[key]['global_step'].values)
        if new_len > max_len:
            max_len = new_len
            max_val = data[key]['global_step'].values
    return max_val


def make_plot(data, verbs, ylim=None, xlim=None, threshold=None, loc=2,
              inset=None, ax=None):
    """Makes a line plot of the accuracy of trials by quantifier, color coded,
    and with the medians also plotted.

    Args:
        data: the data
        quants: list of quantifier names
        ylim: y-axis boundaries
    """
    assert len(verbs) <= len(COLORS)

    if ax is None:
        _, ax = plt.subplots()

    trials_by_verb = [[] for _ in range(len(verbs))]
    for trial in data:
        steps = data[trial]['global_step'].values
        for idx in range(len(verbs)):
            trials_by_verb[idx].append(smooth_data(
                data[trial][verbs[idx].__name__ + '_accuracy'].values))
            ax.plot(steps, trials_by_verb[idx][-1],
                     COLORS[idx], alpha=0.2)

    # plot median lines
    medians_by_verb = [get_median_diff_lengths(trials_by_verb[idx])
                       for idx in range(len(trials_by_verb))]
    # get x-axis of longest trial
    longest_x = get_max_steps(data)
    for idx in range(len(verbs)):
        ax.plot(longest_x,
                medians_by_verb[idx],
                COLORS[idx],
                label=verbs[idx].__name__,
                linewidth=2.75)

    if threshold:
        max_x = max([len(ls) for ls in medians_by_verb])
        ax.plot(longest_x, [threshold for _ in range(max_x)],
                linestyle='dashed', color='grey', alpha=0.5)

    if ylim:
        ax.set_ylim(ylim)

    if xlim:
        # _, xmax = plt.xlim()
        ax.set_xlim(xlim)

    if loc:
        ax.legend(loc=loc)

    if inset:
        axins = zoomed_inset_axes(ax, inset['zoom'], loc=4)
        for trial in data:
            steps = data[trial]['global_step'].values
            for idx in range(len(verbs)):
                axins.plot(steps, trials_by_verb[idx][trial],
                           COLORS[idx], alpha=0.25)
        for idx in range(len(verbs)):
            axins.plot(longest_x,
                     medians_by_verb[idx],
                     COLORS[idx],
                     label=verbs[idx].__name__,
                     linewidth=2.5)
        axins.set_xlim(inset['xlim'])
        axins.set_ylim(inset['ylim'])
        axins.set_xticks([])
        axins.set_yticks([])

    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")


def get_median_diff_lengths(trials):
    """Get the point-wise median of a list of lists of possibly
    different lengths.

    Args:
        trials: a list of lists, corresponding to trials

    Returns:
        a list, of the same length as the longest list in trials,
        where the list at index i contains the median of all of the
        lists in trials that are at least i long
    """
    max_len = np.max([len(trial) for trial in trials])
    # pad trials with NaN values to length of longest trial
    trials = np.asarray(
        [np.pad(trial, (0, max_len - len(trial)),
                'constant', constant_values=np.nan)
         for trial in trials])
    return np.nanmedian(trials, axis=0)


def make_boxplots(convergence_points, verbs):
    """Makes box plots of some data.

    Args:
        convergence_points: dictionary of quantifier convergence points
        quants: names of quantifiers
    """
    plt.boxplot([convergence_points[verb] for verb in verbs])
    plt.xticks(range(1, len(verbs)+1), [verb.__name__ for verb in verbs])
    plt.show()


def make_barplots(convergence_points, quants):
    """Makes bar plots, with confidence intervals, of some data.

    Args:
        convergence_points: dictionary of quantifier convergence points
        quants: names of quantifiers
    """
    pairs = list(it.combinations(quants, 2))
    assert len(pairs) <= len(COLORS)

    diffs = {pair: diff(convergence_points[pair[0]],
                        convergence_points[pair[1]])
             for pair in pairs}
    means = {pair: np.mean(diffs[pair]) for pair in pairs}
    stds = {pair: np.std(diffs[pair]) for pair in pairs}
    intervals = {pair: stats.norm.interval(
        0.95, loc=means[pair],
        scale=stds[pair]/np.sqrt(len(diffs[pair])))
        for pair in pairs}

    # plotting info
    index = np.arange(len(pairs))
    bar_width = 0.75
    # reshape intervals to be fed to pyplot
    yerrs = [[means[pair] - intervals[pair][0] for pair in pairs],
             [intervals[pair][1] - means[pair] for pair in pairs]]

    plt.bar(index, [means[pair] for pair in pairs], bar_width, yerr=yerrs,
            color=[COLORS[idx] for idx in range(len(pairs))],
            ecolor='black', align='center')
    plt.xticks(index, pairs)
    plt.show()


def smooth_data(data, smooth_weight=0.85):
    """Smooths out a series of data which might otherwise be choppy.

    Args:
        data: a line to smooth out
        smooth_weight: between 0 and 1, for 0 being no change and
            1 a flat line.  Higher values are smoother curves.

    Returns:
        a list of the same length as data, containing the smooth version.
    """
    prev = data[0]
    smoothed = []
    for point in data:
        smoothed.append(prev*smooth_weight + point*(1-smooth_weight))
        prev = smoothed[-1]
    return smoothed


if __name__ == '__main__':
    experiment_analysis('../data-with-confusion/', verbs.get_all_verbs(),
                        plots=True)
