# -*- coding: utf-8 -*-
"""
Plots of relations between data PDFs and fits.
"""
from __future__ import generator_stop

import logging
import itertools
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors, ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats
import pandas as pd

from reportengine.figure import figure, figuregen
from reportengine.checks import make_check, CheckError, make_argcheck, check
from reportengine.floatformatting import format_number
from reportengine import collect
from reportengine.table import table

from validphys.core import MCStats, cut_mask, CutsPolicy
from validphys.results import chi2_stat_labels
from validphys.plotoptions import get_info, kitable, transform_result
from validphys import plotutils
from validphys.utils import sane_groupby_iter, split_ranges, scale_from_grid

log = logging.getLogger(__name__)




@figuregen
def plot_nd_fit_cfactors(read_fit_cfactors):
    """Plot a histogram for each fit_cfactor coefficient.
    The nd is used for n-dimensional, if two fit cfactors
    are present: use instead :py:func:`validphys.results.plot_2d_fit_cfactors`
    """
    for label, column in read_fit_cfactors.iteritems():
        # TODO: surely there is a better way
        if label == 'Cb':
            label = r"$\mathbf{C}_{33}^{D\mu}$"
        fig, ax = plt.subplots()

        ax.hist(column.values)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.set_title(f"Distribution for {label} coefficient")
        ax.set_ylabel("Count")
        ax.set_xlabel(label)
        ax.grid(False)

        yield fig

@figuregen
def plot_kde_fit_cfactors(read_fit_cfactors):
    """
    Plots the kernel estimation density for distributions
    of Wilson coefficients. 
    Parameters
    ----------
        read_fit_cfactors: pd.DataFrame
    """
    for label, column in read_fit_cfactors.iteritems():
        # Initialise Axes instance
        fig, ax = plt.subplots()
        # populate the Axes with the KDE
        ax = plotutils.kde_plot(column.values)

        # Format of the plot
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.set_title(f"KDE for {label} coefficient")
        ax.set_ylabel("Density")
        ax.set_xlabel(label)
        ax.grid(True)

        yield fig


@make_argcheck
def _check_two_fitted_cfactors(fit):
    cf = fit.as_input().get("fit_cfactors", [])
    l = len(cf)
    check(
        l == 2,
        "Exactly two elements are required in "
        f"`fit_cfactors_list` for fit '{fit}', but {l} found.",
    )

@figure
@_check_two_fitted_cfactors
def plot_2d_fit_cfactors(read_fit_cfactors, replica_data):
    """Plot two dimensional distributions of the fit cfactors"""
    labels = read_fit_cfactors.columns
    assert len(labels) == 2

    fig, ax = plt.subplots()

    chi2 = [info.chi2 for info in replica_data]

    scatter_plot = ax.scatter(
        read_fit_cfactors.iloc[:, 0], read_fit_cfactors.iloc[:, 1], c=chi2
    )

    # create new axes to the bottom of the scatter plot
    # for the colourbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="15%", pad=0.7)
    fig.colorbar(scatter_plot, cax=cax, label=r"$\chi^2$", orientation='horizontal')

    # set scientific notation for thei scatter plot
    ax.ticklabel_format(
        axis='both', scilimits=(0, 0), style='sci', useOffset=True
    )

    # append axes to the top and to the right for the histograms 
    ax_histx = divider.append_axes("top", 0.5, pad=0.5, sharex=ax)
    ax_histy = divider.append_axes("right", 0.5, pad=0.3, sharey=ax)

    # Make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    # populate the histograms
    ax_histx.hist(read_fit_cfactors.iloc[:, 0])
    ax_histy.hist(read_fit_cfactors.iloc[:, 1], orientation='horizontal')

    ax_histx.grid(False)
    ax_histy.grid(False)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    return fig

@figure
def plot_chi2_fit_cfactors(read_fit_cfactors, replica_data):
    """
    Generates fitcfactor-chi2 scatter plots for all replicas
    in a fit. 
    """

    chi2 = [info.chi2 for info in replica_data]

    for label, column in read_fit_cfactors.iteritems():

        fig, ax = plt.subplots()

        ax.scatter(
            column, chi2, s=40, alpha=0.8
        )

        # set scientific notation for the scatter plot
        ax.ticklabel_format(
            axis='both', scilimits=(0, 0), style='sci', useOffset=True
        )

        ax.set_xlabel(label)
        ax.set_ylabel(r"$\chi^2$")
        ax.set_axisbelow(True)
        ax.grid(True)

        return fig


@table
def fit_cfactor_results_table(read_fit_cfactors):
    """Table generator to summarise information about
    the fit cfactors.
    The returned table contains information about the mean
    and standard deviation of the fit cfactors, as well as showing the
    68% (95%) confidence level by computing mean ± std (mean ± 2*std).
    """
    # Get the numbers from the dataframe
    means = read_fit_cfactors.mean()
    stds = read_fit_cfactors.std()
    
    cl68_lower, cl68_upper = (means - stds, means + stds)
    cl95_lower, cl95_upper = (means - 2 * stds, means + 2 * stds)

    # Format the numbers to display 
    means_disp = list(map(lambda x: "{:.2e}".format(x) , list(means)))
    stds_disp = list(map(lambda x: "{:.2e}".format(x) , list(stds)))
    
    cl68_lower_disp = list(map(lambda x: "{:.2e}".format(x) , list(cl68_lower)))
    cl68_upper_disp = list(map(lambda x: "{:.2e}".format(x) , list(cl68_upper)))
    
    cl95_lower_disp = list(map(lambda x: "{:.2e}".format(x) , list(cl95_lower)))
    cl95_upper_disp = list(map(lambda x: "{:.2e}".format(x) , list(cl95_upper)))

    # fill the dataframe
    df = pd.DataFrame(index=read_fit_cfactors.columns)
    df['68cl bounds'] = list(zip(cl68_lower_disp, cl68_upper_disp))
    df['95cl bounds'] = list(zip(cl95_lower_disp, cl95_upper_disp))
    df['mean'] = means_disp
    df['std'] = stds_disp
    
    return df