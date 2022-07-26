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