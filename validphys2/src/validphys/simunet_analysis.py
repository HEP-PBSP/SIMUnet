# -*- coding: utf-8 -*-
"""
Plots and analysis tools for SIMUnet.
"""
from __future__ import generator_stop

import logging

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

from reportengine.figure import figure, figuregen
from reportengine.checks import make_check, CheckError, make_argcheck, check
from reportengine import collect
from reportengine.table import table

from validphys import plotutils

log = logging.getLogger(__name__)


@figuregen
def plot_nd_bsm_facs(read_bsm_facs):
    """Plot a histogram for each BSM coefficient.
    The nd is used for n-dimensional, if two fit cfactors
    are present: use instead :py:func:`validphys.results.plot_2d_fit_cfactors`
    """
    for label, column in read_bsm_facs.iteritems():
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
def plot_kde_bsm_facs(read_bsm_facs):
    """
    Plots the kernel estimation density for a distribution
    of BSM coefficients. 
    Parameters
    ----------
        read_bsm_facs: pd.DataFrame
    """
    for label, column in read_bsm_facs.iteritems():
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
def _check_two_bsm_facs(fit):
    cf = fit.as_input().get("bsm_fac_data", [])
    l = len(cf)
    check(
        l == 2,
        "Exactly two elements are required in "
        f"`fit_cfactors_list` for fit '{fit}', but {l} found.",
    )

@figure
@_check_two_bsm_facs
def plot_2d_bsm_facs(read_bsm_facs, replica_data):
    """Plot two dimensional distributions of the fit cfactors"""
    labels = read_bsm_facs.columns
    assert len(labels) == 2

    fig, ax = plt.subplots()

    chi2 = [info.chi2 for info in replica_data]

    scatter_plot = ax.scatter(
        read_bsm_facs.iloc[:, 0], read_bsm_facs.iloc[:, 1], c=chi2
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
    ax_histx.hist(read_bsm_facs.iloc[:, 0])
    ax_histy.hist(read_bsm_facs.iloc[:, 1], orientation='horizontal')

    ax_histx.grid(False)
    ax_histy.grid(False)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    return fig

@figure
def plot_chi2_bsm_facs(read_bsm_facs, replica_data):
    """
    Generates bsm_fac value - chi2 scatter plots for all replicas
    in a fit. 
    """

    chi2 = [info.chi2 for info in replica_data]

    for label, column in read_bsm_facs.iteritems():

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
def bsm_facs_bounds(read_bsm_facs):
    """Table generator to summarise information about
    the fit cfactors.
    The returned table contains information about the mean
    and standard deviation of the BSM coefficients in the fit, 
    as well as showing the 68% (95%) confidence level by 
    computing mean ± std (mean ± 2*std).
    """ 
    # Get the numbers from the dataframe
    means = read_bsm_facs.mean()
    stds = read_bsm_facs.std()
    
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
    df = pd.DataFrame(index=read_bsm_facs.columns)
    df['68cl bounds'] = list(zip(cl68_lower_disp, cl68_upper_disp))
    df['95cl bounds'] = list(zip(cl95_lower_disp, cl95_upper_disp))
    df['mean'] = means_disp
    df['std'] = stds_disp
    
    return df

_read_pdf_cfactors = collect("read_bsm_facs", ("pdffit",))

def read_pdf_cfactors(_read_pdf_cfactors, pdf):
    return _read_pdf_cfactors[0]

def dataset_inputs_scaled_fit_cfactor(data, pdf, read_pdf_cfactors, quad_cfacs):
    """Same as :py:func:`validphys.results.dataset_scaled_fit_cfactor`
    but for a list of dataset inputs.
    """
    res =  np.concatenate(
        [dataset_scaled_fit_cfactor(dataset, pdf, read_pdf_cfactors, quad_cfacs) for dataset in data.datasets]
    )
    return res

def dataset_scaled_fit_cfactor(dataset, pdf, read_pdf_cfactors, quad_cfacs):
    """For each replica of ``pdf``, scale the fit cfactors by
    the best fit value.
    Returns
    -------
    res: np.arrays
        An ``ndat`` x ``nrep`` array containing the scaled fit cfactors.
    """
    parsed_cfacs = parse_fit_cfac(dataset.fit_cfac, dataset.cuts)
    if parsed_cfacs is None or not read_pdf_cfactors.values.size:
        # We want an array of ones that ndata x nrep
        # where ndata is the number of post cut datapoints
        ndata = len(dataset.load().get_cv())
        nrep = len(pdf) - 1
        return np.ones((ndata, nrep))
    log.debug("Scaling results using linear cfactors")
    fit_cfac_df = pd.DataFrame(
        {k: v.central_value.squeeze() for k, v in parsed_cfacs.items()}
    )
    scaled_replicas = read_pdf_cfactors.values * fit_cfac_df.values[:, np.newaxis]
    if quad_cfacs:
        log.debug("Scaling results using quadratic cfactors")
        parsed_quads = parse_quad_cfacs(dataset.fit_cfac, dataset.cuts, quad_cfacs)
        quad_cfac_df = pd.DataFrame(
            {k: v.central_value.squeeze() for k, v in parsed_quads.items()}
        )
        scaled_replicas += (read_pdf_cfactors.values**2) * quad_cfac_df.values[:, np.newaxis]

    return 1 + np.sum(scaled_replicas, axis=2)
