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
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import itertools

from reportengine.figure import figure, figuregen
from reportengine.checks import make_check, CheckError, make_argcheck, check
from reportengine import collect
from reportengine.table import table
from reportengine.floatformatting import format_number

from validphys import plotutils

from validphys.fitdata import replica_paths
from validphys.fitdata import read_bsm_facs

log = logging.getLogger(__name__)


"""
Format routines
---------------
"""
def display_format(series):
    """
    Determines the format of the BSM factors
    to be displayed in the tables
    Parameters
    ----------
        series: pd.Series
    """
    return [format_number(x, digits=2) for x in series]

def pass_threshold(value, threshold=0.5):
    return np.abs(value) < threshold

"""
---------------
"""

@figuregen
def plot_nd_bsm_facs(read_bsm_facs):
    """Plot a histogram for each BSM coefficient.
    The nd is used for n-dimensional, if two BSM facs 
    are present: use instead :py:func:`validphys.results.plot_2d_bsm_facs`
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
def plot_nd_bsm_facs_fits(fits):
    """
    Compare histograms of BSM factors between different fits 
    in SIMUnet
    """
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.append(bsm_fac_ops)
    # Remove repeated operators
    all_ops = {o for fit_ops in all_ops for o in fit_ops} 
    
    # plot all operators 
    for op in all_ops:
        fig, ax = plt.subplots()
        for fit in fits:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            if bsm_facs_df.get([op]) is not None:
                ax.hist(bsm_facs_df.get([op]).values, alpha=0.5, label=fit.name)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                ax.set_title(f"Distribution for {op} coefficient")
                ax.set_ylabel("Count")
                ax.set_xlabel(op)
                ax.legend()
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
        f"`bsm_fac_data` for fit '{fit}', but {l} found.",
    )

@figure
#@_check_two_bsm_facs
def plot_2d_bsm_facs(read_bsm_facs, replica_data):
    """
    Plot two dimensional distributions of the BSM coefficient
    results
    """
    bsm_facs_df = read_bsm_facs
    labels = bsm_facs_df.columns

    chi2 = [info.chi2 for info in replica_data]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    chi2 = [info.chi2 for info in replica_data]

    scatter_plot = ax.scatter(
        bsm_facs_df.iloc[:, 0], bsm_facs_df.iloc[:, 1], c=chi2
    )

    # create new axes to the bottom of the scatter plot
    # for the colourbar 
    divider = make_axes_locatable(ax)
    # the width of the colorbar can be changed with `size`
    cax = divider.append_axes("bottom", size="8%", pad=0.7)
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
    ax_histx.hist(bsm_facs_df.iloc[:, 0])
    ax_histy.hist(bsm_facs_df.iloc[:, 1], orientation='horizontal')

    ax_histx.grid(False)
    ax_histy.grid(False)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_axisbelow(True)

    return fig

def _select_plot_2d_bsm_facs(read_bsm_facs, replica_data, pair):
    """
    Auxiliary function to plot 2D plots
    of pair of operators in a N-dimensional fits
    with BSM factors
    """
    op_1, op_2 = pair
    bsm_facs_df = read_bsm_facs
    bsm_facs_df = bsm_facs_df[[op_1, op_2]]
    labels = bsm_facs_df.columns

    chi2 = [info.chi2 for info in replica_data]

    # we use this figsize to have a square scatter plot
    # smaller values do not display too well
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    chi2 = [info.chi2 for info in replica_data]

    scatter_plot = ax.scatter(
        bsm_facs_df.iloc[:, 0], bsm_facs_df.iloc[:, 1], c=chi2, s=40
    )

    # create new axes to the bottom of the scatter plot
    # for the colourbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="9%", pad=0.7)
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
    ax_histx.hist(bsm_facs_df.iloc[:, 0])
    ax_histy.hist(bsm_facs_df.iloc[:, 1], orientation='horizontal')

    ax_histx.grid(False)
    ax_histy.grid(False)

    ax.set_xlabel(labels[0], fontsize=15)
    ax.set_ylabel(labels[1], fontsize=15)

    ax.set_axisbelow(True)

    return fig

@figuregen
def plot_bsm_2d_combs(read_bsm_facs, replica_data):
    """
    Plot two dimensional distributions for all pairs
    of BSM coefficients in a fit
    Parameters
    ----------
        read_bsm_facs: pd.Dataframe 
        replica_data : list
    """
    bsm_facs_df = read_bsm_facs
    labels = bsm_facs_df.columns 

    combs = itertools.combinations(labels, 2)
    for comb in combs:
        fig = _select_plot_2d_bsm_facs(bsm_facs_df, replica_data, pair=comb)
        yield fig 

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
    """
    Table generator to summarise information about
    the BSM coefficient results.
    Paramaters
    ----------
        read_bsm_facs: pd.Dataframe
    The returned table contains information about the mean
    and standard deviation of the BSM coefficients in the fit, 
    as well as showing the 68% (95%) confidence level by 
    computing mean ± std (mean ± 2*std).
    """ 
    bsm_facs_df = read_bsm_facs

    # Get the numbers from the dataframe
    means = bsm_facs_df.mean()
    stds = bsm_facs_df.std()
    
    cl68_lower, cl68_upper = (means - stds, means + stds)
    cl95_lower, cl95_upper = (means - 2 * stds, means + 2 * stds)

    # Format the numbers to display 
    means_disp = display_format(means) 
    stds_disp = display_format(stds) 
    
    cl68_lower_disp = display_format(cl68_lower)
    cl68_upper_disp = display_format(cl68_upper) 

    cl95_lower_disp = display_format(cl95_lower)
    cl95_upper_disp = display_format(cl95_upper) 
    
    # fill the dataframe
    df = pd.DataFrame(index=bsm_facs_df.columns)
    df['68% CL bounds'] = list(zip(cl68_lower_disp, cl68_upper_disp))
    df['95% CL bounds'] = list(zip(cl95_lower_disp, cl95_upper_disp))
    df['Mean'] = means_disp
    df['Std'] = stds_disp
    
    return df

@figure
def plot_bsm_corr(read_bsm_facs):
    """
    Correlation matrix to summarise information about
    the BSM coefficient results.
    Paramaters
    ----------
        read_bsm_facs: pd.Dataframe
    """

    # figsize (11, 9) has good proportions
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    # set background colour
    ax.set_facecolor("0.9")

    # read dataframe and round numbers
    bsm_facs_df = read_bsm_facs
    corr_mat = bsm_facs_df.corr()
    round(corr_mat, 1)

    # Generate a mask
    mask = pass_threshold(corr_mat)

    # create new colourmap
    # https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
    # select cmap and define number of colours to display
    n_colours = 256
    viridis = plt.get_cmap('viridis', n_colours)
    new_colors = viridis(np.linspace(0, 1, n_colours))
    grey = np.array([0.9 * 256/n_colours, 0.9 * 256/n_colours, 0.9 * 256/n_colours, 1])
    new_colors[64:192, :] = grey
    new_cmap = ListedColormap(new_colors)

    # formatting
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')

    # create heatmap
    ax = sns.heatmap(corr_mat, mask=mask,
    vmin=-1.0, vmax=1.0, linewidths=.5, square=True, cmap=new_cmap);

    return fig

@figuregen
def plot_2d_bsm_facs_fits(fits):
    """
    Compare histograms of BSM factors between different fits
    in SIMUnet
    """
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.append(bsm_fac_ops)
    # Remove repeated operators
    all_ops = {o for fit_ops in all_ops for o in fit_ops}
    # get all pairs
    pairs = itertools.combinations(all_ops, 2)
    # plot all pairs of operators
    for pair in pairs:
        op_1, op_2 = pair
        # use this size to keep them sqaure
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.ticklabel_format(
            axis='both', scilimits=(0, 0), style='sci', useOffset=True
        )

        divider = make_axes_locatable(ax)
        # append axes to the top and to the right for the histograms
        ax_histx = divider.append_axes("top", 0.5, pad=0.5, sharex=ax)
        ax_histy = divider.append_axes("right", 0.5, pad=0.3, sharey=ax)

        # Make some labels invisible
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)

        for fit in fits:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            # display the result in the figure only if the fit has the two operators in the pair
            if bsm_facs_df.get([op_1]) is not None and bsm_facs_df.get([op_2]) is not None:
                ax.scatter(
                    bsm_facs_df.get([op_1]), bsm_facs_df.get([op_2]), label=fit.name, alpha=0.5, s=40
                )
                # populate the histograms
                ax_histx.hist(bsm_facs_df.get([op_1]), alpha=0.5)
                ax_histy.hist(bsm_facs_df.get([op_2]), orientation='horizontal', alpha=0.5)

        ax_histx.grid(False)
        ax_histy.grid(False)

        ax.set_xlabel(op_1)
        ax.set_ylabel(op_2)
        ax.legend()
        ax.set_axisbelow(True)

        yield fig

@table
def bsm_facs_bounds_fits(fits, n_sigma):
    """
    Table generator to summarise information about
    the BSM coefficient results.
    Paramaters
    ----------
        fits: NSList of FitSpec 
    The returned table contains information about the mean
    and standard deviation of the BSM coefficients in the fit, 
    as well as showing the confidence levels by 
    computing mean ± n_sigma * std.
    """ 
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.extend(bsm_fac_ops)
    all_ops = list(dict.fromkeys(all_ops))

    fit_names =  [fit.name for fit in fits]
    extra_metrics = ['Best-fit shift', 'Broadening']
    # include extra metrics in columns
    fit_names.extend(extra_metrics)

    # Initialise df 
    df = pd.DataFrame(index=all_ops, columns=fit_names)
    
    # plot all operators 
    for op in all_ops:
        best_fits = []
        bound_lengths = []
        for fit in fits:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            if bsm_facs_df.get([op]) is not None:
                values = bsm_facs_df[op]
                mean =  values.mean()
                std = values.std()
                cl_lower, cl_upper = (mean - n_sigma * std, mean + n_sigma * std)
                lower_dis = format_number(cl_lower, digits=2)
                upper_dis = format_number(cl_upper, digits=2)
                df[fit.name].loc[op] = f"({lower_dis}, {upper_dis})"
                # best-fit value
                best_fits.append(mean)
                # calculate bound length
                length = cl_upper - cl_lower
                bound_lengths.append(length)
            else:
                df[fit.name].loc[op] = 'Not in fit'
                # if the operator is not in the fit, then assume SM
                # for best-fit value
                best_fits.append(0.0)
                bound_lengths.append(0.0)
        # best-fit shift column
        df['Best-fit shift'].loc[op] = format_number(best_fits[0] - best_fits[1], digits=2)
        # broadening column
        curr_len, ref_len = bound_lengths
        if ref_len > 0:
            df['Broadening'].loc[op] = str((curr_len - ref_len) / ref_len * 100.0) + '%'
        else:
            df['Broadening'].loc[op] = 'n/a'

    # formatting columns
    for column in df.columns[:2]:
        if n_sigma == 1:
            df = df.rename(columns={column: f'68% CL - {column}'})
        elif n_sigma == 2:
            df = df.rename(columns={column: f'95% CL - {column}'})

    mapping = {df.columns[0]: '(Current) ' + df.columns[0],
    df.columns[1]: '(Reference) ' + df.columns[1]}

    df = df.rename(columns=mapping)

    return df

@table
def bsm_facs_68bounds_fits(fits):
    """
    Table generator to obtain the 68% CL
    for BSM factors while comparing fits.
    Parameters
    ----------
        fits: NSList of FitSpec 
    """ 
    return bsm_facs_bounds_fits(fits, n_sigma=1)

@table
def bsm_facs_95bounds_fits(fits):
    """
    Table generator to obtain the 95% CL
    for BSM factors while comparing fits.
    Parameters
    ----------
        fits: NSList of FitSpec 
    """ 
    return bsm_facs_bounds_fits(fits, n_sigma=2)

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
