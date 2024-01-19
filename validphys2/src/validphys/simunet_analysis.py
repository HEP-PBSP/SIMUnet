# -*- coding: utf-8 -*-
"""
Plots and analysis tools for SIMUnet.
"""
from __future__ import generator_stop

import logging

import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.colors as colors
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
from validphys.plotutils import grey_centre_cmap
from validphys.pdfbases import PDG_PARTONS
from validphys.utils import split_ranges

from validphys.loader import Loader
from validphys.n3fit_data_utils import parse_simu_parameters_names_CF
from validphys.loader import _get_nnpdf_profile

from validphys.convolution import central_predictions

log = logging.getLogger(__name__)

l = Loader()

"""
Format routines
---------------
"""
def display_format(series):
    """
    Determines the format of the BSM factors to be displayed in the tables.

    Parameters
    ----------
    series : pd.Series
        Series containing the BSM factors.

    Returns
    -------
    list
        A list of formatted numbers representing the BSM factors.
    """
    return [format_number(x, digits=2) for x in series]

BSM_FAC_DISPLAY = ['OtZ', 'OtW', 'OtG', 
'Opt', 'O3pQ3', 'O3pq', 'OpQM', 'OpqMi', 'Opui', 'Opdi', 'O3pl', 'Opl', 'Ope',
'O1qd', 'O1qu', 'O1dt', 'O1qt', 'O1ut', 'O11qq', 'O13qq',
'O8qd', 'O8qu', 'O8dt', 'O8qt', 'O8ut', 'O81qq', 'O83qq',
'OQt8', 'OQQ1', 'OQQ8', 'OQt1', 'Ott1', 'Oeu', 'Olu', 'Oed',
'Olq3', 'Olq1', 'Oqe', 'Old', 'Oll',  'Omup', 'Otap', 'Otp',
'Obp', 'Ocp', 'OG', 'OWWW', 'OpG', 'OpW', 'OpB', 'OpWB', 'Opd', 'OpD',]

def reorder_cols(cols):
    """
    Reorders columns based on predefined BSM factor display order.

    Parameters
    ----------
    cols : list
        List of column names to be reordered.

    Returns
    -------
    list
        Reordered list of columns.
    """
    return sorted(cols, key=BSM_FAC_DISPLAY.index)

def format_residuals(residuals):
    """
    Formats residuals for display, splitting them into intervals.

    Parameters
    ----------
    residuals : list of float
        List of residual values (mean/std) for BSM factors.

    Returns
    -------
    list of list
        List of intervals for display of residuals.
    """
    new_residuals = []
    for residual in residuals:
        if residual >= 0:
            new_residuals.append([0.0, residual])
        else:
            new_residuals.append([residual, 0])
    return new_residuals

"""
---------------
"""

@figuregen
def plot_nd_bsm_facs(read_bsm_facs, bsm_names_to_latex, posterior_plots_settings=None):
    """
    Plot a histogram for each BSM coefficient.
    The nd is used for n-dimensional, if two BSM facs are present: use instead :py:func:`validphys.results.plot_2d_bsm_facs`

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing the BSM factors.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to LaTeX representations.
    posterior_plots_settings : dict, optional
        Settings for posterior plots, such as number of bins and range settings.

    Yields
    ------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for the histogram.
    """
    # extract settings
    if posterior_plots_settings is None:
        n_bins = 10
        rangex = None
        rangey = None
    else:
        try:
            n_bins = posterior_plots_settings["n_bins"]
        except KeyError:
            n_bins = 10
        try:
            rangex = posterior_plots_settings["rangex"]
        except KeyError:
            rangex = None
        try:
            rangey = posterior_plots_settings["rangey"]
        except KeyError:
            rangey = None

    for label, column in read_bsm_facs.items():
        # TODO: surely there is a better way
        fig, ax = plt.subplots()

        ax.hist(column.values, density=True, bins=n_bins)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.set_title(f"Distribution for {bsm_names_to_latex[label]} coefficient")
        ax.set_ylabel("Prob. density", fontsize=14)
        ax.set_xlabel(bsm_names_to_latex[label] + r"$/\Lambda^2$ [TeV$^{-2}$]", fontsize=16)
        ax.grid(False)

        if rangex is not None:
            ax.set_xlim(rangex)
        if rangey is not None:
            ax.set_ylim(rangey)

        yield fig

@figuregen
def plot_nd_bsm_facs_fits(fits, bsm_names_to_latex, posterior_plots_settings=None):
    """
    Compare histograms of BSM factors between different fits in SIMUnet.

    Parameters
    ----------
    fits : NSList
        List of FitSpec to be compared.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to their LaTeX representations.
    posterior_plots_settings : dict, optional
        Dictionary containing settings for posterior plots such as 'n_bins', 'rangex', 'rangey', and 'same_bins'.

    Yields
    ------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for each BSM coefficient comparison.
    """
    # extract settings
    if posterior_plots_settings is None:
        same_bins = False
        n_bins = 10
        rangex = None
        rangey = None
    else:
        try:
            same_bins = posterior_plots_settings["same_bins"]
        except KeyError:
            same_bins = False
        try:
            n_bins = posterior_plots_settings["n_bins"]
        except KeyError:
            n_bins = 10
        try:
            rangex = posterior_plots_settings["rangex"]
        except KeyError:
            rangex = None
        try:
            rangey = posterior_plots_settings["rangey"]
        except KeyError:
            rangey = None

    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.append(bsm_fac_ops)
    # Remove repeated operators
    all_ops = {o for fit_ops in all_ops for o in fit_ops}

    # If same_bins=True, create binnings
    if same_bins:
        min_bins = pd.Series(dict(zip(list(all_ops), np.full(len(all_ops), np.inf))))
        max_bins = pd.Series(dict(zip(list(all_ops), np.full(len(all_ops), -np.inf))))
        for fit in fits:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            min_df = bsm_facs_df.min()
            max_df = bsm_facs_df.max()

            min_bins = pd.concat([min_bins, min_df], axis=1).min(axis=1)
            max_bins = pd.concat([max_bins, max_df], axis=1).max(axis=1)

    # plot all operators 
    for op in all_ops:
        fig, ax = plt.subplots()
        for fit in fits:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)

            if same_bins:
                bins = np.linspace(min_bins.loc[op], max_bins.loc[op], n_bins)
            else:
                bins = n_bins

            if bsm_facs_df.get([op]) is not None:
                ax.hist(bsm_facs_df.get([op]).values, bins=bins, density=True, alpha=0.5, label=fit.label)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                ax.set_ylabel("Prob. density", fontsize=14)
                if bsm_names_to_latex is None:
                    ax.set_xlabel(op, fontsize=14)
                else:
                    ax.set_xlabel(bsm_names_to_latex[op] + r"$/\Lambda^2$ [TeV$^{-2}$]", fontsize=16)
                ax.legend()
                ax.grid(False)
                if rangex is not None:
                    ax.set_xlim(rangex)
                if rangey is not None:
                    ax.set_ylim(rangey)
        yield fig

@figuregen
def plot_kde_bsm_facs(read_bsm_facs, bsm_names_to_latex):
    """
    Plots the kernel estimation density for a distribution of BSM coefficients.

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing the BSM factors. 
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to LaTeX representations.

    Yields
    ------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for the KDE plot.
    """
    for label, column in read_bsm_facs.items():
        # Initialise Axes instance
        fig, ax = plt.subplots()
        # populate the Axes with the KDE
        ax = plotutils.kde_plot(column.values)

        # Format of the plot
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.set_title(f"KDE for {bsm_names_to_latex[label]} coefficient")
        ax.set_ylabel("Prob. density", fontsize=14)
        ax.set_xlabel(bsm_names_to_latex[label] + r"$/\Lambda^2$ [TeV$^{-2}]$", fontsize=14)
        ax.grid(True)

        yield fig


@make_argcheck
def _check_two_bsm_facs(fit):
    cf = fit.as_input().get("simu_parameters", [])
    l = len(cf)
    check(
        l == 2,
        "Exactly two elements are required in "
        f"`simu_parameters` for fit '{fit}', but {l} found.",
    )

@figure
#@_check_two_bsm_facs
def plot_2d_bsm_facs(read_bsm_facs, replica_data):
    """
    Plot two-dimensional distributions of the BSM coefficient results.

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing the BSM factors.
    replica_data : list
        List of FitInfo.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for the 2D distribution plot.
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

def _select_plot_2d_bsm_facs(read_bsm_facs, replica_data, bsm_names_to_latex, pair):
    """
    Auxiliary function to plot 2D plots of pair of operators in a N-dimensional fits with BSM factors.

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing BSM factors. 
    replica_data : list
        List of FitInfo.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to LaTeX representations.
    pair : tuple
        Pair of operators to be plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for the 2D plot.
    """
    op_1, op_2 = pair
    bsm_facs_df = read_bsm_facs
    if op_1 != op_2:
        bsm_facs_df = bsm_facs_df[[op_1, op_2]]
        labels = bsm_facs_df.columns
    else:
        bsm_facs_df = bsm_facs_df[[op_1]]
        labels = [bsm_facs_df.columns]*2

    chi2 = [info.chi2 for info in replica_data]

    # we use this figsize to have a square scatter plot
    # smaller values do not display too well
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    chi2 = [info.chi2 for info in replica_data]

    if op_1 != op_2:
        scatter_plot = ax.scatter(
            bsm_facs_df.iloc[:, 0], bsm_facs_df.iloc[:, 1], c=chi2, s=40
        )
    else:
        scatter_plot = ax.scatter(
            bsm_facs_df.values, bsm_facs_df.values, c=chi2, s=40
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
    ax_histx.hist(bsm_facs_df.iloc[:, 0], density=True)
    if op_1 != op_2:
        ax_histy.hist(bsm_facs_df.iloc[:, 1], orientation='horizontal', density=True)
    else:
        ax_histy.hist(bsm_facs_df.iloc[:, 0], orientation='horizontal', density=True)

    ax_histx.grid(False)
    ax_histy.grid(False)

    ax.set_xlabel(bsm_names_to_latex[labels[0]] + r"$/\Lambda^2$ [TeV$^{-2}]$", fontsize=15)
    ax.set_ylabel(bsm_names_to_latex[labels[1]] + r"$/\Lambda^2$ [TeV$^{-2}]$", fontsize=15)

    ax.set_axisbelow(True)

    return fig

@figuregen
def plot_bsm_2d_combs(read_bsm_facs, replica_data, bsm_names_to_latex):
    """
    Plot two-dimensional distributions for all pairs of BSM coefficients in a fit.

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing BSM factors.
    replica_data : list
        List of FitInfo.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to LaTeX representations.

    Yields
    ------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for each pair of BSM coefficient combination.
    """
    bsm_facs_df = read_bsm_facs
    labels = bsm_facs_df.columns 

    combs = itertools.combinations(labels, 2)
    for comb in combs:
        fig = _select_plot_2d_bsm_facs(bsm_facs_df, replica_data, bsm_names_to_latex, pair=comb)
        yield fig

@figure
def plot_chi2_bsm_facs(read_bsm_facs, replica_data):
    """
    Generates bsm_fac value - chi2 scatter plots for all replicas in a fit.

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing BSM factors.
    replica_data : list
        List of FitInfo.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure object for the chi2 scatter plot.
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


@figuregen
def plot_tr_val_epoch(fit, replica_paths):
    """
    Plot the average across replicas of training and validation chi2 
    for a given epoch.

    Parameters
    ----------
    fit : FitSpec
        Object containing the specifications of the fit.
    replica_paths : list
        List of paths to the replica data.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    paths = [p / 'chi2exps.log' for p in replica_paths]
    # initialise dataframe
    all_cols = pd.concat([pd.read_json(i).loc['total'] for i in paths], axis=1)
    # get training and validation data
    tr_data = all_cols.applymap(lambda x: x['training'], na_action='ignore')
    val_data = all_cols.applymap(lambda x: x['validation'], na_action='ignore')

    tr_chi2 = tr_data.mean(axis=1)
    val_chi2 = val_data.mean(axis=1)

    fig, ax = plt.subplots()
    # formatting
    ax.plot(tr_chi2.index, tr_chi2, label=r'Training $\chi^2$')
    ax.plot(val_chi2.index, val_chi2, label=r'Validation $\chi^2$')
    ax.legend()
    ax.grid(True)
    ax.set_title(fit.label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\chi^2$', rotation='horizontal', labelpad=10.0)

    yield fig

@table
def bsm_facs_bounds(read_bsm_facs, bsm_names_to_latex):
    """
    Table generator to summarise information about the BSM coefficient results.

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing BSM factors.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to LaTeX representations.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing the bounds and statistics of BSM coefficients.
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
    df.index = [bsm_names_to_latex[i] for i in df.index]
    df['68% CL bounds'] = list(zip(cl68_lower_disp, cl68_upper_disp))
    df['95% CL bounds'] = list(zip(cl95_lower_disp, cl95_upper_disp))
    df['Mean'] = means_disp
    df['Std'] = stds_disp
    
    return df

@table
def tabulate_bsm_corr(fit, read_bsm_facs):
    """
    Generate a correlation table for BSM coefficients similar to the corresponding plot.

    Parameters
    ----------
    fit : FitSpec
        Object containing the specifications of the fit.
    read_bsm_facs : pd.DataFrame
        DataFrame containing BSM factors.

    Returns
    -------
    pd.DataFrame
        The correlation matrix as a DataFrame.
    """
    bsm_facs_df = read_bsm_facs
    bsm_facs_df = bsm_facs_df.reindex(columns=reorder_cols(bsm_facs_df.columns))
    corr_mat = bsm_facs_df.corr()
    round(corr_mat, 1)

    return corr_mat

@figure
def plot_2d_bsm_facs_pair(read_bsm_facs, replica_data, bsm_names_to_latex, op1, op2):
    """
    Auxiliary function to plot 2D plots of a pair of operators in a N-dimensional fits with BSM factors.

    Parameters
    ----------
    read_bsm_facs : pd.DataFrame
        DataFrame containing BSM factors.
    replica_data : list
        List of FitInfo.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to LaTeX representations.
    op1 : str
        The first operator name.
    op2 : str
        The second operator name.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the 2D plot.
    """
    return _select_plot_2d_bsm_facs(read_bsm_facs, replica_data, bsm_names_to_latex, (op1, op2))

@figure
def plot_bsm_corr(fit, read_bsm_facs, bsm_names_to_latex, corr_threshold=0.5):
    """
    Plot a correlation matrix to summarise information about the BSM coefficient results.

    Parameters
    ----------
    fit : FitSpec
        Object containing the specifications of the fit.
    read_bsm_facs : pd.DataFrame
        DataFrame containing BSM factors.
    bsm_names_to_latex : dict
        Dictionary mapping BSM names to LaTeX representations.
    corr_threshold : float, optional
        Threshold for coloration in the correlation matrix.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the correlation matrix plot.
    """
    # figsize (11, 9) has good proportions
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    # set background colour
    ax.set_facecolor("0.9")

    # read dataframe and round numbers
    bsm_facs_df = read_bsm_facs
    bsm_facs_df = bsm_facs_df.reindex(columns=reorder_cols(bsm_facs_df.columns))
    # note that bsm_names_to_latex can be None
    if bsm_names_to_latex:
        bsm_facs_df.columns = [bsm_names_to_latex[col] for col in bsm_facs_df.columns]
    corr_mat = bsm_facs_df.corr()
    round(corr_mat, 1)

    # create colourmap with gret in the centre for colourbar
    new_cmap = grey_centre_cmap(frac=corr_threshold)

    # formatting
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.set_title(fit.label, fontsize=20, pad=20)

    # create heatmap
    ax = sns.heatmap(corr_mat,
    vmin=-1.0, vmax=1.0, linewidths=.5, square=True, cmap=new_cmap);

    return fig

@figuregen
def plot_bsm_pdf_corr(
    pdf,
    read_bsm_facs,
    xplotting_grid,
    Q,
    bsm_names_to_latex,
    mark_threshold: float = 0.9,
    ymin: (float, type(None)) = None,
    ymax: (float, type(None)) = None,
    dashed_line_flavours: (list, type(None)) = None,
):
    """
    Plot the correlation between BSM factors and a PDF.

    Parameters
    ----------
    pdf : PDF object
        The parton distribution function being analyzed.
    read_bsm_facs : DataFrame or callable
        Data or function representing BSM factors.
    xplotting_grid : XGrid object
        Object containing the plotting grid information.
    Q : float
        Momentum transfer value in GeV.
    bsm_names_to_latex : dict
        Mapping of BSM factor names to their LaTeX representations.
    mark_threshold : float, optional
        Threshold for marking significant correlations, by default 0.9.
    ymin : float or None, optional
        Minimum y-axis value, by default None.
    ymax : float or None, optional
        Maximum y-axis value, by default None.
    dashed_line_flavours : list of str or None, optional
        List of flavours to be plotted with dashed lines, by default None.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the plot.
    bsm_fac : str
        Name of the BSM factor being plotted.
    """
    # read dataframe
    bsm_facs_df = read_bsm_facs
    # reorder BSM facs
    bsm_facs_df = bsm_facs_df.reindex(columns=reorder_cols(bsm_facs_df.columns))
    # get xplotting_grid
    # x_grid_obj = xplotting_grid(pdf, Q, basis=Basespecs[0]["basis"])
    x_grid_obj = xplotting_grid
    if dashed_line_flavours is None:
        dashed_line_flavours = []

    for bsm_fac in bsm_facs_df.columns:
        # get the values of the BSM factors
        bsm_fac_vals = bsm_facs_df[bsm_fac].values
        # Initialise axes
        fig, ax = plt.subplots()
        # Define xgrid and scale
        xgrid = x_grid_obj.xgrid
        scale = x_grid_obj.scale
        # get grid values
        gv = x_grid_obj.grid_values.error_members()
        for index, flavour in enumerate(x_grid_obj.flavours):
            flavour_label = flavour
            parton_grids = gv[:, index, ...]
            # calculate correlation
            num = np.mean(bsm_fac_vals.reshape(-1, 1) * parton_grids, axis=0) - np.mean(parton_grids, axis=0) * np.mean(bsm_fac_vals)
            den = np.sqrt(np.mean(bsm_fac_vals**2) - np.mean(bsm_fac_vals)**2) * np.sqrt(np.mean(parton_grids**2, axis=0)- np.mean(parton_grids, axis=0)**2)
            corr = num / den
            if flavour in dashed_line_flavours:
                style = "--"
            else:
                style = "-"
            ax.plot(xgrid, corr, style, label=fr'${flavour_label}$')
            # Plot threshold
            mask = np.abs(corr) > mark_threshold
            ranges = split_ranges(xgrid, mask, filter_falses=True)
            for r in ranges:
                ax.axvspan(r[0], r[-1], color='#eeeeff')

        ax.set_xscale(scale)
        ax.set_xlim(xgrid[0], xgrid[-1])
        ax.set_xlabel(r'$x$')
        ax.set_title(f'Correlation {bsm_names_to_latex[bsm_fac]} - {pdf.label}\nQ = {Q} GeV')


        ax.set_ylim(ymin, ymax)

        ax.legend(loc="best")
        ax.grid(True)
        #ax.set_axisbelow(True)
        #ax.set_adjustable("datalim")
        yield fig, bsm_fac

@figuregen
def plot_bsm_pdf_corr_fits(fits, pdfs, xplotting_grids, Q, bsm_names_to_latex):
    """
    Plot correlations between BSM factors and multiple PDFs.

    Parameters
    ----------
    fits : NSList
        List of FitSpec to be compared.
    pdfs : list of PDF objects
        List of parton distribution functions corresponding to the fits.
    xplotting_grids : list of XGrid objects
        List of plotting grid objects corresponding to each fit.
    Q : float
        Momentum transfer value in GeV.
    bsm_names_to_latex : dict
        Mapping of BSM factor names to their LaTeX representations.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the plot.
    """
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.append(bsm_fac_ops)
    # Remove repeated operators
    all_ops = reorder_cols({o for fit_ops in all_ops for o in fit_ops})
    # plot correlation per operator
    # if an operator is not in the fit then it is 
    # simply not plotted
    for bsm_fac in all_ops:
        # Initialise axes
        fig, ax = plt.subplots()
        for fit in fits:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            # get PDF
            pdf = pdfs[fits.index(fit)]
            # get x_object
            x_grid_obj = xplotting_grids[fits.index(fit)] 
            if bsm_facs_df.get([bsm_fac]) is not None:
                bsm_fac_vals = bsm_facs_df[bsm_fac].values
                # Define xgrid and scale
                xgrid = x_grid_obj.xgrid
                scale = x_grid_obj.scale
                # get grid values
                gv = x_grid_obj.grid_values.error_members()
                for flavour in x_grid_obj.flavours:
                    flavour_label = flavour
                    index = tuple(x_grid_obj.flavours).index(flavour)
                    parton_grids = gv[:, index, ...]
                    # calculate correlation
                    num = np.mean(bsm_fac_vals.reshape(-1, 1) * parton_grids, axis=0) - np.mean(parton_grids, axis=0) * np.mean(bsm_fac_vals)
                    den = np.sqrt(np.mean(bsm_fac_vals**2) - np.mean(bsm_fac_vals)**2) * np.sqrt(np.mean(parton_grids**2, axis=0)- np.mean(parton_grids, axis=0)**2)
                    corr = num / den
                    ax.plot(xgrid, corr, label=fr'$\rho({flavour_label},$ ' + bsm_names_to_latex[bsm_fac] + f') {pdf.label}')
                ax.set_xscale(scale)
                ax.set_xlabel(r'$x$')
                ax.set_title(f'Correlation {bsm_names_to_latex[bsm_fac]} - PDFs ' + f'(Q = {Q} GeV)')
                
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17))
                ax.grid(True)
                ax.set_axisbelow(True)
                ax.set_adjustable("datalim")
        yield fig

@figuregen
def plot_2d_bsm_facs_fits(fits, bsm_names_to_latex):
    """
    Generate 2D scatter plots and histograms comparing BSM factor values across different fits.

    This function takes a set of fits and compares the BSM factors between them. For each pair
    of BSM factors, it creates a 2D scatter plot with corresponding histograms on the x and y axes.

    Parameters
    ----------
    fits : NSList
        List of FitSpec to be compared.
    bsm_names_to_latex : dict
        Dictionary mapping BSM factor names to their LaTeX string representations.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for each pair of BSM factors.
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
                    bsm_facs_df.get([op_1]), bsm_facs_df.get([op_2]), label=fit.label, alpha=0.5, s=40
                )
                # populate the histograms
                ax_histx.hist(bsm_facs_df.get([op_1]), alpha=0.5, density=True)
                ax_histy.hist(bsm_facs_df.get([op_2]), orientation='horizontal', alpha=0.5, density=True)

        ax_histx.grid(False)
        ax_histy.grid(False)

        ax.set_xlabel(bsm_names_to_latex[op_1] + r"$/\Lambda^2$ [TeV$^{-2}]$", fontsize=14)
        ax.set_ylabel(bsm_names_to_latex[op_2] + r"$/\Lambda^2$ [TeV$^{-2}]$", fontsize=14)
        ax.legend()
        ax.set_axisbelow(True)

        yield fig

@table
def bsm_facs_bounds_fits(fits, bsm_names_to_latex, n_sigma=2):
    """
    Generate a table summarizing the bounds of BSM coefficients in different fits.

    This function processes a list of fits, extracting the BSM coefficients and
    summarizing their mean, standard deviation, and confidence levels. The confidence
    levels are computed as mean ± n_sigma * std.

    Parameters
    ----------
    fits : NSList
        List of FitSpec to be compared.
    bsm_names_to_latex : dict
        A dictionary mapping BSM factor names to their LaTeX representations.
    n_sigma : int, optional
        The multiplier for the standard deviation to define confidence level bounds, by default 2.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the bounds for each BSM factor in each fit, along with
        additional metrics like 'Best-fit shift' and 'Broadening'.
    """
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.extend(bsm_fac_ops)
    all_ops = list(dict.fromkeys(all_ops))

    fit_names =  [fit.label for fit in fits]
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
                df[fit.label].loc[op] = f"({lower_dis}, {upper_dis})"
                # best-fit value
                best_fits.append(mean)
                # calculate bound length
                length = cl_upper - cl_lower
                bound_lengths.append(length)
            else:
                df[fit.label].loc[op] = 'Not in fit'
                # if the operator is not in the fit, then append None
                best_fits.append(np.nan)
                bound_lengths.append(np.nan)
        # best-fit shift column
        df['Best-fit shift'].loc[op] = format_number(best_fits[0] - best_fits[1], digits=2)
        # broadening column
        curr_len, ref_len = bound_lengths
        if ref_len > 0:
            df['Broadening'].loc[op] = str(np.round((curr_len - ref_len) / ref_len * 100.0, decimals=2)) + '%'
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
    df.index = [bsm_names_to_latex[i] for i in df.index]

    return df

@table
def bsm_facs_68bounds_fits(fits, bsm_names_to_latex,):
    """
    Generate a table summarizing the 68% confidence level (CL) bounds for BSM factors from various fits.

    This function processes a list of fits and utilizes `bsm_facs_bounds_fits` to calculate and
    return the 68% CL bounds for BSM coefficients.

    Parameters
    ----------
    fits : NSList
        List of FitSpec to be compared.
    bsm_names_to_latex : dict
        A dictionary mapping BSM factor names to their LaTeX representations.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the 68% CL bounds for each BSM factor in each fit.
    """
    return bsm_facs_bounds_fits(fits, bsm_names_to_latex, n_sigma=1)

@table
def bsm_facs_95bounds_fits(fits, bsm_names_to_latex):
    """
    Generate a table summarizing the 95% confidence level (CL) bounds for BSM factors from various fits.

    This function processes a list of fits and utilizes `bsm_facs_bounds_fits` to calculate and
    return the 95% CL bounds for BSM coefficients.

    Parameters
    ----------
    fits : NSList
        List of FitSpec to be compared. 
    bsm_names_to_latex : dict
        A dictionary mapping BSM factor names to their LaTeX representations.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the 95% CL bounds for each BSM factor in each fit.
    """
    return bsm_facs_bounds_fits(fits, bsm_names_to_latex, n_sigma=2)

@figuregen
def plot_smefit_internal_comparison(bsm_names_to_latex, smefit_reference_1, smefit_reference_2, bsm_names_to_plot_scales, smefit_labels):
    """
    Generates comparison plots between SMEFiT fits.

    This function creates plots to compare two different SMEFiT fits. It plots the best fit values
    and the bounds for BSM (Beyond the Standard Model) coefficients, allowing for an easy comparison
    of the two fits. It supports both linear and symmetric logarithmic scales.

    Parameters
    ----------
    bsm_names_to_latex : dict
        Dictionary mapping BSM factor names to their LaTeX representations.
    smefit_reference_1 : list of dicts
        List of dictionaries containing BSM coefficient information for the first SMEFiT reference.
    smefit_reference_2 : list of dicts
        List of dictionaries containing BSM coefficient information for the second SMEFiT reference.
    bsm_names_to_plot_scales : dict
        Dictionary to scale the BSM names for plotting.
    smefit_labels : list of str
        Labels for the SMEFiT references to be used in the plot.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the comparison plot.
    """
    # extract all operators in the SMEFiT fits
    all_ops = []
    for fit in [smefit_reference_1, smefit_reference_2]:
        ops_list = []
        for entry in fit:
            ops_list += [entry['name']]
        all_ops.append(ops_list)
    # Remove repeated operators and reorder
    all_ops = reorder_cols({o for fit_ops in all_ops for o in fit_ops})

    # store the relevant values
    bounds_dict = {}
    best_fits_dict ={} 

    # Now extend the bounds_dict and best_fits_dict with SMEFiT stuff
    bounds_1 = []
    best_fits_1 = []
    for op in all_ops:
        best_fits_1 += [bsm_names_to_plot_scales[op]*smefit_reference_1[x]['best'] for x in range(len(smefit_reference_1)) if smefit_reference_1[x]['name'] == op]
        bounds_1 += [[bsm_names_to_plot_scales[op]*smefit_reference_1[x]['lower_bound'], bsm_names_to_plot_scales[op]*smefit_reference_1[x]['upper_bound']] for x in range(len(smefit_reference_1)) if smefit_reference_1[x]['name'] == op]

    bounds_dict[smefit_labels[0]] = bounds_1
    best_fits_dict[smefit_labels[0]] = best_fits_1

    # Now extend the bounds_dict and best_fits_dict with SMEFiT stuff
    bounds_2 = []
    best_fits_2 = []
    for op in all_ops:
        best_fits_2 += [bsm_names_to_plot_scales[op]*smefit_reference_2[x]['best'] for x in range(len(smefit_reference_2)) if smefit_reference_2[x]['name'] == op]
        bounds_2 += [[bsm_names_to_plot_scales[op]*smefit_reference_2[x]['lower_bound'], bsm_names_to_plot_scales[op]*smefit_reference_2[x]['upper_bound']] for x in range(len(smefit_reference_2)) if smefit_reference_2[x]['name'] == op]

    bounds_dict[smefit_labels[1]] = bounds_2
    best_fits_dict[smefit_labels[1]] = best_fits_2

    # plot parameters
    scales= ['linear', 'symlog']
    colour_key = ['#66C2A5', '#FC8D62', '#8DA0CB']

    for scale in scales:
        # initialise plots
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # line for SM prediction
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.3, label='SM')

        labels = smefit_labels

        idx = 0
        for label in labels:
            bounds = bounds_dict[label]
            best_fits = best_fits_dict[label]
            x_coords = [i - 0.1 + 0.2*idx for i in range(len(all_ops))] 
            bounds_min = [bound[0] for bound in bounds]
            bounds_max= [bound[1] for bound in bounds]
            ax.scatter(x_coords, best_fits, color=colour_key[idx])
            ax.vlines(x=x_coords, ymin=bounds_min, ymax=bounds_max, label='95% CL ' + label,
            color=colour_key[idx], lw=2.0)
            idx += 1

        # set x positions for labels and labels
        ax.set_xticks(np.arange(len(all_ops)))
        bsm_latex_names = []
        for op in all_ops:
            if bsm_names_to_plot_scales[op] != 1:
                bsm_latex_names += [str(bsm_names_to_plot_scales[op]) + '$\cdot$' + bsm_names_to_latex[op]]
            else:
                bsm_latex_names += [bsm_names_to_latex[op]]
        ax.set_xticklabels(bsm_latex_names, rotation='vertical', fontsize=10)

        # set y labels
        ax.set_ylabel(r'$c_i / \Lambda^2 \ \ [ \operatorname{TeV}^{-2} ] $', fontsize=10)

        # treatment of the symmetric log scale
        if scale == 'symlog':
            ax.set_yscale(scale, linthresh=0.1)

            # turn off scientific notation
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)

            y_values = [-100, -10, -1, -0.1, 0.0, 0.1, 1, 10, 100] 
            ax.set_yticks(y_values)

            # get rid of scientific notation in y axis and
            # get rid of '.0' for floats bigger than 1
            ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',') if abs(x) >= 1 else x))

        # treatment of linear scale
        else:
            ax.set_yscale(scale)

        # final formatting
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_adjustable("datalim")

        # Load image and add it to the plot
        #file_name = "logo_black.png"
        #logo = image.imread(file_name)

        #The OffsetBox is a simple container artist.
        #The child artists are meant to be drawn at a relative position to its #parent.
        #imagebox = OffsetImage(logo, zoom = 0.15)

        #Container for the imagebox referring to a specific position *xy*.
        #ab = AnnotationBbox(imagebox, (20, -5), frameon = False)
        #ax.add_artist(ab)

        # frames on all sides
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        yield fig

@figuregen
def plot_smefit_comparison(fits, bsm_names_to_latex, smefit_reference, bsm_names_to_plot_scales, smefit_label):
    """
    Generates plots comparing bounds obtained from SIMUnet fits with those obtained by SMEFiT.

    This function takes a set of SIMUnet fits and a SMEFiT reference, extracting BSM coefficients
    from each. It plots the mean and standard deviation of the BSM coefficients for each fit,
    along with confidence levels calculated as mean ± 2*std. It supports both linear and symmetric
    logarithmic scales for the plots.

    Parameters
    ----------
    fits: NSList
        List of FitSpec to be compared.
    bsm_names_to_latex : dict
        Dictionary mapping BSM factor names to their LaTeX representations.
    smefit_reference : list of dicts
        List of dictionaries containing BSM coefficient information from a SMEFiT reference.
    bsm_names_to_plot_scales : dict
        Dictionary to scale the BSM names for plotting.
    smefit_label : str
        Label for the SMEFiT reference to be used in the plot.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the comparison plot.
    """
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.append(bsm_fac_ops)
    # Remove repeated operators and reorder
    all_ops = reorder_cols({o for fit_ops in all_ops for o in fit_ops})

    # store the relevant values
    bounds_dict = {}
    best_fits_dict ={} 

    for fit in fits:
        bounds = []
        best_fits = []
        for op in all_ops:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            if bsm_facs_df.get([op]) is not None:
                values = bsm_names_to_plot_scales[op]*bsm_facs_df[op]
                mean =  values.mean()
                std = values.std()
                cl_lower, cl_upper = (mean - 2*std, mean + 2*std)
                # best-fit value
                best_fits.append(mean)
                # append bounds
                bounds.append([cl_lower, cl_upper])
            else:
                # if the operator is not in the fit, then assume SM
                best_fits.append(np.nan)
                bounds.append([np.nan, np.nan])

        bounds_dict[fit.label] = bounds
        best_fits_dict[fit.label] = best_fits

    # Now extend the bounds_dict and best_fits_dict with SMEFiT stuff
    bounds = []
    best_fits = []
    for op in all_ops:
        best_fits += [bsm_names_to_plot_scales[op]*smefit_reference[x]['best'] for x in range(len(smefit_reference)) if smefit_reference[x]['name'] == op]
        bounds += [[bsm_names_to_plot_scales[op]*smefit_reference[x]['lower_bound'], bsm_names_to_plot_scales[op]*smefit_reference[x]['upper_bound']] for x in range(len(smefit_reference)) if smefit_reference[x]['name'] == op]

    bounds_dict[smefit_label] = bounds
    best_fits_dict[smefit_label] = best_fits

    # plot parameters
    scales= ['linear', 'symlog']
    colour_key = ['#66C2A5', '#FC8D62', '#8DA0CB']

    for scale in scales:
        # initialise plots
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # line for SM prediction
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.3, label='SM')

        labels = [fit.label for fit in fits] + [smefit_label]

        idx = 0
        for label in labels:
            bounds = bounds_dict[label]
            best_fits = best_fits_dict[label]
            x_coords = [i - 0.1 + 0.2*idx for i in range(len(all_ops))] 
            bounds_min = [bound[0] for bound in bounds]
            bounds_max= [bound[1] for bound in bounds]
            ax.scatter(x_coords, best_fits, color=colour_key[idx])
            ax.vlines(x=x_coords, ymin=bounds_min, ymax=bounds_max, label='95% CL ' + label,
            color=colour_key[idx], lw=2.0)
            idx += 1

        # set x positions for labels and labels
        ax.set_xticks(np.arange(len(all_ops)))
        bsm_latex_names = []
        for op in all_ops:
            if bsm_names_to_plot_scales[op] != 1:
                bsm_latex_names += [str(bsm_names_to_plot_scales[op]) + '$\cdot$' + bsm_names_to_latex[op]]
            else:
                bsm_latex_names += [bsm_names_to_latex[op]]
        ax.set_xticklabels(bsm_latex_names, rotation='vertical', fontsize=10)

        # set y labels
        ax.set_ylabel(r'$c_i / \Lambda^2 \ \ [ \operatorname{TeV}^{-2} ] $', fontsize=10)

        # treatment of the symmetric log scale
        if scale == 'symlog':
            ax.set_yscale(scale, linthresh=0.1)

            # turn off scientific notation
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)

            y_values = [-100, -10, -1, -0.1, 0.0, 0.1, 1, 10, 100] 
            ax.set_yticks(y_values)

            # get rid of scientific notation in y axis and
            # get rid of '.0' for floats bigger than 1
            ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',') if abs(x) >= 1 else x))

        # treatment of linear scale
        else:
            ax.set_yscale(scale)

        # final formatting
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_adjustable("datalim")

        # Load image and add it to the plot
        #file_name = "logo_black.png"
        #logo = image.imread(file_name)

        #The OffsetBox is a simple container artist.
        #The child artists are meant to be drawn at a relative position to its #parent.
        #imagebox = OffsetImage(logo, zoom = 0.15)

        #Container for the imagebox referring to a specific position *xy*.
        #ab = AnnotationBbox(imagebox, (20, -5), frameon = False)
        #ax.add_artist(ab)

        # frames on all sides
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        yield fig

@figuregen
def plot_bsm_facs_bounds(fits, bsm_names_to_latex, bsm_names_to_plot_scales):
    """
    Generates plots of the bounds for BSM coefficients from various fits.

    This function takes a list of fits and generates plots showing the mean, standard deviation,
    and confidence levels of the BSM coefficients. The confidence levels are calculated as mean ± 2*std.
    It supports both linear and symmetric logarithmic scales for the plots.

    Parameters
    ----------
    fits: NSList
        List of FitSpec to be compared.
    bsm_names_to_latex : dict
        Dictionary mapping BSM factor names to their LaTeX representations.
    bsm_names_to_plot_scales : dict
        Dictionary to scale the BSM names for plotting.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for each generated plot.
    """
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.append(bsm_fac_ops)
    # Remove repeated operators and reorder
    all_ops = reorder_cols({o for fit_ops in all_ops for o in fit_ops})

    # store the relevant values
    bounds_dict = {}
    best_fits_dict ={} 

    for fit in fits:
        bounds = []
        best_fits = []
        for op in all_ops:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            if bsm_facs_df.get([op]) is not None:
                # note that bsm_names_to_plot_scales can be None
                if bsm_names_to_plot_scales:
                    values = bsm_names_to_plot_scales[op]*bsm_facs_df[op]
                else:
                    values = bsm_facs_df[op]
                mean = values.mean()
                std = values.std()
                cl_lower, cl_upper = (mean - 2*std, mean + 2*std)
                # best-fit value
                best_fits.append(mean)
                # append bounds
                bounds.append([cl_lower, cl_upper])
            else:
                # if the operator is not in the fit, add np.nan
                best_fits.append(np.nan)
                bounds.append([np.nan, np.nan])

        bounds_dict[fit.label] = bounds
        best_fits_dict[fit.label] = best_fits

    # plot parameters
    scales = ['linear', 'symlog']
    colour_key = ['#66C2A5', '#FC8D62', '#8DA0CB']

    for scale in scales:
        # initialise plots
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        # line for SM prediction
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.3, label='SM')

        for fit in fits:
            bounds = bounds_dict[fit.label]
            best_fits = best_fits_dict[fit.label]
            x_coords = [i - 0.1 + 0.2*fits.index(fit) for i in range(len(all_ops))] 
            bounds_min = [bound[0] for bound in bounds]
            bounds_max= [bound[1] for bound in bounds]
            ax.scatter(x_coords, best_fits, color=colour_key[fits.index(fit)])
            ax.vlines(x=x_coords, ymin=bounds_min, ymax=bounds_max, label='95% CL ' + fit.label,
            color=colour_key[fits.index(fit)], lw=2.0)

        # set x positions for labels and labels
        ax.set_xticks(np.arange(len(all_ops)))
        bsm_latex_names = []
        for op in all_ops:
            # note that bsm_names_to_plot_scales can be None
            if bsm_names_to_plot_scales:
                if bsm_names_to_plot_scales[op] != 1:
                    bsm_latex_names += [str(bsm_names_to_plot_scales[op]) + '$\cdot$' + bsm_names_to_latex[op]]
                else:
                    bsm_latex_names += [bsm_names_to_latex[op]]
        ax.set_xticklabels(bsm_latex_names, rotation='vertical', fontsize=10)

        # set y labels
        ax.set_ylabel(r'$c_i / \Lambda^2 \ \ [ \operatorname{TeV}^{-2} ] $', fontsize=10)

        # treatment of the symmetric log scale
        if scale == 'symlog':
            ax.set_yscale(scale, linthresh=0.1)

            # turn off scientific notation
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)

            y_values = [-100, -10, -1, -0.1, 0.0, 0.1, 1, 10, 100] 
            ax.set_yticks(y_values)

            # get rid of scientific notation in y axis and
            # get rid of '.0' for floats bigger than 1
            ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',') if abs(x) >= 1 else x))

        # treatment of linear scale
        else:
            ax.set_yscale(scale)

        # final formatting
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_adjustable("datalim")

        # frames on all sides
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        yield fig

@figuregen
def plot_bsm_facs_68res(fits, bsm_names_to_latex):
    """
    Generates plots of the 68% residuals of BSM coefficients for various fits.

    This function processes a set of fits and generates plots showing the 68% residuals of the
    BSM (Beyond the Standard Model) coefficients. The residuals are calculated as the mean of
    the coefficients divided by their standard deviation, providing a measure of deviation
    from the Standard Model.

    Parameters
    ----------
    fits: NSList
        List of FitSpec to be compared.
    bsm_names_to_latex : dict
        Dictionary mapping BSM factor names to their LaTeX representations.

    Yields
    ------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the residual plot.
    """
    # extract all operators in the fits
    all_ops = []
    for fit in fits:
        paths = replica_paths(fit)
        bsm_facs_df = read_bsm_facs(paths)
        bsm_fac_ops = bsm_facs_df.columns.tolist()
        all_ops.append(bsm_fac_ops)

    # Remove repeated operators and reorder
    all_ops = reorder_cols({o for fit_ops in all_ops for o in fit_ops})

    # store the relevant values
    residuals_dict = {}

    for fit in fits:
        residuals = []
        for op in all_ops:
            paths = replica_paths(fit)
            bsm_facs_df = read_bsm_facs(paths)
            if bsm_facs_df.get([op]) is not None:
                values = bsm_facs_df[op]
                mean = values.mean()
                std = values.std()
                # append residual 
                residuals.append(mean / std)
            else:
                # if the operator is not in the fit, then assume SM
                residuals.append(0.0)

        residuals_dict[fit.label] = residuals

    # plotting specs
    colour_key = ['#66C2A5', '#FC8D62', '#8DA0CB']

    # initialise plots
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # line for 0
    ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.8)

    # line for +-1 residual
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3)

    for fit in fits:
        residuals = residuals_dict[fit.label]
        ordered_residuals = format_residuals(residuals)
        x_coords = [i - 0.1 + 0.2*fits.index(fit) for i in range(len(all_ops))] 
        residuals_min = [residual[0] for residual in ordered_residuals]
        residuals_max = [residual[1] for residual in ordered_residuals]
        ax.vlines(x=x_coords, ymin=residuals_min, ymax=residuals_max, label=fit.label,
                  color=colour_key[fits.index(fit)], lw=4.0)

    # set x positions for labels and labels
    ax.set_xticks(np.arange(len(all_ops)))
    bsm_latex_names = []
    for op in all_ops:
        bsm_latex_names += [bsm_names_to_latex[op]]
    ax.set_xticklabels(bsm_latex_names, rotation='vertical', fontsize=10)

    # set y scale
    ax.set_yscale('linear')

    # set y labels
    ax.set_ylabel(r'Residuals (68%)', fontsize=10)

    # final formatting
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_adjustable("datalim")

    # frames on all sides
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    yield fig

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

"""
Principal component analysis
"""

@table
def fisher_information_matrix(dataset_inputs, groups_index, theoryid, groups_covmat, simu_parameters_names, pdf):
    """
    Obtains the full Fisher information matrix for the BSM parameters.

    This function computes the Fisher information matrix for Beyond the Standard Model (BSM) parameters
    given a dataset. It utilizes an internal function `_compute_fisher_information_matrix` to perform the computation.

    Parameters
    ----------
    dataset_inputs : array-like
        The inputs from the dataset used for computing the Fisher information matrix.
    groups_index : array-like
        Indexes representing different groups in the dataset.
    theoryid : array-like
        Array of theory identifiers.
    groups_covmat : array-like
        Covariance matrices for the groups in the dataset.
    simu_parameters_names : list
        List of names of the simulation parameters.
    pdf : PDF object
        The parton distribution function object.

    Returns
    -------
    array-like
        The computed Fisher information matrix.
    """
    return _compute_fisher_information_matrix(dataset_inputs, theoryid, groups_covmat, simu_parameters_names, pdf)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a colormap to a specific range.

    This function creates a new colormap based on a given colormap but truncated to the range specified
    by minval and maxval. This is useful for adjusting the range of colors used in a plot.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The original colormap to be truncated.
    minval : float, optional
        The minimum value of the new colormap, by default 0.0.
    maxval : float, optional
        The maximum value of the new colormap, by default 1.0.
    n : int, optional
        The number of discrete colors in the new colormap, by default 100.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The truncated colormap.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

@figure
def plot_fisher_information_by_sector(fisher_information_by_sector, bsm_names_to_latex, bsm_sectors_to_latex):
    """
    Produces a heatmap plot from the Fisher information by sector table.

    This function creates a heatmap visualizing the Fisher information matrix, where rows correspond to BSM coefficients
    and columns correspond to different sectors. This visualization helps in understanding the impact of various
    sectors on the coefficients.

    Parameters
    ----------
    fisher_information_by_sector : pandas.DataFrame
        The Fisher information matrix with sectors as columns and BSM coefficients as rows.
    bsm_names_to_latex : dict
        Dictionary mapping BSM factor names to their LaTeX representations.
    bsm_sectors_to_latex : dict
        Dictionary mapping sector names to their LaTeX representations.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the heatmap plot of the Fisher information.
    """
    f = fisher_information_by_sector

    coeff_names = [bsm_names_to_latex[x] for x in f.index]
    sector_names = [bsm_sectors_to_latex[x] for x in f.columns]

    ncoeff, ndata = f.shape

    fig, ax = plt.subplots(figsize=(15,5))
    ax.set_xlim([-1.5,ncoeff-1.5])
    ax.set_ylim([-1.5,ndata-1.5])
    ax.xaxis.set_major_locator(MultipleLocator(1.))
    ax.yaxis.set_major_locator(MultipleLocator(1.))

    old_cmap =  plt.get_cmap('YlGnBu')
    new_cmap  = truncate_colormap(old_cmap, minval=0.0, maxval=0.65)

    ax = sns.heatmap(f.T,vmin=0.0, vmax=100.0,cmap=new_cmap,cbar=False);

    ax.set_xticklabels(coeff_names, rotation=0., va='top', ha='center', fontsize=14)
    ax.set_yticklabels(sector_names, rotation=0., va='center', ha='right', fontsize=14)

    for y,val in enumerate(f.index):
        ax.plot([y, y],[-1.5, ndata+1], ls='solid', c='lightgray', lw=0.8)

    fisher_rounded = np.round(f.to_numpy(),0)

    #Plot numbers
    nrow, ncol = np.shape(f.T)
    for i in range(nrow):
        for j in range(ncol):
            if fisher_rounded.T[i,j]!=0:
                plt.text(x=j+0.15, y=i+0.6, s=str(fisher_rounded.T[i,j]),fontsize=10)

    plt.tight_layout()

    return fig

@table
def fisher_information_by_sector(dataset_inputs, theoryid, groups_covmat, simu_parameters_names, pdf):
    """
    Obtains the Fisher information matrices for each of the BSM sectors.

    This function computes the Fisher information matrix for each sector in a dataset, providing a
    measure of the amount of information each sector contributes to the parameters. The function
    accumulates datasets by sectors, calculates the reduced covariance matrices for each sector,
    and computes the Fisher information matrices. The diagonal elements of these matrices are
    extracted and normalized to provide a comparative view across sectors.

    Parameters
    ----------
    dataset_inputs : list of Dataset objects
        The datasets used for computing the Fisher information matrices.
    theoryid : int or array-like
        Theory identifier(s) associated with the datasets.
    groups_covmat : pd.DataFrame
        Covariance matrices for the groups in the datasets.
    simu_parameters_names : list of str
        List of names of the simulation parameters.
    pdf : PDF object
        The parton distribution function object.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the normalized diagonal elements of the Fisher information matrices,
        indexed by simulation parameters and with sectors as columns.
    """
    # First, get the names of the BSM sectors.

    bsm_dataset_inputs_sectors = {} 

    for dataset in dataset_inputs:
        if dataset.bsm_sector in bsm_dataset_inputs_sectors.keys():
            bsm_dataset_inputs_sectors[dataset.bsm_sector] += [dataset]
        else:
            bsm_dataset_inputs_sectors[dataset.bsm_sector] = [dataset]
    
    all_sectors_duplicates = list(bsm_dataset_inputs_sectors.keys())
    all_sectors = []
    [all_sectors.append(x) for x in all_sectors_duplicates if x not in all_sectors]

    fisher_by_sector = []

    for sec in all_sectors:
        if sec in bsm_dataset_inputs_sectors.keys():
            datasets = bsm_dataset_inputs_sectors[sec]
            dataset_names = [ds.name for ds in datasets]
        else:
            datasets = None
            dataset_names = []

        ds_and_fo_names = dataset_names

        # Take correct submatrix of groups_covmat
        reduced_covmats = []
        for name in ds_and_fo_names:
            reduced_covmats += [groups_covmat.xs(name, axis=1, level=1, drop_level=False)]
        
        reduced_covmat = pd.concat(reduced_covmats, axis=1)

        reduced_covmats = []
        for name in ds_and_fo_names:
            reduced_covmats += [reduced_covmat.T.xs(name, axis=1, level=1, drop_level=False)]

        reduced_covmat = pd.concat(reduced_covmats, axis=1)

        # Hence construct the Fisher matrices
        fisher_by_sector += [_compute_fisher_information_matrix(datasets, theoryid, reduced_covmat, simu_parameters_names, pdf)]

    # Now go through the matrices one-by-one, and take the diagonal
    fisher_diags_by_sector = []

    for matrix in fisher_by_sector:
        diagonal = np.diagonal(matrix.to_numpy())
        fisher_diags_by_sector += [diagonal.tolist()] 

    # Rescale array
    array = np.array(fisher_diags_by_sector).T
    sums = np.sum(array, axis=1)
    rows, columns = array.shape
    for i in range(rows):
        array[i,:] = array[i,:] / sums[i]*100

    df = pd.DataFrame(array, columns=all_sectors, index=simu_parameters_names)
    
    return df

def _compute_fisher_information_matrix(dataset_inputs, theoryid, groups_covmat, simu_parameters_names, pdf):
    """
    Computes the Fisher information matrix for a given set of datasets and simulation parameters.

    This function calculates the Fisher information matrix, which quantifies the amount of information
    that an observable random variable carries about an unknown parameter upon which the probability
    of the random variable depends. It takes into account the datasets, theory IDs, groups covariance
    matrix, simulation parameters, and the parton distribution function (PDF).

    Parameters
    ----------
    dataset_inputs : list of Dataset objects
        The datasets used for computing the Fisher information matrix.
    theoryid : int or array-like
        Theory identifier(s) associated with the datasets.
    groups_covmat : pd.DataFrame
        Covariance matrices for the groups in the datasets.
    simu_parameters_names : list of str
        List of names of the simulation parameters.
    pdf : PDF object
        The parton distribution function object.

    Returns
    -------
    pd.DataFrame
        The computed Fisher information matrix as a pandas DataFrame.
    """
    bsm_factors = []
    if dataset_inputs is not None:
        for dataset in dataset_inputs:
            ds = l.check_dataset(name=dataset.name, theoryid=theoryid, cfac=dataset.cfac, simu_parameters_names=dataset.simu_parameters_names, simu_parameters_linear_combinations=dataset.simu_parameters_linear_combinations, use_fixed_predictions=dataset.use_fixed_predictions)
            bsm_fac = parse_simu_parameters_names_CF(ds.simu_parameters_names_CF, ds.simu_parameters_linear_combinations, cuts=ds.cuts)
            central_sm = central_predictions(ds, pdf)
            coefficients = central_sm.to_numpy().T * np.array([i.central_value for i in bsm_fac.values()])
            bsm_factors += [coefficients] 

    # Make bsm_factors into a nice numpy array. 
    bsm_factors = np.concatenate(bsm_factors, axis=1).T

    # The rows are the data, the columns are the operator
    cov = groups_covmat.to_numpy()
    inv_cov = np.linalg.inv(cov)
    fisher = bsm_factors.T @ inv_cov @ bsm_factors

    fisher = pd.DataFrame(fisher, index=simu_parameters_names)
    fisher = fisher.T
    fisher.index = simu_parameters_names

    return fisher

@table
def principal_component_values(fisher_information_matrix):
    """
    Returns the eigenvalues corresponding to the various principal directions.

    This function performs a principal component analysis (PCA) on the Fisher information matrix
    to return the eigenvalues. These eigenvalues represent the variance of the data along the
    principal components, which can be used to understand the dominant directions in the data.

    Parameters
    ----------
    fisher_information_matrix : pd.DataFrame
        The Fisher information matrix.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the eigenvalues of the Fisher information matrix.
    """
    fisher = fisher_information_matrix.to_numpy()
    fisher = fisher - fisher.mean(axis=0)
    _, values, _ = np.linalg.svd(fisher)
    values = pd.DataFrame(values)
    return values

@table
def principal_component_vectors(fisher_information_matrix, simu_parameters_names):
    """
    Performs a principal component analysis to obtain the principal directions (vectors).

    This function calculates the principal component vectors from the Fisher information matrix,
    providing insights into the flat directions of the parameter space. These directions can
    indicate combinations of parameters that are less constrained by the data.

    Parameters
    ----------
    fisher_information_matrix : pd.DataFrame
        The Fisher information matrix.
    simu_parameters_names : list of str
        Names of the simulation parameters.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the principal component vectors, indexed by the simulation parameters.
    """
    fisher = fisher_information_matrix.to_numpy()
    fisher = fisher - fisher.mean(axis=0)
    _, _, vectors = np.linalg.svd(fisher)
    vectors = pd.DataFrame(vectors, columns=simu_parameters_names)
    return vectors
