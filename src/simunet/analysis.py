from reportengine.figure import figuregen
from simunet.loader import SIMUnetLoader
from validphys.dataplots import plot_fancy
import logging
import numpy as np
import os
from validphys.utils import yaml_safe

from validphys.commondata import loaded_commondata_with_cuts
from validphys.covmats import sqrt_covmat

from reportengine.table import table
from reportengine.figure import figure, figuregen
from validphys.convolution import central_predictions

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import seaborn as sns
log = logging.getLogger(__name__)
l = SIMUnetLoader()


def load_datasets_contamination(data):
    """
    Parameters
    ----------

    contamination_parameters: dict with

    theoryid: TheoryIDSpec

    dataset_inputs: NSList of DataSetInput objects

    Returns
    -------

    dict
        dictionary of BSM k-factors to apply on certain datasets

    """

    bsm_dict = {}

    for dataset in data.datasets:
        if dataset.cuts is not None:
            cuts = dataset.cuts.load()
        cont_params = dataset.contamination_data

        simu_dict = l.get_simu_parameters_name_dict(
            dataset.name, simu_parameters_names=[dataset.contamination]
        )
        cont_path = list(simu_dict.values())[0]
        cont_params = dataset.contamination_data
        cont_order = dataset.contamination

        if cont_order == None:
            log.warning(f"{dataset.name} is not contaminated. Is it right?")

            bsm_dict[dataset.name] = np.ones(dataset.commondata.ndata)

        elif not os.path.exists(cont_path):
            log.error(
                f"Could not find a BSM-factor for {dataset.name}. Are you sure they exist in the given theory?"
            )
            bsm_dict[dataset.name] = np.ones(dataset.commondata.ndata)
        else:
            log.info(f"Loading {dataset.name}.")
            with open(cont_path, "r+") as stream:
                simu_card = yaml_safe.load(stream)
            stream.close()

            k_factors = np.zeros(len(cuts))
            for param in cont_params:
                value = param["value"]
                lin_comb = param["linear_combination"]
                bsm_xs = np.zeros(len(cuts))
                for op in lin_comb:
                    if op in simu_card[dataset.contamination]:
                        bsm_xs += (
                            lin_comb.get(op, 0)
                            * np.array(simu_card[dataset.contamination][op])[cuts]
                        )
                    else:
                        # Log a warning and keep SMEFT K-factor to zero
                        log.warning(
                            f"Operator '{op}' not found for {dataset.name}. Setting K-factor to zero."
                        )
                k_factors += (
                    value
                    * bsm_xs
                    / np.array(simu_card[dataset.contamination]["SM"])[cuts]
                )
            bsm_dict[dataset.name] = k_factors

    return bsm_dict


@figuregen
def plot_data_theory_contaminated(simunet_one_or_more_results, commondata, cuts):
    return plot_fancy(simunet_one_or_more_results, commondata, cuts)


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
        # import IPython; IPython.embed()
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
            ds = l.check_dataset(name=dataset.name, theoryid=theoryid, cfac=dataset.cfac, simu_parameters_names=dataset.simu_parameters_names, simu_parameters_linear_combinations=dataset.simu_parameters_linear_combinations, use_fixed_predictions=dataset.use_fixed_predictions, variant=dataset.variant)
            cuts = ds.cuts.load()
            ndat = len(cuts)
            simu_dict =l.get_simu_parameters_name_dict(
                ds.name, simu_parameters_names=ds.simu_parameters_names
            )
            simu_path = list(simu_dict.values())[0]
            with open(simu_path, "rb") as stream:
                simu_info = yaml_safe.load(stream)
            central_sm = central_predictions(ds, pdf)
            # bsm_factors = []
            operator_central_values = []
            for param, lincomb in ds.simu_parameters_linear_combinations.items():

                model = "_".join(param.split("_")[:-1])

                numerator = np.zeros(ndat)
                for op, coeff in lincomb.items():
                    if op in simu_info[model]:
                        numerator += coeff * np.array(simu_info[model][op])[cuts]
                sm_prediction = np.array(simu_info[model]["SM"])[cuts]
                central = numerator / sm_prediction
                operator_central_values.append(central)
          
            coefficients = central_sm.to_numpy().T * np.array(operator_central_values)
            bsm_factors+= [coefficients]
            # bsm_fac = parse_simu_parameters_names_CF(ds.simu_parameters_names_CF, ds.simu_parameters_linear_combinations, cuts=ds.cuts)
            # central_sm = central_predictions(ds, pdf)
            # coefficients = central_sm.to_numpy().T * np.array([i.central_value for i in bsm_fac.values()])
            # bsm_factors += [coefficients] 

    # Make bsm_factors into a nice numpy array. 
    # The rows are the data, the columns are the operator
    bsm_factors = np.concatenate(bsm_factors, axis=1).T
    cov = groups_covmat.to_numpy()
    inv_cov = np.linalg.inv(cov)
    fisher = bsm_factors.T @ inv_cov @ bsm_factors

    fisher = pd.DataFrame(fisher, index=simu_parameters_names)
    fisher = fisher.T
    fisher.index = simu_parameters_names

    return fisher

@table
def fisher_information_matrix(dataset_inputs, theoryid, groups_covmat, simu_parameters_names, pdf):
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