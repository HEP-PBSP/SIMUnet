"""
Contains the providers that n3fit will use and that we want to override
"""

import numpy as np

from validphys.n3fit_data import fittable_datasets_masked as vanilla_fittable_datasets_masked
from validphys.utils import yaml_safe
from simunet import simufit
from validphys.core import PDF
from simunet.results import SIMUnetThPredictionsResult
import pandas as pd
import scipy as sp

from simunet.loader import SIMUnetLoader
import logging

l = SIMUnetLoader()
log = logging.getLogger(__name__)


def analytic_solution(data, theorySM, theorylin, covmat):
    """
    Returns the minimum of the chi2 function:

      chi2 = (data - theorySM - theorylin c)^T invcovmat (data - theorySM - theorylin c),

    """

    diff = data - theorySM

    theorylin = theorylin

    part1 = np.linalg.solve(covmat, theorylin)
    part2 = np.linalg.solve(covmat, diff)

    sol = np.linalg.solve(theorylin.T @ part1, theorylin.T @ part2)

    minval = (diff - theorylin @ sol).T @ np.linalg.solve(covmat, diff - theorylin @ sol)
    minval = minval / len(diff)

    return (sol, minval)


def construct_analytic_initialisation(
    data,
    theoryid,
    replica,
    analytic_initialisation_pdf,
    make_replica,
    groups_covmat,
    simu_parameters,
    use_th_covmat=False,
):
    """
    Constructs the analytic initialisation for the simu_parameters.
    """
    sm_predictions = []
    linear_bsm = []
    th_covmat = []
    exp_data = make_replica
    # TODO: Check that this changes with contamination
    nop = len(simu_parameters)
    for ds in data:
        dataset_spec = l.check_dataset(
            name=ds.name,
            theoryid=theoryid,
            cfac=ds.cfac,
            contamination=ds.contamination,
            simu_parameters_names=ds.simu_parameters_names,
            simu_parameters_linear_combinations=ds.simu_parameters_linear_combinations,
        )
        cuts = dataset_spec.cuts.load()
        ndat = len(cuts)
        pred_values = SIMUnetThPredictionsResult.from_convolution(
            PDF(analytic_initialisation_pdf), dataset_spec, load_dataset_contamination=None
        ).error_members[replica][cuts]
        sm_predictions.append(pred_values)

        if ds.simu_parameters_names is not None:
            simu_dict = l.get_simu_parameters_name_dict(
                ds.name, simu_parameters_names=ds.simu_parameters_names
            )
            simu_path = list(simu_dict.values())[0]
            with open(simu_path, "rb") as stream:
                simu_info = yaml_safe.load(stream)
            columns = []
            for param in ds.simu_parameters_linear_combinations:
                model = "_".join(param.split("_")[:-1])
                column = np.zeros((ndat,))
                for key in ds.simu_parameters_linear_combinations[param]:
                    if key in simu_info[model].keys():
                        model_values = [simu_info[model][key][i] for i in cuts]
                        column += np.array(
                            model_values * ds.simu_parameters_linear_combinations[param][key]
                        )
                column = column / np.array([simu_info[model]["SM"][i] for i in cuts]) * pred_values
                columns += [column]
            linear_bsm.append(np.array(columns).T)

            if (
                use_th_covmat == True
                and "theory_cov" in simu_info.keys()
                and len(simu_info["theory_cov"]) > 0
            ):
                th_covmat += [np.array(simu_info["theory_cov"])]
            else:
                th_covmat += [np.zeros((ndat, ndat))]
        else:

            linear_bsm.append(np.zeros((ndat, nop)))
            th_covmat += [np.zeros((ndat, ndat))]

    sm_predictions = np.concatenate(sm_predictions)
    linear_bsm = np.concatenate(linear_bsm)

    th_covmat = sp.linalg.block_diag(*th_covmat)
    th_covmat = th_covmat.T
    total_covmat = groups_covmat + th_covmat

    sol, minval = analytic_solution(exp_data, sm_predictions, linear_bsm, total_covmat)
    simu_parameters_scales = [1 / abs(ini) for ini in sol]
    log.info("The analytic solution is " + str(sol))
    log.info("The minimum is achieved at chi2=" + str(minval))
    for param, scale, init in zip(simu_parameters, simu_parameters_scales, sol):
        param["scale"] = float(scale)
        param["initialisation"] = {"type": "constant", "value": float(init)}
    return simu_parameters


# I'm assuming the information necessary is in the data and needs to be propagated to the fittable dataset
# minimal changes are necessary if instead we need to propagate this to the fktable instead


def fittable_datasets_masked(
    data,
    make_replica,
    replica,
    theoryid,
    groups_covmat,
    simu_layer=None,
    simu_parameters=None,
    analytic_initialisation=False,
    analytic_initialisation_pdf=None,
    use_th_covmat=False,
):
    # TODO: set analytic_intialisation to True if initialisation in runcard is analytic
    # TODO: Looks at use_th_covmat
    """Note: for anayltic solution the data must be grouped together (default in simunet: ALL)."""

    ret = vanilla_fittable_datasets_masked(data)
    if simu_layer is None:
        return ret

    if simufit._REGISTRY.get("layer") is None:
        if analytic_initialisation:
            simu_parameters = construct_analytic_initialisation(
                data,
                theoryid,
                replica,
                analytic_initialisation_pdf,
                make_replica,
                groups_covmat,
                simu_parameters,
                use_th_covmat=use_th_covmat,
            )
        simu_layer_generated = simu_layer(simu_parameters)
        simufit._REGISTRY["layer"] = simu_layer_generated
    else:
        simu_layer_generated = simufit._REGISTRY["layer"]

    # At this point we have the information on the simunet parameters twice
    # once in the `simu_layer` and once in
    # data[X].simu_parameters_linear_combinations
    # but `simu_layer` is the right one (they could be made to be the same)

    # The cfactors themselves must be part of the fittable dataset (because they are cfactors applied to the whole dataset)
    # so this function must
    # 1) Read all cfactors that are asked by `simu_fac` of each dataset
    # 2) Pass a dictionary of [key] : [value] to the fittable dataset

    # Loop over the SIMUnetDataSetSpec
    for data_input, dataset, fittable_dataset in zip(data, data.datasets, ret):
        if dataset.simu_parameters_names_CF is None:
            # Nothing to do here
            continue

        # Loop over the cfactors that have been parsed
        # TODO (will there be ever more than one? if so... how to deal with it?)
        cuts = dataset.cuts.load().tolist()
        for cfac_file in dataset.simu_parameters_names_CF.values():
            with open(cfac_file, "rb") as stream:
                cfac_data = yaml_safe.load(stream)

            cfactors_raw = simu_layer_generated.apply_linear_comb(cfac_data[data_input.simu_fac])
            cfactors = [np.take(i, indices=cuts, mode="clip") for i in cfactors_raw]
            break

        # TODO this is ugly, but needs to be beautified in n3fit not here
        fittable_dataset.fktables_data[0].simunet_cfactors = cfactors
        fittable_dataset.fktables_data[0].simunet_layer = simu_layer_generated

    return ret
