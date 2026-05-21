from reportengine.figure import figuregen
from simunet.loader import SIMUnetLoader
from validphys.dataplots import plot_fancy
import logging
import numpy as np
import os
from validphys.utils import yaml_safe

from validphys.commondata import loaded_commondata_with_cuts
from validphys.covmats import sqrt_covmat

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
