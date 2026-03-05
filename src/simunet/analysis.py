from reportengine.figure import figuregen
from simunet.loader import SIMUnetLoader
from validphys.dataplots import plot_fancy
import logging
import numpy as np
import os
from validphys.utils import yaml_safe

log = logging.getLogger(__name__)
l = SIMUnetLoader()


def load_datasets_contamination(contamination_parameters, theoryid, dataset_inputs):
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

    cont_path = l.simudata_path / "simu_factors"

    cont_names, cont_values, cont_lin_combs = [], [], []
    for c in contamination_parameters:
        cont_names.append(c["name"])
        cont_values.append(c["value"])
        cont_lin_combs.append(c["linear_combination"])

    bsm_dict = {}

    for dataset in dataset_inputs:

        bsmfile = cont_path / f"SIMU_{dataset.name}.yaml"

        cont_order = dataset.contamination

        if cont_order == None:
            log.warning(f"{dataset.name} is not contaminated. Is it right?")
            try:
                bsm_dict[dataset.name] = np.ones(dataset.commondata.ndata)
            except AttributeError:
                data = l.check_dataset(
                    dataset.name,
                    cfac=dataset.cfac,
                    theoryid=theoryid,
                    new_commondata=dataset.new_commondata,
                )
                bsm_dict[dataset.name] = np.ones(data.commondata.ndata)

        elif not os.path.exists(bsmfile):
            log.error(
                f"Could not find a BSM-factor for {dataset.name}. Are you sure they exist in the given theory?"
            )
            bsm_dict[dataset.name] = np.ones(dataset.commondata.ndata)
        else:
            log.info(f"Loading {dataset.name}.")
            with open(bsmfile, "r+") as stream:
                simu_card = yaml_safe.load(stream)
            stream.close()

            k_factors = np.ones(len(simu_card[cont_order]["SM"]))
            for cont_value, cont_lin_comb in zip(cont_values, cont_lin_combs):
                k_fac = np.zeros(len(simu_card[cont_order]["SM"]))
                for op in cont_lin_comb:
                    # Check if the operator exists in simu_card[dataset.contamination]
                    if op in simu_card[dataset.contamination]:
                        k_fac += cont_lin_comb[op] * np.array(simu_card[cont_order][op])
                    else:
                        # Log a warning and keep SMEFT K-factor to zero
                        log.warning(
                            f"Operator '{op}' not found for {dataset.name}. Setting K-factor to zero."
                        )
                k_factors += k_fac * cont_value / np.array(simu_card[cont_order]["SM"])

            bsm_dict[dataset.name] = k_factors

    return bsm_dict


@figuregen
def plot_data_theory_contaminated(simunet_one_or_more_results, commondata, cuts):
    # Add contamination to theory predictions in results
    return plot_fancy(simunet_one_or_more_results, commondata, cuts)
