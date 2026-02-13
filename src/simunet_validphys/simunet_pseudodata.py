from simunet_validphys.simunet_loader import SIMUnetLoader
from validphys.utils import yaml_safe
import logging
import numpy as np

log = logging.getLogger(__name__)

l = SIMUnetLoader()


def level0_commondata_wc(data, fakepdf):
    """
    Given a validphys.core.DataGroupSpec object, load commondata and
    generate a new commondata instance with central values replaced
    by fakepdf prediction

    Parameters
    ----------

    data : validphys.core.DataGroupSpec
    which contains simunet_validphys.simunet_core.SIMUnetDataSetSpec datasets

    fakepdf: validphys.core.PDF

    Returns
    -------
    list
        list of validphys.coredata.CommonData instances corresponding to
        all datasets within one experiment. The central value is replaced
        by Level 0 fake data.

    Example
    -------

    """
    from validphys.covmats import dataset_t0_predictions

    level0_commondata_instances_wc = []

    for dataset in data.datasets:
        commondata_wc = dataset.commondata.load()
        if dataset.cuts is not None:
            cuts = dataset.cuts.load()
            commondata_wc = commondata_wc.with_cuts(cuts=cuts)

        # == Generate a new CommonData instance with central value given by Level 0 data generated with fakepdf ==#
        t0_prediction = dataset_t0_predictions(t0dataset=dataset, t0set=fakepdf)
        # Contamination

        if dataset.contamination:
            simu_dict = l.get_simu_parameters_name_dict(
                dataset.name,
                simu_parameters_names=[dataset.contamination],  # dummy list
            )

            # extract the path
            cont_path = list(simu_dict.values())[0]

            # load contamination parameters
            cont_params = dataset.contamination_data

            # load simu_card file
            with open(cont_path, "r+") as stream:
                simu_card = yaml_safe.load(stream)
            stream.close()
            # K-factors loading
            k_factor = np.zeros(len(t0_prediction))
            if cont_params:
                for param in cont_params:
                    # load the k_fac value
                    value = param["value"]
                    # load the linear combination coefficients
                    lin_comb = param["linear_combination"]
                    # load the BMS cross-section
                    bsm_xs = np.zeros(len(t0_prediction))
                    for op in lin_comb:
                        # Check if the operator exists in simu_card[dataset.contamination]
                        if op in simu_card[dataset.contamination]:
                            bsm_xs += (
                                lin_comb.get(op, 0)
                                * np.array(simu_card[dataset.contamination][op])[cuts]
                            )
                        else:
                            # Log a warning or handle the missing operator
                            log.warning(
                                f"Operator '{op}' not found for {dataset.name}. Setting K-factor to zero."
                            )
                    # compute the K-factor correction
                    k_factor += (
                        value
                        * bsm_xs
                        / np.array(simu_card[dataset.contamination]["SM"])[cuts]
                    )
            # update t0 prediction to BSM t0 prediction
            t0_prediction = t0_prediction * (1.0 + k_factor)

        # N.B. cuts already applied to th. pred.
        level0_commondata_instances_wc.append(
            commondata_wc.with_central_value(t0_prediction)
        )

    return level0_commondata_instances_wc
