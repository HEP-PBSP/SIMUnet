"""
Contains the providers that n3fit will use and that we want to override
"""

import numpy as np

from validphys.n3fit_data import fittable_datasets_masked as vanilla_fittable_datasets_masked
from validphys.utils import yaml_safe
from simunet import simufit

# I'm assuming the information necessary is in the data and needs to be propagated to the fittable dataset
# minimal changes are necessary if instead we need to propagate this to the fktable instead


def fittable_datasets_masked(data, simu_layer=None, simu_parameters=None, analytic_initialisation=False):
    """Note: for anayltic solution the data must be grouped together (default in simunet: ALL)."""

    ret = vanilla_fittable_datasets_masked(data)
    if simu_layer is None:
        return ret

    if simufit._REGISTRY.get("layer") is None:
        if analytic_initialisation:
            # TODO: modify `simu_parameters`
            # TODO: check that after the dictionary is modified here, it is also modified inside the layer
            # otherwise a `layer._update_parameters` method needs to be added
            pass
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
