"`""Helpers for parsing bsm"""

import pandas as pd
import numpy as np


def linear_datum_to_op(name:str):
    """Convert something like "None_OtZ" to OtZ"""
    return name.rsplit("_", 1)[1]

def get_bsm_data(
    simu_fac,
    simu_parameters,
    bsm_fac_data_names,
    n_bsm_fac_data
):
    """
    Given BSM specifications within dataset_inputs specs
    return names of bsm kfactors stored in the data/bsm_factors
    folder.

    The output is used by config.parse_dataset_input to add
    bsm_fac_data_names to the DataSetInput class constructor.

    Parameters
    ----------
    simu_fac : str,
            specifies the order of the bsm k-factors

    simu_parameters : list
    bsm_fac_data_names : list
                        list containing names of the dimension 6 operators
                        read from the runcard with production rule
    n_bsm_fac_data: int

    Returns
    -------
    dict
        - bsm_fac_data_names : list containing names of bsm k-factors

        - bsm_sector : str
    

    """
    # default value
    new_bsm_fac_data_names = None

    if simu_parameters is not None and simu_fac is not None:
        new_bsm_fac_data_names = [
        simu_fac + "_" + op for op in bsm_fac_data_names
        ]
        
    return {
        "bsm_fac_data_names": new_bsm_fac_data_names,
    }
