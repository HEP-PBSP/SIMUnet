"`""Helpers for parsing bsm"""

import pandas as pd
import numpy as np


def linear_datum_to_op(name:str):
    """Convert something like "None_OtZ" to OtZ"""
    return name.rsplit("_", 1)[1]

def get_bsm_data(
    simu_fac,
    simu_parameters,
    simu_parameters_names,
    n_simu_parameters
):
    """
    Given BSM specifications within dataset_inputs specs
    return names of bsm kfactors stored in the data/bsm_factors
    folder.

    The output is used by config.parse_dataset_input to add
    simu_parameters_names to the DataSetInput class constructor.

    Parameters
    ----------
    simu_fac : str,
            specifies the order of the bsm k-factors

    simu_parameters : list
    simu_parameters_names : list
                        list containing names of the dimension 6 operators
                        read from the runcard with production rule
    n_simu_parameters: int

    Returns
    -------
    dict
        - simu_parameters_names : list containing names of bsm k-factors

        - bsm_sector : str
    

    """
    # default value
    new_simu_parameters_names = None

    if simu_parameters is not None and simu_fac is not None:
        new_simu_parameters_names = [
        simu_fac + "_" + op for op in simu_parameters_names
        ]
        
    return {
        "simu_parameters_names": new_simu_parameters_names,
    }
