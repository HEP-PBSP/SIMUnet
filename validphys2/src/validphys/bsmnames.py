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
    n_simu_parameters,
    simu_parameters_linear_combinations,
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
    simu_parameters_linear_combinations: list of dictionaries
                        list containing the linear combinations that each of
                        the parameters are defined in terms of

    Returns
    -------
    dict
        - simu_parameters_names : list containing names of bsm k-factors
        - simu_parameters_linear_combinations: list containing the linear combinations
            that each of the parameters are defined in terms of, now with order attached

    """
    # default value
    new_simu_parameters_names = None
    new_simu_parameters_linear_combinations = None

    if simu_parameters is not None and simu_fac is not None:
        new_simu_parameters_names = [
        simu_fac + "_" + op for op in simu_parameters_names
        ]

        new_simu_parameters_linear_combinations = {}
        index = 0
        for dictionary in simu_parameters_linear_combinations:
            new_simu_parameters_linear_combinations[simu_fac + "_" +simu_parameters_names[index]] = dictionary
            index += 1

    return {
        "simu_parameters_names": new_simu_parameters_names,
        "simu_parameters_linear_combinations": new_simu_parameters_linear_combinations,
    }
