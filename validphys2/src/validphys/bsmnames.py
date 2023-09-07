"`""Helpers for parsing bsm"""

import pandas as pd
import numpy as np


def linear_datum_to_op(name:str):
    """Convert something like "None_OtZ" to OtZ"""
    return name.rsplit("_", 1)[1]

def get_bsm_data(
    bsm_order,
    bsm_fac_data,
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
    bsm_order : str,
            specifies the order of the bsm k-factors

    bsm_fac_data : list
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

    if bsm_fac_data is not None and bsm_order is not None:
        new_bsm_fac_data_names = [
        bsm_order + "_" + op for op in bsm_fac_data_names
        ]
        
    return {
        "bsm_fac_data_names": new_bsm_fac_data_names,
    }
