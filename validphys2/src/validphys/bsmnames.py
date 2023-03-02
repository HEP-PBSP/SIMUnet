"`""Helpers for parsing bsm"""

import pandas as pd
import numpy as np

def linear_datum_to_op(name:str):
    """Convert something like "None_OtZ" to OtZ"""
    return name.rsplit("_", 1)[1]

def get_bsm_data(
    bsm_sector,
    bsm_order,
    bsm_fac_data,
    bsm_sector_data,
    bsm_fac_data_names,
    n_bsm_fac_data,
):
    if bsm_fac_data is not None:
        if bsm_order is not None:
            new_bsm_fac_data_names = []
            # Now go through all operators, and see which are in the appropriate sector.
            for op in bsm_fac_data_names:
                if op in bsm_sector_data[bsm_sector]:
                    if bsm_order == "LO_QUAD":
                        new_bsm_fac_data_names += ["LO_LIN_" + op]
                    elif bsm_order == "NLO_QUAD":
                        new_bsm_fac_data_names += ["NLO_LIN_" + op]
                    else:
                        new_bsm_fac_data_names += [bsm_order + "_" + op]
                else:
                    new_bsm_fac_data_names += ["None_" + op]

            # This takes care of linear. Now need to do the same for quadratic, if it's
            # switched on...

            if bsm_order in ["LO_QUAD", "NLO_QUAD"]:
                # Some care needed here...
                new_bsm_fac_quad_names = pd.DataFrame(
                    index=range(n_bsm_fac_data), columns=range(n_bsm_fac_data)
                )

                for i in range(n_bsm_fac_data):
                    for j in range(n_bsm_fac_data):
                        if bsm_fac_data_names[i] in bsm_sector_data[bsm_sector]:
                            if bsm_fac_data_names[j] in bsm_sector_data[bsm_sector]:
                                if i == j:
                                    new_bsm_fac_quad_names.iloc[i, j] = (
                                        bsm_order + "_" + bsm_fac_data_names[i]
                                    )
                                else:
                                    new_bsm_fac_quad_names.iloc[i, j] = (
                                        bsm_order
                                        + "_"
                                        + bsm_fac_data_names[i]
                                        + "*"
                                        + bsm_fac_data_names[j]
                                    )
                                    new_bsm_fac_quad_names.iloc[j, i] = (
                                        bsm_order
                                        + "_"
                                        + bsm_fac_data_names[j]
                                        + "*"
                                        + bsm_fac_data_names[i]
                                    )
                            else:
                                if i == j:
                                    new_bsm_fac_quad_names.iloc[i, j] = (
                                        "None_" + bsm_fac_data_names[i]
                                    )
                                else:
                                    # Ignore contribution entirely
                                    new_bsm_fac_quad_names.iloc[i, j] = (
                                        "None_"
                                        + bsm_fac_data_names[i]
                                        + "*"
                                        + bsm_fac_data_names[j]
                                    )
                                    new_bsm_fac_quad_names.iloc[j, i] = (
                                        "None_"
                                        + bsm_fac_data_names[j]
                                        + "*"
                                        + bsm_fac_data_names[i]
                                    )
                        else:
                            if i == j:
                                new_bsm_fac_quad_names.iloc[i, j] = (
                                    "None_" + bsm_fac_data_names[i]
                                )
                            else:
                                new_bsm_fac_quad_names.iloc[i, j] = (
                                    "None_"
                                    + bsm_fac_data_names[i]
                                    + "*"
                                    + bsm_fac_data_names[j]
                                )
                                new_bsm_fac_quad_names.iloc[j, i] = (
                                    "None_"
                                    + bsm_fac_data_names[j]
                                    + "*"
                                    + bsm_fac_data_names[i]
                                )
                new_bsm_fac_quad_names = new_bsm_fac_quad_names.values.tolist()
            else:
                new_bsm_fac_quad_names = None

        else:
            new_bsm_fac_data_names = None
            new_bsm_fac_quad_names = None

    else:
        new_bsm_fac_data_names = None
        new_bsm_fac_quad_names = None

    return {
        "bsm_fac_data_names": new_bsm_fac_data_names,
        "bsm_fac_quad_names": new_bsm_fac_quad_names,
        "bsm_sector": bsm_sector,
    }

def get_bsm_coefs(ds):
    bsm_fac_data_names_CF = ds.bsm_fac_data_names_CF
    bsm_fac_quad_names_CF = ds.bsm_fac_quad_names_CF
    bsm_fac_data_names = ds.bsm_fac_data_names
    bsm_fac_quad_names = ds.bsm_fac_quad_names

    # It's useful to flatten the list of quadratic names first
    if bsm_fac_quad_names_CF is not None:
        flat_bsm_fac_quad_names = []
        for i in range(len(bsm_fac_quad_names)):
            for j in range(len(bsm_fac_quad_names)):
                flat_bsm_fac_quad_names += [bsm_fac_quad_names[i][j]]

    if bsm_fac_data_names_CF is not None:
        coefficients = np.array(
            [bsm_fac_data_names_CF[i].central_value for i in bsm_fac_data_names]
        )
        if bsm_fac_quad_names_CF is not None:
            quad_coefficients = np.array(
                [
                    bsm_fac_quad_names_CF[i].central_value
                    for i in flat_bsm_fac_quad_names
                ]
            )
        else:
            nops, ndat = coefficients.shape
            quad_coefficients = np.zeros((nops**2, ndat))
    return coefficients, quad_coefficients
