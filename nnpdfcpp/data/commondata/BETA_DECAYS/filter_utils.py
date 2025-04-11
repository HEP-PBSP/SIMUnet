"""
This module contains helper functions that are used to extract the data values 
from the rawdata files.
"""

import yaml
import pandas as pd
import numpy as np


def get_data_values():
    """
    returns the central data values in the form of a list.
    """

    data_central = []

    raw_data = f"rawdata/central.yaml"

    with open(raw_data, 'r') as file:
        input = yaml.safe_load(file)

    values_string = input['central_vals']
    values =  [float(val.strip()) for val in values_string.split(",")]

    for value in values:
        data_central.append(value)

    return data_central


def get_kinematics():
    """
    returns the kinematics in the form of a list of dictionaries.
    """
    kin = []

    raw_data = f"rawdata/central.yaml"

    with open(raw_data, 'r') as file:
        input = yaml.safe_load(file)

    values_string = input['central_vals']
    values =  [float(val.strip()) for val in values_string.split(",")]
    num_of_bins = len(values)

    for i in range(num_of_bins):
        kin_value = {
            'abs_eta': {'min': None, 'mid': 1 ,'max': None},
            'm_W2': {'min': None, 'mid': 1, 'max': None},
            'sqrts': {'min': None, 'mid': 1, 'max': None},
        }
        kin.append(kin_value)

    return kin


def get_uncertainties():
    """
    returns the uncertainties.
    """
    covmat_path = 'rawdata/covmat.yaml'

    uncertainties = []

    data_central = get_data_values()

    with open(covmat_path, 'r') as file:
        data = yaml.safe_load(file)

    stats = []
    for row in data['experimental_cov']:
        for value in row:
            if value !=0:
                unc = round(np.sqrt(value),10)
                stats.append(unc)
   
    uncertainties.append([{"name": 'stat', "values": stats}])
    uncertainties.append([{"name":'uncor',"values": np.zeros(43)}])

    return uncertainties

if __name__ == "__main__":
    get_kinematics()