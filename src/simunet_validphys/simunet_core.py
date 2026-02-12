from validphys.core import DataSetSpec, FKTableSpec
import functools
import numpy as np
# from NNPDF import (LHAPDFSet as libNNPDF_LHAPDFSet,
#     CommonData,
#     FKTable,
#     FKSet,
#     DataSet,
#     Experiment,
#     PositivitySet,)
from validphys.utils import yaml_safe

class SIMUnetDataSetSpec(DataSetSpec):
    def __init__(self, *, name, commondata, fkspecs, thspec, cuts,
                 frac=1, op=None, weight=1, simu_parameters_names_CF=None, simu_parameters_names=None, simu_parameters_linear_combinations=None, use_fixed_predictions=False, contamination=None, contamination_data=None):
        super().__init__(
            name=name,
            commondata=commondata,
            fkspecs=fkspecs,
            thspec=thspec,
            cuts=cuts,
            frac=frac,
            op=op,
            weight=weight,
            rules=()
        )
        self.simu_parameters_names_CF = simu_parameters_names_CF 

        # These are important because they are ORDERED correctly, but the dictionaries might not be
        self.simu_parameters_names = simu_parameters_names
        self.simu_parameters_linear_combinations = simu_parameters_linear_combinations
        self.use_fixed_predictions = use_fixed_predictions
        self.contamination = contamination
        self.contamination_data = contamination_data
