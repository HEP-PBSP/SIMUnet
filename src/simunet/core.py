import functools

from validphys.core import DataSetInput, DataSetSpec, FKTableSpec


class SIMUnetDataSetSpec(DataSetSpec):
    def __init__(
        self,
        *,
        name,
        commondata,
        fkspecs,
        thspec,
        cuts,
        frac=1,
        op=None,
        weight=1,
        simu_parameters_names_CF=None,
        simu_parameters_names=None,
        simu_parameters_linear_combinations=None,
        use_fixed_predictions=False,
        contamination=None,
        contamination_data=None,
    ):
        super().__init__(
            name=name,
            commondata=commondata,
            fkspecs=fkspecs,
            thspec=thspec,
            cuts=cuts,
            frac=frac,
            op=op,
            weight=weight,
            rules=(),
        )
        self.simu_parameters_names_CF = simu_parameters_names_CF

        self.simu_parameters_names = simu_parameters_names
        self.simu_parameters_linear_combinations = simu_parameters_linear_combinations
        self.use_fixed_predictions = use_fixed_predictions
        self.contamination = contamination
        self.contamination_data = contamination_data

    @functools.lru_cache
    def load_commondata(self):
        """Attaches contamination to the loaded commondata"""

        cd = super().load_commondata()

        cd.contamination = self.contamination
        cd.contamination_data = self.contamination_data

        return cd


class SIMUnetFKTableSpec(FKTableSpec):
    def __init__(
        self,
        fkpath,
        cfactors,
        metadata=None,
        use_fixed_predictions=False,
        fixed_predictions_path=None,
        contamination=None,
    ):
        super().__init__(fkpath=fkpath, cfactors=cfactors, metadata=metadata)
        self.use_fixed_predictions = use_fixed_predictions
        self.fixed_predictions_path = fixed_predictions_path
        self.contamination = contamination


class SIMUnetDataSetInput(DataSetInput):
    def __init__(
        self,
        *,
        name,
        cfac,
        frac,
        weight,
        custom_group,
        variant,
        simu_parameters_names,
        simu_parameters_linear_combinations,
        use_fixed_predictions,
        contamination,
        simu_fac,
    ):
        super().__init__(
            name=name,
            cfac=cfac,
            frac=frac,
            weight=weight,
            custom_group=custom_group,
            variant=variant,
        )

        self.simu_parameters_names = simu_parameters_names
        self.simu_parameters_linear_combinations = simu_parameters_linear_combinations
        self.use_fixed_predictions = use_fixed_predictions
        self.contamination = contamination
        self.simu_fac = simu_fac
