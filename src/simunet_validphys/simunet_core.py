from validphys.core import DataSetSpec, FKTableSpec, DataSetInput
import functools


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

        # These are important because they are ORDERED correctly, but the dictionaries might not be
        self.simu_parameters_names = simu_parameters_names
        self.simu_parameters_linear_combinations = simu_parameters_linear_combinations
        self.use_fixed_predictions = use_fixed_predictions
        self.contamination = contamination
        self.contamination_data = contamination_data

    @functools.lru_cache
    def load_commondata(self):
        """Strips the commondata loading from `load`"""

        cd = self.commondata.load()

        if self.cuts is not None:
            loaded_cuts = self.cuts.load()
            if not (hasattr(loaded_cuts, "_full") and loaded_cuts._full):
                intmask = [int(ele) for ele in loaded_cuts]
                cd = cd.with_cuts(intmask)

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

        self.legacy = False

        # NOTE: The legacy interface is currently used by fkparser to decide
        # whether to read an FKTable using the old parser or the pineappl parser
        # this attribute (and the difference between both) might be removed in future
        # releases of NNPDF so please don't write code that relies on it
        if not isinstance(fkpath, (tuple, list)):
            self.legacy = True
        else:
            # Make it into a tuple only for the new format
            fkpath = tuple(fkpath)

        self.fkpath = fkpath
        self.metadata = metadata

        # For non-legacy theory, add the metadata since it defines how the theory is to be loaded
        # and thus, it should also define the hash of the class
        if not self.legacy:
            super().__init__(fkpath, cfactors, self.metadata)
        else:
            super().__init__(fkpath, cfactors)


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
