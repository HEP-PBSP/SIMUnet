from collections.abc import Mapping, Sequence
import logging
import numbers

from nnpdf_data import legacy_to_new_map
from reportengine import report
from reportengine.configparser import ConfigError, element_of
from reportengine.environment import EnvironmentError_
import simunet.bsmnames as bsmnames
from simunet.core import SIMUnetDataSetInput
from simunet.loader import SIMUFallbackLoader, SIMUnetLoader
from validphys.config import CoreConfig, Environment
from validphys.loader import DataNotFoundError, LoaderError, LoadFailedError
from validphys.plotoptions.core import get_info

log = logging.getLogger(__name__)


class SIMUEnvironment(Environment):
    def __init__(self, *, this_folder=None, net=True, upload=False, dry=False, **kwargs):
        super().__init__(this_folder=this_folder, net=net, upload=upload, dry=dry, **kwargs)
        if not net:
            loader_class = SIMUnetLoader
        elif dry and net:
            log.warning(
                "The --dry flag overrides the --net flag. No resources will be downloaded "
                "while executing a dry run"
            )

            loader_class = SIMUnetLoader
        else:
            loader_class = SIMUFallbackLoader
        try:

            self.loader = loader_class()
        except LoaderError as e:
            log.error("Failed to find the paths. These are configured " "in the nnprofile settings")
            raise EnvironmentError_(e) from e

        self.results_path = self.loader.resultspath
        self.data_paths = self.loader.commondata_folders


class SIMUCoreConfig(CoreConfig):

    @property
    def loader(self):
        return self.environment.loader

    def produce_dataset(
        self,
        *,
        dataset_input,
        theoryid,
        cuts,
        use_fitcommondata=False,
        fit=None,
        check_plotting: bool = False,
        contamination_data=None,  # What is contamination data?
    ):
        """Dataset specification from the theory and CommonData.
        Use the cuts from the fit, if provided. If check_plotting is set to
        True, attempt to lod and check the PLOTTING files
        (note this may cause a noticeable slowdown in general)."""
        log.info("Producing dataset with SIMUCoreConfig")
        name = dataset_input.name
        cfac = dataset_input.cfac
        frac = dataset_input.frac
        weight = dataset_input.weight
        variant = dataset_input.variant
        simu_parameters_names = dataset_input.simu_parameters_names
        simu_parameters_linear_combinations = dataset_input.simu_parameters_linear_combinations
        use_fixed_predictions = dataset_input.use_fixed_predictions
        contamination = dataset_input.contamination
        contamination_data = contamination_data

        try:
            ds = self.loader.check_dataset(
                name=name,
                theoryid=theoryid,
                cfac=cfac,
                cuts=cuts,
                frac=frac,
                use_fitcommondata=use_fitcommondata,
                fit=fit,
                weight=weight,
                variant=variant,
                simu_parameters_names=simu_parameters_names,
                simu_parameters_linear_combinations=simu_parameters_linear_combinations,
                use_fixed_predictions=use_fixed_predictions,
                contamination=contamination,
                contamination_data=contamination_data,
            )
        except DataNotFoundError as e:
            raise ConfigError(str(e), name, self.loader.available_datasets)

        except LoadFailedError as e:
            raise ConfigError(e)
        if check_plotting:
            # normalize=True should check for more stuff
            get_info(ds, normalize=True)
            if not ds.commondata.plotfiles:
                log.warning(f"Plotting files not found for: {ds}")
        return ds

    def produce_simu_parameters_names(self, simu_parameters=None):
        """
        Produces the list of the names of the
        BSM coefficients to include in the fit.
        """
        if simu_parameters is not None:
            simu_parameters_names = []
            for entry in simu_parameters:
                simu_parameters_names += [entry["name"]]
            return simu_parameters_names
        return []

    def produce_simu_parameters_scales(self, simu_parameters=None):
        """Produces the list of rescaling values used to multiply predictions going into the fit."""
        if simu_parameters is not None:
            simu_parameters_scales = []
            for entry in simu_parameters:
                simu_parameters_scales += [entry["scale"]]
            return simu_parameters_scales
        return []

    def produce_n_simu_parameters(self, simu_parameters=None):
        """
        Produces the number of BSM coefficients to include in the fit.
        """
        if simu_parameters is not None:
            return len(simu_parameters)
        return 0

    def produce_contamination_data(self, closuretest):
        """
        Produces the contamination data diction from the closuretest runcard entry

        Example in the runcard:
        -----------------------
        closuretest:
        contamination_parameters:
            - name: 'W'
            value: 0.00008
            linear_combination:
                'Olq3': -15.94
            - name: 'Y'
                value: 0.05
                linear_combination:
                Olq1: 1.51606
                Oed: -6.0606
                Oeu: 12.1394
                Olu: 6.0606
                Old: -3.0394
                Oqe: 3.0394
        """
        log.info("Producing contamination data")
        if "contamination_parameters" in closuretest.keys():
            return closuretest["contamination_parameters"]
        else:
            return None

    def produce_simu_parameters_linear_combinations(self, simu_parameters=None):
        """Produces the list of linear combinations for each of the parameters entering the
        simultaneous fit.
        """
        if simu_parameters is not None:
            simu_parameters_linear_combinations = []
            for entry in simu_parameters:
                if "linear_combination" in entry.keys():
                    simu_parameters_linear_combinations += [entry["linear_combination"]]
                else:
                    simu_parameters_linear_combinations += [{entry["name"]: 1}]
            return simu_parameters_linear_combinations
        return []

    def load_default_data_grouping(self, spec):
        """Load the default grouping of data"""
        return "ALL"

    @element_of("dataset_inputs")
    def parse_dataset_input(
        self,
        dataset: Mapping,
        simu_parameters_names,
        simu_parameters_linear_combinations,
        simu_parameters=None,
        allow_legacy_names: bool = True,
    ):
        """The mapping that corresponds to the dataset specifications in the fit files"""
        accepted_keys = {
            "dataset",
            "sys",
            "cfac",
            "frac",
            "weight",
            "custom_group",
            "variant",
            "simu_fac",
            "use_fixed_predictions",
            "contamination",
        }
        try:
            name = dataset["dataset"]
            if not isinstance(name, str):
                raise ConfigError(f"'dataset' must be a string, not {type(name)}")
            # Check whether this is an integrability or positivity dataset (in the only way we know?)
            if name.startswith(("NNPDF_INTEG", "NNPDF_POS", "POS", "INTEG")):
                if name.startswith(("INTEG", "NNPDF_INTEG")):
                    raise ConfigError("Please, use `integdataset` for integrability")
                if name.startswith(("POS", "NNPDF_POS")):
                    raise ConfigError("Please, use `posdataset` for positivity")
        except KeyError:
            raise ConfigError("'dataset' must be a mapping with " "'dataset' and 'sysnum'")

        # Ensure that we can actually read the `dataset_input` before failure
        kdiff = dataset.keys() - accepted_keys
        for k in kdiff:
            # Abuse ConfigError to get the suggestions.
            log.warning(
                ConfigError(f"Key '{k}' in dataset_input not known ({name}).", k, accepted_keys)
            )

        cfac = dataset.get("cfac", tuple())
        custom_group = str(dataset.get("custom_group", "unset"))

        frac = dataset.get("frac", 1)
        if not isinstance(frac, numbers.Real):
            raise ConfigError(f"'frac' must be a number, not '{frac}' ({name})")
        if frac < 0 or frac > 1:
            raise ConfigError(f"'frac' must be between 0 and 1 not '{frac}' ({name})")

        weight = dataset.get("weight", 1)
        if not isinstance(weight, numbers.Real):
            raise ConfigError(f"'weight' must be a number, not '{weight}' ({name})")
        if weight < 0:
            raise ConfigError(f"'weight' must be greater than zero not '{weight}' ({name})")

        variant = dataset.get("variant")
        sysnum = dataset.get("sys")

        if variant is not None and sysnum is not None:
            raise ConfigError(f"The 'variant' and 'sys' keys cannot be used together ({name})")

        # The old -> new name mapping can only be used with allow_legacy_names = True
        # which from 4.1 will default to False.
        # It can be used in order to be able to use old runcard but it is not recommended.
        if allow_legacy_names:
            name, map_variant = legacy_to_new_map(name, sysnum)
            # legacy_dw trumps everything
            if variant is None or map_variant == "legacy_dw":
                variant = map_variant

            if sysnum is not None:
                log.warning(
                    f"The key 'sys' is deprecated and only used for variant discovery: {variant}"
                )

        use_fixed_predictions = dataset.get("use_fixed_predictions", False)
        contamination = dataset.get("contamination", None)
        simu_fac = dataset.get("simu_fac", None)

        bsm_data = bsmnames.get_bsm_data(
            simu_fac, simu_parameters, simu_parameters_names, simu_parameters_linear_combinations
        )

        return SIMUnetDataSetInput(
            name=name,
            cfac=cfac,
            frac=frac,
            weight=weight,
            custom_group=custom_group,
            variant=variant,
            use_fixed_predictions=use_fixed_predictions,
            contamination=contamination,
            simu_fac=simu_fac,
            **bsm_data,
        )


class SIMUConfig(report.Config, SIMUCoreConfig):
    """..."""
