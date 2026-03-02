import logging
from validphys.loader import (
    Loader,
    CfactorNotFound,
    LoaderError,
    RemoteLoader,
    RemoteLoaderError,
    LoadFailedError,
)
from validphys.core import CutsPolicy, TheoryIDSpec
from validphys.utils import yaml_safe
from .simunet_core import SIMUnetDataSetSpec
import importlib.resources
from pathlib import Path
import yaml

log = logging.getLogger(__name__)


class SIMUnetLoader(Loader):
    def __init__(self, profile=None):
        super().__init__(profile)
        package_root: Path = importlib.resources.files("simunet_validphys")

        project_root = package_root.parent.parent

        simudata_path = project_root / "simudata"

        self.simudata_path = simudata_path

    def get_simu_parameters_name_dict(self, setname, simu_parameters_names):
        """
        Parameters
        ----------
        setname: str
                name of the dataset

        simu_parameters_names: list
                list containing the joined `simu_fac` and operator

        Returns
        -------
        dict

        """
        simu_fac_names_paths = {}

        yaml_path = (
            Path(importlib.resources.files("nnpdf_data"))
            / "commondata"
            / "dataset_names.yml"
        )
        with open(yaml_path, "r") as f:
            dataset_map = yaml.safe_load(f)

        setname_old = None
        for old, value in dataset_map.items():
            if isinstance(value, dict):
                if value.get("dataset") == setname:
                    setname_old = old
                    break
            elif value == setname:
                setname_old = old
                break

        if setname_old is None:
            raise KeyError(
                f"Could not find old dataset name corresponding to '{setname}' in dataset_names.yml"
            )
        simudata_path = self.simudata_path
        simufactorpath = simudata_path / "simu_factors" / f"SIMU_{setname_old}.yaml"

        if not simufactorpath.exists():
            msg = (
                f"Could not find a SIMU factor for setname in {simufactorpath}. "
                f"The path {simufactorpath} does not exist."
            )
            raise CfactorNotFound(msg)

        # test whether all the mandatory keys are present
        with open(simufactorpath, "rb") as stream:
            cfac_file = yaml_safe.load(stream)

        if "metadata" not in cfac_file:
            raise KeyError(
                f"The 'metadata' key is not present in the SIMU file at {simufactorpath}."
            )

        if "SM_fixed" not in cfac_file:
            raise KeyError(
                f"The 'SM_fixed' key is not present in the SIMU file at {simufactorpath}."
            )

        # assign to each operator name the same simufactorpath
        for simu_parameters_name in simu_parameters_names:
            simu_fac_names_paths[simu_parameters_name] = simufactorpath

        return simu_fac_names_paths

    def check_dataset(
        self,
        name,
        *,
        rules=None,
        sysnum=None,
        theoryid,
        cfac=(),
        frac=1,
        cuts=CutsPolicy.INTERNAL,
        use_fitcommondata=False,
        fit=None,
        weight=1,
        simu_parameters_names=None,
        simu_parameters_linear_combinations=None,
        use_fixed_predictions=False,
        contamination=None,
        contamination_data=None,
        variant=None,
    ):

        if not isinstance(theoryid, TheoryIDSpec):
            theoryid = self.check_theoryID(theoryid)

        commondata = self.check_commondata(
            name, sysnum, use_fitcommondata=use_fitcommondata, fit=fit, variant=variant
        )

        fkspec, op = self._check_theory_old_or_new(theoryid, commondata, cfac)

        # Note this is simply for convenience when scripting. The config will
        # construct the actual Cuts object by itself
        if isinstance(cuts, str):
            cuts = CutsPolicy(cuts)
        if isinstance(cuts, CutsPolicy):
            if cuts is CutsPolicy.NOCUTS:
                cuts = None
            elif cuts is CutsPolicy.FROMFIT:
                cuts = self.check_fit_cuts(commondata, fit)
            elif cuts is CutsPolicy.INTERNAL:
                if rules is None:
                    rules = self.check_default_filter_rules(theoryid)
                cuts = self.check_internal_cuts(commondata, rules)
            elif cuts is CutsPolicy.FROM_CUT_INTERSECTION_NAMESPACE:
                raise LoaderError(f"Intersection cuts not supported in loader calls.")

        if simu_parameters_names is not None:
            simu_parameters_names_CF = self.get_simu_parameters_name_dict(
                name, simu_parameters_names
            )
        else:
            simu_parameters_names_CF = None

        return SIMUnetDataSetSpec(
            name=name,
            commondata=commondata,
            fkspecs=fkspec,
            thspec=theoryid,
            cuts=cuts,
            frac=frac,
            op=op,
            weight=weight,
            simu_parameters_names_CF=simu_parameters_names_CF,
            simu_parameters_names=simu_parameters_names,
            simu_parameters_linear_combinations=simu_parameters_linear_combinations,
            use_fixed_predictions=use_fixed_predictions,
            contamination=contamination,
            contamination_data=contamination_data,
        )


# Can I simplify this? I only need to change the loader which is parsed into this
class SIMUFallbackLoader(SIMUnetLoader, RemoteLoader):
    """
    A loader that first tries to find resources locally
    (using SIMUnetLoader.check_*) and if it fails,
    it tries to download them (using RemoteLoader.download_*).
    """

    pass
