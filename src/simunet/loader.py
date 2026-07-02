import importlib.resources
import logging
from pathlib import Path

import yaml

from nnpdf_data.validphys_compatibility import new_to_legacy_map
from validphys.core import CutsPolicy, TheoryIDSpec
from validphys.loader import CfactorNotFound, FallbackLoader, Loader, LoaderError
from validphys.utils import yaml_safe

from .core import SIMUnetDataSetSpec

log = logging.getLogger(__name__)


class SIMUnetLoader(Loader):
    def __init__(self, profile=None):
        super().__init__(profile)
        package_root: Path = importlib.resources.files("simunet")

        project_root = package_root.parent.parent

        simudata_path = project_root / "simudata"
        local_commondata_path = simudata_path / "commondata"
        if local_commondata_path.exists():
            self.commondata_folders = (local_commondata_path, *self.commondata_folders)

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
        simufactorpath = self.simudata_path / "simu_factors" / f"SIMU_{setname}.yaml"

        # First check whether the simunet factor exists with the new name, otherwise try the old one
        if not simufactorpath.exists():
            setname_old = new_to_legacy_map(setname, "legacy")[0]

            if setname_old is None:
                raise CfactorNotFound(f"Could not find a SIMU factor for {setname}")

            simufactorpath = self.simudata_path / "simu_factors" / f"SIMU_{setname_old}.yaml"

            if not simufactorpath.exists():
                msg = (
                    f"Could not find a SIMU factor for {setname_old}. "
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

        # TODO: to ask, why can't we read here the file directly instead of doing it in the provider
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


class SIMUFallbackLoader(SIMUnetLoader, FallbackLoader):
    """
    A loader that first tries to find resources locally
    (using SIMUnetLoader.check_*) and if it fails,
    it tries to download them (using RemoteLoader.download_*).
    """

    pass
