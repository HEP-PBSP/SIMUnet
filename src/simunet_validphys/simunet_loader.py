from validphys.loader import Loader, CfactorNotFound, LoaderError, FKTableNotFound, InconsistentMetaDataError
from validphys.core import CutsPolicy, TheoryIDSpec
from validphys.utils import yaml_safe
from .simunet_core import SIMUnetDataSetSpec, SIMUnetFKTableSpec
from .simunet_pineparser import parse_theory_meta
from validphys.pineparser import  TheoryMeta
import importlib.resources
from pathlib import Path
import yaml
import os

class SIMUnetLoader(Loader):
    def get_simu_parameters_name_dict(self, setname, simu_parameters_names, theoryid):
        """
        Parameters
        ----------
        setname: str
                name of the dataset

        simu_parameters_names: list
                list containing the joined `simu_fac` and operator

        theoryid: str

        Returns
        -------
        dict

        """

        _, theopath = self.check_theoryID(theoryid)
        simu_fac_names_paths = {}

        package_root: Path = importlib.resources.files("simunet_validphys")

        project_root = package_root.parent.parent

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

        simufactorpath = (
            project_root / "simudata" / "simu_factors" / f"SIMU_{setname_old}.yaml"
        )
        if not simufactorpath.exists():
            msg = (
                f"Could not find a SIMU factor for setname in {theopath}. "
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
        # new_commondata=False,
    ):

        if not isinstance(theoryid, TheoryIDSpec):
            theoryid = self.check_theoryID(theoryid)

        theoryno, theopath = theoryid

        commondata = self.check_commondata(
            name, sysnum, use_fitcommondata=use_fitcommondata, fit=fit
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
                name, simu_parameters_names, theoryno
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
    
    def check_fktable(self, theoryid, setname, cfac, use_fixed_predictions=False, new_commondata=False, is_compound=False):
        _, theopath = self.check_theoryID(theoryid)

        if use_fixed_predictions:
            fkpath = theopath/ 'fastkernel' / ('FK_FAKEKTABLE.dat')
            if not fkpath.exists():
                raise FKTableNotFound("Could not find the fake FK-table for fixed observables!")
            # Also set the fixed predictions path
            fixed_predictions_path = theopath/ 'simu_factors' / ('SIMU_%s.yaml' % setname)
            cfactors = self.check_cfactor(theoryid, setname, cfac)
            return SIMUnetFKTableSpec(fkpath, cfactors, use_fixed_predictions=True, fixed_predictions_path=fixed_predictions_path)
        # Only works if cfactor name is same as setname
        cfactors = self.check_cfactor(theoryid, setname, cfac)
        
        # use different file name for the FK table if the commondata is new
        if new_commondata:
            # Need to pass a TheoryMeta object to FKTableSpec
            path_metadata = theopath / 'fastkernel' / f'{setname}_metadata.yaml'
            if not path_metadata.exists():
                raise InconsistentMetaDataError(f"Could not find '_metadata.yaml' file for set {setname}."
                                                f"File '{path_metadata}' not found.")
            # get observable name from the setname
            with open(path_metadata, 'r') as f:
                metadata = yaml_safe.load(f)
            # NOTE: write a "_metadata.yaml" file for each observable (then `metadata["implemented_observables"][0]` makes sense)
            fktables = metadata["implemented_observables"][0]["theory"]["FK_tables"][0]
            fkpath = tuple([theopath/ 'fastkernel' / (f'{fktable}.pineappl.lz4') for fktable in fktables])
            for path in fkpath:
                if not path.exists():
                    raise FKTableNotFound(("Could not find FKTable for set '%s'. "
                    "File '%s' not found") % (setname, path) )
                
            common_prefix = os.path.commonprefix([metadata['setname'], setname])
            
            observable_name = setname[len(common_prefix):]
            if observable_name.startswith('_'):
                observable_name = observable_name[1:]
            if is_compound:
                theory_meta = TheoryMeta(FK_tables=[fkpath], operation="NULL", conversion_factor=1., shifts=None, normalization=None, comment=None)
            else:
                theory_meta = parse_theory_meta(path_metadata, observable_name=observable_name)
            
            return SIMUnetFKTableSpec(fkpath, cfactors, 
                               theory_meta=theory_meta,
                               legacy=False)
        else:
            fkpath = theopath/ 'fastkernel' / ('FK_%s.dat' % setname)

            if not fkpath.exists():
                raise FKTableNotFound(("Could not find FKTable for set '%s'. "
                "File '%s' not found") % (setname, fkpath) )
            
            return SIMUnetFKTableSpec(fkpath, cfactors)
