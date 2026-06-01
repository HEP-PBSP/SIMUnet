#!/usr/bin/env python3
"""
This script converts old data files from NNPDF into the new format.

Note that it takes some shortcuts, such as using kin1, kin2, kin3, as the kinematic variables.
Some actions from validphys no longer accept non-descriptive variables for the kinematics so datasets
cannot be automatically ported to the NNPDF repo without manual action.
Fits can be run, but cuts need to either be applied to the `kin1`, `kin2` and `kin3` kinematics or the variables modified.
"""

import functools
import re
import subprocess as sp
import traceback
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from validphys.api import API
from yaml import safe_dump, safe_load

_tev_searcher = re.compile(r"^\d+TEV$")


def _autoname(old_name):
    """Generate a new "NNPDF-like" name.

    It keeps the EXP_ name from simunet.
    It searchers for a XTEV string and puts it in the energy field.
    Then leaves the rest the same.
    If no observable is available it adds a generic `_OBS`.

    E.g.
        ATLAS_SINGLETOP_SCH_13TEV_TOTAL -> ATLAS_SINGLETOP_13TEV_SCH_TOTAL
        ATLAS_WHEL_13TEV -> ATLAS_WHEL_13TEV_OBS
        ATLAS_SSINC_RUNII_ZGAM -> ATLAS_SSINC_NOTFIXED_RUNII_ZGAM
    """
    EXPERIMENTS = tuple([f"{i}_" for i in ("ATLAS", "CMS", "LHCB", "ATLAS_CMS", "LEP")])
    if not old_name.startswith(EXPERIMENTS):
        raise NotImplementedError(f"Not sure what to do with {old_name}")
    new_name = old_name.replace("ATLAS_CMS", "ATLASCMS")

    split_parts = new_name.split("_")
    if len(split_parts) < 3:
        split_parts.append("OBS")

    if _tev_searcher.match(split_parts[2]) is None:
        # So there is no energy value at the expected point...
        for i, part in enumerate(split_parts):
            if _tev_searcher.match(part) is not None:
                # Oh, but we found one somewhere else!
                split_parts.pop(i)
                # Remove it and insert it back at position 2!
                split_parts.insert(2, part)
                break
        else:
            # If we don't find any energy value, assume it is not fixed
            split_parts.insert(2, "NOTFIXED")

    if len(split_parts) < 4:
        split_parts.append("OBS")

    return "_".join(split_parts)


def read_commondata_csv(commondatafile):
    """Read the old format commondata which were csv files we sparkling formatting
    This is directly taken from validphys
    """
    commondatatable = pd.read_csv(commondatafile, sep=r"\s+", skiprows=1, header=None)
    # Do we have NaNs? files with wrong formatting?
    commondataheader = ["entry", "process", "kin1", "kin2", "kin3", "data", "stat"]
    nsys = (commondatatable.shape[1] - len(commondataheader)) // 2

    commondataheader += ["ADD", "MULT"] * nsys
    commondatatable.columns = commondataheader
    commondatatable.set_index("entry", inplace=True)
    return commondatatable


def create_data(commondata_df):
    """Given a commondata dataframe, extract the central data and create a dictionary
    ready to be yamld'd"""
    data = commondata_df["data"].values
    return {"data_central": data.tolist()}


def create_kinematics(df):
    """Create kinematics dictionary with kin1, kin2, kin3 . . ."""
    kin_df = df[["kin1", "kin2", "kin3"]]
    bins = []
    for _, b in kin_df.T.items():
        tmp = {}
        for k, val in b.items():
            tmp[f"k{k[-1]}"] = {"min": None, "mid": val, "max": None}
        bins.append(tmp)
    return {"bins": bins}


def create_uncertainties(df, systype_file, is_default=False, use_multiplicative=False):
    """Create the uncertainties dictionary from the old cd information

    We first clean the dataframe to have only the systematic uncertainties
    and then even-indexes will be ADD uncertainties and odd-indexes MULT uncertainties

    Such that e.g., unc 4 in the systype file, if it is MULT, will correspond to index 7
    """
    stat = df["stat"].values.tolist()
    data = df["data"].values

    to_drop = ["process", "kin1", "kin2", "kin3", "data", "stat"]
    unc_df = df.drop(to_drop, axis=1)

    sys_df = pd.read_csv(systype_file, sep=r"\s+", skiprows=1, header=None, index_col=0)
    definitions = {}

    for i, unc_type in sys_df.T.items():
        definitions[f"sys_corr_{i}"] = {
            "description": f"Sys uncertainty idx: {i}",
            "treatment": unc_type[1],
            "type": unc_type[2],
        }

    # Check whether the number of uncertainties in the systype file is consistent with the df
    if len(unc_df.columns) != len(sys_df) * 2:
        # If this happened and this is the (true) DEFAULT, crash
        if is_default:
            raise ValueError("Different number of systematics in systype file and commondata")
        return {}

    bins = []
    for idx, bin_data in unc_df.T.items():
        tmp = {"stat": stat[idx - 1]}
        for n, (key, info) in enumerate(definitions.items()):
            if info["treatment"] not in ["ADD", "MULT"]:
                raise ValueError(f"Treatment type: {info['treatment']} not recognized")
            if use_multiplicative:
                tmp[key] = float(bin_data[2 * n + 1] * data[idx - 1] / 100.0)
            else:
                tmp[key] = float(bin_data[2 * n])
        bins.append(tmp)

    # Now add stat to the definitions
    definitions_out = {
        "stat": {
            "description": "Uncorrelated statistical uncertainties",
            "treatment": "ADD",
            "type": "UNCORR",
        },
        **definitions,
    }

    return {"definitions": definitions_out, "bins": bins}


def create_plotting(plotting_file, plotting_type=None):
    """Create the plotting dictionary by merging the plotting file of the commondata
    and the plotting type associated to it
    """
    type_info = {}
    if plotting_type is not None and plotting_type.exists():
        type_info = safe_load(plotting_type.read_text())

    plotting_data = safe_load(plotting_file.read_text())

    plotting_dict = {**type_info, **plotting_data}

    plotting_dict["plot_x"] = plotting_dict.pop("x", "idat")
    plotting_dict.pop("kinematics_override", None)
    return plotting_dict


def _gen_k(kx):
    """Generate k{x} variable"""
    return {"description": f"Variable {kx}", "label": f"{kx}", "units": ""}


def create_obs_dict(commondata_df, plotting_dict, theory_dict, obs_name="PLACEHOLDER"):
    """Create the observable dictionary by combining available information
    in the commondata dataframe and the plotting_dict

    It doesn't fill any data files (i.e., uncertainties, data or kinematics)
    """
    final_plotting_dict = dict(plotting_dict)

    # Extract necessary information
    ndata = len(commondata_df)
    process_type = commondata_df["process"][1]

    description = final_plotting_dict.pop("process_description", "DESCRIPTION_PLACEHOLDER")
    label = final_plotting_dict["dataset_label"]
    units = ""

    final_plotting_dict.pop("nnpdf31_process")
    final_plotting_dict.pop("experiment")

    # Sub-dicts
    observable = {"description": description, "label": label, "units": units}

    coverage = ["k1", "k2", "k3"]
    kinematics = {"variables": {i: _gen_k(i) for i in coverage}}

    return {
        "observable_name": obs_name,
        "observable": observable,
        "process_type": process_type,
        "tables": [],
        "npoints": [],
        "ndata": ndata,
        "plotting": final_plotting_dict,
        "kinematic_coverage": coverage,
        "kinematics": kinematics,
        "theory": theory_dict,
        "data_uncertainties": [],
    }


def yaml_dump_wrapper(data, target_file, dry=False, **kwargs):
    """Wrapper around safe_sump in order to use the dry flag"""
    if dry:
        return None
    safe_dump(data, target_file.open("w", encoding="utf-8"), **kwargs)


def convert_old_to_new(
    old_file,
    plotting_file,
    sys_file,
    new_name,
    new_variant=None,
    merge_exists=False,
    output_folder=Path("converted_commondata"),
    dry=False,
    variant=None,
    compound=None,
    theory_conversion=None,
):
    """
    Converts the old dataset defined by the old data, plotting and sys file into the new format.
    Use ``new_name`` (which will be broken down as <EXPERIMENT>_<ENERGY>_<PROCESS>_<OBS>)

    Note, plotting-type file by process is being ignored in this implementation.

    If new_variant is given, the data and uncertainties will fall into the given variant.
    If ``merge_exists`` is True then the dataset will be searched for in the validphys database
    and use as the basis of the new one.
    """
    if merge_exists:
        raise NotImplementedError("Not implemented yet")

    if new_variant is None:
        variant_name = "DEFAULT"
    else:
        variant_name = new_variant

    yaml_safe_dump = functools.partial(yaml_dump_wrapper, dry=dry)

    obs_name = new_name.rsplit("_", 1)[-1]
    set_name = new_name.replace(f"_{obs_name}", "")

    set_folder = output_folder / set_name
    set_folder.mkdir(exist_ok=True, parents=True)

    # Read the commondata file
    commondata_df = read_commondata_csv(old_file)

    kinematics_dict = create_kinematics(commondata_df)
    data_dict = create_data(commondata_df)
    plotting_dict = create_plotting(plotting_file)
    if compound is not None:
        fks = []
        for line in Path(compound).read_text().split("\n"):
            info = safe_load(line)
            if isinstance(info, dict):
                if "FK" in info:
                    fks.append([info["FK"].replace(".dat", "")])
                elif "OP" in info:
                    op = info["OP"]
        theory_dict = {"FK_tables": fks, "operation": op}
    else:
        theory_dict = {"FK_tables": [[old_file.stem.replace("DATA_", "FK_")]]}

    uncertainties_dict = create_uncertainties(commondata_df, sys_file, is_default=True)
    obs_dict = create_obs_dict(commondata_df, plotting_dict, theory_dict, obs_name=obs_name)

    metadata_path = set_folder / "metadata.yaml"
    if metadata_path.exists():
        metadata = safe_load(metadata_path.read_text())
        # Perform sanity checks
        nnpdf_md = metadata["nnpdf_metadata"]
        try:
            assert nnpdf_md["experiment"] == plotting_dict["experiment"]
            assert nnpdf_md["nnpdf31_process"] == plotting_dict["nnpdf31_process"]
            assert metadata.get("setname") == set_name
        except AssertionError:
            print(traceback.format_exc())
            import ipdb

            ipdb.set_trace()

        # Check whether the observable already exists
        already_implemented = [i["observable_name"] for i in metadata["implemented_observables"]]
        if obs_name in already_implemented:
            raise ValueError(f"{obs_name} already implemented for {set_name}")
    else:
        # Create it from scratch
        # Create it anew!
        nnpdf_md = {
            "nnpdf31_process": plotting_dict["nnpdf31_process"],
            "experiment": plotting_dict["experiment"],
        }
        metadata = {
            "setname": set_name,
            "version": 1,
            "version_comment": "Port of old commondata",
            "nnpdf_metadata": nnpdf_md,
            "arXiv": {"url": ""},
            "iNSPIRE": {"url": ""},
            "hepdata": {"url": "", "version": -1},
            "implemented_observables": [],
        }

    kin_path = set_folder / f"kinematics_{obs_name}.yaml"
    yaml_safe_dump(kinematics_dict, kin_path, sort_keys=False)
    obs_dict["kinematics"]["file"] = kin_path.name

    # The lines below should be different for positivity datasets
    data_path = set_folder / f"data_{variant_name}_{obs_name}.yaml"
    unc_path = set_folder / f"uncertainties_{variant_name}_{obs_name}.yaml"

    yaml_safe_dump(data_dict, data_path)
    yaml_safe_dump(uncertainties_dict, unc_path, sort_keys=False)

    if new_variant is None:
        obs_dict["data_uncertainties"] = [unc_path.name]
        obs_dict["data_central"] = data_path.name
    else:
        new_var = {
            "data_uncertainties": [unc_path.name],
            "data_central": data_path.name,
        }
        if "variants" not in obs_dict:
            obs_dict["variants"] = {}
        obs_dict["variants"][variant_name] = new_var

    metadata["implemented_observables"].append(obs_dict)
    yaml_safe_dump(metadata, metadata_path, sort_keys=False)
    print(f"Written new cd for {set_name}_{obs_name} to {set_folder}")

    if theory_conversion is not None:
        theory_path = API.theoryid(theoryid=theory_conversion).path
        fkfolder = theory_path / "fastkernel"
        pinefolder = theory_path / "pineappl_version"
        pinefolder.mkdir(exist_ok=True)
        for operator in theory_dict["FK_tables"]:
            for fk_table in operator:
                fk_path = fkfolder / f"{fk_table}.dat"
                pi_path = pinefolder / f"{fk_table}.pineappl.lz4"
                if not fk_path.exists():
                    print(f" > Not found {fk_path} for {new_name}")
                else:
                    sp.run(
                        [
                            "pineappl",
                            "import",
                            fk_path,
                            pi_path,
                            "NNPDF40_nnlo_as_01180",
                        ],
                        capture_output=True,
                    )
                    print(f" > Converted {fk_table} to {pi_path}")


def main(args):
    """Run the script."""
    old_file = args.old_dat_file
    old_name = args.old_dat_file.stem.replace("DATA_", "")

    if (pfile := args.old_plotting_file) is None:
        pfile = old_file.parent / f"PLOTTING_{old_name}.yaml"

    if (sfile := args.old_sys_file) is None:
        sfile = old_file.parent / "systypes" / f"SYSTYPE_{old_name}_DEFAULT.dat"

    # Check whether we can automagically find the plotting and systype file or whether we need to ask for clarifications
    for check_me in [old_file, pfile, sfile]:
        if not check_me.exists():
            raise FileNotFoundError(f"Couldn't find {check_me}")

    if args.dataset_name is None:
        dataset_name = _autoname(old_name)
        # print(f"Converting {old_name} into {dataset_name}")
    else:
        dataset_name = args.dataset_name

    convert_old_to_new(
        old_file,
        pfile,
        sfile,
        dataset_name,
        variant=args.variant,
        merge_exists=args.merge_exists,
        compound=args.old_compound,
        theory_conversion=args.theory_conversion,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("old_dat_file", help=".dat file of the old dataset", type=Path)
    parser.add_argument(
        "--dataset_name",
        help="Target name of the dataset, if not given it will be automagically generated",
        type=str,
    )
    parser.add_argument(
        "--old_plotting_file",
        help="plotting .yaml file of the old dataset (by default it will be autodiscovered from the dataset)",
        type=Path,
    )
    parser.add_argument(
        "--old_sys_file",
        help="systype file of the old dataset (by default it will be autodiscovered from the dataset)",
        type=Path,
    )
    parser.add_argument(
        "--old_compound",
        help="Old compound file, if not given we will assume it gets the default name",
    )
    parser.add_argument("--variant", help="Create the new dataset as the given variant", type=str)
    parser.add_argument(
        "--merge_exists",
        help="If a dataset already exists in validphys with this name, merge the new info with this dataset in the output",
        action="store_true",
    )
    parser.add_argument(
        "--theory_conversion",
        help="Takes as value a <theoryId>, if given, will try to convert said theory to the new format",
        type=int,
    )

    args = parser.parse_args()
    try:
        main(args)
    except NotImplementedError:
        print(f"Not sure how to deal with {args.old_dat_file}")
    except FileNotFoundError:
        print(f"{args.old_dat_file} not found")
    except IndexError:
        print(f"{args.old_dat_file}")
