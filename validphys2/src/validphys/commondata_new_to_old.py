"""
    Commondata converter script from new to old format:
        it must be run in an up to date simunet environment, in the `commondata_converter_new_to_old` branch.
"""

import os
import sys
import yaml
from validphys.utils import uncertainty_yaml_to_systype, convert_new_data_to_old

# test whether the runcard is passed
if len(sys.argv) != 2:
    raise Exception("No runcard is passed!")
card_name = sys.argv[1]
if not os.path.isfile(card_name):
    raise Exception("Runcard does not exist!")
# load runcard
with open(card_name, "rb") as stream:
    runcard = yaml.safe_load(stream)
# load datasets to convert
datasets = runcard["dataset_inputs"]

# create test directory if it does not already exist
test_dir = "test_utils"
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

# changed by the user
nnpdf_path = "/Users/teto/Software/nnpdf_git/nnpdf"
# new commondata path
new_commondata = f"{nnpdf_path}/nnpdf_data/nnpdf_data/commondata"
# open conversion dictionary
with open(f"{new_commondata}/dataset_names.yml", "rb") as stream:
    conversion = yaml.safe_load(stream)

# old format
old_format_names = list(conversion.keys())
# new format
new_format_names = []
for c in conversion:
    try:
        new_format_names.append(conversion[c]["dataset"])
    except TypeError:
        new_format_names.append(conversion[c])

# prepare list of the datasets to be converted
conversion_ds = []
for ds in datasets:
    if ds["dataset"] in old_format_names:
        conversion_ds.append(conversion[ds["dataset"]])
    elif ds["dataset"] in new_format_names:
        conversion_ds.append({"dataset": ds["dataset"], "variant": "legacy"})
    else:
        conversion_ds.append({"dataset": ds["dataset"], "variant": None})

# separate the dataset & the observable names
for ds in conversion_ds:
    s = ds["dataset"]
    ds["dataset"] = s[:s.rfind("_")]
    ds["obs"] = s[s.rfind("_")+1:]

# convert
for i, ds in enumerate(conversion_ds):
    var_int, obs_ind = "variant", "obs"
    # if only in the new format
    if not ds[var_int]:
        path_data_yaml = new_commondata+"/"+ds["dataset"]+f"/data.yaml"
        path_unc_file = new_commondata+"/"+ds["dataset"]+f"/uncertainties.yaml"
        path_kin = new_commondata+"/"+ds["dataset"]+f"/kinematics.yaml"
    # if also in the old format (legacy variants)
    else:
        if os.path.isfile(new_commondata+"/"+ds["dataset"]+f"/data_{ds[var_int]}_{ds[obs_ind]}.yaml"):
            path_data_yaml = new_commondata+"/"+ds["dataset"]+f"/data_{ds[var_int]}_{ds[obs_ind]}.yaml"
        else:
            path_data_yaml = new_commondata+"/"+ds["dataset"]+f"/data_legacy_{ds[obs_ind]}.yaml"
        path_unc_file = new_commondata+"/"+ds["dataset"]+f"/uncertainties_{ds[var_int]}_{ds[obs_ind]}.yaml"
        path_kin = new_commondata+"/"+ds["dataset"]+f"/kinematics_{ds[obs_ind]}.yaml"
    # load metadata file
    path_metadata = new_commondata+"/"+ds["dataset"]+f"/metadata.yaml"
    # write uncertainty files
    uncertainty_yaml_to_systype(path_unc_file,
                                name_dataset=ds["dataset"],
                                observable=ds["obs"],
                                path_systype=test_dir)
    # write commondata files
    convert_new_data_to_old(path_data_yaml,
                            path_unc_file,
                            path_kin,
                            path_metadata,
                            name_dataset=ds["dataset"],
                            observable=ds["obs"],
                            path_DATA=test_dir)
    # output
    name = ds["dataset"]+"_"+ds["obs"] 
    print(f"{i+1:>2}. {name:>40} converted!")

# write check runcard
with open("test_utils/check_commondata_new_to_old.yaml", "w") as stream:
    yaml.safe_dump(conversion_ds, stream)