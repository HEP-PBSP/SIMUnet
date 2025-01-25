import os
import sys
import yaml
import shutil
import filecmp

# simunet environment commondata path
old_commondata = "/Users/teto/miniconda3/envs/simunet_release/share/NNPDF/data/commondata"
# nnpdf commondata path
new_commondata = "/Users/teto/Software/nnpdf_git/nnpdf/nnpdf_data/nnpdf_data/commondata"
# test whether the runcard is passed
if len(sys.argv) != 2:
    raise Exception("No runcard is passed!")
card_name = sys.argv[1]
if not os.path.isfile(card_name):
    raise Exception("Runcard does not exist!")
# load runcard
with open(card_name, "rb") as stream:
    card = yaml.safe_load(stream)
# load conversion dictionary
with open(new_commondata+"/dataset_names.yml", "rb") as stream:
    conv = yaml.safe_load(stream)
# load datasets to convert
datasets = card["dataset_inputs"]
# temporary list
temp = []
# back conversion map
back_conv = {}
# loop over datasets to convert
for ds in datasets:
    ds_name = ds["dataset"]
    if ds_name in list(conv.keys()) and "-" in ds_name:
        # save the datasets to map
        temp.append(conv[ds_name])
        # print(f"{ds_name} is in the old format with a new name! (Do it manually)")
    else:
        for cds in conv:
            try:
                flag = ds_name == conv[cds]["dataset"]
            except TypeError:
                flag = ds_name == conv[cds]
            if flag:
                back_conv[ds_name] = cds
# loop over the datasets that we still have to convert
for ds in temp:
    ds_name, ds_var = ds["dataset"], ds["variant"]
    back_conv[ds_name] = []
    for cds in conv:
        try:
            flag = (ds_name == conv[cds]["dataset"]) and (ds_var == conv[cds]["variant"] and "-" not in cds)
        except TypeError:
            flag = ds_name == conv[cds]
        if flag:
            back_conv[ds_name] = cds
# copy
for i, bc in enumerate(back_conv):
    # new file name
    filename_new = f"test_utils/PLOTTING_{bc}.yml"
    # old file name
    if os.path.isfile(old_commondata+f"/PLOTTING_{back_conv[bc]}.yml"):
        filename_old = old_commondata+f"/PLOTTING_{back_conv[bc]}.yml"
    elif os.path.isfile(old_commondata+f"/PLOTTING_{back_conv[bc]}.yaml"):
        filename_old = old_commondata+f"/PLOTTING_{back_conv[bc]}.yaml"
    else:
        print(f"Missing PLOTTING file for {back_conv[bc]}!")
    # copy
    shutil.copy(filename_old, filename_new)
    # test the copies
    if filecmp.cmp(filename_old, filename_new):
        print(f"{i+1:>2}. Copied plotting file {back_conv[bc]:>40} -> {bc:>40}!")
    else:
        print(f"{i+1:>2}. Error during copy of plotting file {back_conv[bc]:>40} -> {bc:>40}!")