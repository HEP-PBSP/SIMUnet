"""
    Test the commondata converter from new to old format:
        it must be run in an up to date nnpdf environment.
"""

import yaml
from numpy import allclose
from validphys.commondataparser import parse_set_metadata, load_commondata_new, load_commondata_old
from validphys.covmats import covmat_from_systematics

# nnpdf path
nnpdf_path = "/Users/teto/Software/nnpdf_git/nnpdf"
# open the yaml file created by commondata_new_to_old script
with open("test_utils/check_commondata_new_to_old.yaml", "rb") as stream:
    datasets = yaml.safe_load(stream)
# silly dictionary to output if the feature is sound or not
ok = {1: "OK :D", 0: "NOT OK :C"}
# fake dataset input for covmat_from_systematics
inp = None
# list to store the implementation errors, useful for IPython debug
cd_errors, cm_errors = [], []
# loop over the selected datasets
for i, ds in enumerate(datasets):
    # dataset name, observable name, and dataset variant
    setname, observable, variant = ds["dataset"], ds["obs"], ds["variant"]
    # old commondata
    cd_old = load_commondata_old(commondatafile=f"test_utils/DATA_{setname}_{observable}.dat",
                                 systypefile=f"test_utils/SYSTYPE_{setname}_{observable}_DEFAULT.dat",
                                 setname=setname)
    # load metadata of the new commondata
    metadata = parse_set_metadata(metadata_file=f"{nnpdf_path}/nnpdf_data/nnpdf_data/commondata/{setname}/metadata.yaml")
    # new commondata
    if variant:
        cd_new = load_commondata_new(metadata=metadata.select_observable(observable).apply_variant(variant))
    else:
        cd_new = load_commondata_new(metadata=metadata.select_observable(observable))
    # load covariance matrices
    covmat_old = covmat_from_systematics(loaded_commondata_with_cuts=cd_old,
                                         dataset_input=inp,
                                         use_weights_in_covmat=False)
    covmat_new = covmat_from_systematics(loaded_commondata_with_cuts=cd_new,
                                         dataset_input=inp,
                                         use_weights_in_covmat=False)
    # test central values
    ds["commondata"] = allclose(cd_old.central_values, cd_new.central_values)
    if not ds["commondata"]:
        cd_errors.append({"old": cd_old, "new": cd_new})
    # test covariance matrix
    ds["covmat"] = allclose(covmat_old, covmat_new)
    if not ds["covmat"]:
        cm_errors.append({"old": covmat_old, "new": covmat_new})
    # output
    cd, cm = ds["commondata"], ds["covmat"]
    name = f"{setname}_{observable}"
    print(f"{i+1:2}. {name:>40} -> commondata is {ok[cd]:>9} & covariance matrix is {ok[cm]:>9}")