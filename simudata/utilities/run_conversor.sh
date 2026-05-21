#!/usr/bin/env bash

OLD_SIMUNET=../../../original_simunet_repo
ORIGINAL_DATA_FOLDER=${OLD_SIMUNET}/nnpdfcpp/data/commondata/

# Start with one that looks sensible
python conversor.py CMS_TTBAR_13TEV_LJETS_TOTAL ${ORIGINAL_DATA_FOLDER}/DATA_CMS_TTBAR_13TEV_LJETS_TOTAL.dat

exit 0
CSV_INFO=./data_maps.csv
while IFS=, read -r col1 col2 rest
do
    if [[ "${col2}" =~ "NOT FOUND" ]]
    then
        # TODO Maybe think about the naming before making it automatic...
        python conversor.py ${col1} ${ORIGINAL_DATA_FOLDER}/DATA_${col1}.dat
    fi
done < $CSV_INFO
