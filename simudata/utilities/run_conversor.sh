#!/usr/bin/env bash

OLD_SIMUNET=../../../original_simunet_repo
ORIGINAL_DATA_FOLDER=${OLD_SIMUNET}/nnpdfcpp/data/commondata/
SIMUNET_THEORY=270

# Start with one that looks sensible
# python conversor.py CMS_TTBAR_13TEV_LJETS_TOTAL ${ORIGINAL_DATA_FOLDER}/DATA_CMS_TTBAR_13TEV_LJETS_TOTAL.dat

CSV_INFO=./data_maps.csv
while IFS=, read -r col1 col2 rest
do
    if [[ "${col2}" =~ "NOT FOUND" ]]
    then
        DATA_FILE=${ORIGINAL_DATA_FOLDER}/DATA_${col1}.dat

        # Check whether we have a compound file for this one
        COMPOUND=~/.local/share/NNPDF/theories/theory_${SIMUNET_THEORY}/compound/FK_${col1}-COMPOUND.dat
        if [[ -f ${COMPOUND} ]]
        then
            extra_args="--old_compound ${COMPOUND}"
        fi
        python conversor.py ${DATA_FILE} ${extra_args} --theory_conversion ${SIMUNET_THEORY} < /dev/tty
        unset extra_args
    fi
done < $CSV_INFO
