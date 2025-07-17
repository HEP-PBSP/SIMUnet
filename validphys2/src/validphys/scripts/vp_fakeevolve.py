"""
vp-fakeevolve

Command line tool to copy across all PDF information in a fixed PDF fit.

The script:
- Updates a parameter-only fit using simunet with all the PDF information of another fixed PDF fit.
"""

import argparse
import pathlib
import sys
import logging
import prompt_toolkit
import os

import shutil

from reportengine import colors
from validphys.utils import yaml_safe

from validphys.api import API

from validphys.loader import Loader

# Take command line arguments
def process_args():
    parser = argparse.ArgumentParser(
        description="Script to updated PDF information of a fixed PDF fit."
    )
    parser.add_argument("fixed_simunet_fit", help="Name of the BSM-parameter only simunet fit.")
    parser.add_argument("num_reps", help="Number of replicas to copy across.")
    args = parser.parse_args()
    return args


def main():
    args = process_args()

    input_fit = args.fixed_simunet_fit
    input_dir = pathlib.Path(input_fit)

    with open(input_dir / "filter.yml", 'r') as file:
        input_info = yaml_safe.load(file)

    fixed_fit = input_info['load_weights_from_fit']
    l = Loader()
    fixed_fit_dir = l.resultspath / fixed_fit

    for i in range(int(args.num_reps)):
        source_file = fixed_fit_dir / 'postfit' / ('replica_' + str(i+1)) / (fixed_fit + '.dat')
        destination_file = input_dir / 'nnfit' / ('replica_' + str(i+1)) / (input_fit + '.dat')

        if not os.path.exists(source_file):
            print(source_file)
            print(destination_file)
            logging.warning(f"Replica {str(i+1)} not found. Skipping.")
            continue

        shutil.copy(source_file, destination_file)
    # Copy the info file too
    shutil.copy(fixed_fit_dir / 'nnfit' / (fixed_fit + '.info'), input_dir / 'nnfit' / (input_fit + '.info'))

if __name__ == "__main__":
    main()
