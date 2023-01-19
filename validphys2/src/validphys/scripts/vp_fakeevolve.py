"""
vp-fakeevolve

Command line tool to copy across all PDF information in a fixed PDF fit.

The script:
- Updates a parameter-only fit using simunet with all the PDF information of another fixed PDF fit.
"""

import argparse
import os
import pathlib
import sys
import logging
import prompt_toolkit

from reportengine import colors
from reportengine.compat import yaml

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
    # Logger for writing to screen
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(colors.ColorHandler())

    args = process_args()

    input_fit = args.fixed_simunet_fit
    input_dir = pathlib.Path(input_fit)

    with open(input_dir / "filter.yml", 'r') as file:
        input_info = yaml.safe_load(file)

    fixed_fit = input_info['load_weights_from_fit']
    l = Loader()
    fixed_fit_dir = l.resultspath / fixed_fit

    os.system('mkdir tmp')
    os.system('cp -r ' + str(fixed_fit_dir) + ' ' + 'tmp/.')
    os.system('vp-fitrename tmp/' + str(fixed_fit) + ' ' + str(input_fit))
    # Remove existing postfit folder
    os.system('rm -r tmp/' + str(input_fit) + '/postfit'  )

    # Remove any replicas which are beyond the replicas we care about
    rep_names = os.listdir('tmp/' + str(input_fit) + '/nnfit')
    rep_numbers = [x[8:] for x in rep_names]    
    rep_numbers = [int(x) for x in rep_numbers if x[-5:] != '.info' and x != '']
    num_fixed_reps = max(rep_numbers)

    num_reps = int(args.num_reps)

    for i in range(num_fixed_reps):
        if i+1 > num_reps:
            os.system('rm -r tmp/' + str(input_fit) + '/nnfit/replica_' + str(i+1))

    for i in range(num_reps):
        if not os.path.exists(str(input_fit) + '/nnfit/replica_' + str(i+1)):
            os.system('rm -r tmp')
            log.error("Too many replicas requested.")
            sys.exit(1)

        os.system('cp ' + str(input_fit) + '/nnfit/replica_' + str(i+1) + '/bsm_fac.csv ' + 'tmp/' + str(input_fit) + '/nnfit/replica_' + str(i+1) + '/.')
        os.system('cp ' + str(input_fit) + '/nnfit/replica_' + str(i+1) + '/chi2exps.log ' + 'tmp/' + str(input_fit) + '/nnfit/replica_' + str(i+1) + '/.')
        os.system('cp ' + str(input_fit) + '/nnfit/replica_' + str(i+1) + '/' + str(input_fit) + '.json ' + 'tmp/' + str(input_fit) + '/nnfit/replica_' + str(i+1) + '/.')

    # The fake fit is now completely prepared. Remove the original and copy the fake to the
    # current directory.
    os.system('rm -r ' + str(input_fit))
    os.system('cp -r tmp/' + str(input_fit) + ' .')
    os.system('rm -r tmp')

if __name__ == "__main__":
    main()
