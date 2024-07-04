# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:12:41 2016

@author: Zahari Kassabov
"""
import contextlib
import shutil
import pathlib
import tempfile

import numpy as np

from validobj import parse_input, ValidationError
from reportengine.compat import yaml


def parse_yaml_inp(inp, spec, path):
    """Helper function to parse yaml using the `validobj` library and print
    useful error messages in case of a parsing error.

    https://validobj.readthedocs.io/en/latest/examples.html#yaml-line-numbers
    """
    try:
        return parse_input(inp, spec)
    except ValidationError as e:
        current_exc = e
        current_inp = inp
        error_text_lines = []
        while current_exc:
            if hasattr(current_exc, 'wrong_field'):
                wrong_field = current_exc.wrong_field
                # Mappings compping from ``round_trip_load`` have an
                # ``lc`` attribute that gives a tuple of
                # ``(line_number, column)`` for a given item in
                # the mapping.
                line = current_inp.lc.item(wrong_field)[0]
                error_text_lines.append(f"Problem processing key at line {line} in {path}:")
                current_inp = current_inp[wrong_field]
            elif hasattr(current_exc, 'wrong_index'):
                wrong_index = current_exc.wrong_index
                # Similarly lists allow to retrieve the line number for
                # a given item.
                line = current_inp.lc.item(wrong_index)[0]
                current_inp = current_inp[wrong_index]
                error_text_lines.append(f"Problem processing list item at line {line} in {path}:")
            elif hasattr(current_exc, 'unknown'):
                unknown_lines = []
                for u in current_exc.unknown:
                    unknown_lines.append((current_inp.lc.item(u)[0], u))
                unknown_lines.sort()
                for line, key in unknown_lines:
                    error_text_lines.append(
                        f"Unknown key {key!r} defined at line {line} in {path}:"
                    )
            error_text_lines.append(str(current_exc))
            current_exc = current_exc.__cause__
        raise ValidationError('\n'.join(error_text_lines)) from e

        
@contextlib.contextmanager
def tempfile_cleaner(root, exit_func, exc, prefix=None, **kwargs):
    """A context manager to handle temporary directory creation and
    clean-up upon raising an expected exception.

    Parameters
    ----------
    root: str
        The root directory to create the temporary directory in.
    exit_func: Callable
        The exit function to call upon exiting the context manager.
        Usually one of ``shutil.move`` or ``shutil.rmtree``. Use the former
        if the temporary directory will be the final result directory and the
        latter if the temporary directory will contain the result directory, for
        example when downloading a resource.
    exc: Exception
        The exception to catch within the ``with`` block.
    prefix: optional[str]
        A prefix to prepend to the temporary directory.
    **kwargs: dict
        Keyword arguments to provide to ``exit_func``.

    Returns
    -------
    tempdir: pathlib.Path
        The path to the temporary directory.

    Example
    -------
    The following example creates a temporary directory prepended with
    ``tutorial_`` in the ``/tmp`` directory. The context manager will listen
    for a ``KeyboardInterrupt`` and will clean up if this exception is
    raised. Upon completion of the ``with`` block, it will rename the
    temporary to ``completed`` as the ``dst``, using ``shutil.move``. The
    final directory will contain an empty file called ``new_file``, which
    we created within the ``with`` block.

        .. code-block:: python
          :linenos:

            import shutil

            from validphys.utils import tempfile_cleaner

            with tempfile_cleaner(
                root="/tmp",
                exit_func=shutil.move,
                exc=KeyboardInterrupt,
                prefix="tutorial_",
                dst="completed",
            ) as tempdir:

                new_file = tempdir / "new_file"
                input("Press enter to continue or Ctrl-C to interrupt:\\n")
                new_file.touch()
    """
    try:
        tempdir = pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        yield tempdir
    except exc:
        shutil.rmtree(tempdir)
        raise
    else:
        # e.g shutil.rmtree, shutil.move etc
        exit_func(tempdir, **kwargs)


def experiments_to_dataset_inputs(experiments_list):
    """Flatten a list of old style experiment inputs
    to the new, flat, ``dataset_inputs`` style.

    Example
    -------
    >>> from validphys.api import API
    >>> from validphys.utils import experiments_to_dataset_inputs
    >>> fit = API.fit(fit='NNPDF31_nnlo_as_0118_1000')
    >>> experiments = fit.as_input()['experiments']
    >>> dataset_inputs = experiments_to_dataset_inputs(experiments)
    >>> dataset_inputs[:3]
    [{'dataset': 'NMCPD', 'frac': 0.5},
     {'dataset': 'NMC', 'frac': 0.5},
     {'dataset': 'SLACP', 'frac': 0.5}]
    """
    dataset_inputs = []
    for experiment in experiments_list:
        dataset_inputs.extend(experiment['datasets'])

    return dataset_inputs

def split_by(it, crit):
    """Split ``it`` in two lists, the first is such that ``crit`` evaluates to
    True and the second such it doesn't. Crit can be either a function or an
    iterable (in this case the original ``it`` will be sliced if the length of
    ``crit`` is smaller)."""

    true, false = [], []
    if callable(crit):
        for ele in it:
            if crit(ele):
                true.append(ele)
            else:
                false.append(ele)
    elif hasattr(crit, '__iter__'):
        for keep, ele in zip(crit,it):
            if keep:
                true.append(ele)
            else:
                false.append(ele)
    else:
        raise TypeError("Crit must be  a function or a sequence")

    return true, false

#Copied from smpdf.utils
def split_ranges(a,cond=None,*, filter_falses=False):
    """Split ``a`` so that each range has the same
    value for ``cond`` . If ``filter_falses`` is true, only the ranges
    for which the
    condition is true will be returned."""
    if cond is None:
        cond = a
    cond = cond.astype(bool)
    d = np.r_[False, cond[1:]^cond[:-1]]
    split_at = np.argwhere(d)
    splits = np.split(a, np.ravel(split_at))
    if filter_falses:
        #Evaluate condition at split points
        it = iter(cond[np.r_[0, np.ravel(split_at)]])
        return [s for s in splits if next(it)]
    else:
        return splits


def sane_groupby_iter(df, by, *args, **kwargs):
    """Iterate groupby in such a way that  first value is always the tuple of
    the common values.

    As a concenience for plotting, if by is None, yield the empty string and
    the whole dataframe.
    """
    if by is None or not by:
        yield ('',), df
        return
    gb = df.groupby(by, *args,**kwargs)
    for same_vals, table in gb:
        if not isinstance(same_vals, tuple):
            same_vals = (same_vals,)
        yield same_vals, table

def common_prefix(*s):
    """Return the longest string that is a prefix to both s1 and s2"""
    small, big = min(s), max(s)
    for i, c in enumerate(small):
        if big[i] != c:
            return small[:i]
    return small

def scale_from_grid(grid):
    """Guess the appropriate matplotlib scale from a grid object.
    Returns ``'linear'`` if the scale of the grid object is linear,
    and otherwise ``' log'``."""
    return 'linear' if grid.scale == 'linear' else 'log'


def uncertainty_yaml_to_systype(path_uncertainty_yaml, name_dataset, path_systype=None, write_to_file=True):
    """
    Convert the new style uncertainty yaml file to the old style systype.
    Writes 

    Parameters
    ----------
    path_uncertainty_yaml : str, or Path
        Path to the new style uncertainty yaml file to be converted
    
    path_systype : str, or Path, optional
        path to the output systype file
    
    Returns
    -------
    n_sys : int
        Number of systematics in the systype file
    """
    # open the uncertainty yaml file
    with open(path_uncertainty_yaml) as f:
        uncertainty = yaml.safe_load(f)
    
    # get uncertainty definitions
    uncertainty_definitions = uncertainty['definitions']

    # check whether path_systype is provided else save it in the same directory in which the uncertainty yaml file is
    if path_systype is None:
        if isinstance(path_uncertainty_yaml, str):
            path_uncertainty_yaml = pathlib.Path(path_uncertainty_yaml)
        path_systype = path_uncertainty_yaml.parent / f"SYSTYPE_{name_dataset}_DEFAULT.dat"
    else:
        path_systype = pathlib.Path(path_systype) / f"SYSTYPE_{name_dataset}_DEFAULT.dat"
    
    # get number of sys (note: stat is not included in the sys)
    if 'stat' in uncertainty_definitions.keys():
        n_sys = len(uncertainty_definitions.keys()) - 1
    else:
        n_sys = len(uncertainty_definitions.keys())

    if write_to_file:
        # open the systype file for writing
        with open(path_systype, 'w') as stream:
            
            # header: number of sys 
            stream.write(f"{n_sys}\n")

            # write the systype treatments

            # remove stat from the uncertainty definitions
            uncertainty_definitions.pop('stat', None)
            
            for i, (_, sys_dict) in enumerate(uncertainty_definitions.items()):
                # four spaces seems to be the standard format (has to be checked for other datasets than CMS_1JET_8TEV)
                stream.write(f"{i+1}    {sys_dict['treatment']}    {sys_dict['type']}\n")

    return n_sys


def convert_new_data_to_old(path_data_yaml, path_uncertainty_yaml, path_kinematics, path_metadata, name_dataset, path_DATA=None):
    """
    Convert the new data format into the old data format
    """

    # open the metadata yaml file
    with open(path_metadata) as f:
        metadata = yaml.safe_load(f)

    # open the data yaml file
    with open(path_data_yaml) as f:
        data = yaml.safe_load(f)
    
    # open the uncertainty yaml file
    with open(path_uncertainty_yaml) as f:
        uncertainty = yaml.safe_load(f)

    # open the kinematics yaml file
    with open(path_kinematics) as f:
        kinematics = yaml.safe_load(f)
    
    # get uncertainty definitions and values
    uncertainty_definitions = uncertainty['definitions']
    uncertainty_values = uncertainty['bins']
    n_sys = uncertainty_yaml_to_systype(path_uncertainty_yaml, name_dataset, write_to_file=False)
    stats = []
    for entr in uncertainty_values:
        try: stats.append(entr["stat"])
        except KeyError: stats.append(0.)
    stats = np.array(stats)

    # get data values
    data_values = data['data_central']
    
    # check whether path_DATA is provided else save it in the same directory in which the uncertainty yaml file is
    if path_DATA is None:
        if isinstance(path_uncertainty_yaml, str):
            path_uncertainty_yaml = pathlib.Path(path_uncertainty_yaml)
        path_DATA = path_uncertainty_yaml.parent / f"DATA_{name_dataset}.dat"
    else:
        path_DATA = pathlib.Path(path_DATA) / f"DATA_{name_dataset}.dat"

    kin_names = list(kinematics['bins'][0].keys())
    kin_values = kinematics['bins']
    # open the DATA file for writing
    with open(path_DATA, 'w') as stream:
        
        # write the header: Dataset name, number of sys errors, and number of data points, whitespace separated
        stream.write(f"{name_dataset} {n_sys} {len(data_values)}\n")

        for i, data_value in enumerate(data_values):
            cd_line = f"{i+1:6}\t{metadata['implemented_observables'][0]['process_type']:6}\t"

            for index in [2, 1, 0]:
                if kin_values[i][kin_names[index]]['mid'] == None:
                    kin_values[i][kin_names[index]]['mid'] = (kin_values[i][kin_names[index]]['min'] + kin_values[i][kin_names[index]]['max']) / 2
                if kin_names[index] == "pT":
                    cd_line += f"{kin_values[i][kin_names[index]]['mid']**2:20.12e}\t"
                else:
                    cd_line += f"{kin_values[i][kin_names[index]]['mid']:20.12e}\t"

            cd_line += f"\t{data_value:20.12e}\t{stats[i]:20.12e}\t"

            # for j, sys in enumerate(uncertainty_values):
            sys = uncertainty_values[i]
            for j, (sys_name, sys_val) in enumerate(sys.items()):
                if sys_name == 'stat':
                    continue

                add_sys = sys_val
                if data_value != 0.0:
                    mult_sys = add_sys * 100.0 / data_value 
                else:
                    mult_sys = 0.0

                if j == len(sys)-1:
                    cd_line += f"{add_sys:20.12e}\t {mult_sys:20.12e}\n"
                else:
                    cd_line += f"{add_sys:20.12e}\t {mult_sys:20.12e}\t"

            stream.write(cd_line)

        

if __name__ == '__main__':
    new_commondata    = "/Users/teto/Software/nnpdf_git/nnpdf/nnpdf_data/nnpdf_data/new_commondata"
    test_dir          = "/Users/teto/Software/simunet_git/SIMUnet/validphys2/src/validphys/test_utils"
    name_dataset      = "CMS_1JET_13TEV_DIF"
    path_unc_file     = new_commondata+"/"+name_dataset+"/uncertainties_r04.yaml"
    path_data_yaml    = new_commondata+"/"+name_dataset+"/data_r04.yaml"
    path_kin          = new_commondata+"/"+name_dataset+"/kinematics_r04.yaml"
    path_metadata     = new_commondata+"/"+name_dataset+"/metadata.yaml"
    uncertainty_yaml_to_systype(path_unc_file, name_dataset=name_dataset, path_systype=test_dir)
    convert_new_data_to_old(path_data_yaml, path_unc_file, path_kin, path_metadata, name_dataset=name_dataset, path_DATA=test_dir)