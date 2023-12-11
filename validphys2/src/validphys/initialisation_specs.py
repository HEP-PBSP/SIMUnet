"""Module containing representations of initialisation strategies for fit
parameters. The actual sampling of the parameters is not implemented here.
However, there are checks to ensure the consitency of the parameters and obtain
them from a runcard.

The ``Initialisation`` type can be used to dispach an arbitrary dictionary to
one of the the initialisation methods, using the ``validobj`` library.

Example
-------

>>> from validphys.initialisation_specs import Initialisation
>>> import validobj
>>> validobj.parse_input({"type": "gaussian", "mean": 0, "std_dev": 1}, Initialisati
... on)
GaussianInitialisation(type='gaussian', mean=0, std_dev=1)

"""
import dataclasses
import typing

import validobj

import numpy as np

__all__ = [
    "UniformInitialisation",
    "GaussianInitialisation",
    "ConstantInitialisation",
    "AnalyticInitialisation",
    "Initialisation",
]

Number = typing.Union[float, int]


@dataclasses.dataclass
class UniformInitialisation:
    type: typing.Literal["uniform"]
    maxval: Number
    minval: Number

    def __post_init__(self):
        if not self.minval <= self.maxval:
            raise validobj.ValidationError("minval should be smaller than maxval")


@dataclasses.dataclass
class GaussianInitialisation:
    type: typing.Literal["gaussian"]
    mean: Number
    std_dev: Number

    def __post_init__(self):
        if not self.std_dev > 0:
            raise validobj.ValidationError("std_dev should bs positive")


@dataclasses.dataclass
class ConstantInitialisation:
    type: typing.Literal["constant"]
    value: Number

@dataclasses.dataclass
class AnalyticInitialisation:
    type: typing.Literal["analytic"]

Initialisation = typing.Union[
    UniformInitialisation, GaussianInitialisation, ConstantInitialisation, AnalyticInitialisation
]
