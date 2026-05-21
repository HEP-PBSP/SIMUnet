"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
simunet_nnpdf providers, SIMUConfig and SIMUEnvironment.

# To do: Add example usage

"""
import logging

from reportengine import api
from validphys.app import providers
from simunet.app import simunet_providers
from simunet.config import SIMUConfig, SIMUEnvironment

log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(simunet_providers + providers , SIMUConfig, SIMUEnvironment)
