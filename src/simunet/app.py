"""
simunet.app.py

"""

from validphys.app import App
import validphys.commondata
import validphys.results
from simunet.config import SIMUConfig, SIMUEnvironment

simunet_providers = [
    "simunet.pseudodata",
    "simunet.analysis",
    "simunet.results",
    "reportengine.report",
    validphys.commondata,
    validphys.results,
]  # Does the order here matter?


class SIMUnetApp(App):
    config_class = SIMUConfig
    environment_class = SIMUEnvironment


def main():
    a = SIMUnetApp(name="simunet", providers=simunet_providers)
    a.main()


if __name__ == "__main__":
    main()
