"""
simunet.app.py

"""

from validphys.app import App
from simunet.config import SIMUConfig, SIMUEnvironment

simunet_providers = [
    "simunet.pseudodata",
    "simunet.analysis",
    "simunet.results",
    "reportengine.report",
]


class SIMUnetApp(App):
    config_class = SIMUConfig
    environment_class = SIMUEnvironment


def main():
    a = SIMUnetApp(name="simunet", providers=simunet_providers)
    a.main()


if __name__ == "__main__":
    main()
