"""
simunet_validphys.app.py

"""

from validphys.app import App
from simunet_validphys.simunet_config import SIMUConfig


simunet_providers = [
    "simunet_validphys.simunet_pseudodata",
]


class SIMUnetApp(App):
    config_class = SIMUConfig


def main():
    a = SIMUnetApp(name="simunet", providers=simunet_providers)
    a.main()


if __name__ == "__main__":
    main()