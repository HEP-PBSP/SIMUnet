"""
n3fit version of simunet
"""

from n3fit.scripts.n3fit_exec import N3FIT_PROVIDERS, N3FitApp, N3FitConfig, N3FitEnvironment

# Import simufit to override the wrapper
from simunet import simufit
from simunet.config import SIMUConfig, SIMUEnvironment

SIMUNET_PROVIDERS = N3FIT_PROVIDERS + ["simunet.n3fit_providers"]


class SimufitEnvironment(SIMUEnvironment, N3FitEnvironment):
    pass


class SimufitConfig(SIMUConfig, N3FitConfig):

    def produce_simu_layer(self, simu_parameters=None):
        """
        Parses the simu_parameters dictionary and
        generates the simunet layer that will be applied to all obserables.
        """
        if simu_parameters is None:
            return None

        from simunet.simufit.combine_cfac import CombineCfacLayer

        return CombineCfacLayer(simu_parameters)


class SimunfitApp(N3FitApp):
    environment_class = SimufitEnvironment
    config_class = SimufitConfig

    def __init__(self):
        super(N3FitApp, self).__init__(name="SimunfitApp", providers=SIMUNET_PROVIDERS)


def main():
    SimunfitApp().main()


if __name__ == "__main__":
    main()
