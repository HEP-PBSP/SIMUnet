import numpy as np

from n3fit.backends import MetaLayer


def _choose_initializer(ini_dict, scale=1.0):
    # TODO change to a provider
    if ini_dict["type"] == "uniform":
        max_v = ini_dict["maxval"] / scale
        min_v = ini_dict["minval"] / scale
        return MetaLayer.select_initializer("random_uniform", minval=min_v, maxval=max_v)
    elif ini_dict["type"] == "constant":
        return MetaLayer.init_constant(ini_dict["value"] / scale)
    raise ValueError(f"{ini_dict} not understood")


class CombineCfacLayer(MetaLayer):

    def __init__(self, simu_parameters, name="SimunetFactor", **kwargs):
        self._simu_parameters = simu_parameters
        self._kernel = []
        super().__init__(name=name, **kwargs)

    def apply_linear_comb(self, cfactors=None):
        """Take all cfactors and returns a list of pre-computed values to call this function with."""
        if cfactors is None:
            return [0.0] * len(self._simu_parameters)

        lin_comb = []
        for parameter in self._simu_parameters:
            linear_combination = parameter.get("linear_combination", {parameter["name"]: 1.0})
            scale = parameter.get("scale", 1.0)
            tmp = 0.0
            for k, v in linear_combination.items():
                tmp += np.array(cfactors.get(k, 0.0)) / np.array(cfactors["SM"]) * v
            lin_comb.append(tmp / scale)
        return lin_comb

    @property
    def scales(self):
        """Return the scales in self._simu_parameters as an iterable."""
        for parameter in self._simu_parameters:
            yield parameter.get("scale", 1.0)

    def build(self, input_shape):
        """Build stage should only be run at compile time or first inference.""" 
        for parameter in self._simu_parameters:
            initializer = _choose_initializer(parameter["initialisation"])
            ker = self.builder_helper(
                name=parameter["name"],
                kernel_shape=(1,),  # TODO here we could have a different one per replica
                initializer=initializer,
                trainable=True,
            )
            self._kernel.append(ker)
        super().build(input_shape)

    def call(self, linear_comb, observables):
        wsum = 0.0
        for kernel, arr in zip(self._kernel, linear_comb):
            wsum += kernel * arr
        return (1.0 + wsum) * observables
