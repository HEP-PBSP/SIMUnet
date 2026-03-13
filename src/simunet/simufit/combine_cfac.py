import numpy as np

from n3fit.backends import MetaLayer


class CombineCfacLayer(MetaLayer):

    def __init__(self, simu_parameters, **kwargs):
        self._simu_parameters = simu_parameters
        self._kernel = []
        self._linear_comb = [i["linear_combination"] for i in simu_parameters]
        super().__init__(**kwargs)

    def build(self, input_shape):
        for parameter in self._simu_parameters:
            ini = parameter["initialisation"]
            initializer = MetaLayer.select_initializer(
                "random_uniform", minval=ini["minval"], maxval=ini["maxval"]
            )
            # TODO: deal with seeds and initialization
            ker = self.builder_helper(
                name=parameter["name"],
                kernel_shape=(1,),  # TODO here we could have a different one per replica
                initializer=initializer,
                trainable=True,
            )
            self._kernel.append(ker)

    def apply_linear_comb(self, cfactors=None):
        """Take all cfactors and returns a list of pre-computed values to call this function with."""
        if cfactors is None:
            return [0.0] * len(self._linear_comb)

        lin_comb = []
        for linear_combination in self._linear_comb:
            tmp = 0.0
            for k, v in linear_combination.items():
                tmp += np.array(cfactors[k]) * v
            lin_comb.append(tmp)
        return lin_comb

    def call(self, linear_comb):
        wsum = 0.0
        for kernel, arr in zip(self._kernel, linear_comb):
            wsum += kernel * arr
        return 1.0 + wsum
