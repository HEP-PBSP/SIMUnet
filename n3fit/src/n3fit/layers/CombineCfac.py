import tensorflow as tf
from tensorflow.keras.layers import Layer

import numpy as np

class CombineCfacLayer(Layer):
    """
    Creates the combination layer of SIMUnet.
    """

    def __init__(
        self,
        n_bsm_fac_data,
        bsm_fac_data_scales,
        bsm_fac_quad_scales,
        bsm_fac_data_names,
        bsm_fac_quad_names,
    ):
        """
        Parameters
        ----------
            n_bsm_fac_data: int 
                number of bsm parameters in the fit 
        """
        # Initialise a Layer instance
        super(CombineCfacLayer, self).__init__()

        init_value = tf.random_normal_initializer()
        self.w = tf.Variable(
            #initial_value=init_value(shape=(ncfacs,), dtype="float32"),
            initial_value=tf.zeros(shape=(n_bsm_fac_data,), dtype="float32"),
            trainable=True,
        )
        self.bsm_fac_data_names= bsm_fac_data_names  
        self.bsm_fac_data_scales = bsm_fac_data_scales 
        self.bsm_fac_quad_names = bsm_fac_quad_names
        self.bsm_fac_quad_scales = bsm_fac_quad_scales

    @property
    def flat_quadnames(self):
        return [item for sublist in self.bsm_fac_quad_names for item in sublist]

    def call(self, inputs, bsm_factor_values, quad_bsm_factor_values):
        """
        Makes the forward pass to map the SM observable to the EFT one.
        Parameters
        ----------
            inputs: number
                This is the SM theoretical prediction that comes after the FK convolution.
            bsm_factor_values: np.array
                Array of BSM C-factors for a given dataset, whose length is the
                number of datapoints.
        Returns
        -------
            output: tf.Tensor
               tensor of shape `(ndatapoints, )` with the EFT prediction
        """
        # 1) tensor[:, tf.newaxis] adds an extra dimension to the end of the tensor.
        # 2) tensor_1 * cfactor_values return a tensor of dimensions `(ncfacs, ncfactors)`
        # 3) tf.reduce_sum(tensor, axis=i) sums over the `i` dimension and gets rid of it

        # Convert the BSM factor scales
        # Sort coefficients in canonical order
        if (a := set(bsm_factor_values)) != (b := set(self.bsm_fac_data_names)):
            raise ValueError(f"BSM factor names don't match {a}, {b}")

        if (a := set(quad_bsm_factor_values)) != (b := set(self.flat_quadnames)):
            raise ValueError(f"BSM quad factor names don't match {a} {b}")

        bsm_factor_values = np.array(
            [bsm_factor_values[k] for k in self.bsm_fac_data_names],
            dtype=np.float32,
        )
        quad_bsm_factor_values = np.array(
            [quad_bsm_factor_values[k] for k in self.flat_quadnames],
            dtype=np.float32,
        )

        nops, ndata = bsm_factor_values.shape
        scale_reciprocals = [1 / scale for scale in self.bsm_fac_data_scales]
        scales = np.array(scale_reciprocals)
        scales = np.tile(scales, (ndata, 1)).T
        scales = tf.constant(scales.tolist(), dtype=float)

        # Convert the quadratic BSM factor scales
        # It's useful to flatten the BSM factor scales first
        flat_quad_scales = []
        for i in range(nops):
            for j in range(nops):
                # Now is our chance to eliminate duplicates (e.g. Oi*Oj and Oj*Oi should not both
                # enter the predictions if i is different from j).
                if i > j:
                    flat_quad_scales += [0.0]
                else:
                    flat_quad_scales += [self.bsm_fac_quad_scales[i][j]]

        quad_scale_reciprocals = []
        for i in range(len(flat_quad_scales)):
            if flat_quad_scales[i] == 0.0:
                quad_scale_reciprocals += [0.0]
            else:
                quad_scale_reciprocals += [1 / flat_quad_scales[i]]

        quad_scales = np.array(quad_scale_reciprocals)
        quad_scales = np.tile(quad_scales, (ndata, 1)).T
        quad_scales = tf.constant(quad_scales.tolist(), dtype=float)

        # Multiply by the scales
        bsm_factor_values = tf.multiply(bsm_factor_values, scales)
        quad_bsm_factor_values = tf.multiply(quad_bsm_factor_values, quad_scales)

        linear = tf.reduce_sum(self.w[:, tf.newaxis] * bsm_factor_values, axis=0)
        quad_weights = tf.stack(
            [self.w[i] * self.w[j] for i in range(nops) for j in range(nops)]
        )
        quadratic = tf.reduce_sum(
            quad_weights[:, tf.newaxis] * quad_bsm_factor_values, axis=0
        )


        return (1 + linear + quadratic) * inputs
