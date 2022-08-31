import tensorflow as tf
from tensorflow.keras.layers import Layer

import numpy as np

class CombineCfacLayer(Layer):
    """
    Creates the combination layer of SIMUnet. 
    """

    def __init__(self, n_bsm_fac_data, bsm_fac_data_scales, bsm_fac_data_names):
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

    def call(self, inputs, bsm_factor_values):
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
        _, ndata = bsm_factor_values.shape
        scale_reciprocals = [1/scale for scale in self.bsm_fac_data_scales]
        scales = np.array(scale_reciprocals)
        scales = np.tile(scales,(ndata,1)).T
        scales = tf.constant(scales.tolist(), dtype=float)

        bsm_factor_values = tf.multiply(bsm_factor_values,scales)

        ret = (1 + tf.reduce_sum(self.w[:, tf.newaxis] * bsm_factor_values, axis=0)) * inputs

        return ret
