import tensorflow as tf
from tensorflow.keras.layers import Layer


class CombineCfacLayer(Layer):
    """
    Creates the combination layer of SIMUnet. 
    """

    def __init__(self, n_bsm_fac_data, scale, fit_cfactors):
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
        self.fit_cfactors= fit_cfactors  
        self.scale = scale 

    def call(self, inputs, cfactor_values):
        """
        Makes the forward pass to map the SM observable to the EFT one. 
        Parameters
        ----------
            inputs: number
                This is the SM theoretical prediction that comes after the FK convolution.
            cfactor_values: np.array 
                Array of SMEFT C-factors for a given dataset, whose length is the 
                number of datapoints. 
        Returns
        -------
            output: tf.Tensor 
               tensor of shape `(ndatapoints, )` with the EFT prediction
        """
        # 1) tensor[:, tf.newaxis] adds an extra dimension to the end of the tensor. 
        # 2) tensor_1 * cfactor_values return a tensor of dimensions `(ncfacs, ncfactors)`
        # 3) tf.reduce_sum(tensor, axis=i) sums over the `i` dimension and gets rid of it 
        ret = (1 + tf.reduce_sum(self.w[:, tf.newaxis] * cfactor_values, axis=0) / self.scale) * inputs

        return ret