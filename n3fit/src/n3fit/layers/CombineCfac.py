import tensorflow as tf
from tensorflow.keras.layers import Layer

import numpy as np
import hashlib

from validphys import initialisation_specs

class CombineCfacLayer(Layer):
    """
    Creates the combination layer of SIMUnet.
    """

    def __init__(
        self,
        scales,
        linear_names,
        initialisations,
        analytic_initialisation,
        initialisation_seed,
        replica_number,
    ):
        """
        Parameters
        ----------
            scales: np.array
                The scales by which to divide each contribution.
            linear_names: list[str]
                A list of names for the operators
            initialisations: list[dict]
                A list of dictionaries containing all the initialisation info.
        """
        # Initialise a Layer instance
        if len(scales) != len(linear_names):
            raise ValueError("Scales and linear_names must have the same length")
        super().__init__()

        # At this point, create a tf object with the correct random initialisation.
        initial_values = []
        assert len(initialisations) == len(linear_names)
        index = 0
        for ini, name in zip(initialisations, linear_names):
            hash_value = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (10 ** 18)
            seed = np.int32((initialisation_seed + replica_number) ^ hash_value)

            if isinstance(ini, initialisation_specs.ConstantInitialisation):
                val = tf.constant(ini.value, dtype='float32', shape=(1,))
            elif isinstance(ini, initialisation_specs.UniformInitialisation):
                val = tf.random_uniform_initializer(
                    minval=ini.minval,
                    maxval=ini.maxval,
                    seed=seed,
                )(shape=(1,))
            elif isinstance(ini, initialisation_specs.GaussianInitialisation):
                tf.random.set_seed(seed)
                val = tf.random.normal([1], ini.mean, ini.std_dev, tf.float32)
            elif isinstance(ini, initialisation_specs.AnalyticInitialisation):
                val = float(analytic_initialisation[index])
            else:
                raise RuntimeError(
                    "Invalid initialisation: choose form constant, uniform or Gaussian."
                )
            index += 1
            initial_values.append(val)

        num_initial = len(initial_values)

        if num_initial > 0:
            initial_values = tf.concat(initial_values, 0)
            initial_values = tf.reshape(initial_values, (num_initial,))
            self.w = tf.Variable(
                initial_value=initial_values,
                trainable=True,
            )
        else:
            self.w = tf.Variable(
                initial_value=tf.zeros(shape=(num_initial,), dtype="float32"),
                trainable=True,
            )

        self.scales = np.array(scales, dtype=np.float32)
        self.linear_names = linear_names

    def _compute_linear(self, linear_values):
        scaled_values = linear_values/self.scales[:, np.newaxis]
        return tf.reduce_sum(self.w[:, tf.newaxis] * scaled_values, axis=0)

    def call(self, inputs, linear_values):
        """
        Makes the forward pass to map the SM observable to the EFT one.

        Parameters
        ----------
            inputs: number
                This is the SM theoretical prediction that comes after the FK convolution.
            linear_values: dict[str, np.array]
                Dictionary mapping the names of the linear BSM C-factors for a
                given dataset, to arrays with the cfactor values, whose length
                is the number of datapoints. If not empty, the keys of the
                dictionary must match the ``linear_names`` attribute.

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
        if linear_values:
            if (a := set(linear_values)) != (b := set(self.linear_names)):
                raise ValueError(f"BSM factor names don't match {a}, {b}")
            linear_values = np.array(
                [linear_values[k] for k in self.linear_names],
                dtype=np.float32,
            )
            linear = self._compute_linear(linear_values)
            return (1 + linear) * inputs

        return inputs
