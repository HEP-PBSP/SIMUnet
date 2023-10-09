"""
    Fixed layer

    This layer produces a Fixed observable, which consists of NO FK-tables, only the 
    fixed predictions.
    The rationale behind this layer is to keep all required operation in one single place
    such that is easier to optimize or modify.
"""

import numpy as np
from .observable import Observable
from n3fit.backends import operations as op
import tensorflow as tf

class Fixed(Observable):

    def gen_mask(self, basis):
        """
            Receives a list of active flavours and generates a boolean mask tensor
            Of course, this is completely pointless for the Fixed observables

            Parameters
            ----------
                basis: list(int)
                    list of active flavours
        """
        if basis is None:
            self.basis = np.ones(self.nfl, dtype=bool)
        else:
            basis_mask = np.zeros(self.nfl, dtype=bool)
            for i in basis:
                basis_mask[i] = True
        return op.numpy_to_tensor(basis_mask, dtype=bool)

    def call(self, pdf):
        """
            This function perform the fktable \otimes pdf convolution, which in this case
            is completely trivial. 

            Parameters
            ----------
                pdf:  backend tensor
                    rank 4 tensor (batch_size, xgrid, flavours, replicas)

            Returns
            -------
                result: backend tensor
                    rank 3 tensor (batchsize, replicas, ndata)
        """
        return tf.constant(self.fixed_predictions.reshape((1, 1, -1)))
