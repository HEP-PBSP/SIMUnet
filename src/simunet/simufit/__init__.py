""" """

# TODO
# Things to fix in n3fit' observables
# 1) It should take the FittableDataset directly instead of the content
# 2) It should be more amenable to post-processing (and pre-processing!)
# 3) A short-term solution is to take the model, inject the layer, then let the model continue though

from n3fit.layers.observable import Observable
from n3fit.stopping import Stopping

original_init = Observable.__init__
original_call = Observable.call


def _init_patch(self, fktable_data, *args, **kwargs):

    original_init(self, fktable_data, *args, **kwargs)
    # Save the contamination layer that is coming as part of fktable_data (because of reasons)
    try:
        self.simunet_layer = fktable_data[0].simunet_layer
        self.simunet_cfactors = fktable_data[0].simunet_cfactors
    except AttributeError:  # for positivity/integrability that also pass through here
        self.simunet_layer = None
        self.simunet_cfactors = None


def _call_patch(self, pdf):
    observables = original_call(self, pdf)
    # Here do what's now in simunet's model_gen
    # NB: training/validation is now done after the forward pass so the whole cfactor is to be applied
    # something like
    if self.simunet_layer is None:
        return observables

    return self.simunet_layer(self.simunet_cfactors) * observables


Observable.__init__ = _init_patch
Observable.call = _call_patch

original_monitor = Stopping.monitor_chi2


def monitor_chi2_patch(self, training_info, epoch, print_stats=True):
    ret = original_monitor(self, training_info, epoch, print_stats=print_stats)
    print(self._validation.trainable_weights[-1])


Stopping.monitor_chi2 = monitor_chi2_patch
