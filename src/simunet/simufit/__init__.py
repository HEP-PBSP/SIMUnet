""" """

# TODO
# Things to fix in n3fit' observables
# 1) It should take the FittableDataset directly instead of the content
# 2) It should be more amenable to post-processing (and pre-processing!)
# 3) A short-term solution is to take the model, inject the layer, then let the model continue though

from n3fit.backends.keras_backend.MetaModel import MetaModel
from n3fit.layers.observable import Observable
from n3fit.stopping import Stopping

original_init = Observable.__init__
original_call = Observable.call

# Define a simunet registry for keeping track of stuff we are injecting into validphys/n3fit
_REGISTRY = {}


def _patch_me_up():

    if _REGISTRY.get("patched", False):
        return

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

        return self.simunet_layer(self.simunet_cfactors, observables)

    Observable.__init__ = _init_patch
    Observable.call = _call_patch

    original_monitor = Stopping.monitor_chi2
    original_stopping_init = Stopping.__init__

    def stopping_init_patch(self, *args, **kwargs):
        """Freezing the model must happen here if freezing is necessary."""
        original_stopping_init(self, *args, **kwargs)
        if _REGISTRY.get("freeze", False):
            self._pdf_model.trainable = False

    def monitor_chi2_patch(self, training_info, epoch, print_stats=True):
        ret = original_monitor(self, training_info, epoch, print_stats=print_stats)
        if not ret:
            return False

        # Keep track of whether the best epoch changed
        if not hasattr(self, "simunet_best_epoch"):
            self.simunet_best = -1

        # TODO: at the moment this is only working for single replicas
        # For multireplica fits simunet_best need to be a list and check the entire _best_epoch list
        # then in the cfactor layer, we'll need a weight per replica for the initialization to be ok
        if self.simunet_best != self._best_epochs[0]:
            self.simunet_best = self._best_epochs[0]
            _REGISTRY["best_weights"] = _REGISTRY["layer"].get_weights()

        return ret

    Stopping.monitor_chi2 = monitor_chi2_patch
    Stopping.__init__ = stopping_init_patch

    _REGISTRY["patched"] = True


_patch_me_up()
