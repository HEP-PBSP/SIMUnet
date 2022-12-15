"""
    Library of functions which generate the NN objects

    Contains:
        # observable_generator:
            Generates the output layers as functions
        # pdfNN_layer_generator:
            Generates the PDF NN layer to be fitted


"""
from dataclasses import dataclass, field
from importlib.util import spec_from_loader
import numpy as np
import tensorflow as tf

from validphys import bsmnames

from n3fit.msr import msr_impose
from n3fit.layers import DIS, DY, ObsRotation, losses
from n3fit.layers import Preprocessing, FkRotation, FlavourToEvolution
from n3fit.backends import MetaModel, Input
from n3fit.backends import operations as op
from n3fit.backends import MetaLayer, Lambda
from n3fit.backends import base_layer_selector, regularizer_selector
from n3fit.layers.CombineCfac import CombineCfacLayer

import logging
log = logging.getLogger(__name__)



@dataclass
class ObservableWrapper:
    """Wrapper to generate the observable layer once the PDF model is prepared
    It can take normal datasets or Lagrange-multiplier-like datasets
    (such as positivity or integrability)
    """

    # IDEALLY:
    # In principle this is something that could be automatically provided by validphyts
    # __but__ it requires backend (i.e., tensorflow) information
    # but maybe it can be constructed in such a way that the backend is lazyly imported
    # and make this part of the experiment spec

    name: str
    observables: list
    dataset_xsizes: list
    invcovmat: np.array = None
    covmat: np.array = None
    multiplier: float = 1.0
    integrability: bool = False
    positivity: bool = False
    fixed: list = field(default_factory=list)
    data: np.array = None
    spec_dict: dict = None
    rotation: ObsRotation = None  # only used for diagonal covmat
    split: str = None
    post_observable: CombineCfacLayer = None


    def make_fixed_observable_inputs(self):
        """Generate a dictionary with values corresponding to the predictions
        for the fixed observables. Use the values to call the model.
        """
        # Note: Even though the inputs are constant, we still need to generate
        # them in one step and pass them to the model in another. See
        #
        # https://github.com/keras-team/keras/issues/11912
        #
        # Also we need the raw tensorflow tensor because there is no way to get
        # it from the Keras wrapper.
        res = {}
        for fo in self.fixed:
            inp = tf.constant(fo.prediction.central_value.reshape((1, 1, -1)))
            kinp = tf.keras.Input(tensor=inp)
            res[fo.commondata.setname] = (kinp, inp)
        return res

    def _all_data(self):
        """Concatenate experimental data from datasets and fixed observables"""
        return np.concatenate(
            [
                np.atleast_2d(self.data),
                *[np.atleast_2d(fo.commondata.central_values) for fo in self.fixed],
            ],
            axis=1,
        )


    def _generate_loss(self, mask=None):
        """Generates the corresponding loss function depending on the values the wrapper
        was initialized with"""
        if self.invcovmat is not None:
            loss = losses.LossInvcovmat(
                self.invcovmat,
                self._all_data(),
                mask,
                covmat=self.covmat,
                name=self.name,
            )
        elif self.positivity:
            loss = losses.LossPositivity(name=self.name, c=self.multiplier)
        elif self.integrability:
            loss = losses.LossIntegrability(name=self.name, c=self.multiplier)
        return loss


    def _generate_experimental_layer(self, pdf, fixed_inputs):
        """Generates the experimental layer from the PDF"""
        # First split the layer into the different datasets (if needed!)
        if len(self.dataset_xsizes) > 1:
            splitting_layer = op.as_layer(
                op.split,
                op_args=[self.dataset_xsizes],
                op_kwargs={"axis": 1},
                name=f"{self.name}_split",
            )
            split_pdf = splitting_layer(pdf)
        else:
            split_pdf = [pdf]
        # Every obs gets its share of the split
        output_layers = [obs(p_pdf) for p_pdf, obs in zip(split_pdf, self.observables)]

        for idx, (dataset_dict, output_layer) in enumerate(
            zip(self.spec_dict['datasets'], output_layers)
        ):
            # Use get here to prevent having to worry about POSDATSETS
            bsm_fac_data_names_CF = dataset_dict.get('bsm_fac_data_names_CF')
            bsm_fac_quad_names_CF = dataset_dict.get('bsm_fac_quad_names_CF')
            bsm_fac_data_names = dataset_dict.get('bsm_fac_data_names')
            bsm_fac_quad_names = dataset_dict.get('bsm_fac_quad_names')

            # It's useful to flatten the list of quadratic names first

            if bsm_fac_data_names_CF is not None:

                # coefficients = np.array([bsm_fac_data_names_CF[i].central_value for i in bsm_fac_data_names])
                coefficients = {
                    bsmnames.linear_datum_to_op(k): v.central_value
                    for k, v in bsm_fac_data_names_CF.items()
                }

                if bsm_fac_quad_names_CF is not None:
                    quad_coefficients = {
                        bsmnames.linear_datum_to_op(k): v.central_value
                        for k, v in bsm_fac_quad_names_CF.items()
                    }
                else:
                    ndat = dataset_dict["ndata"]
                    quad_coefficients = {
                        name: np.zeros(ndat)
                        for name in self.post_observable.quad_names
                    }

                if self.split == 'ex':
                    cfacs = coefficients
                    quad_cfacs = quad_coefficients
                elif self.split == 'tr':
                    cfacs = {
                        k: v[dataset_dict["ds_tr_mask"]]
                        for k, v in coefficients.items()
                    }
                    quad_cfacs = {
                        k: v[dataset_dict["ds_tr_mask"]]
                        for k, v in quad_coefficients.items()
                    }
                elif self.split == 'vl':
                    # cfacs = coefficients[:, ~dataset_dict['ds_tr_mask']]
                    cfacs = {
                        k: v[~dataset_dict["ds_tr_mask"]]
                        for k, v in coefficients.items()
                    }
                    quad_cfacs = {
                        k: v[~dataset_dict["ds_tr_mask"]]
                        for k, v in quad_coefficients.items()
                    }
                log.info("Applying combination layer")

                output_layers[idx] = self.post_observable(
                    output_layer,
                    linear_values=cfacs,
                    quad_values=quad_cfacs,
                )

        for fo, inp in zip(self.fixed, fixed_inputs):
            linear = {
                bsmnames.linear_datum_to_op(k): v.central_value
                for k, v in fo.linear_bsm.items()
            }
            quad = {bsmnames.linear_datum_to_op(k): v.central_value for k, v in fo.quad_bsm.items()}

            # NOTE: Generating the input like
            #
            # inp = tf.keras.Input(tensor=tf.constant(fo.prediction.central_value))
            #
            # doesn't work becuse it needs to be seen by the model instance (despite being constant). See
            # https://github.com/keras-team/keras/issues/11912
            output_layers.append(
                self.post_observable(
                    inputs=inp,
                    linear_values=linear,
                    quad_values=quad,
                )
            )

        # Concatenate all datasets (so that experiments are one single entity)
        ret = op.concatenate(output_layers, axis=2)

        if self.rotation is not None:
            ret = self.rotation(ret)
        return ret

    def __call__(self, pdf_layer, fixed_inputs, mask=None):
        loss_f = self._generate_loss(mask)
        experiment_prediction = self._generate_experimental_layer(pdf_layer, fixed_inputs)
        return loss_f(experiment_prediction)


def observable_generator(
    spec_dict, positivity_initial=1.0, integrability=False, post_observable=None
):  # pylint: disable=too-many-locals
    """
    This function generates the observable model for each experiment.
    These are models which takes as input a PDF tensor (1 x size_of_xgrid x flavours) and outputs
    the result of the observable for each contained dataset (n_points,)

    An experiment contains an fktable, which is loaded by the convolution layer
    (be it hadronic or DIS) and a inv covmat which loaded by the loss.

    This function also outputs three "output objects" (which are functions that generate layers)
    that use the training and validation mask to create a training_output, validation_output
    and experimental_output

    If the dataset is a positivity dataset acts in consequence.

    The output is a dictionary (`layer_info`), each one of the three output functions
    have a signature:

        `def out_tr(pdf_layer, dataset_out=None)`

    The `pdf_layer` must be a layer of shape (1, size_of_xgrid, flavours)
    `datasets_out` is the list of dataset to be masked to 0 when generating the layer

    Parameters
    ----------
        spec_dict: dict
            a dictionary-like object containing the information of the experiment
        positivity_initial: float
            set the positivity lagrange multiplier for epoch 1

    Returns
    ------
        layer_info: dict
            a dictionary with:
            - `inputs`: input layer
            - `output`: output layer (unmasked)
            - `output_tr`: output layer (training)
            - `output_vl`: output layer (validation)
            - `experiment_xsize`: int (size of the output array)
    """
    spec_name = spec_dict["name"]
    dataset_xsizes = []
    model_obs_tr = []
    model_obs_vl = []
    model_obs_ex = []
    model_inputs = []
    # The first step is to compute the observable for each of the datasets
    for dataset_dict in spec_dict["datasets"]:
        # Get the generic information of the dataset
        dataset_name = dataset_dict["name"]

        # Look at what kind of layer do we need for this dataset
        if dataset_dict["hadronic"]:
            Obs_Layer = DY
        else:
            Obs_Layer = DIS

        # Set the operation (if any) to be applied to the fktables of this dataset
        operation_name = dataset_dict["operation"]

        # Now generate the observable layer, which takes the following information:
        # operation name
        # dataset name
        # list of fktable_dictionaries
        #   these will then be used to check how many different pdf inputs are needed
        #   (and convolutions if given the case)

        if spec_dict["positivity"]:
            # Positivity (and integrability, which is a special kind of positivity...)
            # enters only at the "training" part of the models
            obs_layer_tr = Obs_Layer(
                dataset_dict["fktables"],
                dataset_dict["tr_fktables"],
                operation_name,
                name=f"dat_{dataset_name}",
            )
            obs_layer_ex = obs_layer_vl = None
        elif spec_dict.get("data_transformation_tr") is not None:
            # Data transformation needs access to the full array of output data
            obs_layer_ex = Obs_Layer(
                dataset_dict["fktables"],
                dataset_dict["ex_fktables"],
                operation_name,
                name=f"exp_{dataset_name}",
            )
            obs_layer_tr = obs_layer_vl = obs_layer_ex
        else:
            obs_layer_tr = Obs_Layer(
                dataset_dict["fktables"],
                dataset_dict["tr_fktables"],
                operation_name,
                name=f"dat_{dataset_name}",
            )
            obs_layer_ex = Obs_Layer(
                dataset_dict["fktables"],
                dataset_dict["ex_fktables"],
                operation_name,
                name=f"exp_{dataset_name}",
            )
            obs_layer_vl = Obs_Layer(
                dataset_dict["fktables"],
                dataset_dict["vl_fktables"],
                operation_name,
                name=f"val_{dataset_name}",
            )

        # To know how many xpoints we compute we are duplicating functionality from obs_layer
        if obs_layer_tr.splitting is None:
            xgrid = dataset_dict["fktables"][0]["xgrid"]
            model_inputs.append(xgrid)
            dataset_xsizes.append(xgrid.shape[1])
        else:
            xgrids = [i["xgrid"] for i in dataset_dict["fktables"]]
            model_inputs += xgrids
            dataset_xsizes.append(sum([i.shape[1] for i in xgrids]))

        model_obs_tr.append(obs_layer_tr)
        model_obs_vl.append(obs_layer_vl)
        model_obs_ex.append(obs_layer_ex)

    full_nx = sum(dataset_xsizes)
    if spec_dict["positivity"]:
        out_positivity = ObservableWrapper(
            spec_name,
            model_obs_tr,
            dataset_xsizes,
            multiplier=positivity_initial,
            positivity=not integrability,
            integrability=integrability,
            spec_dict=spec_dict,
            split='ex',
            post_observable=post_observable
        )

        layer_info = {
            "inputs": model_inputs,
            "output_tr": out_positivity,
            "experiment_xsize": full_nx,
        }
        # For positivity we end here
        return layer_info

    # Generate the loss function and rotations of the final data (if any)
    if spec_dict.get("data_transformation_tr") is not None:
        obsrot_tr = ObsRotation(spec_dict.get("data_transformation_tr"))
        obsrot_vl = ObsRotation(spec_dict.get("data_transformation_vl"))
    else:
        obsrot_tr = None
        obsrot_vl = None


    out_tr = ObservableWrapper(
        spec_name,
        model_obs_tr,
        dataset_xsizes,
        invcovmat=spec_dict["invcovmat"],
        data=spec_dict["expdata"],
        rotation=obsrot_tr,
        spec_dict=spec_dict,
        split='tr',
        post_observable=post_observable,
        fixed=spec_dict["fixed"],
    )
    out_vl = ObservableWrapper(
        f"{spec_name}_val",
        model_obs_vl,
        dataset_xsizes,
        invcovmat=spec_dict["invcovmat_vl"],
        data=spec_dict["expdata_vl"],
        rotation=obsrot_vl,
        spec_dict=spec_dict,
        split='vl',
        post_observable=post_observable,
        fixed=spec_dict["fixed_vl"],
    )
    out_exp = ObservableWrapper(
        f"{spec_name}_exp",
        model_obs_ex,
        dataset_xsizes,
        invcovmat=spec_dict["invcovmat_true"],
        covmat=spec_dict["covmat"],
        data=spec_dict["expdata_true"],
        rotation=None,
        spec_dict=spec_dict,
        split='ex',
        post_observable=post_observable,
        fixed=spec_dict["fixed_true"],
    )

    layer_info = {
        "inputs": model_inputs,
        "output": out_exp,
        "output_tr": out_tr,
        "output_vl": out_vl,
        "experiment_xsize": full_nx,
    }
    return layer_info


# Network generation functions
def generate_dense_network(
    nodes_in,
    nodes,
    activations,
    initializer_name="glorot_normal",
    seed=0,
    dropout_rate=0.0,
    regularizer=None,
):
    """
    Generates a dense network

    the dropout rate, if selected, is set
    for the next to last layer (i.e., the last layer of the dense network before getting to
    the output layer for the basis choice)
    """
    list_of_pdf_layers = []
    number_of_layers = len(nodes)
    if dropout_rate > 0:
        dropout_layer = number_of_layers - 2
    else:
        dropout_layer = -1
    for i, (nodes_out, activation) in enumerate(zip(nodes, activations)):
        # if we have dropout set up, add it to the list
        if dropout_rate > 0 and i == dropout_layer:
            list_of_pdf_layers.append(base_layer_selector("dropout", rate=dropout_rate))

        # select the initializer and move the seed
        init = MetaLayer.select_initializer(initializer_name, seed=seed + i)

        # set the arguments that will define the layer
        arguments = {
            "kernel_initializer": init,
            "units": int(nodes_out),
            "activation": activation,
            "input_shape": (nodes_in,),
            "kernel_regularizer": regularizer,
        }

        layer = base_layer_selector("dense", **arguments)

        list_of_pdf_layers.append(layer)
        nodes_in = int(nodes_out)
    return list_of_pdf_layers


def generate_dense_per_flavour_network(
    nodes_in, nodes, activations, initializer_name="glorot_normal", seed=0, basis_size=8
):
    """
    For each flavour generates a dense network of the chosen size

    """
    list_of_pdf_layers = []
    number_of_layers = len(nodes)
    current_seed = seed
    for i, (nodes_out, activation) in enumerate(zip(nodes, activations)):

        initializers = []
        for _ in range(basis_size):
            # select the initializer and move the seed
            initializers.append(MetaLayer.select_initializer(initializer_name, seed=current_seed))
            current_seed += 1

        # set the arguments that will define the layer
        # but careful, the last layer must be nodes = 1
        # TODO the mismatch is due to the fact that basis_size
        # is set to the number of nodes of the last layer when it should
        # come from the runcard
        if i == number_of_layers - 1:
            nodes_out = 1
        arguments = {
            "kernel_initializer": initializers,
            "units": nodes_out,
            "activation": activation,
            "input_shape": (nodes_in,),
            "basis_size": basis_size,
        }

        layer = base_layer_selector("dense_per_flavour", **arguments)

        if i == number_of_layers - 1:
            # For the last layer, apply concatenate
            concat = base_layer_selector("concatenate")

            def output_layer(ilayer):
                result = layer(ilayer)
                return concat(result)

            list_of_pdf_layers.append(output_layer)
        else:
            list_of_pdf_layers.append(layer)

        nodes_in = int(nodes_out)
    return list_of_pdf_layers


def pdfNN_layer_generator(
    inp=2,
    nodes=None,
    activations=None,
    initializer_name="glorot_normal",
    layer_type="dense",
    flav_info=None,
    fitbasis="NN31IC",
    out=14,
    seed=None,
    dropout=0.0,
    regularizer=None,
    regularizer_args=None,
    impose_sumrule=None,
    scaler=None,
    parallel_models=1,
):  # pylint: disable=too-many-locals
    """
    Generates the PDF model which takes as input a point in x (from 0 to 1)
    and outputs a basis of 14 PDFs.
    It generates the preprocessing of the x into a set (x, log(x)),
    the arbitrary NN to fit the form of the PDF
    and the preprocessing factors.

    The funtional form of the output of this function is of:

        f_{i}(x) = R_{ji} N(x)_{j} * x^{1-alpha_{j}} * (1-x)^{beta_{j}}

    Where i goes from 1 to 14 while j goes from 1 to the size of the basis. R_{ji}
    is the rotation from the fitting basis to the physical basis needed for the
    convolution with the fktables.


    `layer_type` defines the architecture of the Neural Network, currently
    the following two options are implemented:
        - `dense`
            it will generate a densely connected networks where all nodes of layer n
            are connected with all nodes of layer n+1 (and n-1), the output layer is
            of the size of the last item in the `nodes` list (usually 8)
        - `dense_per_flavour`
            similar to `dense` but the nodes are disconnected in a per-flavour basis
            This means at the end of the PDF (and before the preprocessing)
            we have 8 (or whatever number) nodes just as before, but from the input
            to the output the nodes are disconnected

    This is a complicated function comprised of several sections:

    1. As initialization a number of checks are carried outs to ensure the number of layers
    and the number of activation functions match up.

    2. The neural network NN(x) is constructed depending on the requested layer_type

    3. Break the input into (x, log(x)) and passes down to the NN.
    The output of the NN in the previous step is a list of independent layers.
    A function is constructed that joins all those layers. The function takes a
    tensor as the input and applies all layers for NN in order.

    4. Create a preprocessing layer (that takes as input the same tensor x as the NN)
    and multiply it to the NN. We have now:
        N(x)_{j} * x^{1-alpha_{j}} * (1-x)^{beta_{j}}

    5. Create a rotation layer (that takes as input the previous step) and applies it
    so that we get f(x)_{i}

    Finally we output the final answer as well as the list of all generating functions
    in the model for easy usage within `n3fit`.

    Example
    -------

    >>> import numpy as np
    >>> from n3fit.vpinterface import N3PDF
    >>> from n3fit.model_gen import pdfNN_layer_generator
    >>> from validphys.pdfgrids import xplotting_grid
    >>> fake_fl = [{'fl' : i, 'largex' : [0,1], 'smallx': [1,2]} for i in ['u', 'ubar', 'd', 'dbar', 'c', 'cbar', 's', 'sbar']]
    >>> fake_x = np.linspace(1e-3,0.8,3)
    >>> pdf_model = pdfNN_layer_generator(nodes=[8], activations=['linear'], seed=[2,3], flav_info=fake_fl, parallel_models=2)

    Parameters
    ----------
        inp: int
            dimension of the xgrid. If inp=2, turns the x point into a (x, log(x)) pair
        nodes: list(int)
            list of the number of nodes per layer of the PDF NN. Default: [15,8]
        activation: list
            list of activation functions to apply to each layer. Default: ["tanh", "linear"]
            if the number of activation function does not match the number of layers, it will add
            copies of the first activation function found
        initializer_name: str
            selects the initializer of the weights of the NN. Default: glorot_normal
        layer_type: str
            selects the type of architecture of the NN. Default: dense
        flav_info: dict
            dictionary containing the information about each PDF (basis dictionary in the runcard)
            to be used by Preprocessing
        out: int
            number of output flavours of the model (default 14)
        seed: list(int)
            seed to initialize the NN
        dropout: float
            rate of dropout layer by layer
        impose_sumrule: str
            whether to impose sumrules on the output pdf and which one to impose (All, MSR, VSR)
        scaler: scaler
            Function to apply to the input. If given the input to the model
            will be a (1, None, 2) tensor where dim [:,:,0] is scaled
        parallel_models: int
            How many models should be trained in parallel

    Returns
    -------
       pdf_models: list with a number equal to `parallel_models` of type n3fit.backends.MetaModel
            a model f(x) = y where x is a tensor (1, xgrid, 1) and y a tensor (1, xgrid, out)
    """
    # Parse the input configuration
    if seed is None:
        seed = parallel_models * [None]
    elif isinstance(seed, int):
        seed = parallel_models * [seed]

    if nodes is None:
        nodes = [15, 8]
    ln = len(nodes)

    if impose_sumrule is None:
        impose_sumrule = "All"

    if scaler:
        inp = 1

    if activations is None:
        activations = ["tanh", "linear"]
    elif callable(activations):
        # hyperopt passes down a function to generate dynamically the list of
        # activations functions
        activations = activations(ln)

    if regularizer_args is None:
        regularizer_args = dict()

    number_of_layers = len(nodes)
    # The number of nodes in the last layer is equal to the number of fitted flavours
    last_layer_nodes = nodes[-1]  # (== len(flav_info))

    # Generate the generic layers that will not depend on extra considerations

    # First prepare the input for the PDF model and any scaling if needed
    placeholder_input = Input(shape=(None, 1), batch_size=1)

    subtract_one = False
    process_input = Lambda(lambda x: x)
    input_x_eq_1 = [1.0]
    # When scaler is active we also want to do the subtraction of large x
    # TODO: make it its own option (i.e., one could want to use this without using scaler)
    if scaler:
        # change the input domain [0,1] -> [-1,1]
        process_input = Lambda(lambda x: 2 * x - 1)
        subtract_one = True
        input_x_eq_1 = scaler([1.0])[0]
        placeholder_input = Input(shape=(None, 2), batch_size=1)
    elif inp == 2:
        # If the input is of type (x, logx)
        # create a x --> (x, logx) layer to preppend to everything
        process_input = Lambda(lambda x: op.concatenate([x, op.op_log(x)], axis=-1))

    model_input = {"pdf_input": placeholder_input}
    if subtract_one:
        layer_x_eq_1 = op.numpy_to_input(np.array(input_x_eq_1).reshape(1, 1))
        model_input["layer_x_eq_1"] = layer_x_eq_1

    # Evolution layer
    layer_evln = FkRotation(input_shape=(last_layer_nodes,), output_dim=out)

    # Basis rotation
    basis_rotation = FlavourToEvolution(flav_info=flav_info, fitbasis=fitbasis)

    # Normalization and sum rules
    if impose_sumrule:
        sumrule_layer, integrator_input = msr_impose(mode=impose_sumrule, scaler=scaler)
        model_input["integrator_input"] = integrator_input
    else:
        sumrule_layer = lambda x: x

    # Now we need a trainable network per model to be trained in parallel
    pdf_models = []
    for i, layer_seed in enumerate(seed):
        if layer_type == "dense":
            reg = regularizer_selector(regularizer, **regularizer_args)
            list_of_pdf_layers = generate_dense_network(
                inp,
                nodes,
                activations,
                initializer_name,
                seed=layer_seed,
                dropout_rate=dropout,
                regularizer=reg,
            )
        elif layer_type == "dense_per_flavour":
            # Define the basis size attending to the last layer in the network
            # TODO: this information should come from the basis information
            #       once the basis information is passed to this class
            list_of_pdf_layers = generate_dense_per_flavour_network(
                inp,
                nodes,
                activations,
                initializer_name,
                seed=layer_seed,
                basis_size=last_layer_nodes,
            )

        def dense_me(x):
            """Takes an input tensor `x` and applies all layers
            from the `list_of_pdf_layers` in order"""
            processed_x = process_input(x)
            curr_fun = list_of_pdf_layers[0](processed_x)

            for dense_layer in list_of_pdf_layers[1:]:
                curr_fun = dense_layer(curr_fun)
            return curr_fun

        preproseed = layer_seed + number_of_layers
        layer_preproc = Preprocessing(
            flav_info=flav_info,
            input_shape=(1,),
            name=f"pdf_prepro_{i}",
            seed=preproseed,
            large_x=not subtract_one,
        )

        # Apply preprocessing and basis
        def layer_fitbasis(x):
            """The tensor x has a expected shape of (1, None, {1,2})
            where x[...,0] corresponds to the feature_scaled input and x[...,-1] the original input
            """
            x_scaled = op.op_gather_keep_dims(x, 0, axis=-1)
            x_original = op.op_gather_keep_dims(x, -1, axis=-1)

            nn_output = dense_me(x_scaled)
            if subtract_one:
                nn_at_one = dense_me(layer_x_eq_1)
                nn_output = op.op_subtract([nn_output, nn_at_one])

            ret = op.op_multiply([nn_output, layer_preproc(x_original)])
            if basis_rotation.is_identity():
                # if we don't need to rotate basis we don't want spurious layers
                return ret
            return basis_rotation(ret)

        # Rotation layer, changes from the 8-basis to the 14-basis
        def layer_pdf(x):
            return layer_evln(layer_fitbasis(x))

        # Final PDF (apply normalization)
        final_pdf = sumrule_layer(layer_pdf)

        # Create the model
        pdf_model = MetaModel(
            model_input, final_pdf(placeholder_input), name=f"PDF_{i}", scaler=scaler
        )
        pdf_models.append(pdf_model)
    return pdf_models
