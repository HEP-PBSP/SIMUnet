.. _simufit:

Simultaneous fits
====================

This is a basic tutorial to perform simultaneous PDF-EFT fits with SIMUnet.
The procedure consists of 3 steps: 

1. `Preparing a fit runcard <#preparing-a-fit-runcard>`_
2. `Running the fitting code <#running-the-fitting-code>`_
3. `Uploading and analysing the results of the fit <#upload-and-analyse-the-fit>`_

.. _preparing-a-fit-runcard:

1. Preparing a fit runcard
--------------------------

The runcard is written in YAML. The runcard is the unique identifier of a fit
and contains all required information to perform a fit, which includes the
experimental data, the theory setup and the fitting setup.

A detailed explanation on the parameters accepted by the ``n3fit`` runcards
can be found in the `detailed guide <runcard-detailed>`_.

For newcomers, it is recommended to start from an already existing runcard,
example runcards (and runcard used in NNPDF releases) are available at
`n3fit/runcards <https://github.com/NNPDF/nnpdf/tree/master/n3fit/runcards>`_.
The runcards are mostly self explanatory, see for instance below an
example of the ``parameter`` dictionary that defines the Machine Learning framework.

.. code-block:: yaml

    # runcard example
    ...
    parameters:
      nodes_per_layer: [15, 10, 8]
      activation_per_layer: ['sigmoid', 'sigmoid', 'linear']
      initializer: 'glorot_normal'
      optimizer:
        optimizer_name: 'RMSprop'
        learning_rate: 0.01
        clipnorm: 1.0
      epochs: 900
      positivity:
        multiplier: 1.05
        threshold: 1e-5
      stopping_patience: 0.30 # Ratio of the number of epochs
      layer_type: 'dense'
      dropout: 0.0
    ...

The runcard system is designed such that the user can utilize the program without having to
tinker with the codebase.
One can simply modify the options in ``parameters`` to specify the
desired architecture of the Neural Network as well as the settings for the optimization algorithm.

An important feature of ``n3fit`` is the ability to perform `hyperparameter scans <hyperoptimization>`_,
for this we have also introduced a ``hyperscan_config`` key which specifies
the trial ranges for the hyperparameter scan procedure.
See the following self-explanatory example:

.. code-block:: yaml

    hyperscan_config:
        stopping: # setup for stopping scan
            min_epochs: 5e2  # minimum number of epochs
            max_epochs: 40e2 # maximum number of epochs
            min_patience: 0.10 # minimum stop patience
            max_patience: 0.40 # maximum stop patience
        positivity: # setup for the positivity scan
            min_multiplier: 1.04 # minimum lagrange multiplier coeff.
            max_multiplier: 1.1 # maximum lagrange multiplier coeff.
            min_initial: 1.0 # minimum initial penalty
            max_initial: 5.0 # maximum initial penalty
        optimizer: # setup for the optimizer scan
            - optimizer_name: 'Adadelta'
              learning_rate:
                min: 0.5
                max: 1.5
            - optimizer_name: 'Adam'
              learning_rate:
                min: 0.5
                max: 1.5
        architecture: # setup for the architecture scan
            initializers: 'ALL' # Use all implemented initializers from keras
            max_drop: 0.15 # maximum dropout probability
            n_layers: [2,3,4] # number of layers
            min_units: 5 # minimum number of nodes
            max_units: 50 # maximum number of nodes
            activations: ['sigmoid', 'tanh'] # list of activation functions

It is also possible to take the configuration of the hyperparameter scan from a previous
run in the NNPDF server by using the key `from_hyperscan`:

.. code-block:: yaml

    hyperscan_config:
      from_hyperscan: 'some_previous_hyperscan'

or to directly take the trials from said hyperscan:

.. code-block:: yaml

    hyperscan_config:
      use_tries_from: 'some_previous_hyperscan'

.. _run-n3fit-fit:

2. Running the fitting code
---------------------------

After successfully installing the ``n3fit`` package and preparing a runcard
following the points presented above you can proceed with a fit.

1. Prepare the fit: ``vp-setupfit runcard.yml``. This command will generate a
    folder with the same name as the runcard (minus the file extension) in the
    current directory, which will contain a copy of the original YAML runcard.
    The required resources (such as the theory and t0 PDF set) will be
    downloaded automatically. Alternatively they can be obtained with the
    ``vp-get`` tool.

    .. note::
       This step is not strictly necessary when producing a standard fit with
       ``n3fit`` but it is required by :ref:`validphys <vp-index>`
       and it should therefore always be done. Note that :ref:`vp-upload <upload-fit>`
       will fail unless this step has been followed. If necessary, this step can
       be done after the fit has been run.

2. The ``n3fit`` program takes a ``runcard.yml`` as input and a replica number, e.g.
   ``n3fit runcard.yml replica`` where ``replica`` goes from 1-n where n is the
   maximum number of desired replicas. Note that if you desire, for example, a 100
   replica fit you should launch more than 100 replicas (e.g. 130) because not
   all of the replicas will pass the checks in ``postfit``
   (`see here <postfit-selection-criteria>`_ for more info).

3. Wait until you have fit results. Then run the ``evolven3fit`` program once to
   evolve all replicas using DGLAP. The arguments are ``evolven3fit runcard_folder
   number_of_replicas``. Remember to use the total number of replicas run (130 in the
   above example), rather than the number you desire in the final fit.

4. Wait until you have results, then use ``postfit number_of_replicas
   runcard_folder`` to finalize the PDF set by applying post selection criteria.
   This will produce a set of ``number_of_replicas + 1`` replicas. This time the
   number of replicas should be that which you desire in the final fit (100 in the
   above example). Note that the
   standard behaviour of ``postfit`` can be modified by using various flags.
   More information can be found at `Processing a fit <postfit>`_.

It is possible to run more than one replica in one single run of ``n3fit`` by
using the ``--replica_range`` option. Running ``n3fit`` in this way increases the
memory usage as all replicas need to be stored in memory but decreases disk load
as the reading of the datasets and fktables is only done once for all replicas.

If you are planning to perform a hyperparameter scan just perform exactly the
same steps by adding the ``--hyperopt number_of_trials`` argument to ``n3fit``,
where ``number_of_trials`` is the maximum allowed value of trials required by the
fit. Usually when running hyperparameter scan we switch-off the MC replica
generation so different replicas will correspond to different initial points for
the scan, this approach provides faster results. We provide the ``vp-hyperoptplot``
script to analyse the output of the hyperparameter scan.

Output of the fit
-----------------
Every time a replica is finalized, the output is saved to the `runcard/nnfit/replica_$replica`_
folder, which contains a number of files:

- ``chi2exps.log``: a json log file with the χ² of the training every 100 epochs.
- ``runcard.exportgrid``: a file containing the PDF grid.
- ``runcard.json``: Includes information about the fit (metadata, parameters, times) in json format.

.. note:: The reported χ² refers always to the actual χ², i.e., without positivity loss or other penalty terms.

.. _upload-fit:

3. Uploading and analysing the fit
----------------------------------
After obtaining the fit you can proceed with the fit upload and analysis by:

1. Uploading the results using ``vp-upload runcard_folder`` then install the
   fitted set with ``vp-get fit fit_name``.

2. Analysing the results with ``validphys``, see the `vp-guide <../vp/index>`_.
   Consider using the ``vp-comparefits`` tool.

SIMUnet analysis
----------------------

