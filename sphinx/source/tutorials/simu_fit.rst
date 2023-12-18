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

We begin by showing the user an example of a complete runcard. We will go
into the details of each part  later. Here is a SIMUnet runcard:

.. code-block:: yaml

    # Runcard for SIMUnet
    #
    ############################################################
    description: "Example runcard. This one performs a simultaenous PDF-EFT fit using data from different sectors."

    ############################################################
    # frac: training fraction of datapoints for the PDFs
    # QCD: apply QCD K-factors
    # EWK: apply electroweak K-factors
    # simu_fac: fit BSM coefficients using their K-factors in the dataset 
    # use_fixed_predictions:  if set to True it removes the PDF dependence of the dataset
    # sys: systematics treatment (see systypes)

    dataset_inputs:
    # HERA
    - {dataset: HERACOMBNCEP575, frac: 0.75}
    # Drell - Yan
    - {dataset: CMSDY1D12, cfac: ['QCD', 'EWK']}
    # ttbar
    - {dataset: ATLASTTBARTOT7TEV, cfac: [QCD], simu_fac: "EFT_NLO"}
    # ttbar AC
    - {dataset: ATLAS_TTBAR_8TEV_ASY, cfac: [QCD], simu_fac: "EFT_NLO"}
    # TTZ
    - {dataset: ATLAS_TTBARZ_8TEV_TOTAL, simu_fac: "EFT_LO"}
    # TTW
    - {dataset: ATLAS_TTBARW_8TEV_TOTAL, simu_fac: "EFT_LO"}
    # single top
    - {dataset: ATLAS_SINGLETOP_TCH_7TEV_T, cfac: [QCD], simu_fac: "EFT_NLO"}
    # tW
    - {dataset: ATLAS_SINGLETOPW_8TEV_TOTAL, simu_fac: "EFT_NLO"}
    # W helicity
    - {dataset: ATLAS_WHEL_13TEV, simu_fac: "EFT_NLO", use_fixed_predictions: True}
    # ttgamma
    - {dataset: ATLAS_TTBARGAMMA_8TEV_TOTAL, simu_fac: "EFT_LO", use_fixed_predictions: True}
    # tZ
    - {dataset: ATLAS_SINGLETOPZ_13TEV_TOTAL, simu_fac: "EFT_LO", use_fixed_predictions: True}
    # EWPO
    - {dataset: LEP_ZDATA, simu_fac: "EFT_LO", use_fixed_predictions: True}
    # Higgs
    - {dataset: ATLAS_CMS_SSINC_RUNI, simu_fac: "EFT_NLO", use_fixed_predictions: True}
    # Diboson
    - {dataset: LEP_EEWW_182GEV, simu_fac: "EFT_LO", use_fixed_predictions: True}

    ############################################################
    #fixed_pdf_fit: True # If this is uncommented the PDFs are fixed during the fit and only the EFT coefficients are optimised
    #load_weights_from_fit: 221103-jmm-no_top_1000_iterated # If the line above is uncommented, the weights of the PDF are loaded from here
    analytic_initialisation_pdf: 221103-jmm-no_top_1000_iterated

    ############################################################
    simu_parameters:
    # Dipoles
    - {name: "OtG", scale: 0.01, initialisation: {type: uniform, minval: -10, maxval: 10} }
    # Quark Currents
    - {name: "Opt", scale: 0.1, initialisation: {type: gaussian, mean: 0, std_dev: 1} }
    # Lepton currents
    - {name: "O3pl", scale: 1.0, initialisation: {type: constant, value: 0} }
    # 4 Fermions 4Q
    - {name: 'O1qd', scale: 1.0, initialisation: {type: analytic}}
    # linear combination
    - name: 'Y'
      linear_combination:
        'Olq1 ': 1. 51606
        'Oed ': -6. 0606
        'Oeu ': 12. 1394
        'Olu ': 6. 0606
        'Old ': -3. 0394
        'Oqe ': 3. 0394
      scale: 1.0
      initialisation: { type: uniform , minval: -1, maxval: 1}

    ############################################################
    datacuts:
    t0pdfset: 221103-jmm-no_top_1000_iterated # PDF set to generate t0 covmat
    q2min: 3.49                        # Q2 minimum
    w2min: 12.5                        # W2 minimum

    ############################################################
    theory:
    theoryid: 200     # database id

    ############################################################
    trvlseed: 475038818
    nnseed: 2394641471
    mcseed: 1831662593
    save: "weights.h5"
    genrep: true      # true = generate MC replicas, false = use real data

    ############################################################
    parameters: # This defines the parameter dictionary that is passed to the Model Trainer
    nodes_per_layer: [25, 20, 8]
    activation_per_layer: [tanh, tanh, linear]
    initializer: glorot_normal
    optimizer:
        clipnorm: 6.073e-6
        learning_rate: 2.621e-3
        optimizer_name: Nadam
    epochs: 1000
    positivity:
        initial: 184.8
        multiplier:
    integrability:
        initial: 184.8
        multiplier:
    stopping_patience: 1.0
    layer_type: dense
    dropout: 0.0
    threshold_chi2: 3.5

    fitting:
    # EVOL(QED) = sng=0,g=1,v=2,v3=3,v8=4,t3=5,t8=6,(pht=7)
    # EVOLS(QED)= sng=0,g=1,v=2,v8=4,t3=4,t8=5,ds=6,(pht=7)
    # FLVR(QED) = g=0, u=1, ubar=2, d=3, dbar=4, s=5, sbar=6, (pht=7)
    fitbasis: EVOL  # EVOL (7), EVOLQED (8), etc.
    basis:
    - {fl: sng, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        1.093, 1.121], largex: [1.486, 3.287]}
    - {fl: g, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.8329, 1.071], largex: [3.084, 6.767]}
    - {fl: v, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.5202, 0.7431], largex: [1.556, 3.639]}
    - {fl: v3, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.1205, 0.4839], largex: [1.736, 3.622]}
    - {fl: v8, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.5864, 0.7987], largex: [1.559, 3.569]}
    - {fl: t3, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        -0.5019, 1.126], largex: [1.754, 3.479]}
    - {fl: t8, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        0.6305, 0.8806], largex: [1.544, 3.481]}
    - {fl: t15, pos: false, trainable: false, mutsize: [15], mutprob: [0.05], smallx: [
        1.087, 1.139], largex: [1.48, 3.365]}

    ############################################################
    positivity:
    posdatasets:
    - {dataset: POSF2U, maxlambda: 1e6}        # Positivity Lagrange Multiplier
    - {dataset: POSF2DW, maxlambda: 1e6}
    - {dataset: POSF2S, maxlambda: 1e6}
    - {dataset: POSFLL, maxlambda: 1e6}
    - {dataset: POSDYU, maxlambda: 1e10}
    - {dataset: POSDYD, maxlambda: 1e10}
    - {dataset: POSDYS, maxlambda: 1e10}
    - {dataset: POSF2C, maxlambda: 1e6}
    - {dataset: POSXUQ, maxlambda: 1e6}        # Positivity of MSbar PDFs
    - {dataset: POSXUB, maxlambda: 1e6}
    - {dataset: POSXDQ, maxlambda: 1e6}
    - {dataset: POSXDB, maxlambda: 1e6}
    - {dataset: POSXSQ, maxlambda: 1e6}
    - {dataset: POSXSB, maxlambda: 1e6}
    - {dataset: POSXGL, maxlambda: 1e6}

    ############################################################
    integrability:
    integdatasets:
    - {dataset: INTEGXT8, maxlambda: 1e2}
    - {dataset: INTEGXT3, maxlambda: 1e2}

    ############################################################
    debug: false
    maxcores: 4

For newcomers, it is recommended to start from an already existing runcard,
example runcards (and runcard used in NNPDF releases) are available at
`n3fit/runcards <https://github.com/NNPDF/nnpdf/tree/master/n3fit/runcards>`_.
The runcards are mostly self explanatory, see for instance below an
example of the ``parameter`` dictionary that defines the Machine Learning framework.

.. code-block:: yaml

    # runcard example
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

