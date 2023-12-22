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
into the details of each part  later. Here is a complete SIMUnet runcard:

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

    ############################################################
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

The structure of the runcard is similar to the one that is used in the NNPDF methodology.
So, in this tutorial we will mostly adress the new syntax and features of SIMUnet. 

We begin by looking at the following section of the runcard:

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

It contains the description of the runcard and some short comments about new keys
of SIMUnet. The user should always provide a useful ``description`` of the runcard as
it will appear when running analyses and can provide information to other people studying the fit.

Now we consider the following fraction of the runcard:

.. code-block:: yaml

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

The ``dataset_inputs`` key contains the datasets that will be used to peform the
simultaneous PDF-EFT fit. The first two datasets, ``HERACOMBNCEP575`` and
``CMSDY1D12``, are included in the same way as in a NNPDF fit, and are
used only to fit PDFs. All the other datasets have the key ``simu_fac`` set to either
``EFT_LO`` or ``EFT_NLO``. This means that SIMUnet will use those datasets to fit
EFT coefficients at the desired accuracy, LO or NLO. The fit requires EFT K-factors for all
the datasets that have the ``simu_fac`` key. Additinally, some datasets have the key ``use_fixed_predictions``
set to ``True``. This means that the PDF dependence is removed from this dataset and, effectively,
the dataset becomes PDF-independent.

Now, we take a look of this part of the runcard:

.. code-block:: yaml

    #fixed_pdf_fit: True # If this is uncommented the PDFs are fixed during the fit and only the EFT coefficients are optimised
    #load_weights_from_fit: 221103-jmm-no_top_1000_iterated # If the line above is uncommented, the weights of the PDF are loaded from here
    analytic_initialisation_pdf: 221103-jmm-no_top_1000_iterated

These keys, if uncommented, allow the user to perform a fixed-PDF fit. This means that only
the EFT coefficients are found during the optimisation. If ``fixed_pdf_fit: True``, the PDF weights
are loaded from the fit ``221103-jmm-no_top_1000_iterated``. Still have to comment on the analytic
initialisation of the PDFs.

We move on to this part of the runcard:

.. code-block:: yaml

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

This block contains the EFT coefficients that are going to be fitted. Each one
of them has a key ``name``. The name usually resembles the notation of the Warsaw
basis, and they have to match the name of the EFT operators that were used
to produce the K-factors of the datasets in the previous section. 

Also, each EFT coefficient has a ``scale``. This scale is used to modify the size of the learning
rate for this coefficient within the SIMUnet framework. The size of the ``scale``
for an EFT coefficient can speed up the training and, in the case, of a big K-factor,
the convergence to the minimum of the loss function without going over it.

There are several types of initialisation.

.. _run-n3fit-fit:

2. Running the fitting code
---------------------------

After preparing a SIMUnet runcard ``runcard.yml``, we are now ready to run a fit. The pipeline
is similar to the NNPDF framework but some additional features can be included.

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
   all of the replicas will pass the checks in ``postfit``.


3. Wait until you have fit results. In the case of a simultaneous PDF-EFT fit,
   run the ``evolven3fit`` program once to evolve all replicas using DGLAP.
   The arguments are ``evolven3fit runcard_folder number_of_replicas``. Remember
   to use the total number of replicas run (130 in the
   above example), rather than the number you desire in the final fit.

    .. note::
       In the case of a fixed-PDF fit, since the PDFs have already been found,
       the user can simply run ``vp-fakeevolve`` instead
       of ``evolven3fit`` (using the same syntax). This will be much faster and will leave
       the fit directory ready for the next step. 

4. Wait until you have results, then use ``postfit number_of_replicas
   runcard_folder`` to finalize the PDF set by applying post selection criteria.
   This will produce a set of ``number_of_replicas + 1`` replicas. This time the
   number of replicas should be that which you desire in the final fit (100 in the
   above example). Note that the
   standard behaviour of ``postfit`` can be modified by using various flags.
   More information can be found at `Processing a fit <postfit>`_.

Output of the fit
-----------------
Every time a replica is finalized, the output is saved to the `runcard/nnfit/replica_$replica`_
folder, which contains a number of files:

- ``chi2exps.log``: a json log file with the χ² of the training every 100 epochs.
- ``runcard.exportgrid``: a file containing the PDF grid.
- ``runcard.json``: Includes information about the fit (metadata, parameters, times) in json format.
- ``bsm_fac.csv``: Contains the values of the EFT coefficients for this replica.

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