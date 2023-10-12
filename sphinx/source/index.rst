.. simunet documentation master file, created by
   sphinx-quickstart on Sun Jul  2 20:53:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



===================================
SIMUnet
===================================

The :math:`\text{SIMUnet}` methodology :cite:`Iranipour:2022iak` is an open-source methodology to perform simultaneous PDF-EFT fits and fixed-PDF EFT fits. 

It extends the :math:`\text{NNPDF}` framework :cite:`NNPDF:2021njg, NNPDF:2021uiq` to account for the EFT dependence (or, in principle, any parametric dependence) of the theory cross-sections entering the PDF determination.

This is achieved by adding an extra layer to the :math:`\text{NNPDF}` neural network to encapsulate the dependence of the theory predictions on the EFT coefficients, including the free parameters in the general optimization procedure. This results in a simultaneous fit of the PDF as well as EFT coefficients to the input data.

As in the NNPDF methodology, the error uncertainty estimation makes use of the Monte Carlo replica method, which yields an uncertainty estimate on both PDF and EFT parameters.

The SM theoretical observables are encoded using interpolation grids, known as *FK*-tables :cite:`Ball:2010de,Ball:2012cx,Bertone:2016lga`, which encode the contribution of both the DGLAP evolution and the hard-scattering matrix elements and interface it with the initial-scale PDFs in a fast and efficient way.


.. figure:: images/simunet_figure.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *example figure*
    :label: fig-simunet

The simultaneous fit is represented as a neural network using the
``Tensorflow`` :cite:`tensorflow2015:whitepaper` and
``Keras`` :cite:`chollet2015keras` libraries. The architecture is
schematically represented in Fig. :ref:`fig-simunet`.

Trainable weights are represented by solid arrows, and
non-trainable weights by dashed arrows. Through a
forward pass across the network, the inputs (:math:`x`-Bjorken and its logarithm) proceed through
hidden layers to output the eight fitted PDFs at
the initial parametrisation scale :math:`Q_0`.

For each of the experimental observables entering the fit, these
PDFs are then combined into a partonic luminosity :math:`\mathcal{L}^{(0)}` at :math:`Q_0`,
which is convolved with the precomputed **FK**-tables :math:`\Sigma` to obtain the SM
theoretical prediction :math:`\mathcal{T}^\text{SM}`.

Subsequently, the effects of the :math:`N` EFT coefficients :math:`\boldsymbol{c}=(c_1,\ldots,c_N)`,
associated with the operator basis considered,
are accounted for by means of an extra layer, resulting in the
final prediction for the observable :math:`\mathcal{T}` entering the SMEFT-PDF fit.

The :math:`\text{SIMUnet}` code allows for both linear and quadratic dependence on the EFT
coefficients. In linear EFT fits, the last layer consists of :math:`N` trainable
weights to account for each Wilson coefficient. In quadratic EFT fits, in
addition to the :math:`N` trainable weights, a set of :math:`N(N+1)/2` non-trainable
parameters, which are functions of the trainable weights, is included to account
for all diagonal and non-diagonal contributions of EFT-EFT interference to the
cross-sections. The results obtained with the quadratic functionality
of :math:`\text{SIMUnet}` are, however, not displayed in this work, for the reasons explained in
App. :ref:`app-quad`. 

The PDF parameters :math:`\boldsymbol{\theta}`
and the EFT coefficients :math:`\boldsymbol{c}` entering the evaluation
of the SMEFT observable in Fig. :ref:`fig-architecture` are then determined
simultaneously from the minimisation of the fit figure of merit (also
known as loss function).

The :math:`\text{SIMUnet}` architecture can be minimally modified
to deal with the fixed-PDF case, in which only the EFT coefficients
are treated as free parameters in the optimization process.

This can be achieved by freezing the PDF-related weights in the network
architecture to the values obtained in some previous fit, for example a SM-PDF
determination based on :math:`\text{nnpdf}`.

In this manner, :math:`\text{SIMUnet}` can also be used to carry out traditional EFT fits where the
PDF dependence of the theory predictions is neglected.

Furthermore,
for PDF-independent observables, computing an FK-table :math:`\Sigma` is not required
and the SM cross-section :math:`\mathcal{T}^\text{SM}` can be evaluated separately
and stored to be used in the fit.

As illustrated in Fig. :ref:`fig: architecture`, within
the :math:`\text{SIMUnet}` framework a single neural network
encapsulates both the PDF and the EFT dependence of physical observables,
with the corresponding parameters being simultaneously constrained from the experimental
data included in the fit.

Specifically, we denote the prediction of the neural network as:

.. math::

   \mathcal{T} = \mathcal{T}(\boldsymbol{\hat\theta})= \left( T_1(\boldsymbol{\hat\theta}),\ldots, T_n(\boldsymbol{\hat\theta}) \right) \, ,

with :math:`n=n_{\rm dat}` and 
:math:`\boldsymbol{\hat{\theta}} = (\boldsymbol{\theta}, \boldsymbol{c})`, where :math:`\boldsymbol{\theta}` and :math:`\boldsymbol{c}=(c_1, \ldots, c_N)` represent
the weights associated to the PDF nodes
of the network, and to the :math:`N` Wilson coefficients from
the operator basis, respectively.

The uncertainty estimation uses the Monte Carlo replica method, where a large number :math:`N_{\rm rep}`
of replicas :math:`D^{(k)} = \left( D_1^{(k)}, \ldots, D_n^{(k)} \right)`
of the experimental measurements :math:`D = \left( D_1, \ldots, D_n \right)` are sampled from the
distribution of experimental uncertainties with :math:`k = 1, \ldots, N_{\rm rep}`.

The optimal values for the fit parameters :math:`\boldsymbol{\hat{\theta}}^{(k)}` associated
to each replica are obtained by means of a Stochastic Gradient Descent (SGD) algorithm
that minimizes the corresponding figure of merit:

.. math::
   :label: eq:simunet_loss

   E_{\rm tot}^{(k)} \left( \boldsymbol{\hat{\theta}} \right) = \frac{1}{n_{\rm dat}}\sum_{i,j=1}^{n_{\rm dat}} \left( D_i^{(k)} - T_i(\boldsymbol{\hat\theta}) \right) \left( {\rm cov}_{t^0}^{-1} \right)_{ij}
   \left( D_j^{(k)} - T_j(\boldsymbol{\hat\theta}) \right) \, ,

where the covariance matrix in Eq. :eq:`eq:simunet_loss`
is the :math:`t_0` covariance matrix, which is constructed from all sources of statistical and
systematic uncertainties that are made available by the experiments
with correlated multiplicative uncertainties treated via the 't0' rescription :cite:`Ball:2009qv`
in the fit to avoid fitting bias associated with multiplicative uncertainties.



.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. toctree::
   :maxdepth: 4
   :caption: Tutorial
   :hidden:

   tutorial.rst

.. toctree::
   :maxdepth: 4
   :caption: Results
   :hidden:

   results/overview.rst

.. toctree::
   :maxdepth: 4
   :caption: Simunet analysis files
   :hidden:

   simunet_analysis.rst

.. toctree::
   :maxdepth: 1
   :caption: Bibliography
   :hidden:

   bibliography.rst




The PBSP team
----------------------------------
The team is currently composed by the following members: 

- Mark Costantini - University of Cambridge
- Elie Hammou - University of Cambridge
- Maeve Madigan - University of Heidelberg
- Luca Mantani - University of Cambridge
- James Moore - University of Cambridge
- Manuel Morales-Alvarado - University of Cambridge
- Maria Ubiali - University of Cambridge

Citation policy
----------------------------------
If you use the :math:`\text{SIMUnet}` code in a scientific publication, please make sure to cite: :cite:`Iranipour:2022iak` and ...



Indices and tables
=====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



