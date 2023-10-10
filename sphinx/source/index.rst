.. simunet documentation master file, created by
   sphinx-quickstart on Sun Jul  2 20:53:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



===================================
SIMUnet
===================================

simunet is .....

hello i am testing this. This is math notation: :math:`t \overline{t}` production.

The :math:`SIMUnet` methodology :cite:`Iranipour:2022iak` extends the :math:`\text{NNPDF}` framework :cite:`NNPDF:2021njg, NNPDF:2021uiq` to account for the EFT dependence (or, in principle, any parametric dependence) of the theory cross-sections entering the PDF determination.

This is achieved by adding an extra layer to the :math:`\text{NNPDF}` neural network to encapsulate the dependence of the theory predictions on the EFT coefficients, including the free parameters in the general optimization procedure. This results in a simultaneous fit of the PDF as well as EFT coefficients to the input data.

As in the NNPDF methodology, the error uncertainty estimation makes use of the Monte Carlo replica method, which yields an uncertainty estimate on both PDF and EFT parameters.

The SM theoretical observables are encoded using interpolation grids, known as *FK*-tables :cite:`Ball:2010de,Ball:2012cx,Bertone:2016lga`, which encode the contribution of both the DGLAP evolution and the hard-scattering matrix elements and interface it with the initial-scale PDFs in a fast and efficient way.


.. figure:: images/simunet_figure.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *example figure*




.. toctree::
   :maxdepth: 2
   :caption: Contents:


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
If you use the SIMUnet code in a scientific publication, please make sure to cite: :cite:`Iranipour:2022iak` and ...



Indices and tables
=====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



