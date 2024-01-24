.. simunet documentation master file, created by
   sphinx-quickstart on Sun Jul  2 20:53:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
SIMUnet
===================================

:math:`\text{SIMUnet}` is a machine-learning based framework to explore 
the interplay between the Parton Distribution Functions (PDFs) of the proton and possible signs of 
New Physics that might appear in the experimental data at hadron colliders. 
:math:`\text{SIMUnet}` builds upon the first tagged version of the `NNPDF code <https://docs.nnpdf.science/>`_ by adding 
the fitting and analysis features that allow users to simultaneously determine the PDFs and the Wilson Coefficients of an EFT 
expansion and to perform Closure Tests with New Physics signals injected into the data, to check whether PDFs can absorb the signs 
of New Physics associated to any given New Physics model.

The code
--------
:math:`\text{SIMUnet}` is released as an open-source, flexible code that allows users to explore the interplay between PDFs and 
New Physics. Users can use SIMUnet to perform simultaneous PDF and SMEFT fits and fixed-PDF SMEFT fits using a global set of data 
from hadron and lepton colliders.
It also provides analysis tools to evaluate different metrics of these fits, including quality metrics, 
uncertainties, PDF and SMEFT correlations, etc.

Citation policy
---------------
We encourage all who use the :math:`\text{SIMUnet}` code for a scientific publication to cite the following relevant publications: 

- :cite:`current`: Manual and release paper

- :cite:`NNPDF:2021uiq`: NNPDF open-source code release paper

- :cite:`Iranipour:2022iak`: Seminal paper presenting the methodology and applying it to high invariant mass Drell-Yan distributions

- :cite:`Kassabov:2023hbm`: Application of the :math:`\text{SIMUnet}` methodology to a global fit of Run-II top quark data

- :cite:`Hammou:2023heg`: Closure test with New Physics injection and methodology to assess whether New Physics signals can be fitted away by the PDFs.

===================================
PBSP
===================================

SIMUnet is the first public tool provided by the `PBSP <https://www.pbsp.org.uk/>`_ (Physics Beyond the Standard Proton) collaboration.
PBSP is an ERC-funded project based at the Department of Applied Mathematics 
and Theoretical Physics at the University of Cambridge. 
The projects focuses on the global interpretation of the LHC data in terms of indirect searches for New Physics, 
by providing a robust framework to globally interpret all subtle deviations from the SM predictions that might arise at 
colliders and characterising their interplay with the Parton Distribution Fuctions (PDFs) of the proton. 

The PBSP team
----------------------------------
The team is led by P.I. Maria Ubiali and is currently composed by the following members: 

- Mark Costantini - University of Cambridge

- Elie Hammou - University of Cambridge

- Maeve Madigan - University of Heidelberg

- Luca Mantani - University of Cambridge

- James Moore - University of Cambridge

- Manuel Morales-Alvarado - University of Cambridge

- Maria Ubiali - University of Cambridge

We thank the former members: Shayan Iranipour, Zahari Kassabov and Cameron Voisey. Without their work and dedication the 
outcome that we present here would have not been achieved.

Contents
========
.. toctree::
   :maxdepth: 2

   methodology.rst
   features.rst
   tutorials/tutorials_overview.rst
   datasets.rst
   results/overview.rst
   simunet_analysis.rst
   bibliography.rst



Indices and tables
=====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



