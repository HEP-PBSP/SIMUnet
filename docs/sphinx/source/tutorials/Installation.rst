.. _simu_installation:

Installing SIMUnet
==================

The installation process for SIMUnet is very similar to the one for NNPDF, more details can be found on their `website <https://docs.nnpdf.science/get-started/installation.html>`_. The only difference is that you need to clone the SIMUnet repository instead of the NNPDF one. The following instructions have been tested on a Linux system.

.. _dependencies-label:

Dependencies installation
-------------------------

The dependencies need by to be installed via conda:

.. code-block:: yaml

    conda create -n simunet
    conda activate simunet
    conda install --only-deps nnpdf=4.0.5
    conda install gxx_linux-64
    conda install pkg-config swig cmake
    conda install sysroot_linux-64=2.17


.. _simunet-compilation-label:

Code compilation
----------------

The SIMUnet code can be downloaded from github:

.. code-block:: yaml

    mkdir simunet_git
    cd simunet_git
    git clone https://github.com/HEP-PBSP/SIMUnet.git
    git clone https://github.com/scarrazza/apfel.git

The code can then be compiled and installed with the following commands:

.. code-block:: yaml

    simunet_git$ cd simunet_release
    mkdir conda-bld
    cd conda-bld
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
    make
    make install

The SIMUnet code, in addition to the regular files that are needed in the NNPDF methodology to produce `theoretical predictions <https://docs.nnpdf.science/theory/index.html>`_,
requires K-factors to account for the effect of SMEFT operators. These K-factors are implemented in ``simu_fac`` files, which
exists for each dataset in the SIMUnet methodology. For a given dataset, the ``simu_fac`` file includes the SM theory prediction, and the SMEFT
theory prediction at LO and/or NLO, if applicable. These K-factors are hosted in the NNPDF ``theory_270`` folder, which will be automatically
downloaded when required by the user's runcard.
