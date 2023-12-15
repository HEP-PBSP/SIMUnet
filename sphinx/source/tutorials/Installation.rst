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
    git clone git@github.com:LucaMantani/simunet_release.git
    git clone https://github.com/scarrazza/apfel.git

The code can then be compiled and installed with the following commands:

.. code-block:: yaml

    simunet_git$ cd simunet_release
    mkdir conda-bld
    cd conda-bld
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
    make
    make install





        