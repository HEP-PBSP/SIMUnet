.. _simu_installation:

SIMUnet installation guide
==========================

The installation process for SIMUnet is very similar to the one for NNPDF, more details can be found on their `website <https://docs.nnpdf.science/get-started/installation.html>`_. The only difference is that you need to clone the SIMUnet repository instead of the NNPDF one.

.. _linux-installation:

Linux
-------------------------

 The following instructions have been tested on a Linux system.

.. _dependencies-label-linux:

Dependencies installation
~~~~~~~~~~~~~~~~~~~~~~~~~

The dependencies need by to be installed via conda:

.. code-block:: bash

    conda create -n simunet
    conda activate simunet
    conda install --only-deps nnpdf=4.0.5
    conda install gxx_linux-64
    conda install pkg-config swig cmake
    conda install sysroot_linux-64=2.17


.. _simunet-compilation-label-linux:

Code compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

The SIMUnet code can be downloaded from GitHub:

.. code-block:: bash

    mkdir simunet_git
    cd simunet_git
    git clone https://github.com/HEP-PBSP/SIMUnet.git

The code can then be compiled and installed with the following commands:

.. code-block:: bash

    simunet_git$ cd simunet_release
    mkdir conda-bld
    cd conda-bld
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
    make
    make install

The SIMUnet code, in addition to the regular files that are needed in the NNPDF methodology to produce `theoretical predictions <https://docs.nnpdf.science/theory/index.html>`_, requires K-factors to account for the effect of SMEFT operators. These K-factors are implemented in ``simu_fac`` files, which exists for each dataset in the SIMUnet methodology. For a given dataset, the ``simu_fac`` file includes the SM theory prediction, and the SMEFT theory prediction at LO and/or NLO, if applicable. These K-factors are hosted in the NNPDF ``theory_270`` folder, which will be automatically downloaded when required by the user's runcard.

.. _macos-arm-installation:

M1/M2/M3 MacOS
-------------------

The following instructions have been tested on MacOS systems with ``arm`` processors like M1, M2, and M3.

.. _dependencies-label-macos:

Dependencies installation
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to install SIMUnet on arm64 MacOS machines using ``conda``, we must create an environment which is able to accomodate a C++ compiler, e.g. ``clangxx_osx-64`` as we will see later. We must create a ``conda`` environment setting the ``CONDA_SUBDIR=osx-64`` since we want a ``x86_64`` environment, moreover, we will use the following ``yml`` file to build the environment with the right dependencies:

.. code-block:: yaml

    name: simunet
    channels:
        - anaconda
        - https://packages.nnpdf.science/public
        - defaults
        - conda-forge
    dependencies:
        - _tflow_select=2.2.0=eigen
        - abseil-cpp=20220623.0=hddbf539_6
        - absl-py=1.4.0=py310hecd8cb5_0
        - accessible-pygments=0.0.4=pyhd8ed1ab_0
        - aiohttp=3.8.3=py310h6c40b1e_0
        - aiosignal=1.2.0=pyhd3eb1b0_0
        - alabaster=0.7.12=pyhd3eb1b0_0
        - anyio=3.5.0=py310hecd8cb5_0
        - apfel=3.0.6.3=h8d2ef1a_9
        - appdirs=1.4.4=pyhd3eb1b0_0
        - appnope=0.1.2=py310hecd8cb5_1001
        - argon2-cffi=20.1.0=py310hca72f7f_1
        - asttokens=2.0.5=pyhd3eb1b0_0
        - astunparse=1.6.3=py_0
        - async-timeout=4.0.2=py310hecd8cb5_0
        - babel=2.11.0=py310hecd8cb5_0
        - backcall=0.2.0=pyhd3eb1b0_0
        - banana-hep=0.6.8=pyhd8ed1ab_0
        - beautifulsoup4=4.12.2=py310hecd8cb5_0
        - blas=1.0=openblas
        - bleach=4.1.0=pyhd3eb1b0_0
        - blessings=1.7=py310hecd8cb5_1002
        - blinker=1.4=py310hecd8cb5_0
        - bokeh=2.4.3=py310hecd8cb5_0
        - bottleneck=1.3.5=py310h4e76f89_0
        - brotli=1.0.9=hca72f7f_7
        - brotli-bin=1.0.9=hca72f7f_7
        - brotlipy=0.7.0=py310hca72f7f_1002
        - bzip2=1.0.8=h1de35cc_0
        - c-ares=1.19.0=h6c40b1e_0
        - ca-certificates=2023.08.22=hecd8cb5_0
        - cachetools=4.2.2=pyhd3eb1b0_0
        - cctools=949.0.1=h9abeeb2_25
        - cctools_osx-64=949.0.1=hc7db93f_25
        - certifi=2023.11.17=py310hecd8cb5_0
        - cffi=1.15.1=py310h6c40b1e_3
        - chardet=4.0.0=py310hecd8cb5_1003
        - charset-normalizer=2.0.4=pyhd3eb1b0_0
        - clang=14.0.6=hecd8cb5_1
        - clang-14=14.0.6=default_hd95374b_1
        - clang_osx-64=14.0.6=hb1e4b1b_0
        - clangxx=14.0.6=default_hd95374b_1
        - clangxx_osx-64=14.0.6=hd8b9576_0
        - click=8.0.4=py310hecd8cb5_0
        - cloudpickle=2.2.1=py310hecd8cb5_0
        - cmake=3.22.1=hbfa4a85_0
        - colorama=0.4.6=py310hecd8cb5_0
        - comm=0.1.2=py310hecd8cb5_0
        - commonmark=0.9.1=pyhd3eb1b0_0
        - compiler-rt=14.0.6=hda8b6b8_0
        - compiler-rt_osx-64=14.0.6=h8d5cb93_0
        - conda=22.11.1=py310h2ec42d9_1
        - conda-build=3.27.0=py310hecd8cb5_0
        - conda-index=0.3.0=py310hecd8cb5_0
        - conda-package-handling=2.2.0=py310hecd8cb5_0
        - conda-package-streaming=0.9.0=py310hecd8cb5_0
        - contourpy=1.0.5=py310haf03e11_0
        - cryptography=39.0.1=py310hf6deb26_2
        - curio=1.4=pyhd3eb1b0_0
        - cycler=0.11.0=pyhd3eb1b0_0
        - cytoolz=0.12.0=py310hca72f7f_0
        - dask=2023.4.1=py310hecd8cb5_1
        - dask-core=2023.4.1=py310hecd8cb5_0
        - dataclasses=0.8=pyh6d0b6a4_7
        - debugpy=1.6.7=py310hcec6c5f_0
        - decorator=5.1.1=pyhd3eb1b0_0
        - defusedxml=0.7.1=pyhd3eb1b0_0
        - distributed=2023.4.1=py310hecd8cb5_1
        - docutils=0.18.1=py310hecd8cb5_3
        - eko=0.13.5=pyhd8ed1ab_0
        - exceptiongroup=1.2.0=py310hecd8cb5_0
        - executing=0.8.3=pyhd3eb1b0_0
        - expat=2.4.9=he9d5cce_0
        - fiatlux=0.1.2=py310h0eb4f65_0
        - filelock=3.9.0=py310hecd8cb5_0
        - flatbuffers=22.12.06=hf0c8a7f_2
        - fonttools=4.25.0=pyhd3eb1b0_0
        - freetype=2.12.1=hd8bbffd_0
        - frozenlist=1.3.3=py310h6c40b1e_0
        - fsspec=2023.9.2=py310hecd8cb5_0
        - future=0.18.3=py310hecd8cb5_0
        - gast=0.4.0=pyhd3eb1b0_0
        - giflib=5.2.1=h6c40b1e_3
        - google-auth=2.6.0=pyhd3eb1b0_0
        - google-auth-oauthlib=0.4.4=pyhd3eb1b0_0
        - google-pasta=0.2.0=pyhd3eb1b0_0
        - greenlet=2.0.1=py310hcec6c5f_0
        - grpc-cpp=1.51.1=h88f4db0_1
        - grpcio=1.51.1=py310hdfcfac3_1
        - gsl=2.7.1=hdbe807d_1
        - h5py=3.7.0=py310h6c517f8_0
        - hdf5=1.10.6=h10fe05b_1
        - heapdict=1.0.1=pyhd3eb1b0_0
        - hyperopt=0.2.7=pyhd8ed1ab_0
        - icu=70.1=h96cf925_0
        - idna=3.4=py310hecd8cb5_0
        - imagesize=1.4.1=py310hecd8cb5_0
        - importlib-metadata=6.0.0=py310hecd8cb5_0
        - importlib_metadata=6.0.0=hd3eb1b0_0
        - ipykernel=6.25.0=py310h20db666_0
        - ipython=8.12.0=py310hecd8cb5_0
        - ipython_genutils=0.2.0=pyhd3eb1b0_1
        - ipywidgets=8.0.4=py310hecd8cb5_0
        - jedi=0.18.1=py310hecd8cb5_1
        - jinja2=3.1.2=py310hecd8cb5_0
        - jpeg=9e=h6c40b1e_1
        - jsonschema=4.17.3=py310hecd8cb5_0
        - jupyter=1.0.0=pyhd8ed1ab_10
        - jupyter_client=8.6.0=py310hecd8cb5_0
        - jupyter_console=6.6.3=py310hecd8cb5_0
        - jupyter_core=5.5.0=py310hecd8cb5_0
        - jupyter_server=1.23.4=py310hecd8cb5_0
        - jupyterlab_pygments=0.2.2=py310hecd8cb5_0
        - jupyterlab_widgets=3.0.9=py310hecd8cb5_0
        - keras=2.11.0=py310_0
        - keras-preprocessing=1.1.2=pyhd3eb1b0_0
        - kiwisolver=1.4.4=py310hcec6c5f_0
        - krb5=1.20.1=h428f121_1
        - latexcodec=2.0.1=pyh9f0ad1d_0
        - lcms2=2.12=hf1fd2bf_0
        - ld64=530=h20443b4_25
        - ld64_osx-64=530=h70f3046_25
        - ldid=2.1.5=hc58f1be_3
        - lerc=3.0=he9d5cce_0
        - lhapdf=6.5.0=py310ha23aa8a_1
        - libabseil=20220623.0=cxx17_h844d122_6
        - libarchive=3.4.2=ha0e9c3a_2
        - libbrotlicommon=1.0.9=hca72f7f_7
        - libbrotlidec=1.0.9=hca72f7f_7
        - libbrotlienc=1.0.9=hca72f7f_7
        - libclang-cpp14=14.0.6=default_hd95374b_1
        - libcurl=8.1.1=hf20ceda_1
        - libcxx=14.0.6=h9765a3e_0
        - libdeflate=1.17=hb664fd8_0
        - libedit=3.1.20221030=h6c40b1e_0
        - libev=4.33=h9ed2024_1
        - libffi=3.4.4=hecd8cb5_0
        - libgfortran=5.0.0=11_3_0_hecd8cb5_28
        - libgfortran5=11.3.0=h9dfd629_28
        - libgrpc=1.51.1=h1ddfa78_1
        - libiconv=1.16=hca72f7f_2
        - liblief=0.12.3=hcec6c5f_0
        - libllvm14=14.0.6=h91fad77_3
        - libnghttp2=1.52.0=h9beae6a_1
        - libopenblas=0.3.21=h54e7dc3_0
        - libpng=1.6.39=h6c40b1e_0
        - libprotobuf=3.21.12=hbc0c0cd_0
        - libsodium=1.0.18=h1de35cc_0
        - libsqlite=3.42.0=h58db7d2_0
        - libssh2=1.10.0=h04015c4_2
        - libtiff=4.5.0=hcec6c5f_2
        - libuv=1.44.2=h6c40b1e_0
        - libwebp=1.2.4=hf6ce154_1
        - libwebp-base=1.2.4=h6c40b1e_1
        - libxml2=2.9.14=hea49891_4
        - libzlib=1.2.13=h8a1eda9_5
        - llvm-openmp=14.0.6=h0dcd299_0
        - llvm-tools=14.0.6=he0576d7_3
        - llvmlite=0.40.0=py310hfff2838_0
        - locket=1.0.0=py310hecd8cb5_0
        - lz4=4.3.2=py310h6c40b1e_0
        - lz4-c=1.9.4=hcec6c5f_0
        - markdown=3.4.1=py310hecd8cb5_0
        - markupsafe=2.1.1=py310hca72f7f_0
        - matplotlib=3.7.1=py310hecd8cb5_1
        - matplotlib-base=3.7.1=py310ha533b9c_1
        - matplotlib-inline=0.1.6=py310hecd8cb5_0
        - mistune=2.0.4=py310hecd8cb5_0
        - more-itertools=8.12.0=pyhd3eb1b0_0
        - msgpack-python=1.0.3=py310haf03e11_0
        - multidict=6.0.2=py310hca72f7f_0
        - munkres=1.1.4=py_0
        - mypy_extensions=1.0.0=py310hecd8cb5_0
        - nbclassic=1.0.0=py310hecd8cb5_0
        - nbclient=0.8.0=py310hecd8cb5_0
        - nbconvert=7.10.0=py310hecd8cb5_0
        - nbformat=5.9.2=py310hecd8cb5_0
        - ncurses=6.4=hcec6c5f_0
        - nest-asyncio=1.5.6=py310hecd8cb5_0
        - networkx=2.8.4=py310hecd8cb5_1
        - notebook=6.5.4=py310hecd8cb5_0
        - notebook-shim=0.2.3=py310hecd8cb5_0
        - numba=0.57.0=py310h3ea8b11_0
        - numexpr=2.8.4=py310he50c29a_1
        - numpy=1.24.3=py310he50c29a_0
        - numpy-base=1.24.3=py310h992e150_0
        - oauthlib=3.2.2=py310hecd8cb5_0
        - openssl=3.0.12=hca72f7f_0
        - opt_einsum=3.3.0=pyhd3eb1b0_1
        - packaging=23.0=py310hecd8cb5_0
        - pandas=1.5.3=py310h3ea8b11_0
        - pandoc=2.12=hecd8cb5_3
        - pandocfilters=1.5.0=pyhd3eb1b0_0
        - parso=0.8.3=pyhd3eb1b0_0
        - partd=1.4.1=py310hecd8cb5_0
        - patch=2.7.6=h1de35cc_1001
        - pcre=8.45=h23ab428_0
        - pendulum=2.1.2=pyhd3eb1b0_1
        - pexpect=4.8.0=pyhd3eb1b0_3
        - pickleshare=0.7.5=pyhd3eb1b0_1003
        - pillow=9.4.0=py310hcec6c5f_0
        - pineappl=0.6.0=py310h3461e44_0
        - pip=23.1.2=py310hecd8cb5_0
        - pkg-config=0.29.2=h3efe00b_8
        - pkginfo=1.9.6=py310hecd8cb5_0
        - platformdirs=3.10.0=py310hecd8cb5_0
        - pluggy=1.0.0=py310hecd8cb5_1
        - pooch=1.4.0=pyhd3eb1b0_0
        - prometheus_client=0.14.1=py310hecd8cb5_0
        - prompt-toolkit=3.0.36=py310hecd8cb5_0
        - prompt_toolkit=3.0.36=hd3eb1b0_0
        - protobuf=4.21.12=py310h7a76584_0
        - psutil=5.9.0=py310hca72f7f_0
        - ptyprocess=0.7.0=pyhd3eb1b0_2
        - pure_eval=0.2.2=pyhd3eb1b0_0
        - py-lief=0.12.3=py310hcec6c5f_0
        - py4j=0.10.9.3=py310hecd8cb5_0
        - pyasn1=0.4.8=pyhd3eb1b0_0
        - pyasn1-modules=0.2.8=py_0
        - pybtex=0.24.0=pyhd8ed1ab_2
        - pybtex-docutils=1.0.2=py310h2ec42d9_2
        - pycosat=0.6.6=py310h6c40b1e_0
        - pycparser=2.21=pyhd3eb1b0_0
        - pydata-sphinx-theme=0.13.3=pyhd8ed1ab_0
        - pygments=2.15.1=py310hecd8cb5_1
        - pyjwt=2.4.0=py310hecd8cb5_0
        - pyopenssl=23.0.0=py310hecd8cb5_0
        - pyparsing=3.0.9=py310hecd8cb5_0
        - pyrsistent=0.18.0=py310hca72f7f_0
        - pysocks=1.7.1=py310hecd8cb5_0
        - python=3.10.9=he7542f4_0_cpython
        - python-dateutil=2.8.2=pyhd3eb1b0_0
        - python-fastjsonschema=2.16.2=py310hecd8cb5_0
        - python-flatbuffers=2.0=pyhd3eb1b0_0
        - python-libarchive-c=2.9=pyhd3eb1b0_1
        - python-lmdb=1.4.1=py310hcec6c5f_0
        - python_abi=3.10=2_cp310
        - pytz=2022.7=py310hecd8cb5_0
        - pytzdata=2020.1=pyhd3eb1b0_0
        - pyyaml=6.0=py310h6c40b1e_1
        - pyzmq=25.1.0=py310hcec6c5f_0
        - qtconsole-base=5.5.1=pyha770c72_0
        - qtpy=2.4.1=py310hecd8cb5_0
        - re2=2023.02.01=hf0c8a7f_0
        - readline=8.2=hca72f7f_0
        - recommonmark=0.6.0=pyhd3eb1b0_0
        - reportengine=0.30.28=py_0
        - requests=2.29.0=py310hecd8cb5_0
        - requests-oauthlib=1.3.0=py_0
        - rhash=1.4.3=h04015c4_0
        - rich=12.5.1=py310hecd8cb5_0
        - rsa=4.7.2=pyhd3eb1b0_1
        - ruamel.yaml=0.17.21=py310hca72f7f_0
        - ruamel.yaml.clib=0.2.7=py310h6729b98_2
        - ruamel_yaml=0.15.100=py310hca72f7f_0
        - scipy=1.10.1=py310ha516a68_1
        - seaborn=0.12.2=py310hecd8cb5_0
        - send2trash=1.8.2=py310hecd8cb5_0
        - setuptools=67.8.0=py310hecd8cb5_0
        - six=1.16.0=pyhd3eb1b0_1
        - snappy=1.1.9=he9d5cce_0
        - sniffio=1.2.0=py310hecd8cb5_1
        - snowballstemmer=2.2.0=pyhd3eb1b0_0
        - sortedcontainers=2.4.0=pyhd3eb1b0_0
        - soupsieve=2.4=py310hecd8cb5_0
        - sphinx=5.0.2=py310hecd8cb5_0
        - sphinx-book-theme=1.0.1=pyhd8ed1ab_0
        - sphinx_rtd_theme=1.1.1=py310hecd8cb5_0
        - sphinxcontrib-applehelp=1.0.2=pyhd3eb1b0_0
        - sphinxcontrib-bibtex=2.5.0=pyhd8ed1ab_0
        - sphinxcontrib-devhelp=1.0.2=pyhd3eb1b0_0
        - sphinxcontrib-htmlhelp=2.0.0=pyhd3eb1b0_0
        - sphinxcontrib-jsmath=1.0.1=pyhd3eb1b0_0
        - sphinxcontrib-qthelp=1.0.3=pyhd3eb1b0_0
        - sphinxcontrib-serializinghtml=1.1.5=pyhd3eb1b0_0
        - sqlalchemy=1.4.39=py310hca72f7f_0
        - sqlite=3.41.2=h6c40b1e_0
        - stack_data=0.2.0=pyhd3eb1b0_0
        - swig=4.0.2=he9d5cce_4
        - tapi=1000.10.8=ha1b3eb9_0
        - tbb=2021.8.0=ha357a0b_0
        - tblib=1.7.0=pyhd3eb1b0_0
        - tensorboard=2.11.0=py310_0
        - tensorboard-data-server=0.6.1=py310h7242b5c_0
        - tensorboard-plugin-wit=1.6.0=py_0
        - tensorflow=2.11.0=cpu_py310h22f808f_0
        - tensorflow-base=2.11.0=cpu_py310h760b059_0
        - tensorflow-estimator=2.11.0=cpu_py310h5e669bb_0
        - termcolor=2.1.0=py310hecd8cb5_0
        - terminado=0.17.1=py310hecd8cb5_0
        - tinycss2=1.2.1=py310hecd8cb5_0
        - tk=8.6.12=h5d9f67b_0
        - tomli=2.0.1=py310hecd8cb5_0
        - toolz=0.12.0=py310hecd8cb5_0
        - tornado=6.2=py310hca72f7f_0
        - tqdm=4.65.0=py310h20db666_0
        - traitlets=5.7.1=py310hecd8cb5_0
        - typing_extensions=4.6.3=py310hecd8cb5_0
        - tzdata=2023c=h04d1e81_0
        - urllib3=1.26.16=py310hecd8cb5_0
        - validobj=1.0=pyhd8ed1ab_0
        - wcwidth=0.2.5=pyhd3eb1b0_0
        - webencodings=0.5.1=py310hecd8cb5_1
        - websocket-client=0.58.0=py310hecd8cb5_4
        - werkzeug=2.2.3=py310hecd8cb5_0
        - wheel=0.35.1=pyhd3eb1b0_0
        - widgetsnbextension=4.0.5=py310hecd8cb5_0
        - wrapt=1.14.1=py310hca72f7f_0
        - xz=5.2.10=h6c40b1e_1
        - yaml=0.2.5=haf1e3a3_0
        - yaml-cpp=0.7.0=he9d5cce_1
        - yarl=1.8.1=py310hca72f7f_0
        - zeromq=4.3.4=h23ab428_0
        - zict=3.0.0=py310hecd8cb5_0
        - zipp=3.11.0=py310hecd8cb5_0
        - zlib=1.2.13=h8a1eda9_5
        - zstandard=0.19.0=py310h6c40b1e_0
        - zstd=1.5.5=hc035e20_0
        - pip:
            - attrs==23.2.0
            - black==24.4.0
            - hypothesis==6.100.1
            - pathspec==0.12.1
    prefix: <path_to_root_conda_directory>/envs/simunet

where ``<path_to_root_conda_directory>`` can be obtained using the following command line when the ``(base)`` environment is activated, in particular, we will obtain the absolute path to the ``(base)`` environment, `e.g.` the ``miniconda3`` folder if we use it.

.. code-block:: bash

    echo $CONDA_PREFIX

Then, as we anticipated, we must set the ``CONDA_SUBDIR=osx-64``, create the environment from the ``yml`` file, and activate it

.. code-block:: bash

    CONDA_SUBDIR=osx-64 conda env create -f simunet.yml
    conda activate simunet

Another important step is installing a previous version of the MacOSX Software Development Kit (SDK), we can download the chosen version from `this link <https://github.com/phracker/MacOSX-SDKs/releases/>`_, we suggest ``MacOSX10.9.sdk.tar.xz``, which has been tested.

.. code-block:: bash

    curl -L -O https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX10.9.sdk.tar.xz

once we have downloaded the ``tar.xz`` file we must untar it into the ``(base)`` environment folder

.. code-block:: bash

    tar xfz MacOSX10.9.sdk.tar.xz -C <path_to_root_conda_directory>

where ``<path_to_root_conda_directory>`` is the same we obtained before. Then we must run the following command line to set the right path for the SDK that will be used during the installation of the SIMUnet C++ code

.. code-block:: bash

    export CONDA_BUILD_SYSROOT=<path_to_root_conda_directory>/MacOSX10.9.sdk

.. _simunet-compilation-label-macos:

Code compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

The SIMUnet code can be downloaded from GitHub:

.. code-block:: bash

    mkdir simunet_git
    cd simunet_git
    git clone https://github.com/HEP-PBSP/SIMUnet.git

The code can then be compiled and installed with the following commands, first we have to move into the folder downloaded using ``git``, then we must create and place ourselves in the build folder:

.. code-block:: bash

    cd SIMUnet
    mkdir conda-bld
    cd conda-bld

Finally, we can complete the installation with the following three steps:

.. code-block:: bash

    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
    make
    make install

Note that if your CPU has more than one core, which is the case for M1/M2/M3 Macs, the option ``-j4`` can be used to speed up the ``make`` command.

Moreover, the ``make install`` command will raise some non-stopping errors, which do invalidated the installation.

The SIMUnet code, in addition to the regular files that are needed in the NNPDF methodology to produce `theoretical predictions <https://docs.nnpdf.science/theory/index.html>`_, requires K-factors to account for the effect of SMEFT operators. These K-factors are implemented in ``simu_fac`` files, which exists for each dataset in the SIMUnet methodology. For a given dataset, the ``simu_fac`` file includes the SM theory prediction, and the SMEFT theory prediction at LO and/or NLO, if applicable. These K-factors are hosted in the NNPDF ``theory_270`` folder, which will be automatically downloaded when required by the user's runcard.