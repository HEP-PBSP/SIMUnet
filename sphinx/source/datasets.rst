Dataset selection
==================

================================
Top sector
================================

In this section we describe the top quark datasets of SIMUnet.

- :ref:`dataset_sel-label`

.. _dataset_sel-label:

This is another header
----------------------
Some text

.. list-table:: The inclusive cross-sections and differential distributions for top quark pair production from ATLAS and CMS that we consider in this analysis. For each dataset, we indicate the experiment, the centre of mass energy :math:`\sqrt{s}`, the final-state channel, the observable(s) used in the fit, the integrated luminosity :math:`\mathcal{L}` in inverse femtobarns, and the number of data points :math:`n_{\rm dat}`, together with the corresponding publication reference. In the last two columns, we indicate with a :math:`\checkmark` the datasets that are included for the first time here in a global PDF fit and in a SMEFT interpretation, respectively. The sets marked with brackets have already been included in previous studies but here we account for their constraints in different manner (e.g. by changing spectra or normalisation), as indicated in the table and in the text description.
   :widths: 5 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Channel**
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **LEP**
     - 0.250
     - 
     - Z observables
     - 
     - 19
     - LEP_ZDATA
     - `Reference <https://arxiv.org/abs/hep-ex/0509008>`_ (Tables 2.13, 3.6, 4.3, 5.8, and 5.10)
   * - 
     - 0.196
     - 
     - :math:`\mathcal{B}(W \rightarrow e^{-} \bar{v}_e),` :math:`\mathcal{B}(W \rightarrow \mu^{-} \bar{v}_{\mu}),` :math:`\mathcal{B}(W \rightarrow \tau^{-} \bar{v}_{\tau})`
     - 
     - 3
     - LEP_BRW
     - `Reference <https://arxiv.org/abs/1302.3415>`_ (Table E.6)
   * -
     - 0.189
     - 
     - :math:`\sigma(e^+ e^- \rightarrow e^+ e^-)`
     - 
     - 21
     - LEP_BHABHA
     - `Reference <https://arxiv.org/abs/1302.3415>`_ (Table 3.11 and 3.12)
   * -
     - 0.209
     - 
     - :math:`\hat{\alpha}^{(5)}_{\rm}(M_Z)`
     - 
     - 1
     - LEP_ALPHAEW
     - `Reference <https://pdg.lbl.gov/2023/web/viewer.html?file=../reviews/rpp2022-rev-standard-model.pdf>`_ (Equation 10.11)
   * -
     - 0.182
     - 
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.164
     - 10
     - LEP_EEWW_182GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`_ (Table 5.6, line 1)
   * -
     - 0.189
     - 
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.588
     - 10
     - LEP_EEWW_189GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`_ (Table 5.6, line 2)
   * -
     - 0.198
     - 
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.605
     - 10
     - LEP_EEWW_198GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`_ (Table 5.6, line 3)
   * -
     - 0.206
     - 
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.631
     - 10
     - LEP_EEWW_206GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`_ (Table 5.6, line 4)
   * - **ATLAS**
     - 7
     - dilepton
     - :math:`\sigma(t\bar{t})`
     - 4.6
     - 1
     -
     - [Ref](#ATLAS:2014nxi)
   * - 
     - 8
     - dilepton
     - :math:`\sigma(t\bar{t})`
     - 20.3
     - 1
     -
     - [Ref](#ATLAS:2014nxi)
   * - 
     - 
     - 
     - :math:`1/\sigma d\sigma/dm_{t\bar{t}}`
     - 20.2
     - 5
     -
     - [Ref](#Aaboud:2016iot)
   * - 
     - 
     - :math:`\ell+j`
     - :math:`\sigma(t\bar{t})`
     - 20.2
     - 1
     -
     - [Ref](#ATLAS:2017wvi)
   * - 
     - 13
     - Diboson
     - :math:`d \sigma _{W^+W^-}/d m_{e \mu}`
     - 36.1
     - 13
     - ATLAS_WW_13TeV_2016_MEMU
     - `Reference <https://arxiv.org/abs/1905.04242>`_ (Figure 7.b), `HEPdata <https://www.hepdata.net/record/ins1734263>`_ (Table 42)
   * - 
     - 13
     - Diboson
     - :math:`d \sigma _{WZ} / d m_{T}`
     - 36.1
     - 6
     - ATLAS_WZ_13TeV_2016_MTWZ
     - `Reference <https://arxiv.org/abs/1902.05759>`_ (Figure 4), `HEPdata <https://www.hepdata.net/record/ins1720438>`_ (Table 12)
   * - 
     - 13
     - Z production + dijet
     - :math:`d \sigma(Zjj)/d \Delta \phi_{jj}`
     - 139
     - 12
     - ATLAS_Zjj_13TeV_2016
     - `Reference <https://arxiv.org/abs/2006.15458>`_ (Table 8)
   * - **ATLAS and CMS**
     - 7 and 8
     - 
     - Higgs decay
     - 5 and 20
     - 22
     - ATLAS_CMS_SSinc_RunI
     - `Reference <https://arxiv.org/abs/1606.02266>`_ (Table 13)
   * - **CMS**
     - 5
     - combination
     - :math:`\sigma(t\bar{t})`
     - 0.027
     - 1
     -
     - [Ref](#CMS:2017zpm)
   * - 
     - 7
     - combination
     - :math:`\sigma(t\bar{t})`
     - 5.0
     - 1
     -
     - [Ref](#Spannagel:2016cqt)
   * - 
     - 13
     - Diboson
     - :math:`d \sigma _{WZ} / d p_{T}`
     - 35.9
     - 11
     - CMS_WZ_13TeV_2016_PTZ
     - `Reference <https://arxiv.org/abs/1901.03428>`_

... (And so on for the rest of the table rows.)
