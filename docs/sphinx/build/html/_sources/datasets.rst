.. _dataset:

**Dataset selection**
=====================

Here is the list of the datasets we implemented in SIMUnet to fit SMEFT coefficients. For each dataset, we indicate the experiment, the centre of mass energy :math:`\sqrt{s}`, the final-state channel, the observable(s) used in the fit, the integrated luminosity :math:`\mathcal{L}` in inverse femtobarns, and the number of data points :math:`n_{\rm dat}`, together with the corresponding publication reference. The datasets are organised by physical sectors, namely top, Drell-Yan, W helicity, EW precision observables, Higgs and Diboson.

The user can easily include any of the following datasets to perform a fit with :math:`\text{SIMUnet}`. This 
is done by adding the corresponding ``Dataset name`` with its corresponding configuration keys to the ``dataset_input`` in the runcard,
as shown in the example below:

.. code-block:: yaml

    dataset_inputs:
    - {dataset: NMC, frac: 0.75}
    - {dataset: ATLASTTBARTOT7TEV, cfac: [QCD], simu_fac: "EFT_NLO"}
    - {dataset: CMS_SINGLETOPW_8TEV_TOTAL, simu_fac: "EFT_NLO", use_fixed_predictions: True}

The dataset ``NMC`` is a NNPDF dataset, and it is included as it would be on an NNPDF4.0 runcard.
:math:`\text{SIMUnet}` recognises it as a standard dataset with no need for SMEFT modifications.

In contrast, the dataset ``ATLASTTBARTOT7TEV`` includes the additional key ``simu_fac``,
which instructs :math:`\text{SIMUnet}` to calculate theory predictions for this dataset using SMEFT NLO K-factor, as it is clear from its ``EFT_NLO`` value.

The third dataset, ``CMS_SINGLETOPW_8TEV_TOTAL``, includes two keys: ``simu_fac``, which has already been discussed, and ``use_fixed_predictions``. The ``use_fixed_predictions`` key, when set to True,
effectively removes the PDF dependence of the dataset. This approach is appropriate for observables that are either independent of the PDFs (like EWPO or W-helicities in top decays, for example) or are only weakly influenced by them,
and therefore a full computation of FK-tables would be too computationally taxing.

================================
Top sector
================================

TTBAR
----------------------


.. list-table:: 
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
   * - **ATLAS**
     - 7
     - dilepton
     - :math:`\sigma(t\bar{t})`
     - 4.6
     - 1
     - ATLASTTBARTOT7TEV
     - `Reference <https://arxiv.org/abs/1406.5375>`__ 
   * - **ATLAS**
     - 8
     - dilepton
     - :math:`\sigma(t\bar{t})`
     - 20.3
     - 1
     - ATLASTTBARTOT8TEV
     - `Reference <https://arxiv.org/abs/1406.5375>`__ 
   * - **ATLAS**
     - 8
     - dilepton
     - :math:`1/\sigma d\sigma/dm_{t\bar{t}}`
     - 20.2
     - 5
     - ATLAS_TOPDIFF_DILEPT_8TEV_TTMNORM
     - `Reference <https://arxiv.org/abs/1607.07281>`__ 
   * - **ATLAS**
     - 8
     - :math:`\ell \mathrm{+jets}`
     - :math:`\sigma(t\bar{t})`
     - 20.2
     - 1
     - ATLAS_TTBAR_8TEV_LJETS_TOTAL
     - `Reference <https://arxiv.org/abs/1712.06857>`__ 
   * - **ATLAS**
     - 8
     - :math:`\ell \mathrm{+jets}`
     - :math:`1/\sigma d\sigma/d|y_{t}|`
     - 20.3
     - 4
     - ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM
     - `Reference <https://arxiv.org/abs/1511.04716>`__ 
   * - **ATLAS**
     - 8
     - :math:`\ell \mathrm{+jets}`
     - :math:`1/\sigma d\sigma/d|y_{t\bar{t}}|`
     - 20.3
     - 4
     - ATLAS_TTB_DIFF_8TEV_LJ_TTRAPNORM
     - `Reference <https://arxiv.org/abs/1511.04716>`__ 
   * - **ATLAS**
     - 13
     - dilepton
     - :math:`\sigma(t\bar{t})`
     - 36.1
     - 1
     - ATLAS_TTBAR_13TEV_DILEPTON_TOTAL
     - `Reference <https://arxiv.org/abs/1910.08819>`__ 
   * - **ATLAS**
     - 13
     - hadronic
     - :math:`\sigma(t\bar{t})`
     - 36.1
     - 1
     - ATLAS_TTBAR_13TEV_HADRONIC_TOTAL
     - `Reference <https://arxiv.org/abs/2006.09274>`__ 
   * - **ATLAS**
     - 13
     - hadronic
     - :math:`1/\sigma d^2\sigma/d|y_{t\bar{t}}|dm_{t\bar{t}}`
     - 36.1
     - 10
     - ATLAS_TTBAR_13TEV_HADRONIC_2D_TTM_ABSYTTNORM
     - `Reference <https://arxiv.org/abs/2006.09274>`__ 
   * - **ATLAS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`\sigma(t\bar{t})`
     - 139
     - 1
     - ATLAS_TTBAR_13TEV_LJETS_TOTAL
     - `Reference <https://arxiv.org/abs/2006.13076>`__ 
   * - **ATLAS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`1/\sigma d\sigma/dm_{t\bar{t}}`
     - 36
     - 8
     - ATLAS_TTBAR_13TEV_TTMNORM
     - `Reference <https://arxiv.org/abs/1908.07305>`__ 
   * - **CMS**
     - 5
     - combination
     - :math:`\sigma(t\bar{t})`
     - 0.027
     - 1
     - CMSTTBARTOT5TEV
     - `Reference <https://arxiv.org/abs/1711.03143>`__ 
   * - **CMS**
     - 7
     - combination
     - :math:`\sigma(t\bar{t})`
     - 5.0
     - 1
     - CMSTTBARTOT7TEV
     - `Reference <https://arxiv.org/abs/1607.04972>`__ 
   * - **CMS**
     - 8
     - combination
     - :math:`\sigma(t\bar{t})`
     - 19.7
     - 1
     - CMSTTBARTOT8TEV
     - `Reference <https://arxiv.org/abs/1607.04972>`__ 
   * - **CMS**
     - 8
     - dilepton
     - :math:`1/\sigma d^2\sigma/dy_{t\bar{t}}dm_{t\bar{t}}`
     - 19.7
     - 16
     - CMS_TTBAR_2D_DIFF_MTT_TTRAP_NORM
     - `Reference <https://arxiv.org/abs/1703.01630>`__ 
   * - **CMS**
     - 8
     - :math:`\ell \mathrm{+jets}`
     - :math:`1/\sigma d\sigma/dy_{t\bar{t}}`
     - 19.7
     - 9
     - CMSTOPDIFF8TEVTTRAPNORM
     - `Reference <https://arxiv.org/abs/1703.01630>`__ 
   * - **CMS**
     - 13
     - dilepton
     - :math:`\sigma(t\bar{t})`
     - 43
     - 1
     - CMSTTBARTOT13TEV
     - `Reference <https://arxiv.org/abs/1510.05302>`__ 
   * - **CMS**
     - 13
     - dilepton
     - :math:`1/\sigma d\sigma/dy_{t\bar{t}}`
     - 35.9
     - 5
     - CMS_TTB_DIFF_13TEV_2016_2L_TTMNORM
     - `Reference <https://arxiv.org/abs/1811.06625>`__ 
   * - **CMS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`\sigma(t\bar{t})`
     - 137
     - 1
     - CMS_TTBAR_13TEV_LJETS_TOTAL
     - `Reference <https://arxiv.org/abs/2108.02803>`__ 
   * - **CMS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`1/\sigma  d\sigma/dm_{t\bar{t}}`
     - 137
     - 14
     - CMS_TTBAR_13TEV_TTMNORM
     - `Reference <https://arxiv.org/abs/2108.02803>`__ 

TTBAR Asymmetry
----------------------

.. list-table:: 
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
   * - **ATLAS**
     - 8
     - dilepton
     - :math:`A_C`
     - 20.3
     - 1
     - ATLAS_TTBAR_8TEV_ASY
     - `Reference <https://arxiv.org/abs/1604.05538>`__ 
   * - **ATLAS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`A_C`
     - 139
     - 5
     - ATLAS_TTBAR_13TEV_ASY_2022
     - `Reference <https://arxiv.org/abs/2208.12095>`__ 
   * - **CMS**
     - 8
     - dilepton
     - :math:`A_C`
     - 19.5
     - 3
     - CMS_TTBAR_8TEV_ASY
     - `Reference <https://arxiv.org/abs/1603.06221>`__ 
   * - **CMS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`A_C`
     - 138
     - 3
     - CMS_TTBAR_13TEV_ASY
     - `Reference <https://cds.cern.ch/record/2809614>`__ 
   * - **ATLAS and CMS**
     - 8
     - :math:`\ell \mathrm{+jets}`
     - :math:`A_C`
     - 20
     - 6
     - ATLAS_CMS_TTBAR_8TEV_ASY
     - `Reference <https://arxiv.org/abs/1709.05327>`__ 

TTZ
----------------------

.. list-table:: 
   :widths: 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **ATLAS**
     - 8
     - :math:`\sigma(t\bar{t}Z)`
     - 20.3
     - 1
     - ATLAS_TTBARZ_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1509.05276>`__ 
   * - **ATLAS**
     - 13
     - :math:`\sigma(t\bar{t}Z)`
     - 36.1
     - 1
     - ATLAS_TTBARZ_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1901.03584>`__ 
   * - **ATLAS**
     - 13
     - :math:`1/\sigma d\sigma(t\bar{t}Z)/dp_T^Z`
     - 139
     - 6
     - ATLAS_TTBARZ_13TEV_PTZNORM
     - `Reference <https://arxiv.org/abs/2103.12603>`__ 
   * - **CMS**
     - 8
     - :math:`\sigma(t\bar{t}Z)`
     - 19.5
     - 1
     - CMS_TTBARZ_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1510.01131>`__ 
   * - **CMS**
     - 13
     - :math:`\sigma(t\bar{t}Z)`
     - 35.9
     - 1
     - CMS_TTBARZ_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1711.02547>`__ 
   * - **CMS**
     - 13
     - :math:`1/\sigma d\sigma(t\bar{t}Z)/dp_T^Z`
     - 77.5
     - 3
     - CMS_TTBARZ_13TEV_PTZNORM
     - `Reference <https://arxiv.org/abs/1907.11270>`__ 

TTW
----------------------

.. list-table:: 
   :widths: 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **ATLAS**
     - 8
     - :math:`\sigma(t\bar{t}W)`
     - 20.3
     - 1
     - ATLAS_TTBARW_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1509.05276>`__ 
   * - **ATLAS**
     - 13
     - :math:`\sigma(t\bar{t}W)`
     - 36.1
     - 1
     - ATLAS_TTBARW_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1901.03584>`__ 
   * - **CMS**
     - 8
     - :math:`\sigma(t\bar{t}W)`
     - 19.5
     - 1
     - CMS_TTBARW_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1510.01131>`__ 
   * - **CMS**
     - 13
     - :math:`\sigma(t\bar{t}W)`
     - 35.9
     - 1
     - CMS_TTBARW_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1711.02547>`__ 

TTGamma
----------------------

.. list-table:: 
   :widths: 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **ATLAS**
     - 8
     - :math:`\sigma(t\bar{t}\gamma)`
     - 20.2
     - 1
     - ATLAS_TTBARGAMMA_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1706.03046>`__ 
   * - **CMS**
     - 8
     - :math:`\sigma(t\bar{t}\gamma)`
     - 19.7
     - 1
     - CMS_TTBARGAMMA_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1706.08128>`__ 

4 Heavy quarks
----------------------

.. list-table:: 
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
   * - **ATLAS**
     - 13
     - multi-lepton
     - :math:`\sigma_{\text{tot}}(t\bar{t}t\bar{t})`
     - 139
     - 1
     - ATLAS_4TOP_13TEV_MULTILEP_TOTAL
     - `Reference <https://arxiv.org/abs/2007.14858>`__ 
   * - **ATLAS**
     - 13
     - single-lepton
     - :math:`\sigma_{\text{tot}}(t\bar{t}t\bar{t})`
     - 139
     - 1
     - ATLAS_4TOP_13TEV_SLEP_TOTAL
     - `Reference <https://arxiv.org/abs/2106.11683>`__ 
   * - **ATLAS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`\sigma_{\text{tot}}(t\bar{t}b\bar{b})`
     - 36.1
     - 1
     - ATLAS_TTBB_13TEV_LJETS_TOTAL
     - `Reference <https://arxiv.org/abs/1811.12113>`__ 
   * - **CMS**
     - 13
     - multi-lepton
     - :math:`\sigma_{\text{tot}}(t\bar{t}t\bar{t})`
     - 137
     - 1
     - CMS_4TOP_13TEV_MULTILEP_TOTAL
     - `Reference <https://arxiv.org/abs/1908.06463>`__ 
   * - **CMS**
     - 13
     - single-lepton
     - :math:`\sigma_{\text{tot}}(t\bar{t}t\bar{t})`
     - 35.8
     - 1
     - CMS_4TOP_13TEV_SLEP_TOTAL
     - `Reference <https://arxiv.org/abs/1906.02805>`__ 
   * - **CMS**
     - 13
     - all-jet
     - :math:`\sigma_{\text{tot}}(t\bar{t}b\bar{b})`
     - 35.9
     - 1
     - CMS_TTBB_13TEV_ALLJET_TOTAL
     - `Reference <https://arxiv.org/abs/1909.05306>`__ 
   * - **CMS**
     - 13
     - dilepton
     - :math:`\sigma_{\text{tot}}(t\bar{t}b\bar{b})`
     - 35.9
     - 1
     - CMS_TTBB_13TEV_DILEPTON_TOTAL
     - `Reference <https://arxiv.org/abs/2003.06467>`__ 
   * - **CMS**
     - 13
     - :math:`\ell \mathrm{+jets}`
     - :math:`\sigma_{\text{tot}}(t\bar{t}b\bar{b})`
     - 35.9
     - 1
     - CMS_TTBB_13TEV_LJETS_TOTAL
     - `Reference <https://arxiv.org/abs/2003.06467>`__ 


T
----------------------

.. list-table:: 
   :widths: 5 5 8 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Channel**
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **ATLAS**
     - 7
     - t-channel
     - :math:`\sigma_\text{tot}(t)`
     - 4.59
     - 1
     - ATLAS_SINGLETOP_TCH_7TEV_T
     - `Reference <https://arxiv.org/abs/1406.7844>`__ 
   * - **ATLAS**
     - 7
     - t-channel
     - :math:`\sigma_\text{tot}(\bar{t})`
     - 4.59
     - 1
     - ATLAS_SINGLETOP_TCH_7TEV_TB
     - `Reference <https://arxiv.org/abs/1406.7844>`__ 
   * - **ATLAS**
     - 7
     - t-channel
     - :math:`1/\sigma d\sigma(tq)/dy_t`
     - 4.59
     - 3
     - ATLAS_SINGLETOP_TCH_DIFF_7TEV_T_RAP_NORM
     - `Reference <https://arxiv.org/abs/1406.7844>`__ 
   * - **ATLAS**
     - 7
     - t-channel
     - :math:`1/\sigma d\sigma(\bar{t}q)/dy_{\bar{t}}`
     - 4.59
     - 3
     - ATLAS_SINGLETOP_TCH_DIFF_7TEV_TBAR_RAP_NORM
     - `Reference <https://arxiv.org/abs/1406.7844>`__ 
   * - **ATLAS**
     - 8
     - t-channel
     - :math:`\sigma_\text{tot}(t)`
     - 20.2
     - 1
     - ATLAS_SINGLETOP_TCH_8TEV_T
     - `Reference <https://arxiv.org/abs/1702.02859>`__ 
   * - **ATLAS**
     - 8
     - t-channel
     - :math:`\sigma_{\text{tot}}(\bar{t})`
     - 20.2
     - 1
     - ATLAS_SINGLETOP_TCH_8TEV_TB
     - `Reference <https://arxiv.org/abs/1702.02859>`__ 
   * - **ATLAS**
     - 8
     - t-channel
     - :math:`1/\sigma d\sigma(tq)/dy_t`
     - 20.2
     - 1
     - ATLAS_SINGLETOP_TCH_DIFF_8TEV_T_RAP_NORM
     - `Reference <https://arxiv.org/abs/1702.02859>`__ 
   * - **ATLAS**
     - 8
     - t-channel
     - :math:`1/\sigma d\sigma(\bar{t}q)/dy_{\bar{t}}`
     - 20.2
     - 3
     - ATLAS_SINGLETOP_TCH_DIFF_8TEV_TBAR_RAP_NORM
     - `Reference <https://arxiv.org/abs/1702.02859>`__ 
   * - **ATLAS**
     - 8
     - s-channel
     - :math:`\sigma_{\text{tot}}(t + \bar{t})`
     - 20.3
     - 1
     - ATLAS_SINGLETOP_SCH_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1511.05980>`__ 
   * - **ATLAS**
     - 13
     - t-channel
     - :math:`\sigma_\text{tot}(t)`
     - 3.2
     - 1
     - ATLAS_SINGLETOP_TCH_13TEV_T
     - `Reference <https://arxiv.org/abs/1609.03920>`__ 
   * - **ATLAS**
     - 13
     - t-channel
     - :math:`\sigma_{\text{tot}}(\bar{t})`
     - 3.2
     - 1
     - ATLAS_SINGLETOP_TCH_13TEV_TB
     - `Reference <https://arxiv.org/abs/1609.03920>`__ 
   * - **ATLAS**
     - 13
     - s-channel
     - :math:`\sigma_\text{tot}(t+\bar{t})`
     - 139
     - 1
     - ATLAS_SINGLETOP_SCH_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/2209.08990>`__ 
   * - **CMS**
     - 7
     - t-channel
     - :math:`\sigma_\text{tot}(t) + \sigma_{\text{tot}}(\bar{t})`
     - 1.17, 1.56
     - 1
     - CMS_SINGLETOP_TCH_TOT_7TEV
     - `Reference <https://arxiv.org/abs/1209.4533>`__ 
   * - **CMS**
     - 8
     - t-channel
     - :math:`\sigma_\text{tot}(t)`
     - 19.7
     - 1
     - CMS_SINGLETOP_TCH_8TEV_T
     - `Reference <https://arxiv.org/abs/1403.7366>`__ 
   * - **CMS**
     - 8
     - t-channel
     - :math:`\sigma_{\text{tot}}(\bar{t})`
     - 19.7
     - 1
     - CMS_SINGLETOP_TCH_8TEV_TB
     - `Reference <https://arxiv.org/abs/1403.7366>`__ 
   * - **CMS**
     - 8
     - s-channel
     - :math:`\sigma_\text{tot}(t+\bar{t})`
     - 19.7
     - 1
     - CMS_SINGLETOP_SCH_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1603.02555>`__ 
   * - **CMS**
     - 13
     - t-channel
     - :math:`\sigma_\text{tot}(t)`
     - 2.2
     - 1
     - CMS_SINGLETOP_TCH_13TEV_T
     - `Reference <https://arxiv.org/abs/1610.00678>`__ 
   * - **CMS**
     - 13
     - t-channel
     - :math:`\sigma_{\text{tot}}(\bar{t})`
     - 2.2
     - 1
     - CMS_SINGLETOP_TCH_13TEV_TB
     - `Reference <https://arxiv.org/abs/1610.00678>`__ 
   * - **CMS**
     - 13
     - t-channel
     - :math:`1/\sigma d\sigma/d|y^{(t)}|`
     - 35.9
     - 4
     - CMS_SINGLETOP_TCH_13TEV_YTNORM
     - `Reference <https://arxiv.org/abs/1907.08330>`__ 

TW
----------------------

.. list-table:: 
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
   * - **ATLAS**
     - 8
     - dilepton
     - :math:`\sigma_{\text{tot}}(tW)`
     - 20.3
     - 1
     - ATLAS_SINGLETOPW_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1510.03752>`__ 
   * - **ATLAS**
     - 8
     - single-lepton
     - :math:`\sigma_{\text{tot}}(tW)`
     - 20.2
     - 1
     - ATLAS_SINGLETOPW_8TEV_SLEP_TOTAL
     - `Reference <https://arxiv.org/abs/2007.01554>`__ 
   * - **ATLAS**
     - 13
     - dilepton
     - :math:`\sigma_{\text{tot}}(tW)`
     - 3.2
     - 1
     - ATLAS_SINGLETOPW_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1612.07231>`__ 
   * - **CMS**
     - 8
     - dilepton
     - :math:`\sigma_{\text{tot}}(tW)`
     - 12.2
     - 1
     - CMS_SINGLETOPW_8TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1401.2942>`__ 
   * - **CMS**
     - 13
     - dilepton
     - :math:`\sigma_{\text{tot}}(tW)`
     - 35.9
     - 1
     - CMS_SINGLETOPW_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1805.07399>`__ 
   * - **CMS**
     - 13
     - single-lepton
     - :math:`\sigma_{\text{tot}}(tW)`
     - 36
     - 1
     - CMS_SINGLETOPW_13TEV_SLEP_TOTAL
     - `Reference <https://arxiv.org/abs/2109.01706>`__ 

TZ
----------------------

.. list-table:: 
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
   * - **ATLAS**
     - 13
     - dilepton
     - :math:`\sigma_{\text{fid}}(tZj)`
     - 139
     - 1
     - ATLAS_SINGLETOPZ_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/2002.07546>`__ 
   * - **CMS**
     - 13
     - dilepton
     - :math:`\sigma_{\text{fid}}(tZj)`
     - 77.4
     - 1
     - CMS_SINGLETOPZ_13TEV_TOTAL
     - `Reference <https://arxiv.org/abs/1812.05900>`__ 
   * - **CMS**
     - 13
     - dilepton
     - :math:`d\sigma_{\text{fid}}(tZj)/dp_T^t`
     - 138
     - 3
     - CMS_SINGLETOPZ_13TEV_PTT
     - `Reference <https://arxiv.org/abs/2111.02860>`__ 

================================
Drell-Yan
================================

.. list-table:: 
   :widths: 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **ATLAS**
     - 8
     - :math:`d^2\sigma/dm_{\ell\ell}d|y_{\ell\ell}|`
     - 20.3
     - 48
     - ATLASDY2D8TEV
     - `Reference <https://arxiv.org/abs/1606.01736>`__ (Table 3)
   * - **ATLAS**
     - 7
     - :math:`d\sigma_{Z/\gamma^{*}}/dM_{ll}`
     - 4.9
     - 13
     - ATLASZHIGHMASS49FB
     - `Reference <https://arxiv.org/abs/1305.4192>`__ 
   * - **CMS**
     - 7
     - :math:`d\sigma_{Z/\gamma^{*}}/dy`
     - 4.5
     - 132
     - CMSDY2D11
     - `Reference <https://arxiv.org/abs/1310.7291>`__ 
   * - **CMS**
     - 8
     - :math:`d\sigma_{Z/\gamma^{*}}/dy`
     - 19.7
     - 41
     - CMSDY1D12
     - `Reference <https://arxiv.org/abs/1412.1115>`__ 
   * - **CMS**
     - 13
     - :math:`d\sigma/dM_{\ell\ell}`
     - 5.1
     - 43
     - CMS_HMDY_13TEV
     - `Reference <https://arxiv.org/abs/1812.10529v2>`__ 

================================
W helicity
================================

.. list-table:: 
   :widths: 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **ATLAS and CMS**
     - 8
     - :math:`F_0, F_L`
     - 20
     - 2
     - ATLAS_CMS_WHEL_8TEV
     - `Reference <https://arxiv.org/abs/2005.03799>`__ 
   * - **ATLAS**
     - 13
     - :math:`F_0, F_L`
     - 139
     - 2
     - ATLAS_WHEL_13TEV
     - `Reference <https://arxiv.org/abs/2209.14903>`__ 


=================================
Electroweak Precision Observables
=================================

.. list-table:: 
   :widths: 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **LEP**
     - 0.250
     - Z observables
     - 
     - 19
     - LEP_ZDATA
     - `Reference <https://arxiv.org/abs/hep-ex/0509008>`__ (Tables 2.13, 3.6, 4.3, 5.8, and 5.10)
   * - **LEP**
     - 0.196
     - :math:`\mathcal{B}(W \rightarrow e^{-} \bar{v}_e),` :math:`\mathcal{B}(W \rightarrow \mu^{-} \bar{v}_{\mu}),` :math:`\mathcal{B}(W \rightarrow \tau^{-} \bar{v}_{\tau})`
     - 3
     - 3
     - LEP_BRW
     - `Reference <https://arxiv.org/abs/1302.3415>`__ (Table E.6)
   * - **LEP**
     - 0.189
     - :math:`\sigma(e^+ e^- \rightarrow e^+ e^-)`
     - 3
     - 21
     - LEP_BHABHA
     - `Reference <https://arxiv.org/abs/1302.3415>`__ (Table 3.11 and 3.12)
   * - **LEP**
     - 0.209
     - :math:`\hat{\alpha}^{(5)}_{\rm}(M_Z)`
     - 3
     - 1
     - LEP_ALPHAEW
     - `Reference <https://pdg.lbl.gov/2023/web/viewer.html?file=../reviews/rpp2022-rev-standard-model.pdf>`__ (Equation 10.11)


================================
Higgs
================================
.. list-table:: 
   :widths: 5 5 8 5 5 5 5
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **ATLAS and CMS**
     - 7 and 8
     - :math:`\mu_{H \rightarrow \mu^+ \mu^-}`
     - 5 and 20
     - 22
     - ATLAS_CMS_SSinc_RunI
     - `Reference <https://arxiv.org/abs/1606.02266>`__ (Table 13)
   * - **CMS**
     - 13
     - :math:`\mu_{H}`
     - 35.9
     - 24
     - CMS_SSINC_RUNII
     - `Reference <https://arxiv.org/abs/1809.10733>`__
   * - **ATLAS**
     - 13
     - :math:`\mu_{H}`
     - 80
     - 25
     - ATLAS_STXS_RUNII
     - `Reference <https://arxiv.org/abs/1909.02845>`__
   * - **ATLAS**
     - 13
     - :math:`\mu_{H \rightarrow Z \gamma}`
     - 139
     - 1
     - ATLAS_SSINC_RUNII_ZGAM
     - `Reference <https://arxiv.org/abs/2005.05382>`__
   * - **ATLAS**
     - 13
     - :math:`\mu_{H \rightarrow \mu^+ \mu^-}`
     - 139
     - 1
     - ATLAS_SSINC_RUNII_MUMU
     - `Reference <https://arxiv.org/abs/2007.07830>`__

================================
Diboson
================================


.. list-table:: 
   :widths: 5 5 8 5 5 5 9
   :header-rows: 1

   * - **Exp.**
     - :math:`\bf{\sqrt{s}} \textbf{(TeV)}`
     - **Observable**
     - :math:`\mathcal{L} (\text{fb}^{-1})`
     - :math:`\mathbf{n_{\rm dat}}`
     - **Dataset name**
     - **Ref.**
   * - **LEP**
     - 0.182
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.164
     - 10
     - LEP_EEWW_182GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`__ (Table 5.6, line 1)
   * - **LEP**
     - 0.189
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.588
     - 10
     - LEP_EEWW_189GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`__ (Table 5.6, line 2)
   * - **LEP**
     - 0.198
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.605
     - 10
     - LEP_EEWW_198GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`__ (Table 5.6, line 3)
   * - **LEP**
     - 0.206
     - :math:`d \sigma _{WW} / d cos(\theta _W)`
     - 0.631
     - 10
     - LEP_EEWW_206GEV
     - `Reference <https://arxiv.org/abs/1302.3415>`__ (Table 5.6, line 4)
   * - **ATLAS**
     - 13
     - :math:`d \sigma _{W^+W^-}/d m_{e \mu}`
     - 36.1
     - 13
     - ATLAS_WW_13TeV_2016_MEMU
     - `Reference <https://arxiv.org/abs/1905.04242>`__ (Figure 7.b), `HEPdata <https://www.hepdata.net/record/ins1734263>`__ (Table 42)
   * - **ATLAS**
     - 13
     - :math:`d \sigma _{WZ} / d m_{T}`
     - 36.1
     - 6
     - ATLAS_WZ_13TeV_2016_MTWZ
     - `Reference <https://arxiv.org/abs/1902.05759>`__ (Figure 4), `HEPdata <https://www.hepdata.net/record/ins1720438>`__ (Table 12)
   * - **ATLAS**
     - 13
     - :math:`d \sigma(Zjj)/d \Delta \phi_{jj}`
     - 139
     - 12
     - ATLAS_Zjj_13TeV_2016
     - `Reference <https://arxiv.org/abs/2006.15458>`__ (Table 8)
   * - **CMS**
     - 13
     - :math:`d \sigma _{WZ} / d p_{T}`
     - 35.9
     - 11
     - CMS_WZ_13TeV_2016_PTZ
     - `Reference <https://arxiv.org/abs/1901.03428>`__
