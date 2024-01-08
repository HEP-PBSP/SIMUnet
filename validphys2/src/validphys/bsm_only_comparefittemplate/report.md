%SIMUnet report comparing {@ current fit_id @} and {@ reference fit_id @}

Summary
-------

We are comparing:

  - {@ current fit @} (`{@ current fit_id @}`): {@ current description @}
  - {@ reference fit @} (`{@ reference fit_id @}`): {@ reference description @}

Luminosities
------------
{@with Energies@}
[Plots at {@Energytitle@}]({@lumi_report report@})
{@endwith@}

SIMUnet analysis
----------------
### Comparison histograms of BSM factors
{@plot_nd_bsm_facs_fits@}

### Plots of the bounds
{@plot_bsm_facs_bounds@}

### 68% CL bounds comparison
{@bsm_facs_68bounds_fits@}

### 95% CL bounds comparison
{@bsm_facs_95bounds_fits@}

### BSM 68% residuals
{@plot_bsm_facs_68res@}

### BSM correlations  
{@fits plot_bsm_corr@}




Code versions
-------------
{@fits_version_table@}
