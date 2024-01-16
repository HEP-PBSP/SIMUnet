===================================
**Features**
===================================

Here we detail some of the main features and capabilities of the :math:`\text{SIMUnet}` methodology.

PDF and SMEFT fits
------------------

:math:`\text{SIMUnet}` can perform simultaneous PDF-SMEFT fits.
The interplay between PDFs and SMEFT parameters is complex and traditional methods
of treating them separately can lead to inconsistencies and inaccuracies.
By performing simultaneous fits, the :math:`\text{SIMUnet}` framework addresses this challenge head-on,
ensuring that the subtleties and interdependencies are correctly captured.

In Fig. 1 we show an example of a PDF obtained from a simultaneous PDF-SMEFT fit with :math:`\text{SIMUnet}`
and compare it its analogous SM result.

.. figure:: images/sm_smeft_pdf.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *Figure 1:*
    Gluon PDF determined from a simultaneous PDF-SMEFT fit (green)
    and a SM (NNPDF4.0) fit (blue) in the top sector. We also show
    the PDF obtained when no top data is included in the SM fit.

We see how the inclusion of new data shifts the SM PDF, and the impact of performing a simultaneous PDF-SMEFT fit which,
in this case, allows the PDF to accommodate better the inclusion of new data.

Additionally, :math:`\text{SIMUnet}` provides distributions of SMEFT coefficients, as shown in Fig. 2.

.. figure:: images/ctg_plot.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *Figure 2:*
    Distribution of the :math:`c_{tG}` SMEFT coefficient determined from
    a simultaneous PDF-SMEFT fit (green) and a fixed-PDF fit (orange).

In this case, the distribution of the SMEFT coefficient :math:`c_{tG}` changes mildly.
The PDF-SMEFT interplay is not trivial, and in other cases more substantial changes can be found.

:math:`\text{SIMUnet}` also provides complete summaries to show
the behaviour of the SMEFT basis as a whole. In Fig. 3, we show a plot with the bounds for a set of
top operators.

.. figure:: images/wilsons_summary.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *Figure 3:*
    Bounds for a set of 20 SMEFT operators obtained with :math:`\text{SIMUnet}` in a simultaneous PDF-SMEFT fit
    and a fixed-PDF fit.

:math:`\text{SIMUnet}` also provides summary tables, correlations, PCA, among many other things.

In Fig. 4 we show the correlation between a SMEFT coefficient and a set of PDFs.

.. figure:: images/pdf_eft_corr.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *Figure 4:*
    Correlations between the SMEFT coefficients :math:`c_{tG}` and :math:`c_{ut}^{8}` and the PDFs
    of the gluon and some components of the evolution basis.
    
Correlations like the ones shown above can be useful to quantify the strength of the PDF-SMEFT interplay
and the sensitivity of SMEFT operators and PDFs when performing simultaneous and fixed-PDF fits.

Can PDFs absorb new physics?
----------------------------

.. figure:: images/np_values.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *example figure*
    :label: fig-np_values


.. figure:: images/pdf_shift_lumis.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *example figure*
    :label: fig-pdf_shift_lumis


.. figure:: images/cont_effects.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *example figure*
    :label: fig-cont_effects


.. figure:: images/cont_chi2.png
    :width: 100%
    :class: align-center
    :figwidth: 100%
    :figclass: align-center

    *example figure*
    :label: fig-cont_chi2

