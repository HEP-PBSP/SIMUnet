from simunet.core import SIMUnetDataSetSpec
from validphys.core import PDF, DataGroupSpec
from validphys.results import DataResult, ThPredictionsResult
from reportengine.checks import remove_outer, require_one
from validphys.convolution import PredictionsRequireCutsError, central_predictions, predictions

import pandas as pd
import numpy as np

from collections.abc import Sequence


class SIMUnetThPredictionsResult(ThPredictionsResult):
    def __init__(
        self, dataobj, stats_class, datasetnames=None, label=None, pdf=None, theoryid=None
    ):
        super().__init__(
            dataobj=dataobj,
            stats_class=stats_class,
            datasetnames=datasetnames,
            label=label,
            pdf=pdf,
            theoryid=theoryid,
        )

    @classmethod
    def from_convolution(cls, pdf, dataset, load_dataset_contamination, central_only=False):
        # This should work for both single dataset and whole groups
        try:
            datasets = dataset.datasets
        except AttributeError:
            datasets = (dataset,)

        try:
            if central_only:
                preds = [central_predictions(d, pdf) for d in datasets]
            else:
                preds = [predictions(d, pdf) for d in datasets]
            th_predictions = pd.concat(preds)
            if load_dataset_contamination is not None:
                th_predictions *= 1.0 + load_dataset_contamination[dataset.name][:, None]

        except PredictionsRequireCutsError as e:
            raise PredictionsRequireCutsError(
                "Predictions from FKTables always require cuts, "
                "if you want to use the fktable intrinsic cuts set `use_cuts: 'internal'`"
            ) from e

        label = cls.make_label(pdf, dataset)
        thid = dataset.thspec.id
        datasetnames = [i.name for i in datasets]
        return cls(th_predictions, pdf.stats_class, datasetnames, label, pdf=pdf, theoryid=thid)


def simu_results(
    dataset: SIMUnetDataSetSpec,
    pdf: PDF,
    covariance_matrix,
    sqrt_covmat,
    load_datasets_contamination,
):
    """Tuple of data and theory results for a single pdf. The data will have an associated
    covariance matrix, which can include a contribution from the theory covariance matrix which
    is constructed from scale variation.

    The theory is specified as part of the dataset (a remnant of the old C++ layout)
    A group of datasets is also allowed.
    """
    return (
        DataResult(dataset, covariance_matrix, sqrt_covmat),
        SIMUnetThPredictionsResult.from_convolution(pdf, dataset, load_datasets_contamination),
    )


@require_one("pdfs", "pdf")
@remove_outer("pdfs", "pdf")
def simunet_one_or_more_results(
    dataset: (SIMUnetDataSetSpec, DataGroupSpec),
    covariance_matrix,
    sqrt_covmat,
    load_datasets_contamination,
    pdfs: (type(None), Sequence) = None,
    pdf: (type(None), PDF) = None,
):
    if pdf is not None:
        return simu_results(
            dataset, pdf, covariance_matrix, sqrt_covmat, load_datasets_contamination
        )
    raise NotImplementedError(
        "Not yet implemented for SIMUnet, only single PDF results are currently supported."
    )
    # return simu_pdf_results(
    #     dataset, pdfs, covariance_matrix, sqrt_covmat, load_datasets_contamination
    # )
