from simunet.core import SIMUnetDataSetSpec
from validphys.core import PDF, DataGroupSpec
from validphys.results import DataResult, ThPredictionsResult
from reportengine.checks import check_not_empty, remove_outer, require_one

from collections.abc import Sequence


class SIMUnetThPredictionsResult(ThPredictionsResult):
    def __init__(
        self,
        dataobj,
        stats_class,
        datasetnames=None,
        label=None,
        pdf=None,
        theoryid=None,
        load_dataset_contamination=None,
    ):
        self.load_dataset_contamination = load_dataset_contamination

        super().__init__(
            dataobj=dataobj,
            stats_class=stats_class,
            datasetnames=datasetnames,
            label=label,
            pdf=pdf,
            theoryid=theoryid,
        )

    @classmethod
    def convolution(cls, pdf, dataset, central_only=False):
        conv = super().convolution(pdf, dataset, central_only=central_only)
        return conv


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
        SIMUnetThPredictionsResult.from_convolution(
            pdf, dataset, load_datasets_contamination
        ),
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
    return simu_pdf_results(
        dataset, pdfs, covariance_matrix, sqrt_covmat, load_datasets_contamination
    )
