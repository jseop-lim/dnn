from numpy.typing import NDArray

from dnn.libs import np


def compute_error_rate(
    true_labels: NDArray[np.uint8],
    predicted_labels: NDArray[np.uint8],
) -> float:
    """Compute the error rate of the predicted labels.

    Args:
        true_labels: The true labels. shape = (B,)
        predicted_labels: The predicted labels. shape = (B,)

    Returns:
        error_rate: The error rate.
    """
    return float(np.mean(true_labels != predicted_labels))
