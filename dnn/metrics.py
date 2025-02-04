from numpy.typing import NDArray

from dnn.libs import np


def compute_error_rate(
    predicted: NDArray[np.uint8],
    true: NDArray[np.uint8],
) -> float:
    """Compute the error rate of the predicted labels.

    Args:
        predicted: The predicted labels. shape = (B,)
        true: The true labels. shape = (B,)

    Returns:
        error_rate: The error rate.
    """
    return float(np.mean(predicted != true))
