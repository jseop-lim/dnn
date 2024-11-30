from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from numpy.typing import NDArray

from dnn.libs import np


class Dataset(NamedTuple):
    x: NDArray[np.float64]  # shape = (B, I)
    r: NDArray[np.float64]  # shape = (B, 1)


def load_dataset(filepath: Path) -> Dataset:
    """Load a dataset from files."""
    data = _parse_file_to_array(filepath)
    return Dataset(data[:, :-1], data[:, -1].reshape(-1, 1))


def _parse_file_to_array(filepath: Path) -> NDArray[np.float64]:
    """Parse a file with space-separated integers into a numpy array."""
    return np.loadtxt(filepath, dtype=np.float64)  # type: ignore


def generate_random_batches(dataset: Dataset, batch_size: int) -> Iterator[Dataset]:
    """Generate random batches from the dataset."""
    x, r = dataset

    if (data_size := x.shape[0]) != r.shape[0]:
        raise ValueError("The number of features and labels must be the same.")

    indices = np.random.permutation(data_size)
    for i in range(0, data_size, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield Dataset(x[batch_indices], r[batch_indices])
