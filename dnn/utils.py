from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

from matplotlib import pyplot as plt
from numpy.typing import NDArray

from dnn.libs import np


def plot_line_graphs(
    x: Sequence[int | float],
    ys: dict[str, Sequence[int | float]],
    title: str,
    y_label: str,
    x_label: str,
) -> None:
    """Plot a line graph with multiple lines."""
    for name, y in ys.items():
        plt.plot(x, y, label=name, linestyle="-", marker="o")

    plt.xticks(x)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.show()


def parse_file_to_array(filepath: Path) -> NDArray[np.float64]:
    """Parse a file with space-separated integers into a numpy array."""
    return np.loadtxt(filepath, dtype=np.float64)  # type: ignore


class Dataset(NamedTuple):
    x: NDArray[np.float64]  # shape = (B, I)
    r: NDArray[np.float64]  # shape = (B, 1)


def generate_random_batches(dataset: Dataset, batch_size: int) -> Iterator[Dataset]:
    """Generate random batches from the dataset."""
    x, r = dataset

    if (data_size := x.shape[0]) != r.shape[0]:
        raise ValueError("The number of features and labels must be the same.")

    indices = np.random.permutation(data_size)
    for i in range(0, data_size, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield Dataset(x[batch_indices], r[batch_indices])


def print_shape(**kwargs: NDArray[np.generic]) -> None:
    for key, value in kwargs.items():
        print(f"{key}: {value.shape}", end=", ")
    print()
