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


class Batch(NamedTuple):
    x: NDArray[np.float64]  # shape = (B, I)
    r: NDArray[np.float64]  # shape = (B, 1)


def generate_random_batches(
    data: NDArray[np.float64],
    batch_size: int,
) -> Iterator[Batch]:
    """Generate random mini-batches from the data."""
    data_size, _ = data.shape
    indices = np.random.permutation(data_size)
    for start_idx in range(0, data_size, batch_size):
        end_idx = min(start_idx + batch_size, data_size)
        batch_indices = indices[start_idx:end_idx]
        yield Batch(
            data[batch_indices][:, :-1], data[batch_indices][:, -1:].reshape(-1, 1)
        )


def print_shape(**kwargs: NDArray[np.generic]) -> None:
    for key, value in kwargs.items():
        print(f"{key}: {value.shape}", end=", ")
    print()
