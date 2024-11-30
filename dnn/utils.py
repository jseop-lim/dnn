from collections.abc import Sequence

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


def print_shape(**kwargs: NDArray[np.generic]) -> None:
    for key, value in kwargs.items():
        print(f"{key}: {value.shape}", end=", ")
    print()
