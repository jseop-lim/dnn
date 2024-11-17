"""This module implements the linear layer of a neural network.

Symbols used for size of dimensions:
    B: batch size
    I: input size
    O: output size
"""

from numpy.typing import NDArray

from dnn.libs import np


class LinearLayer:
    x: NDArray[np.float64]  # shape = (B, I)
    W: NDArray[np.float64]  # shape = (O, I)
    b: NDArray[np.float64]  # shape = (O, 1)
    dLdW: NDArray[np.float64]  # shape = (O, I)
    dLdb: NDArray[np.float64]  # shape = (1, O)

    def __init__(self, input_size: int, output_size: int, std: float = 0.01) -> None:
        """Initialize the weights and biases of the linear layer.

        Args:
            input_size: The number of input features.
            output_size: The number of output features.
            std: The standard deviation of the normal distribution used to initialize the weights and biases.
        """
        self.W: NDArray[np.float64] = np.random.normal(
            0, std, (output_size, input_size)
        )
        self.b: NDArray[np.float64] = np.random.normal(0, std, (output_size, 1))

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Feed forward for a batch of inputs.

        Args:
            x: The input to the layer. shape = (B, I)

        Returns:
            y: The output of the layer. shape = (B, O)
        """
        self.x = x
        y: NDArray[np.float64] = (self.W @ x.T + self.b).T
        return y

    def backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Error back-propagation for a batch of inputs.

        Args:
            dLdy: The differential of the loss w.r.t. the output of the layer. shape = (B, O)
                transpose of the upstream gradient

        Returns:
            dLdx: The differential of the loss w.r.t. the input to the layer. shape = (B, I)
                transpose of the downstream gradient
        """
        if not hasattr(self, "x"):
            raise RuntimeError("forward() must be called before backward().")

        batch_size, _ = dLdy.shape
        self.dLdW = dLdy.T @ self.x / batch_size
        self.dLdb = dLdy.mean(axis=0, keepdims=True)  # TODO: 검토 필요
        dLdx: NDArray[np.float64] = dLdy @ self.W
        return dLdx

    def update_weights(self, lr: float) -> None:
        """Update the weights and biases of the linear layer.

        Args:
            lr: The positive learning rate.
        """
        if lr <= 0:
            raise ValueError("The learning rate must be positive.")

        self.W -= lr * self.dLdW
        self.b -= lr * self.dLdb.T


# DP나 BFS 이용해서 gradient 구하고, 가중치 갱신하자.
