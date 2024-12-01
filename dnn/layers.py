"""This module implements the linear layer of a neural network.

Symbols used for size of dimensions:
    B: batch size
    I: input size
    O: output size
"""

from abc import ABC, abstractmethod

from numpy.typing import NDArray

from dnn.libs import np


class NNLayer(ABC):
    x: NDArray[np.float64]  # shape = (B, I)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Feed forward for a batch of inputs.

        Args:
            x: The input to the layer. shape = (B, I)

        Returns:
            y: The output of the layer. shape = (B, O)
        """
        self.x = x
        return self._forward()

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
        return self._backward(dLdy)

    @abstractmethod
    def _forward(self) -> NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    @abstractmethod
    def update_weights(self, lr: float) -> None:
        """Update the parameters of the layer.

        Args:
            lr: The learning rate.
        """
        raise NotImplementedError


class LinearLayer(NNLayer):
    W: NDArray[np.float64]  # shape = (O, I)
    b: NDArray[np.float64]  # shape = (O, 1)
    dLdW: NDArray[np.float64]  # shape = (O, I)
    dLdb: NDArray[np.float64]  # shape = (1, O)
    # Warning: The shapes of b and dLdb are transposed.

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

    def _forward(self) -> NDArray[np.float64]:
        y: NDArray[np.float64] = (self.W @ self.x.T + self.b).T
        return y

    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        batch_size, _ = dLdy.shape
        self.dLdW = dLdy.T @ self.x / batch_size
        self.dLdb = dLdy.mean(axis=0, keepdims=True)
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


class SigmoidLayer(NNLayer):
    def _forward(self) -> NDArray[np.float64]:
        return self.sigmoid(self.x)  # shape = (B, I)

    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        y = self.sigmoid(self.x)  # shape = (B, I)
        dLdx: NDArray[np.float64] = dLdy * y * (1 - y)
        return dLdx

    def update_weights(self, lr: float) -> None:
        pass

    @staticmethod
    def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the sigmoid function element-wise."""
        y: NDArray[np.float64] = 1 / (1 + np.exp(-x))
        return y


class ReLULayer(NNLayer):
    def _forward(self) -> NDArray[np.float64]:
        return self.relu(self.x)  # shape = (B, I)

    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        dLdx: NDArray[np.float64] = dLdy * (self.x > 0)
        return dLdx

    def update_weights(self, lr: float) -> None:
        pass

    @staticmethod
    def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the ReLU function element-wise."""
        y: NDArray[np.float64] = np.maximum(0, x)
        return y


class LeakyReLULayer(NNLayer):
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha: float = alpha

    def _forward(self) -> NDArray[np.float64]:
        return self.leaky_relu(self.x)  # shape = (B, I)

    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        dLdx: NDArray[np.float64] = (
            dLdy * (self.x > 0) + dLdy * (self.x <= 0) * self.alpha
        )
        return dLdx

    def update_weights(self, lr: float) -> None:
        pass

    @staticmethod
    def leaky_relu(x: NDArray[np.float64], alpha: float = 0.01) -> NDArray[np.float64]:
        """Compute the Leaky ReLU function element-wise."""
        y: NDArray[np.float64] = np.maximum(alpha * x, x)
        return y


class ELULayer(NNLayer):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha: float = alpha

    def _forward(self) -> NDArray[np.float64]:
        return self.elu(self.x)  # shape = (B, I)

    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        dLdx: NDArray[np.float64] = dLdy * (self.x > 0) + dLdy * (self.x <= 0) * (
            self.elu(self.x) + self.alpha
        )
        return dLdx

    def update_weights(self, lr: float) -> None:
        pass

    @staticmethod
    def elu(x: NDArray[np.float64], alpha: float = 1.0) -> NDArray[np.float64]:
        """Compute the ELU function element-wise."""
        y: NDArray[np.float64] = np.maximum(alpha * (np.exp(x) - 1), x)
        return y


class SoftmaxLayer(NNLayer):
    def _forward(self) -> NDArray[np.float64]:
        return self.softmax(self.x)  # shape = (B, I)

    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        y = self.softmax(self.x)  # shape = (B, I)
        dydx = np.array(
            [np.diagflat(row) - np.outer(row, row) for row in y]
        )  # shape = (B, I, I)
        dLdx: NDArray[np.float64] = np.einsum("bij,bj->bi", dydx, dLdy)
        return dLdx

    def update_weights(self, lr: float) -> None:
        pass

    @staticmethod
    def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the softmax function element-wise for each row."""
        y_numerator: NDArray[np.float64] = np.exp(x - x.max(axis=1, keepdims=True))
        y: NDArray[np.float64] = y_numerator / y_numerator.sum(axis=1, keepdims=True)
        return y


# DP나 BFS 이용해서 gradient 구하고, 가중치 갱신하자.
