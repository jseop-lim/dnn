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


class LossFunction(ABC):
    x: NDArray[np.float64]  # shape = (B, I)
    r: NDArray[np.float64]  # shape = (B, 1)

    def forward(self, x: NDArray[np.float64], r: NDArray[np.float64]) -> np.float64:
        """Compute the loss for a batch of inputs.

        Args:
            x: The input to the loss function. shape = (B, I)
            r: The target labels. shape = (B, 1)

        Returns:
            loss: The loss value. shape = (1,)
        """
        self.x = x
        self.r = r
        return self._forward()

    def backward(self) -> NDArray[np.float64]:
        """Error back-propagation for a batch of inputs.

        Returns:
            dLdx: The differential of the loss w.r.t. the input to the loss function. shape = (B, I)
                transpose of the downstream gradient
        """
        if not hasattr(self, "x"):
            raise RuntimeError("forward() must be called before backward().")
        return self._backward()

    @abstractmethod
    def _forward(self) -> np.float64:
        raise NotImplementedError

    @abstractmethod
    def _backward(self) -> NDArray[np.float64]:
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
        # TODO: feature 변동폭 일정하도록 초기화 방법 수정
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

    @staticmethod
    def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the ReLU function element-wise."""
        y: NDArray[np.float64] = np.maximum(0, x)
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

    @staticmethod
    def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the softmax function element-wise for each row."""
        y_numerator: NDArray[np.float64] = np.exp(x - x.max(axis=1, keepdims=True))
        y: NDArray[np.float64] = y_numerator / y_numerator.sum(axis=1, keepdims=True)
        return y


class CrossEntropyLayer(NNLayer):
    r: NDArray[np.float64]  # shape = (B, 1)

    def __init__(self, r: NDArray[np.float64]) -> None:
        self.r = r

    def _forward(self) -> NDArray[np.float64]:
        batch_size, _ = self.x.shape
        loss: NDArray[np.float64] = (
            np.sum(-np.log(self.x[range(batch_size), self.r.flatten()])) / batch_size
        )
        return loss  # shape = (1,)

    def backward(self, dLdy: NDArray[np.float64] | None = None) -> np.float64:
        # TODO: ISP 위반하므로 NNLayer 상속하지 않도록 리팩터링
        batch_size, input_size = self.x.shape
        indicator: NDArray[np.float64] = np.eye(input_size)[
            self.r.flatten()
        ]  # shape = (B, I)
        posteriors: NDArray[np.float64] = self.x[
            range(batch_size), self.r.flatten()
        ].reshape(-1, 1)  # shape = (B, 1)
        dLdx: NDArray[np.float64] = -indicator / posteriors / batch_size
        return dLdx

    def _backward(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


# DP나 BFS 이용해서 gradient 구하고, 가중치 갱신하자.
