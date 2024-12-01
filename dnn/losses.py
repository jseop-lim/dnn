from abc import ABC, abstractmethod

from numpy.typing import NDArray

from dnn.libs import np


class LossFunction(ABC):
    # NOTE: y는 loss function의 input이지만 NN model의 output이므로 y로 표기
    y: NDArray[np.float64]  # shape = (B, I)
    r: NDArray[np.uint8]  # shape = (B, 1)

    def forward(self, y: NDArray[np.float64], r: NDArray[np.uint8]) -> np.float64:
        """Compute the loss for a batch of outputs.

        Args:
            y: The predicted outputs. shape = (B, I)
            r: The target labels. shape = (B, 1)

        Returns:
            loss: The loss value. shape = (1,)
        """
        self.y = y
        self.r = r
        return self._forward()

    def backward(self) -> NDArray[np.float64]:
        """Error back-propagation for a batch of outputs.

        Returns:
            dLdy: The differential of the loss w.r.t. the output to the layer. shape = (B, I)
                transpose of the downstream gradient
        """
        if not hasattr(self, "y"):
            raise RuntimeError("forward() must be called before backward().")
        return self._backward()

    @abstractmethod
    def _forward(self) -> np.float64:
        raise NotImplementedError

    @abstractmethod
    def _backward(self) -> NDArray[np.float64]:
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):
    def _forward(self) -> np.float64:
        batch_size, _ = self.y.shape
        loss: np.float64 = np.mean(-np.log(self.y[range(batch_size), self.r.flatten()]))
        return loss  # shape = (1,)

    def _backward(self) -> NDArray[np.float64]:
        batch_size, input_size = self.y.shape
        indicator: NDArray[np.float64] = np.eye(input_size)[
            self.r.flatten()
        ]  # shape = (B, I)
        posteriors: NDArray[np.float64] = self.y[
            range(batch_size), self.r.flatten()
        ].reshape(-1, 1)  # shape = (B, 1)
        dLdy: NDArray[np.float64] = -indicator / posteriors / batch_size
        return dLdy
