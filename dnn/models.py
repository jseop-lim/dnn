from dataclasses import dataclass
from functools import reduce

from numpy.typing import NDArray

from dnn.data_processors import Dataset, generate_random_batches
from dnn.layers import NNLayer
from dnn.libs import np
from dnn.losses import LossFunction


def train_mini_batch_sgd(
    models: list[NNLayer],
    loss: LossFunction,
    dataset: Dataset,
    lr: float,
    max_epoch: int,
    batch_size: int,
) -> None:
    """Train the neural network model using mini-batch stochastic gradient descent."""
    for epoch in range(max_epoch):
        for batch in generate_random_batches(dataset, batch_size):
            x, r = batch
            # TODO: reduce로 리팩터링
            for model in models:
                x = model.forward(x)
            loss_value = loss.forward(x, r)
            grad = loss.backward()
            for model in reversed(models):
                grad = model.backward(grad)
                model.update_weights(lr)
            print(f"Epoch {epoch + 1}, Loss: {loss_value}")
    print("Training complete.")


# TODO: Model 추상 클래스를 상속받도록 리팩터링
@dataclass
class MiniBatchSgdNNClassifier:
    layers: list[NNLayer]  # ordered from deepest hidden layer to output layer
    loss_func: LossFunction
    lr: float = 0.1
    max_epoch: int = 100
    batch_size: int = 32

    def train(self, dataset: Dataset) -> NDArray[np.float64]:
        """Train the neural network model using mini-batch stochastic gradient descent.

        Args:
            dataset: The training dataset. x shape = (B, I), r shape = (B, 1)

        Returns:
            losses: The loss values for each weight update. shape = (max_epoch,)
        """
        num_batches = len(dataset.x) // self.batch_size
        losses: NDArray[np.float64] = np.zeros(self.max_epoch * num_batches)
        # TODO: max_epoch 대신 수렴 조건 판별하여 종료하도록 수정
        for epoch in range(self.max_epoch):
            for i, batch in enumerate(
                generate_random_batches(dataset, self.batch_size)
            ):
                # XXX: losses의 index 시작 번호를 1로 맞추기 위해 +1 적용. 그래프 그려보고 판단
                losses[epoch * num_batches + i] = self.loss_func.forward(
                    y=reduce(lambda x, layer: layer.forward(x), self.layers, batch.x),
                    r=batch.r,
                )
                reduce(
                    lambda dLdy, layer: layer.backward(dLdy),
                    reversed(self.layers),
                    self.loss_func.backward(),
                )
                for layer in self.layers:
                    layer.update_weights(self.lr)
        return losses

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the labels for a batch of inputs.

        Args:
            x: The input to the model. shape = (B, I)

        Returns:
            y: The predicted labels. shape = (B, 1)
        """
        posteriors = reduce(
            lambda x, layer: layer.forward(x), self.layers, x
        )  # shape = (B, O)
        y: NDArray[np.float64] = posteriors.argmax(axis=1).reshape(
            -1, 1
        )  # shape = (B, 1)
        return y
