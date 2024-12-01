from dataclasses import dataclass
from functools import reduce

from numpy.typing import NDArray

from dnn.data_processors import Dataset, generate_random_batches
from dnn.layers import NNLayer
from dnn.libs import np
from dnn.losses import LossFunction


# Deprecated
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
    lr: float
    max_epoch: int
    batch_size: int
    threshold: float = 1e-2

    def train(self, dataset: Dataset) -> NDArray[np.float64]:
        """Train the neural network model using mini-batch stochastic gradient descent.

        Args:
            dataset: The training dataset. x shape = (B, I), r shape = (B, 1)

        Returns:
            losses: The loss values for each epoch. shape = (final_epoch,)
        """
        num_batches = len(dataset.x) // self.batch_size + 1
        loss_per_update = np.full((self.max_epoch, num_batches), np.nan)

        for epoch in range(self.max_epoch):
            for i, batch in enumerate(
                generate_random_batches(dataset, self.batch_size)
            ):
                loss_per_update[epoch, i] = self.loss_func.forward(
                    y=self._feed_forward(batch.x), r=batch.r
                )
                if loss_per_update[epoch, i] < self.threshold:
                    break

                self._error_backprop(self.loss_func.backward())
                self._update_weights()

        loss_per_epoch: NDArray[np.float64] = np.nanmean(loss_per_update, axis=1)
        loss_per_epoch = loss_per_epoch[~np.isnan(loss_per_epoch)]  # remove nan
        return loss_per_epoch

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
        predicted_r: NDArray[np.float64] = posteriors.argmax(axis=1).reshape(
            -1, 1
        )  # shape = (B, 1)
        return predicted_r

    def _feed_forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Feed forward for a batch of inputs."""
        return reduce(lambda x, layer: layer.forward(x), self.layers, x)

    def _error_backprop(self, dLdy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Error back-propagation for a batch of inputs."""
        return reduce(
            lambda dLdy, layer: layer.backward(dLdy), reversed(self.layers), dLdy
        )

    def _update_weights(self) -> None:
        """Update the parameters of the model."""
        for layer in self.layers:
            layer.update_weights(self.lr)
