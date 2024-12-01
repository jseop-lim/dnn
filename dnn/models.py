from dataclasses import dataclass
from functools import reduce
from typing import NamedTuple

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


class TrainResult(NamedTuple):
    train_losses: NDArray[np.float64]
    validate_losses: NDArray[np.float64] | None


# TODO: Model 추상 클래스를 상속받도록 리팩터링
@dataclass
class MiniBatchSgdNNClassifier:
    layers: list[NNLayer]  # ordered from deepest hidden layer to output layer
    loss_func: LossFunction
    lr: float
    max_epoch: int
    batch_size: int
    threshold: float = 1e-2

    def train(
        self,
        train_data: Dataset,
        validate_data: Dataset | None = None,
    ) -> TrainResult:
        """Train the neural network model using mini-batch stochastic gradient descent.

        Only train_data is involved in parameter determination.
        If validate_data is given, the loss value for the entire validation_data is measured every epoch.
        Final epoch is determined by the convergence threshold or max_epoch.

        Args:
            train_data: The training dataset. x shape = (B, I), r shape = (B, 1)
            validate_data: The validation dataset. x shape = (B, I), r shape = (B, 1)

        Returns:
            train_losses: The loss values for each epoch of training. shape = (final_epoch,)
            validate_losses: The loss values for each epoch of validation. shape = (final_epoch,)
                None if validate_data is not given.
        """
        num_batches = len(train_data.x) // self.batch_size + 1
        train_loss_per_update = np.full((self.max_epoch, num_batches), np.nan)
        vaildate_loss_per_epoch = np.full((self.max_epoch), np.nan)

        for epoch in range(self.max_epoch):
            for i, batch in enumerate(
                generate_random_batches(train_data, self.batch_size)
            ):
                train_loss_per_update[epoch, i] = self._feed_forward(batch)
                # check convergence
                if train_loss_per_update[epoch, i] < self.threshold:
                    break

                self._error_backprop()
                self._update_weights()

            if validate_data:
                vaildate_loss_per_epoch[epoch] = self._feed_forward(validate_data)

        train_loss_per_epoch = np.nanmean(train_loss_per_update, axis=1)
        return TrainResult(
            train_loss_per_epoch[~np.isnan(train_loss_per_epoch)],
            vaildate_loss_per_epoch[~np.isnan(vaildate_loss_per_epoch)]
            if validate_data
            else None,
        )

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

    def _feed_forward(self, batch: Dataset) -> np.float64:
        """Compute the loss for a batch of inputs."""
        return self.loss_func.forward(
            y=reduce(lambda x, layer: layer.forward(x), self.layers, batch.x),
            r=batch.r,
        )

    def _error_backprop(self) -> NDArray[np.float64]:
        """Error back-propagation for a batch of inputs."""
        return reduce(
            lambda dLdy, layer: layer.backward(dLdy),
            reversed(self.layers),
            self.loss_func.backward(),
        )

    def _update_weights(self) -> None:
        """Update the parameters of all layers."""
        for layer in self.layers:
            layer.update_weights(self.lr)
