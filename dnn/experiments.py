from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import NamedTuple, TypedDict

from numpy.typing import NDArray

from dnn import layers
from dnn.data_processors import Dataset
from dnn.libs import np
from dnn.losses import CrossEntropyLoss
from dnn.metrics import compute_error_rate
from dnn.models import MiniBatchSgdNNClassifier


class ActFunc(str, Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    SOFTPLUS = "softplus"
    LEAKY_RELU = "leakyRelu"
    ELU = "elu"


act_func_map: dict[ActFunc, type[layers.NNLayer]] = {
    ActFunc.SIGMOID: layers.SigmoidLayer,
    ActFunc.RELU: layers.ReLULayer,
    ActFunc.LEAKY_RELU: layers.LeakyReLULayer,
    ActFunc.ELU: layers.ELULayer,
}


@dataclass
class HyperParams:
    class Meta(TypedDict):  # TODO: 안쓰면 삭제
        short_name: str
        to_filename: bool

    lr: float = field(metadata=Meta(short_name="lr", to_filename=True))
    batch_size: int = field(metadata=Meta(short_name="batch", to_filename=True))
    hidden_nodes: list[int] = field(metadata=Meta(short_name="nodes", to_filename=True))
    act_func: ActFunc = field(metadata=Meta(short_name="act", to_filename=True))
    max_epoch: int = field(metadata=Meta(short_name="maxEpoch", to_filename=False))

    def __str__(self) -> str:
        return "_".join(
            [
                f"lr={self.lr}",
                f"batch={self.batch_size}",
                f"nodes={','.join(str(n) for n in self.hidden_nodes)}",
                f"act={self.act_func.value}",
            ]
        )


def generate_layers(
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    act_class: type[layers.NNLayer],
) -> list[layers.NNLayer]:
    dnn_layers: list[layers.NNLayer] = [layers.LinearLayer(input_size, hidden_sizes[0])]

    for i_size, o_size in zip(hidden_sizes[:-1], hidden_sizes[1:] + [output_size]):
        dnn_layers.append(act_class())
        dnn_layers.append(layers.LinearLayer(i_size, o_size))

    dnn_layers.append(layers.SoftmaxLayer())
    return dnn_layers


class ExperimentResult(NamedTuple):
    train_losses: NDArray[np.float64]
    validate_losses: NDArray[np.float64]
    test_error_rate: float


def experiment(
    hyperparams: HyperParams,
    train_data: Dataset,
    test_data: Dataset,
    validate_data: Dataset | None = None,
) -> ExperimentResult:
    validate_data = validate_data or test_data

    _, input_size = train_data.x.shape
    output_size = len(np.unique(train_data.r))

    print("Training started.")

    train_model = MiniBatchSgdNNClassifier(
        layers=generate_layers(
            input_size,
            hyperparams.hidden_nodes,
            output_size,
            act_class=act_func_map[hyperparams.act_func],
        ),
        loss_func=CrossEntropyLoss(),
        lr=hyperparams.lr,
        batch_size=hyperparams.batch_size,
        max_epoch=hyperparams.max_epoch,
    )
    train_loss_per_epoch = train_model.train(train_data)

    gap = 50
    for epoch, loss in enumerate(train_loss_per_epoch[::gap]):
        print(f"Epoch {epoch * gap + 1}, Loss: {loss:.6f}")
    print("Training complete.")

    print("--------------------")
    print("Validation started.")

    validate_model = MiniBatchSgdNNClassifier(
        layers=generate_layers(
            input_size,
            hyperparams.hidden_nodes,
            output_size,
            act_class=act_func_map[hyperparams.act_func],
        ),
        loss_func=CrossEntropyLoss(),
        lr=train_model.lr,
        max_epoch=train_model.max_epoch,
        batch_size=train_model.batch_size,
        threshold=train_model.threshold,
    )
    validate_loss_per_epoch = validate_model.train(validate_data)

    for epoch, loss in enumerate(validate_loss_per_epoch[::gap]):
        print(f"Epoch {epoch * gap + 1}, Loss: {loss:.6f}")
    print("Validation complete.")

    print("--------------------")
    print("Prediction started.")

    test_predicted_r = train_model.predict(test_data.x).astype(np.uint8)
    test_error_rate: float = compute_error_rate(
        predicted=test_predicted_r, true=test_data.r
    )

    print(f"Error rate: {test_error_rate:.2%}")
    print("Prediction complete.")

    return ExperimentResult(
        train_loss_per_epoch, validate_loss_per_epoch, test_error_rate
    )


def export_result(
    hyperparams: HyperParams,
    result: ExperimentResult,
) -> None:
    metadata = {
        "errorRate": f"{result.test_error_rate:.2}",
    }

    metadata_str = "_".join(f"{k}={v}" for k, v in metadata.items())
    now_str = datetime.now().strftime("%y%m%d-%H%M%S")

    result_filepath = f"logs/{hyperparams}_{metadata_str}_{now_str}.csv"

    with open(result_filepath, "w") as f:
        f.write("epoch,train_error,validate_error\n")
        for epoch, (train_loss, validate_loss) in enumerate(
            zip(result.train_losses, result.validate_losses), 1
        ):
            f.write(f"{epoch},{train_loss},{validate_loss}\n")

    print("--------------------")
    print(f"Saved to {result_filepath}")
