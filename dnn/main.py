import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

from dnn import layers
from dnn.data_processors import Dataset, load_dataset
from dnn.libs import np
from dnn.losses import CrossEntropyLoss
from dnn.metrics import compute_error_rate
from dnn.models import MiniBatchSgdNNClassifier

if not (train_data_path := os.getenv("TRAIN_DATA_PATH")):
    raise ValueError("TRAIN_DATA_PATH environment variable is not set")

if not (test_data_path := os.getenv("TEST_DATA_PATH")):
    raise ValueError("TEST_DATA_PATH environment variable is not set")

train_data: Dataset = load_dataset(Path(train_data_path))
# train_data = Dataset(train_data.x[:10000], train_data.r[:10000])  # temp
test_data: Dataset = load_dataset(Path(test_data_path))

train_data_size, input_size = train_data.x.shape
test_data_size, _ = test_data.x.shape
output_size = len(set(train_data.r.flatten()))


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


hidden_sizes = [64, 32]

print("Training started.")

train_model = MiniBatchSgdNNClassifier(
    layers=generate_layers(
        input_size,
        hidden_sizes,
        output_size,
        act_class=layers.ReLULayer,
    ),
    loss_func=CrossEntropyLoss(),
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
        hidden_sizes,
        output_size,
        act_class=layers.ReLULayer,
    ),
    loss_func=CrossEntropyLoss(),
    lr=train_model.lr,
    max_epoch=train_model.max_epoch,
    batch_size=train_model.batch_size,
    threshold=train_model.threshold,
)
validate_loss_per_epoch = validate_model.train(test_data)

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


# Export the training and validation loss values to a CSV file
hyperparams = {
    "lr": train_model.lr,
    "batch": train_model.batch_size,
    "nodes": ",".join(str(n) for n in hidden_sizes),
    "act": train_model.layers[1].__class__.__name__.split("Layer")[0].lower()
    if hidden_sizes
    else None,
}

metadata = {
    "errorRate": f"{test_error_rate:.2}",
}

hyperparams_str = "_".join(f"{k}={v}" for k, v in hyperparams.items())
metadata_str = "_".join(f"{k}={v}" for k, v in metadata.items())
now_str = datetime.now().strftime("%y%m%d-%H%M%S")

filename = f"{hyperparams_str}_{metadata_str}_{now_str}.csv"

with open(f"logs/{filename}", "w") as f:
    f.write("epoch,train_error,validate_error\n")
    for epoch, (train_loss, validate_loss) in enumerate(
        zip(train_loss_per_epoch, validate_loss_per_epoch), 1
    ):
        f.write(f"{epoch},{train_loss},{validate_loss}\n")

print("--------------------")
print(f"Saved to logs/{filename}")
