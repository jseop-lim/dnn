import os
from pathlib import Path

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
test_data: Dataset = load_dataset(Path(test_data_path))

train_data_size, input_size = train_data.x.shape
test_data_size, _ = test_data.x.shape
output_size = len(set(train_data.r.flatten()))


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

model = MiniBatchSgdNNClassifier(
    layers=[
        *generate_layers(
            input_size,
            hidden_sizes,
            output_size,
            act_class=layers.ReLULayer,
        ),
    ],
    loss_func=CrossEntropyLoss(),
)
loss_per_epoch = np.nanmean(model.train(train_data), axis=1)

print("Training complete.")

test_predicted_r = model.predict(test_data.x).astype(np.uint8)
error_rate: float = compute_error_rate(predicted=test_predicted_r, true=test_data.r)

print("Prediction complete.")
print(f"Error rate: {error_rate:.2%}")
