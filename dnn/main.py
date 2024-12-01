import os
from pathlib import Path

from dnn.data_processors import Dataset, load_dataset
from dnn.experiments import ActFunc, HyperParams, experiment, export_result

if not (train_data_path := os.getenv("TRAIN_DATA_PATH")):
    raise ValueError("TRAIN_DATA_PATH environment variable is not set")

if not (test_data_path := os.getenv("TEST_DATA_PATH")):
    raise ValueError("TEST_DATA_PATH environment variable is not set")

train_data: Dataset = load_dataset(Path(train_data_path))
# train_data = Dataset(train_data.x[:10000], train_data.r[:10000])  # temp
test_data: Dataset = load_dataset(Path(test_data_path))

hyperparams = HyperParams(
    lr=0.1,
    max_epoch=100,
    batch_size=32,
    hidden_nodes=[64, 32],
    act_func=ActFunc.RELU,
)
export_result(hyperparams, experiment(hyperparams, train_data, test_data))
