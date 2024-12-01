import os
from dataclasses import replace
from pathlib import Path

from dnn.data_processors import Dataset, load_dataset
from dnn.experiments import ActFunc, HyperParams, experiment, export_result

if not (train_data_path := os.getenv("TRAIN_DATA_PATH")):
    raise ValueError("TRAIN_DATA_PATH environment variable is not set")

if not (test_data_path := os.getenv("TEST_DATA_PATH")):
    raise ValueError("TEST_DATA_PATH environment variable is not set")

train_data: Dataset = load_dataset(Path(train_data_path))
test_data: Dataset = load_dataset(Path(test_data_path))


"""Model Selection"""
# 가변 조건
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]
batch_size_list = [1, 32, 64, 128, train_data.x.shape[0]]
hidden_nodes_list = [[16], [16, 8], [16, 16, 8], [16, 8, 8], [16, 16, 8, 8]]
act_func_list = [ActFunc.SIGMOID, ActFunc.RELU, ActFunc.LEAKY_RELU]

# 불변 조건
base_config = HyperParams(
    lr=1e-2,
    batch_size=32,
    hidden_nodes=[16, 8],
    act_func=ActFunc.RELU,
    max_epoch=250,
)

lr_configs = [
    replace(base_config, lr=lr, max_epoch=max_epoch)
    for lr, max_epoch in zip(lr_list, [250, 250, 250, 250])
]
batch_size_configs = [
    replace(base_config, batch_size=batch_size, max_epoch=max_epoch)
    for batch_size, max_epoch in zip(batch_size_list, [20000, 100, 125, 1000, 1000])
]
hidden_nodes_configs = [
    replace(base_config, hidden_nodes=hidden_nodes, max_epoch=max_epoch)
    for hidden_nodes, max_epoch in zip(hidden_nodes_list, [250, 250, 1000, 1000, 1000])
]
act_func_configs = [
    replace(base_config, act_func=act_func, max_epoch=max_epoch)
    for act_func, max_epoch in zip(act_func_list, [500, 500, 500])
]

print("Base Config")
# export_result(base_config, experiment(base_config, train_data, test_data))

print("Change Learning Rate")
for config in lr_configs:
    # export_result(config, experiment(config, train_data, test_data))
    pass

print("Change Batch Size")
for config in batch_size_configs:
    # export_result(config, experiment(config, train_data, test_data))
    pass

print("Change Hidden Nodes")
for config in hidden_nodes_configs:
    # export_result(config, experiment(config, train_data, test_data))
    pass

print("Change Activation Function")
for config in act_func_configs:
    # export_result(config, experiment(config, train_data, test_data))
    pass


selected_config = HyperParams(
    lr=0.001,
    hidden_nodes=[16, 16, 8, 8],
    batch_size=32,
    act_func=ActFunc.RELU,
    max_epoch=4000,
)
final_result = experiment(selected_config, train_data, test_data)
export_result(selected_config, final_result)
