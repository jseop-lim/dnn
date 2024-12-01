# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
import numpy as np
from matplotlib import pyplot as plt

from dnn.data_processors import Dataset
from dnn.layers import LinearLayer, SigmoidLayer, SoftmaxLayer
from dnn.losses import CrossEntropyLoss
from dnn.models import MiniBatchSgdNNClassifier

num_d = 5

# number of test runs
num_test = 40

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr = 0.1
num_gd_step = 100000

# dataset size
batch_size = 4 * num_d

# number of classes is 2
num_class = 2

# variable to measure accuracy
accuracy = 0

# set this True if want to plot training data
show_train_data = True

# set this True if want to plot loss over gradient descent iteration
show_loss = True

################
# create training data
################

m_d1 = (0, 0)
m_d2 = (1, 1)
m_d3 = (0, 1)
m_d4 = (1, 0)

sig = 0.05
s_d1 = sig**2 * np.eye(2)

d1 = np.random.multivariate_normal(m_d1, s_d1, num_d)
d2 = np.random.multivariate_normal(m_d2, s_d1, num_d)
d3 = np.random.multivariate_normal(m_d3, s_d1, num_d)
d4 = np.random.multivariate_normal(m_d4, s_d1, num_d)

# training data, and has shape (4*num_d,2)
x_train_d = np.vstack((d1, d2, d3, d4))
# training data lables, and has shape (4*num_d,1)
y_train_d = np.vstack(
    (np.zeros((2 * num_d, 1), dtype="uint8"), np.ones((2 * num_d, 1), dtype="uint8"))
)


xor_model = MiniBatchSgdNNClassifier(
    layers=[
        LinearLayer(2, 4, std=1),
        SigmoidLayer(),
        LinearLayer(4, 2, std=1),
        SoftmaxLayer(),
    ],
    loss_func=CrossEntropyLoss(),
    lr=lr,
    max_epoch=num_gd_step,
    batch_size=batch_size,
)

loss_per_update, _ = xor_model.train(Dataset(x_train_d, y_train_d))
print(np.sum(loss_per_update < 1e-3))  # losses 모든 값을 채웠는지 검증

# set show_loss to True to plot the loss over gradient descent iterations
if show_loss:
    plt.figure(1)
    plt.grid()
    plt.plot(range(num_gd_step), loss_per_update, label="loss")
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("cross entropy loss")
    plt.show()


################
# training done
# now testing

num_test = 100
# num_test = 1

for j in range(num_test):
    predicted = np.ones((4,))

    # dispersion of test data
    sig_t = 1e-2

    # generate test data
    # generate 4 samples, each sample nearby (1,1), (0,0), (1,0), (0,1) respectively
    t11 = np.random.multivariate_normal((1, 1), sig_t**2 * np.eye(2), 1)
    t00 = np.random.multivariate_normal((0, 0), sig_t**2 * np.eye(2), 1)
    t10 = np.random.multivariate_normal((1, 0), sig_t**2 * np.eye(2), 1)
    t01 = np.random.multivariate_normal((0, 1), sig_t**2 * np.eye(2), 1)

    test_data = np.vstack((t11, t00, t10, t01))

    predicted = xor_model.predict(test_data).reshape(-1)

    print("total predicted labels:", predicted.astype("uint8"))

    accuracy += (
        (predicted[0] == 0)
        & (predicted[1] == 0)
        & (predicted[2] == 1)
        & (predicted[3] == 1)
    )

    if (j + 1) % 10 == 0:
        print("test iteration:", j + 1)

print("accuracy:", accuracy / num_test * 100, "%")
