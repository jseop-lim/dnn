# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
import numpy as np
from matplotlib import pyplot as plt

from dnn import layers
from dnn.losses import CrossEntropyLoss

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


# model = {
#     "lin1": layers.LinearLayer(2, 4, 1),
#     "sigmoid": layers.SigmoidLayer(),
#     "lin2": layers.LinearLayer(4, 2, 1),
#     "softmax": layers.SoftmaxLayer(),
#     "ce": layers.CrossEntropyLayer(),
# }

linear_layer1 = layers.LinearLayer(2, 4, std=1)
sigmoid_layer = layers.SigmoidLayer()
linear_layer2 = layers.LinearLayer(4, 2, std=1)
softmax_layer = layers.SoftmaxLayer()
ce_loss = CrossEntropyLoss()


loss_per_epoch = np.zeros(num_gd_step)

for i in range(num_gd_step):
    # feed forward
    z_lin = linear_layer1.forward(x_train_d)
    z = sigmoid_layer.forward(z_lin)
    y_lin = linear_layer2.forward(z)
    y = softmax_layer.forward(y_lin)
    loss_per_epoch[i] = ce_loss.forward(y, y_train_d)

    # error backpropagation
    error_ce = ce_loss.backward()
    error_softmax = softmax_layer.backward(error_ce)
    error_lin2 = linear_layer2.backward(error_softmax)
    error_sigmoid = sigmoid_layer.backward(error_lin2)
    error_lin1 = linear_layer1.backward(error_sigmoid)

    # gradient descent
    linear_layer1.update_weights(lr)
    linear_layer2.update_weights(lr)

    if (i + 1) % 2000 == 0:
        print(f"Epoch {i + 1}: Loss = {loss_per_epoch[i]}")


# set show_loss to True to plot the loss over gradient descent iterations
if show_loss:
    plt.figure(1)
    plt.grid()
    plt.plot(range(num_gd_step), loss_per_epoch, label="loss")
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

    # predicting label for test sample nearby (1,1)
    l1_out = linear_layer1.forward(t11)
    a1_out = sigmoid_layer.forward(l1_out)
    l2_out = linear_layer2.forward(a1_out)
    smax_out = softmax_layer.forward(l2_out)
    predicted[0] = np.argmax(smax_out)
    print("softmax out for (1,1)", smax_out, "predicted label:", int(predicted[0]))

    # predicting label for test sample nearby (0,0)
    l1_out = linear_layer1.forward(t00)
    a1_out = sigmoid_layer.forward(l1_out)
    l2_out = linear_layer2.forward(a1_out)
    smax_out = softmax_layer.forward(l2_out)
    predicted[1] = np.argmax(smax_out)
    print("softmax out for (0,0)", smax_out, "predicted label:", int(predicted[1]))

    # predicting label for test sample nearby (1,0)
    l1_out = linear_layer1.forward(t10)
    a1_out = sigmoid_layer.forward(l1_out)
    l2_out = linear_layer2.forward(a1_out)
    smax_out = softmax_layer.forward(l2_out)
    predicted[2] = np.argmax(smax_out)
    print("softmax out for (1,0)", smax_out, "predicted label:", int(predicted[2]))

    # predicting label for test sample nearby (0,1)
    l1_out = linear_layer1.forward(t01)
    a1_out = sigmoid_layer.forward(l1_out)
    l2_out = linear_layer2.forward(a1_out)
    smax_out = softmax_layer.forward(l2_out)
    predicted[3] = np.argmax(smax_out)
    print("softmax out for (0,1)", smax_out, "predicted label:", int(predicted[3]))

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
