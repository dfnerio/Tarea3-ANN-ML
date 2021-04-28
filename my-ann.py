
import numpy as np
import pandas as pd


def fit(X, y, alpha, reg_factor, epochs, hidden, activations, theta):
    for i in range(epochs):
        for i in range(m):
            X_i = X[:, i]
            X_i = X_i.reshape(-1, 1)
            y_i = y[i, :].T
            forward(X_i, hidden, activations, theta)
            backprop(X_i, y_i, hidden, activations, theta, alpha, reg_factor)


def z(input, Theta):
    return np.dot(Theta, input)


def activation(z):
    return 1 / (1 + np.exp(-z))


def initialize_weights(X, num_classes, hidden):
    weights = []
    out_neurons = num_classes
    n = X.shape[0]

    hidden_layers = len(hidden)

    for i in range(0, hidden_layers):
        m = hidden[i]
        cols = n + 1
        weights_layer = np.random.rand(m, cols)
        weights.append(weights_layer)
        n = m

    weights.append(np.random.rand(out_neurons, n + 1))
    return weights


def initialize_activations(X, out_neurons, hidden):
    activations = []
    a1 = X
    biases = np.ones(a1.shape[1])
    a1 = np.vstack((biases, a1))
    activations.append(a1)

    for i in range(len(hidden)):
        ai = np.zeros((hidden[i] + 1, 1))
        for j in range(0, ai.shape[0]):
            ai[j, 0] = 1.0
        activations.append(ai)

    activations.append(np.zeros((out_neurons, 1)))
    return activations


def forward(X, hidden, activations, theta):
    m = X.shape[1]
    for i in range(m):
        for j in range(0, len(hidden)):
            ai = activations[i]
            znext = z(ai, theta[1])
            anext = activation(znext)
            activations[i+1][1:] = anext

        ai = activations[i+1]
        znext = z(ai, theta[i+1])
        anext = activation(znext)
        activations[i+2] = anext


def backprop(X, y, hidden, activations, theta, alpha, reg):
    m = len(y)
    delta = []

    ypred = activations[-1]
    deltai = ypred - y

    delta.append(deltai)

    start = len(activations) - 1

    for i in range(start, 1, -1):
        theta_prev = theta[i-1][:, 1:]
        tmp = np.dot(theta_prev.T, deltai)
        acts = activations[i-1][1:]
        deltai = tmp * (acts * (1 - acts))
        delta.append(deltai)

    delta.reverse()

    n_clases = theta[-1].shape[0]
    Delta = initialize_weights(X, n_clases, hidden)
    start = len(delta) - 1

    for i in range(start, -1, -1):
        current_activations = activations[i][1:, :]
        next_delta = delta[i]
        activations_x_delta = np.dot(current_activations, next_delta.T)
        Delta[i][:, 1:] = activations_x_delta.T

    D = [x/m for x in Delta]

    for i in range(len(D)):
        d = D[i]
        t = theta[i]
        d[:, 1:] += reg * t[:, 1:]

    for i in range(len(theta)):
        t = theta[i]
        d = D[i]
        t[:, 1:] = t[:, 1:] - (alpha * d[:, 1:])


def predict(X, theta):
    ai = X
    for i in range(len(theta)):
        biases = np.ones(ai.shape[1])
        ai = np.vstack((biases, ai))
        znext = z(ai, theta[i])
        ai = activation(znext)
    return ai


def get_config_for_example():
    X = np.array([[0.05], [0.10]])
    y = np.array([[0.01], [0.99]])

    hidden = [2]
    # el dos es por las neuronas en la capa de salida
    theta = initialize_weights(X, 2, hidden)

    theta_1 = theta[0]
    theta_1[0, 0] = 0.35
    theta_1[0, 1] = 0.15
    theta_1[0, 2] = 0.20

    theta_1[1, 0] = 0.35
    theta_1[1, 1] = 0.25
    theta_1[1, 2] = 0.30

    #####
    theta_2 = theta[1]
    theta_2[0, 0] = 0.60
    theta_2[0, 1] = 0.40
    theta_2[0, 2] = 0.45

    theta_2[1, 0] = 0.60
    theta_2[1, 1] = 0.50
    theta_2[1, 2] = 0.55

    return (X, y, hidden, theta)


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

# main


data = pd.read_csv("blobs.csv")
X = data.iloc[:, :-1].to_numpy().T   # all but last column of labels
y = data.iloc[:, -1].to_numpy()      # the last col is class
y = y.reshape(-1, 1)  # to get an mx1 array and not (m,)
unique_classes = len(np.unique(y))
y = get_one_hot(y, unique_classes)

hidden = [2, 5]
theta = initialize_weights(X, unique_classes, hidden)
activations = initialize_activations(X, unique_classes, hidden)

m = X.shape[1]
epochs = 2000

fit(X, y, 0.5, 0, epochs, hidden, activations, theta)

to_pred = X[:, 0]
y_i = y[0, :].T
to_pred = to_pred.reshape(-1, 1)
y_pred = predict(to_pred, theta)
print('{} pred as {}, should be {}'.format(to_pred.T, y_pred.T, y_i.T))


# (X, y, hidden, theta) = get_config_for_example()
# activations = initialize_activations(X, out_neurons=2, hidden=hidden)
# fit(X, y, 0.5, 0, 10000, hidden, activations, theta)
# y_pred = predict(X, theta)
# print('{} pred as {}, should be {}'.format(X.T, y_pred.T, y.T))
