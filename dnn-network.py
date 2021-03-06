import numpy as np
import pickle
import gzip


class Network(object):


    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)
        self.biases = [
            np.random.randn(y, 1) for y in sizes[1:]
        ]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]


    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def evaluate(self, x, y):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        x = np.array([self.feed_forward(l) for l in x])

        num_x = x.shape[0]

        loss = np.sum(np.linalg.norm(x-y)**2) / (2. * num_x)

        x = np.argmax(x, axis=1).reshape((-1,))
        y = np.argmax(y, axis=1).reshape((-1,))

        match = np.sum(np.int8(x == y))

        acc = match / num_x * 100

        return loss, acc, match, num_x


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


    def SGD(self, x, y, epochs, batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent. ``x`` represents the training input data.
        ``y`` is the corresponding output data.  The other non-optional
        parameters are self-explanatory.  If ``test_data`` is provided
        then the network will be evaluated against the test data after
        each epoch, and partial progress printed out.  This is useful
        for tracking progress, but slows things down substantially."""

        # The training_data is a list of tuples
        # (x, y) representing the training inputs
        # and the desired outputs
        training_data = np.array(list(zip(x, y)))
        n = training_data.shape[0]

        if test_data is not None:
            tx, ty = test_data
            n_test = tx.shape[0]

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+batch_size]
                for k in range(0, n, batch_size)
            ]
            for mini_batch in mini_batches:
                #print('.', end='', flush=True)
                self.update_mini_batch(mini_batch, eta)
            #print('') # print new line

            loss, acc, match, total = self.evaluate(x, y)
            print(
                f'Training dataset: {match} / {total},  '
                f'loss: {loss:.7f},  accuracy: {acc:.2f}%'
            )

            if test_data is not None:
                loss, acc, match, total = self.evaluate(tx, ty)
                print(
                    f'Epoch {j:02d}: test dataset {match} / {total},  '
                    f'test loss: {loss:.7f},  test accuracy: {acc:.2f}%'
                )
            else:
                print(f'Epoch {j:02d} complete')


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        n_batch = len(mini_batch)
        self.weights = [
            w - (eta / n_batch) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / n_batch) * nb for b, nb in zip(self.biases, nabla_b)
        ]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feed forward
        activation = x
        activations = [x] # store all the activations, layer by layer
        zs = [] # store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = (
            self.cost_derivative(activations[-1], y) *
            sigmoid_prime(zs[-1])
        )
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            delta = (
                np.dot(self.weights[-l+1].transpose(), delta) *
                sigmoid_prime(zs[-l])
            )
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w


    def load_weights(self, fpath):
        with gzip.open(fpath, 'rb') as f:
            self.biases, self.weights = pickle.load(f, encoding='latin1')


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)


def to_categorical(x):
    """Turns the input data into the categorical vector with the corrsponding
    probability of 1.0 in the desired category."""
    num_categories = np.unique(x).shape[0]
    vector = np.eye(num_categories, dtype='uint8')[x]
    return vector.reshape((-1, 10, 1))


def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f, encoding='latin1')

    return train_data, valid_data, test_data


if __name__ == '__main__':
    sizes = [784, 100, 10]

    dnn = Network(sizes)

    dnn.load_weights('dnn-model-weights.pkl.gz')

    print('Test feed forward with an empty vector')
    inpzero = np.zeros((784, 1))
    res = dnn.feed_forward(inpzero)
    print('Resulted output shape:', res.shape)
    print('Resulted vector:', np.array2string(
        res.reshape((-1,)),
        separator=', ',
        formatter={'float_kind': lambda x: f'{x:0.2f}'}
    ))

    # load data
    print('Loading training, test and validation data sets')
    train_data, valid_data, test_data = load_data()

    x_train, y_train = train_data
    x_test, y_test = test_data

    xx_train = x_train.reshape((-1, 784, 1))
    xx_test = x_test.reshape((-1, 784, 1))

    yy_train = to_categorical(y_train)
    yy_test = to_categorical(y_test)

    print('Apply stochastic gradient decent')
    dnn.SGD(
        x=xx_train,
        y=yy_train,
        epochs=10,
        batch_size=10,
        eta=3.0,
        test_data=(xx_test, yy_test)
    )
