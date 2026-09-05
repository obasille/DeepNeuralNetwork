from typing import NamedTuple

import numpy as np


class Neuron:
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias

    def z(self, x: np.ndarray) -> float:
        return np.dot(self.weights, x) + self.bias


class ReLU:
    def __init__(self):
        self.last_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = np.maximum(0, x)
        return self.last_output

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid:
    def __init__(self):
        self.last_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = 1 / (1 + np.exp(-x))
        return self.last_output

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)


class Softmax:
    def __init__(self):
        self.last_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # subtract max for numerical stability before exponentiating
        e = np.exp(x - np.max(x))
        self.last_output = e / e.sum()
        return self.last_output

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)  # Note: This is not the full Jacobian, just the diagonal


class Identity:
    def __init__(self):
        self.last_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = x
        return self.last_output

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


# Mean Squared Error
class MSE:
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


# Categorical Cross-Entropy
class CrossEntropy:
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(y_pred))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # clip to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -y_true / y_pred


# Usually used for binary classification -> dimension 1 (single output)
class BinaryCrossEntropy:
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # clip to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


class Sequence:
    def __init__(self, steps: list):
        self.steps = steps

    def forward(self, x: np.ndarray) -> np.ndarray:
        for step in self.steps:
            x = step.forward(x)
        return x


class Layer:
    def __init__(self, n_in: int, n_out: int, seed: int):
        rng = np.random.default_rng(seed)
        self.neurons = [
            Neuron(weights=rng.standard_normal(n_in), bias=rng.standard_normal())
            for _ in range(n_out)
        ]
        self.last_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = np.array([neuron.z(x) for neuron in self.neurons])
        return self.last_output

    # W shape (output_dim, input_dim)
    def get_weights(self) -> np.ndarray:
        return np.array([neuron.weights for neuron in self.neurons])

    def set_weights(self, weights: np.ndarray):
        for neuron, w in zip(self.neurons, weights):
            neuron.weights = w

    def get_bias(self) -> np.ndarray:
        return np.array([neuron.bias for neuron in self.neurons])

    def set_bias(self, biases: np.ndarray):
        for neuron, b in zip(self.neurons, biases):
            neuron.bias = b


class LayerActivationPair(NamedTuple):
    layer: Layer
    activation: object


class NeuralNet:
    def __init__(
        self,
        layers: list[int],
        seed: int,
        hidden_activation: type = ReLU,
        output_activation: type = Softmax,
    ):
        steps = []
        pairs = []
        for i, (n_in, n_out) in enumerate(zip(layers, layers[1:])):
            print(f"Creating layer {i} with {n_in} inputs and {n_out} outputs")
            layer = Layer(n_in=n_in, n_out=n_out, seed=seed + i)
            steps.append(layer)
            activation = hidden_activation() if i < len(layers) - 2 else output_activation()
            steps.append(activation)
            pairs.append(LayerActivationPair(layer, activation))
        self.sequence = Sequence(steps)
        self.pairs = pairs  # Store the layer-activation pairs for later use

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.sequence.forward(x)

    # This method returns a list of (layer, activation) tuples
    def get_layer_activation_pairs(self):
        return self.pairs


def testRandom(layers: list[int], seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(layers[0])        # input activations
    y_true = rng.standard_normal(layers[-1])  # target activations

    dnn = NeuralNet(layers=layers, seed=seed)
    y_pred = dnn.forward(x)

    print("input:     ", x)
    print("target:    ", y_true)
    print("predicted: ", y_pred)


def testXOR(seed: int):
    layers = [2, 2, 1]
    dnn = NeuralNet(layers=layers, seed=seed, hidden_activation=Sigmoid, output_activation=Sigmoid)
    Loss = MSE;
    for epoch in range(20000):
        for x, y in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
            x_array = np.array(x)
            print(f"== Epoch {epoch}, input: {x_array}, target: {y} ==")
            y_pred = dnn.forward(x_array)
            y_true = np.array([y])
            print(f" * input: {x_array}, target: {y_true} (shape: {y_true.shape}), predicted: {y_pred} (shape: {y_pred.shape})")
            loss = Loss().loss(y_pred, y_true)
            print(f" * loss: {loss}")
            pairs = dnn.get_layer_activation_pairs()
            # Create a list to hold the deltas for each layer
            deltas = [None] * len(pairs)
            # print(f" * Layer-Activation pairs: {len(pairs)}")
            for i_reverse, (layer, activation) in enumerate(reversed(pairs)):
                i = len(pairs) - 1 - i_reverse
                # print(f" ** Layer {i}: {type(layer).__name__}, Activation: {type(activation).__name__}")
                if i_reverse == 0:
                    # Last layer
                    delta = \
                        Loss().derivative(y_pred, y_true) * \
                        activation.derivative(layer.last_output)
                else:
                    delta = \
                        (pairs[i + 1].layer.get_weights().T @ deltas[i + 1]) * \
                        activation.derivative(layer.last_output)
                deltas[i] = delta
            # Apply the deltas to update weights and biases
            for i, (layer, activation) in enumerate(pairs):
                delta = deltas[i]
                if i == 0:
                    # First layer
                    dloss = np.outer(delta, x_array)
                else:
                    dloss = np.outer(delta, pairs[i - 1].activation.last_output)
                # print(f" ** delta at layer {i}: {delta}, gradient at layer {i}: {dloss}")
                layer.set_weights(layer.get_weights() - 0.1 * dloss)  # simple gradient descent step
                layer.set_bias(layer.get_bias() - 0.1 * delta)  # update bias


if __name__ == "__main__":
    # testRandom(layers=[4, 8, 8, 4], seed=42)
    testXOR(seed=42)

