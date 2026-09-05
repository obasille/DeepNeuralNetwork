import random
from typing import NamedTuple
import numpy as np
import csv


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
        self.last_derivative = (x > 0).astype(float)
        return self.last_output


class Sigmoid:
    def __init__(self):
        self.last_output: np.ndarray | None = None
        self.last_derivative: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = 1 / (1 + np.exp(-x))
        self.last_derivative = self.last_output * (1 - self.last_output)
        return self.last_output


class Softmax:
    def __init__(self):
        self.last_output: np.ndarray | None = None
        self.last_derivative: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # subtract max for numerical stability before exponentiating
        e = np.exp(x - np.max(x))
        self.last_output = e / e.sum()
        # Softmax derivative is not implemented because full Jacobian is complex.
        # Use CrossEntropy loss with Softmax for proper backpropagation.
        # self.last_derivative = self.last_output * (1 - self.last_output)  # Note: This is not the full Jacobian, just the diagonal
        return self.last_output


class Identity:
    def __init__(self):
        self.last_output: np.ndarray | None = None
        self.last_derivative: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = x
        self.last_derivative = np.ones_like(x)
        return self.last_output


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
    def __init__(self, n_in: int, n_out: int, seed: int, Activation: type):
        rng = np.random.default_rng(seed)
        if Activation == ReLU:
            scale = np.sqrt(2.0 / n_in)   # He
        elif Activation in (Sigmoid, Softmax, Identity):
            scale = np.sqrt(1.0 / n_in)   # Xavier
        else:
            raise ValueError(f"No initialization strategy defined for {Activation}")

        self.neurons = [
            Neuron(weights=rng.standard_normal(n_in) * scale, bias=0.0)
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
        seed: int,
        layer_sizes: list[int],
        hidden_activation: type = ReLU,
        output_activation: type = Softmax,
    ):
        steps = []
        pairs = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[0:-1], layer_sizes[1:])):
            print(f"Creating layer {i} with {n_in} inputs and {n_out} outputs")
            Activation = hidden_activation if i < len(layer_sizes) - 2 else output_activation
            layer = Layer(n_in=n_in, n_out=n_out, seed=seed + i, Activation = Activation)
            activation = Activation()
            steps.append(layer)
            steps.append(activation)
            pairs.append(LayerActivationPair(layer, activation))
        self.sequence = Sequence(steps)
        self.pairs = pairs  # Store the layer-activation pairs for later use

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.sequence.forward(x)

    # This method returns a list of (layer, activation) tuples
    def get_layer_activation_pairs(self):
        return self.pairs

    
def train(dnn: NeuralNet, training_data: list[tuple[list[float], float | list[float]]], Loss, seed: int, num_epochs: int = 20000, learning_rate: float = 0.1, target_loss: float = 0.001):
    rng = random.Random(seed)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_data = list(training_data)
        rng.shuffle(epoch_data)
        for x, y in epoch_data:
            x_array = np.array(x)
            y_pred = dnn.forward(x_array)
            y_true = np.array(y if isinstance(y, list) else [y])
            loss = Loss().loss(y_pred, y_true)
            epoch_loss += loss
            pairs = dnn.get_layer_activation_pairs()
            # Create a list to hold the deltas for each layer
            deltas = [None] * len(pairs)
            for i_reverse, (layer, activation) in enumerate(reversed(pairs)):
                i = len(pairs) - 1 - i_reverse
                if i_reverse == 0:
                    # Last layer
                    if isinstance(activation, Softmax) and Loss == CrossEntropy:
                        delta = y_pred - y_true
                    else:
                        delta = Loss().derivative(y_pred, y_true) * activation.last_derivative
                else:
                    next_weights = pairs[i + 1].layer.get_weights()
                    delta = (next_weights.T @ deltas[i + 1]) * activation.last_derivative
                deltas[i] = delta
            # Apply the deltas to update weights and biases
            for i, (layer, activation) in enumerate(pairs):
                delta = deltas[i]
                if i == 0:
                    # First layer
                    dloss = np.outer(delta, x_array)
                else:
                    dloss = np.outer(delta, pairs[i - 1].activation.last_output)
                layer.set_weights(layer.get_weights() - learning_rate * dloss)  # simple gradient descent step
                layer.set_bias(layer.get_bias() - learning_rate * delta)  # update bias
        avg_loss = epoch_loss / len(training_data)
        if (epoch + 1) % 500 == 0:
            print(f"Average loss after epoch {epoch + 1}: {avg_loss}")
        if avg_loss < target_loss:
            print(f"Training stopped early at epoch {epoch + 1} due to low loss: {avg_loss}")
            break


def check_accuracy_XOR(dnn: NeuralNet, test_data: list[tuple[list[float], float | list[float]]], threshold=0.5):
    def predict_class(y_pred):
        return (y_pred >= threshold).astype(int)
    for x, y in test_data:
        x_array = np.array(x)
        y_pred = predict_class(dnn.forward(x_array))
        print(f"Input: {x_array}, Target: {y}, Predicted: {y_pred}")
    correct_predictions = sum(predict_class(dnn.forward(np.array(x))) == y for x, y in test_data)
    accuracy = correct_predictions / len(test_data)
    return accuracy[0]


def check_accuracy(dnn: NeuralNet, test_data: list[tuple[list[float], float | list[float]]]):
    correct = 0
    for x, y in test_data:
        x_array = np.array(x)
        y_pred = dnn.forward(x_array)
        predicted_class = np.argmax(y_pred)
        true_class = np.argmax(y) if isinstance(y, list) else int(y)
        is_correct = predicted_class == true_class
        correct += is_correct
    accuracy = correct / len(test_data)
    return accuracy


def print_final_weights_and_biases(dnn: NeuralNet):
    for i, (layer, activation) in enumerate(dnn.get_layer_activation_pairs()):
        print(f"Layer {i} weights:\n{layer.get_weights()}")
        print(f"Layer {i} biases:\n{layer.get_bias()}")


def testXOR(seed: int):
    dnn = NeuralNet(
        seed=seed,
        layer_sizes=[2, 2, 1],
        hidden_activation=Sigmoid,
        output_activation=Sigmoid,
    )
    Loss = MSE;
    training_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    train(dnn, training_data, Loss, seed)
    accuracy = check_accuracy_XOR(dnn, training_data)
    print(f"Final accuracy on XOR problem: {accuracy * 100:.2f}%")
    print_final_weights_and_biases(dnn)


def testIris(seed: int):
    # Load the Iris dataset from its CSV file (headers are sepal_length,sepal_width,petal_length,petal_width,species)
    iris_data = []
    with open("iris.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iris_data.append(row)
    # Normalize each feature to mean 0, std 1
    for feature in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        values = np.array([float(row[feature]) for row in iris_data])
        mean = np.mean(values)
        std = np.std(values)
        for row in iris_data:
            row[feature] = (float(row[feature]) - mean) / std

    dnn = NeuralNet(
        seed=seed,
        layer_sizes=[4, 8, 3],
        # layer_sizes=[4, 6, 6, 3],
        hidden_activation=ReLU,
        output_activation=Softmax
    )
    Loss = CrossEntropy;
    # Training data is a list of tuples where each tuple contains a list of 4 floats (the features) and a list of 3 floats (the one-hot encoded target)
    training_data = [
        (
            [float(row["sepal_length"]), float(row["sepal_width"]), float(row["petal_length"]), float(row["petal_width"])],
            [1.0 if row["species"] == "setosa" else 0.0,
             1.0 if row["species"] == "versicolor" else 0.0,
             1.0 if row["species"] == "virginica" else 0.0]
        )
        for row in iris_data
    ]
    train(dnn, training_data, Loss, seed, learning_rate=0.01)
    accuracy = check_accuracy(dnn, training_data)
    print(f"Final accuracy on Iris problem: {accuracy * 100:.2f}%")
    # print_final_weights_and_biases(dnn)


if __name__ == "__main__":
    testIris(seed=42)
