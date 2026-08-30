import numpy as np


class Neuron:
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias

    def z(self, x: np.ndarray) -> float:
        return np.dot(self.weights, x) + self.bias



class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sigmoid:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class Softmax:
    def forward(self, x: np.ndarray) -> np.ndarray:
        # subtract max for numerical stability before exponentiating
        e = np.exp(x - np.max(x))
        return e / e.sum()


class Identity:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


# Mean Squared Error
class MSE:
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)


# Categorical Cross-Entropy
class CrossEntropy:
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(y_pred))


# Usually used for binary classification -> dimension 1 (single output)
class BinaryCrossEntropy:
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


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

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.array([neuron.z(x) for neuron in self.neurons])


class NeuralNet:
    def __init__(
        self,
        layers: list[int],
        seed: int,
        hidden_activation: type = ReLU,
        output_activation: type = Softmax,
    ):
        steps = []
        for i, (n_in, n_out) in enumerate(zip(layers, layers[1:])):
            steps.append(Layer(n_in=n_in, n_out=n_out, seed=seed + i))
            if i < len(layers) - 2:
                steps.append(hidden_activation())
        steps.append(output_activation())
        self.sequence = Sequence(steps)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.sequence.forward(x)


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
    layers = [2, 1]
    dnn = NeuralNet(layers=layers, seed=seed, hidden_activation=Sigmoid, output_activation=Sigmoid)
    for x, y in [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]:
        x_array = np.array(x)
        y_pred = dnn.forward(x_array)
        print(f"input: {x_array}, target: {y}, predicted: {y_pred}")
        loss = BinaryCrossEntropy().loss(y_pred, np.array([y]))
        print(f"loss: {loss}")

if __name__ == "__main__":
    # testRandom(layers=[4, 8, 8, 4], seed=42)
    testXOR(seed=42)

