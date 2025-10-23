from typing import Callable
import numpy as np
import numpy.typing as npt

_ActivationFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
def _get_activation(activation: str) -> tuple[_ActivationFunction, _ActivationFunction]:
    match activation:
        case 'sigmoid':
            def sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return 1 / (1 + np.exp(-z))

            def sigmoid_prime(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                s = sigmoid(z)
                return s * (1 - s)

            return sigmoid, sigmoid_prime
        case _:
            raise ValueError(f"No such activation function: {activation}")

class MLP:
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_layers: list[int] | None = None, 
                 weights: list[npt.NDArray[np.float64]] | None = None, 
                 biases: list[npt.NDArray[np.float64]] | None = None, 
                 learning_rate: float = 0.1, 
                 activation: str = 'sigmoid'):

        self.layers_sizes: list[int] = [input_size] + (hidden_layers if hidden_layers else []) + [output_size]
        self.size: int = len(self.layers_sizes)
        self._weight_shapes: list[tuple[int, int]] = list(zip(self.layers_sizes[1:], self.layers_sizes[:-1]))
        self._bias_shapes: list[int] = [size for size in self.layers_sizes[1:]]

        self.learning_rate: float = learning_rate
        self.activation: str = activation

        self._af: _ActivationFunction
        self._af_p: _ActivationFunction
        self._af, self._af_p = _get_activation(activation)

        self._a: list[npt.NDArray[np.float64]] = [np.zeros(size) for size in self.layers_sizes]
        self._a_p: list[npt.NDArray[np.float64]] = [np.zeros(size) for size in self.layers_sizes[1:]]

        self.weights: list[npt.NDArray[np.float64]]
        if weights:
            self._validate_weights(weights)
            self.weights = weights
        else:
            self.weights = self._init_random_weights()
            
        self.biases: list[npt.NDArray[np.float64]]
        if biases:
            self._validate_biases(biases)
            self.biases = biases
        else:
            self.biases = [np.zeros(shape) for shape in self._bias_shapes]
        
    def _feed_forward(self):
        for i in range(0, self.size - 1):
            dot = self.weights[i] @ self._a[i]
            z = dot + self.biases[i]
            self._a[i + 1] = self._af(z)
            self._a_p[i] = self._af_p(z)

    def _backpropagation(self, ground_truth: npt.NDArray[np.float64]):
        deltas: list[npt.NDArray[np.float64]] = [np.zeros(shape) for shape in self._bias_shapes]
        L = self.size - 2
        deltas[L] = self._a_p[L] * (ground_truth - self._a[-1])
        for i in reversed(range(L)):
            deltas[i] = self._a_p[i] * np.sum(self.weights[i + 1] @ deltas[i + 1])
        for i in range(self.size - 1):
            self.weights[i] += self.learning_rate * self._a[i - 1] * deltas[i]
    
    def _validate_weights(self, weights: list[npt.NDArray[np.float64]]):
        if len(weights) != len(self._weight_shapes):
            raise ValueError(f"Expected {len(self._weight_shapes)} weight arrays, got {len(weights)}")
            
        for w, shape in zip(weights, self._weight_shapes):
            if w.ndim != 2:
                raise ValueError(f"Weights must be 2D arrays. Got shape {w.shape}")
            if w.shape != shape:
                raise ValueError(f"Weight shape mismatch. Expected {shape}, got {w.shape}")

    def _validate_biases(self, biases: list[npt.NDArray[np.float64]]):
        if len(biases) != len(self._bias_shapes):
            raise ValueError(f"Expected {len(self._bias_shapes)} bias arrays, got {len(biases)}")
        
        for b, shape in zip(biases, self._bias_shapes):
            if b.ndim != 1:
                raise ValueError(f"Biases must be 1D arrays. Got shape {b.shape}")
            if b.shape[0] != shape:
                raise ValueError(f"Bias shape mismatch. Expected ({shape},), got {b.shape}")

    def _init_random_weights(self) -> list[npt.NDArray[np.float64]]:
        rng = np.random.default_rng()
        weights: list[npt.NDArray[np.float64]] = []
        for (rows, cols) in self._weight_shapes:
            weights.append(rng.standard_normal((rows, cols)) * 0.01)
        return weights

    def _init_random_biases(self) -> list[npt.NDArray[np.float64]]:
        rng = np.random.default_rng()
        biases: list[npt.NDArray[np.float64]] = []
        for rows in self._bias_shapes:
            biases.append(rng.standard_normal(rows) * 0.01)
        return biases
