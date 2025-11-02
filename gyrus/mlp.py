from typing import Callable
import numpy as np
import numpy.typing as npt

_ActivationFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
def _get_activation(activation: str) -> tuple[_ActivationFunction, _ActivationFunction]:
    match activation:
        case 'sigmoid':
            def sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                z = np.clip(z, -500, 500)
                return 1 / (1 + np.exp(-z))

            def sigmoid_prime(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                s = sigmoid(z)
                return s * (1 - s)

            return sigmoid, sigmoid_prime
        case 'tanh':
            def tanh(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return np.tanh(z)
            
            def tanh_prime(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return 1.0 - np.tanh(z) ** 2

            return tanh, tanh_prime
        case 'relu':
            def relu(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return np.maximum(0, z)

            def relu_prime(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                return (z > 0) * np.float64(1.0)

            return relu, relu_prime
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
        self._activation: str = activation

        self._af: _ActivationFunction
        self._af_p: _ActivationFunction
        self._af, self._af_p = _get_activation(activation)

        self._a: list[npt.NDArray[np.float64]] = [np.zeros(size) for size in self.layers_sizes]
        self._a_p: list[npt.NDArray[np.float64]] = [np.zeros(size) for size in self.layers_sizes[1:]]

        self.weights: list[npt.NDArray[np.float64]]
        if weights:
            self.weights = weights
        else:
            self.weights = self._init_random_weights()
        self._validate_weights()

        self.biases: list[npt.NDArray[np.float64]]
        if biases:
            self.biases = biases
        else:
            self.biases = self._init_random_biases()
        self._validate_biases()

    @property
    def activation(self) -> str:
        return self.activation

    @activation.setter
    def activation(self, new_value: str):
        _ = _get_activation(new_value)
        self._activation = new_value

    def feed_forward(self, input_data: npt.NDArray[np.float64]):
        if input_data.shape[0] != self.layers_sizes[0]:
             raise ValueError(f"Input data shape {input_data.shape} does not match input size {self.layers_sizes[0]}")
        self._a[0] = input_data
        for i in range(0, self.size - 1):
            dot = self.weights[i] @ self._a[i]
            z = dot + self.biases[i]
            self._a[i + 1] = self._af(z)
            self._a_p[i] = self._af_p(z)

    def backpropagation(self, ground_truth: npt.NDArray[np.float64]):
        if ground_truth.shape[0] != self.layers_sizes[-1]:
             raise ValueError(f"Ground truth shape {ground_truth.shape} does not match output size {self.layers_sizes[-1]}")
            
        deltas: list[npt.NDArray[np.float64]] = [np.zeros(shape) for shape in self._bias_shapes]
        L = self.size - 2
        
        deltas[L] = self._a_p[L] * (ground_truth - self._a[-1]) * 2
        
        for i in reversed(range(L)):
            deltas[i] = (self.weights[i + 1].T @ deltas[i + 1]) * self._a_p[i]
            
        for i in range(self.size - 1):
            self.weights[i] += self.learning_rate * np.outer(deltas[i], self._a[i])
            self.biases[i] += self.learning_rate * deltas[i]

    def fit(self, 
            X: npt.NDArray[np.float64], 
            y: npt.NDArray[np.float64], 
            epochs: int, 
            print_loss_every: int = 0):
        
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true_item in zip(X, y):
                
                y_true = np.array([y_true_item]) if not isinstance(y_true_item, (np.ndarray, list, tuple)) else np.asarray(y_true_item)
                
                self.feed_forward(x)
                
                loss = np.sum((y_true - self._a[-1]) ** 2)
                total_loss += loss

                self.backpropagation(y_true)
            
            if print_loss_every > 0 and (epoch + 1) % print_loss_every == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if X.ndim == 1:
            X_vals = [X]
        else:
            X_vals = X
            
        predictions = []
        for x in X_vals:
            self.feed_forward(x)
            predictions.append(self._a[-1].copy())
            
        return np.array(predictions)
   
    def _validate_weights(self):
        if len(self.weights) != len(self._weight_shapes):
            raise ValueError(f"Expected {len(self.weights)} weight arrays, got {len(self.weights)}")
            
        for w, shape in zip(self.weights, self._weight_shapes):
            if w.ndim != 2:
                raise ValueError(f"Weights must be 2D arrays. Got shape {w.shape}")
            if w.shape != shape:
                raise ValueError(f"Weight shape mismatch. Expected {shape}, got {w.shape}")

    def _validate_biases(self):
        if len(self.biases) != len(self._bias_shapes):
            raise ValueError(f"Expected {len(self.biases)} bias arrays, got {len(self.biases)}")
        
        for b, shape in zip(self.biases, self._bias_shapes):
            if b.ndim != 1:
                raise ValueError(f"Biases must be 1D arrays. Got shape {b.shape}")
            if b.shape[0] != shape:
                raise ValueError(f"Bias shape mismatch. Expected ({shape},), got {b.shape}")

    def _init_random_weights(self) -> list[npt.NDArray[np.float64]]:
        rng = np.random.default_rng()
        weights: list[npt.NDArray[np.float64]] = []
        for (rows, cols) in self._weight_shapes:
            limit: np.float64 = np.sqrt(6 / (rows + cols))
            weights.append(rng.uniform(-limit, limit, (rows, cols)))
        return weights

    def _init_random_biases(self) -> list[npt.NDArray[np.float64]]:
        rng = np.random.default_rng()
        biases: list[npt.NDArray[np.float64]] = []
        for rows in self._bias_shapes:
            biases.append(rng.standard_normal(rows) * 0.01)
        return biases

def show_results(model: MLP, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    raw_predictions = model.predict(X)
    
    predicted_classes = (raw_predictions > 0.5) * 1
    
    is_correct = (predicted_classes == y)
    
    print(f"{'Input':<12} | {'Expected':<10} | {'Predicted':<10} | {'Raw':<10} | Status")
    print("-" * 60)

    for i in range(len(X)):
        input_str = str(X[i])

        expected_str = str(y[i].item())
        predicted_str = str(predicted_classes[i].item())
        raw_str = f"{raw_predictions[i].item():.4f}"
        
        status = "✅ Right" if is_correct[i] else "❌ Wrong"
        
        print(f"{input_str:<12} | {expected_str:<10} | {predicted_str:<10} | {raw_str:<10} | {status}")

    correct_count = np.sum(is_correct)
    total_count = len(y)
    accuracy = (correct_count / total_count) * 100
    
    print("-" * 60)
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count} correct)")

def show_results_multiclass(model: MLP, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
    raw_predictions = model.predict(X)
    
    predicted_classes = np.argmax(raw_predictions, axis=1)
    
    predicted_confidences = np.max(raw_predictions, axis=1)
    
    expected_classes = np.argmax(y, axis=1)
    
    is_correct = (predicted_classes == expected_classes)
    
    print(f"{'Input':<25}|{'Expected':<10}|{'Predicted':<10}|{'Confidence':<12}|Status")
    print("-" * 75)

    for i in range(len(X)):
        input_str = str(X[i])
        expected_str = str(expected_classes[i])
        predicted_str = str(predicted_classes[i])
        confidence_str = f"{predicted_confidences[i]:.4f}"
        status = "✅ Right" if is_correct[i] else "❌ Wrong"
        
        print(f"{input_str:<25}|{expected_str:<10}|{predicted_str:<10}|{confidence_str:<12}|{status}")

    correct_count = np.sum(is_correct)
    total_count = len(y)
    accuracy = (correct_count / total_count) * 100
    
    print("-" * 75)
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count} correct)")
