import gyrus.mlp as mlp;
import numpy as np;
import numpy.typing as npt;
def main():
    X_perfect = np.array([
        [1, 1, 1, 1, 1, 1, 0],  # 0
        [0, 1, 1, 0, 0, 0, 0],  # 1
        [1, 1, 0, 1, 1, 0, 1],  # 2
        [1, 1, 1, 1, 0, 0, 1],  # 3
        [0, 1, 1, 0, 0, 1, 1],  # 4
        [1, 0, 1, 1, 0, 1, 1],  # 5
        [1, 0, 1, 1, 1, 1, 1],  # 6
        [1, 1, 1, 0, 0, 0, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1]   # 9
    ], dtype=np.float64)
    
    y_perfect = np.eye(10, dtype=np.float64)
    
    X_train = X_perfect
    y_train = y_perfect

    X_noisy = add_noise(X_perfect, noise_level=0.15)
    
    X_test = np.vstack((X_perfect, X_noisy))
    
    y_test = np.vstack((y_perfect, y_perfect))

    print('sigmoid')
    network = mlp.MLP(
        input_size=7,
        output_size=10,
        hidden_layers=[5],
        learning_rate=0.1,
        activation='sigmoid'
    )
    
    network.fit(X_train, y_train, epochs=5000, print_loss_every=1000)
    mlp.show_results_multiclass(network, X_test, y_test)

    print('relu')
    network = mlp.MLP(
        input_size=7,
        output_size=10,
        hidden_layers=[5],
        learning_rate=0.1,
        activation='relu'
    )
    
    network.fit(X_train, y_train, epochs=5000, print_loss_every=1000)
    mlp.show_results_multiclass(network, X_test, y_test)

    print('tanh')
    network = mlp.MLP(
        input_size=7,
        output_size=10,
        hidden_layers=[5],
        learning_rate=0.1,
        activation='tanh'
    )
    
    network.fit(X_train, y_train, epochs=5000, print_loss_every=1000)
    mlp.show_results_multiclass(network, X_test, y_test)

def add_noise(X: npt.NDArray[np.float64], noise_level: float = 0.15) -> npt.NDArray[np.float64]:
    X_noisy = X.copy()
    rng = np.random.default_rng()
    noise_mask = rng.random(X_noisy.shape) < noise_level
    X_noisy[noise_mask] = 1 - X_noisy[noise_mask]
    return X_noisy

if __name__ == '__main__':
    main()
