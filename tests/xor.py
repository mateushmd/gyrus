import gyrus.mlp as mlp;
import numpy as np;

def main():
    network = mlp.MLP(
            input_size=2,
            output_size=1,
            hidden_layers=[2])
    
    network.fit(np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]]),
                np.array([[0],
                          [1],
                          [1],
                          [0]]),
                epochs=20000,
                print_loss_every=2000)
    print(network.predict(np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])))

    mlp.show_results(network, 
                 np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]]),
                 np.array([[0], [1], [1], [0]]))

if __name__ == '__main__':
    main()
