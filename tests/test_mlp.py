import gyrus.mlp as mlp;

def test_import():
    assert mlp is not None

def test_constructor():
    network = mlp.MLP(input_size=2, output_size=1, hidden_layers=[2, 2])
    assert len(network.weights) == 3
    assert len(network.biases) == 3
