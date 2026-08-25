import mmist_loader
import network

training_data, validation_data, test_data = mmist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.stochastic_gradient(
    training_data,
    30,
    10,
    100.0,
    test_data=test_data
)