"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.
"""

import pickle
import os
import numpy as np


def load_data():
    """Return the MNIST data as training, validation and test data."""

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        'mnist.pkl'
    )

    with open(data_path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(
            f,
            encoding='latin1'
        )

    return training_data, validation_data, test_data


def load_data_wrapper():
    """Return MNIST data in a format convenient for the neural network."""

    tr_d, va_d, te_d = load_data()

    # Training data
    training_inputs = [
        np.reshape(x, (784, 1))
        for x in tr_d[0]
    ]

    training_results = [
        vectorized_result(y)
        for y in tr_d[1]
    ]

    training_data = list(
        zip(training_inputs, training_results)
    )

    # Validation data
    validation_inputs = [
        np.reshape(x, (784, 1))
        for x in va_d[0]
    ]

    validation_data = list(
        zip(validation_inputs, va_d[1])
    )

    # Test data
    test_inputs = [
        np.reshape(x, (784, 1))
        for x in te_d[0]
    ]

    test_data = list(
        zip(test_inputs, te_d[1])
    )

    return training_data, validation_data, test_data


def vectorized_result(j):
    """Return a 10-dimensional vector representing digit j."""

    e = np.zeros((10, 1))
    e[j] = 1.0

    return e