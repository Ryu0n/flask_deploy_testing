import tensorflow.keras as keras

from src.patterns.singleton import SingletonInstance


class MNIST(SingletonInstance):
    def __init__(self):
        self.mnist = keras.datasets.mnist

    def load_with_preprocess(self):
        (x_train, y_train), (x_test, y_test) = self.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test
