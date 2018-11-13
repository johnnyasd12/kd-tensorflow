from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist

import numpy as np
# from utils import *

class MyData:

    def __init__(self):
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.scaler = None

    def get_train(self, scaler):
        pass

    def get_val(self, scaler):
        pass

    def get_test(self, scaler):
        pass


class MnistTF:
    def __init__(self):
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    def get_train_data(self):
        return self.mnist.train.images, self.mnist.train.labels

    def get_validation_data(self):
        return self.mnist.validation.images, self.mnist.validation.labels

    def get_test_data(self):
        return self.mnist.test.images, self.mnist.test.labels

    




