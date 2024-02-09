"""
This code is based on code released under the
MIT license --> https://opensource.org/license/mit/
"""

import matplotlib.pyplot as plt
import numpy as np

class perceptron:
    """
    As is the example given to me, this implementation of the
    perceptron algorith will not feature a bias term.
    """

    def __init__(self, iterations=100, learning_rate=.1,
                 plot_data=False, random_w=False, seed=10):
        """
        this constructor will default the value of iterations to 100,
        learning rate to .1, plot_data to False, random_w to False, and
        seed to 10. Note that random_w in this case refers to the weight
        vector orthoganal to the hyperplane that perceptron will create
        """

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.plot_data = plot_data
        self.random_w = random_w
        self.seed = seed

    def fit(self, X, y):
        