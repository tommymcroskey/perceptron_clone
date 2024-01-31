"""This is a clone of the perceptron algorithm heavily influenced by the code
provided to me by my professor of machine learning, Sarath Sreedharan."""

import matplotlib.pyplot as plt
import numpy as np

class perceptron:
    """As is the example given to me, this implementation of the
    perceptron algorith will not feature a bias term."""

    def __init__(self, iterations=100, learning_rate=.1,
                 plot_data=False, random_w=False, seed=10):
        """this constructor will default the value of iterations to 100,
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
        