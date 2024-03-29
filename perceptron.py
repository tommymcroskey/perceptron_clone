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
        seed to 10. Note that w in this case refers to the weight
        vector orthoganal to the hyperplane that perceptron will create
        """

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.plot_data = plot_data
        self.random_w = random_w
        self.seed = seed

    def fit(self, X, y):
        """
        This function serves to define the main converging loop,
        where selectron will rotate a weight vector such that the
        classification error is minimized over the training set.
        """

        if self.random_w:
            rng = np.random.default_rng(self.seed)  # We use a seed for replicablility
            self.w = rng.uniform(-1, 1, len(X[0]))  # Set weights to random values within [-1, 1]
            print("initialized with random weight vector")
        else:
            self.w = np.zeros(len(X[0]))  # Initialize weights to zero
            print("initialized with a zeros weight vector")

        self.wold = self.w  # Store current weights to compare for convergence later
        converged = False  # Flag to check if algorithm has converged
        iteration = 0  # Counter to keep track of iterations

        # Main loop to update weights until convergence or reaching max iterations
        while (not converged and iteration <= self.iterations):
            converged = True  # Assume convergence, prove otherwise
            for i in range(len(X)):  # Iterate over all data points
            # Update weights if classification is incorrect
                if y[i] * self.decision_function(X[i]) <= 0:
                    self.wold = self.w  # Keep track of old weights
                    self.w = self.w + y[i] * self.learning_rate * X[i]  # Update weights
                    converged = False  # Set converged to False as weights were updated
                    if self.plot_data:  # If plotting is enabled, update the plot
                        self.plot_update(X, y, i)
            iteration += 1  # Increment iteration count

        self.converged = converged  # Store convergence status
        if converged:
            print('converged in %d iterations ' % iteration)  # Print if converged within the iteration limit
