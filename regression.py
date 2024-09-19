import numpy as np
import math


class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class LinearRegressionWithRegularization(object):


    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate


    def initialize_weights(self, n_features):
        # Initialize weights randomly to:
            # 1. Break symmetry and avoid identical updates during optimization
            # 2. Explore parameter space for faster convergence
            # 3. Prevent stagnation by ensuring all weights update differently
            # A uniform distribution keeps weights on a consistent scale
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))


    def fit(self, X, y):
        # bias term
        # The bias term adjusts the model's offset, allowing the regression line 
        # to fit data that does not pass through the origin, improving flexibility and accuracy.
        X = np.insert(X, 0, 1, axis=1)

        self.training_errors = []

        self.initialize_weights(n_features=X.shape[1])

        # gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)


if __name__ == '__main__':
    print('here')




