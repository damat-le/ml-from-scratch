import numpy as np
from functools import partial
from src.optimizers import GradientDescent

class PoissonRegression:

    def __init__(self) -> None:
        pass

    def validate_features(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def loss(self, w, X, y):
        z = X @ w
        # c = np.log(np.math.factorial(y))
        pointwise_loss = np.exp(z) - y * z # + c
        return pointwise_loss.sum(axis=0)

    def loss_gradient(self, w, X, y):
        """
        @NOTE: Instead of computing the pointwise loss gradient and then taking the mean over the number of samples, we can directly compute the mean of the pointwise loss gradient as follows:
        n_samples = X.shape[0]
        z = X @ w
        grad = X.T @ (np.exp(z) - y)
        """
        z = X @ w
        pointwise_loss_grad = X * (np.exp(z) - y)
        grad = pointwise_loss_grad.mean(axis=0).reshape(-1, 1)
        return grad
    
    def fit(self, X, y, **opt_params):
        X = self.validate_features(X)
        y = y.reshape(-1, 1)
        self.w = np.zeros((X.shape[1], 1))
        grad = partial(self.loss_gradient, X=X, y=y)
        self.w = GradientDescent(**opt_params).minimize(grad, self.w)
        return None
    
    def predict(self, X):
        X = self.validate_features(X)
        return np.exp(X @ self.w)

    def score(self, X, y):
        X = self.validate_features(X)
        return np.pow(self.predict(X) - y, 2).mean()
    
    @property
    def coef_(self):
        return self.w[1:]
    
    @property
    def intercept_(self):
        return self.w[0]
    