import numpy as np
from itertools import combinations_with_replacement
from .linear_regression import LinearRegression

class PolynomialRegression(LinearRegression):
    """
    @NOTE: scikit-learn does not have a PolynomialRegression class but it implements a PolynomialFeatures class to generate polynomial features as preprocessing step. We can then use these new features with the usual LinearRegression class. In our codebase, we extend the LinearRegression class to implement PolynomialRegression. 
    """

    def __init__(self, degree: int) -> None:
        super().__init__()
        self.degree = degree
        pass
    
    def validate_features(self, X):
        """
        For each d in [0, degree], we generate all possible combinations of features of length d, WITH REPLACEMENT.

        Parameters
        ----------
        X : np.ndarray
            Features matrix of shape (n_samples, n_features)

        Returns
        -------
        Features matrix of shape (n_samples, n_features_new)
        """
        n_samples, n_features = X.shape
        
        combs = [
            comb for d in range(self.degree + 1) \
                    for comb in combinations_with_replacement(range(n_features), d)
        ]
        self.feature_combs = combs

        n_features_new = len(combs)
        phi_X = np.ones(shape=(n_samples, n_features_new))
        for i, comb in enumerate(combs):
            phi_X[:, i] = np.prod(X[:, comb], axis=1)
        return phi_X
    
    @property
    def coef_(self):
        return self.w[1:]
    
    @property
    def intercept_(self):
        return self.w[0]