import numpy as np


class RBF(object):
    def __init__(self, centers):
        self.coefs = np.zeros((len(centers), 1), dtype=float)
        self.centers = centers  # An np.array of shape (N,d) where N is the number of data points and d is the dimension.
        pass

    def kernel(self, x, y):
        pass

    def fit(self, X):
        pass

    def Evaluate(self, x):
        return np.dot(self.coefs, np.array([[self.kernel(x, self.centers)]]))

    def EvaluateKernels(self, x):
        np.array([self.kernel(x, self.centers)])  # TODO: Maybe use double arrays? Fix this

class Gaussian(RBF):
    def __init__(self,centers, gamma):
        RBF.__init__(self,centers)
        self.gamma = gamma

    def kernel(self,x, y):
        diff = x-y # A numpy array of distances. Only works if they're the same size...
        return np.exp(-1.0*self.gamma*np.dot(diff.T, diff)) # Redefine gamma to negative gamma?

