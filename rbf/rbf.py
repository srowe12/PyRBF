import numpy as np
import scipy.linalg as sl
import scipy.spatial.distance as scidist

class RBF(object):
    def __init__(self, centers):
        self.coefs = np.zeros((len(centers), 1), dtype=float)
        self.centers = centers  # An np.array of shape (N,d) where N is the number of data points and d is the dimension.
        pass

    def kernel(self, x, y):
        dist = scidist.cdist(x,y,metric="sqeuclidean")
        return self.KernelRadial(dist)

    def KernelRadial(self,r):
        pass

    def fit(self, Y):
        kernel_matrix = self.EvaluateCentersKernel()
        self.coefs = sl.solve(kernel_matrix, Y, sym_pos=True)

    def Evaluate(self, x):
        return np.dot(self.coefs, self.kernel(x, self.centers))

    def EvaluateCentersKernel(self):
        diffs = scidist.pdist(self.centers,metric='sqeuclidean')
        print("The shape of centers is", np.shape(self.centers))
        print("The shape of diffs is", np.shape(diffs))
        # TODO: Maybe move the squareform into the KernelRadial to save time.
        return self.KernelRadial(scidist.squareform(diffs)) # Convert distance list to matrix.

class Gaussian(RBF):
    def __init__(self,centers, gamma):
        RBF.__init__(self,centers)
        self.gamma = gamma

    # def kernel(self,x, y):
    #     diffs = scidist.cdist(x,y,metric='sqeuclidean') # A numpy array of distances. Only works if they're the same size...
    #     return np.exp(-1.0*self.gamma*diffs) # Redefine gamma to negative gamma?

    def KernelRadial(self,r):
        return np.exp(-1.0*self.gamma*r)



