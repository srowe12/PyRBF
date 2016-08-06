import rbf


class CPDRBF(object):
    """
    A conditionally positive definite (CPD) RBF. A CPD RBF generates a non-singular fit matrix only in the case
    that certain additional polynomial constraints and dimensional constraints are verified.
    """
    def __init__(self, centers, poly_degree, dimension = 2):
        self.coefs = np.zeros((len(centers), 1), dtype=float)
        self.centers = centers  # An np.array of shape (N,d) where N is the number of data points and d is the dimension.

        self.polynomial = Polynomial(poly_degree, dimension)
        pass

    def kernel(self, x, y):
        """
        Returns the squared distance between each pair of points in arrays x and y
        :param x: A vector of input points of dimension N
        :param y: A vector of input points of dimension N
        :return: A vector of squared distances between each pair of points from x and y
        """
        # NOTE: x and y both must be 2d arrays. Maybe [x] is necessary if shape too small.
        dist = scidist.cdist(x,y,metric="sqeuclidean")
        return self.KernelRadial(dist)

    def KernelRadial(self,r):
        """Overloaded in the derived class"""
        pass

    def fit(self, Y):
        """
        Generates the RBF coefficients to fit a set of given data values Y for centers self.centers
        :param Y: A set of dependent data values corresponding to self.centers
        :return: Void, sets the self.coefs values
        """
        kernel_matrix = self.EvaluateCentersKernel()
        self.coefs = sl.solve(kernel_matrix, Y, sym_pos=True)

    def Evaluate(self, x):
        return np.dot(self.coefs, self.kernel(x, self.centers)[0,:]) + self.polynomial.Evaluate(x)

    def EvaluateCentersKernel(self):
        diffs = scidist.pdist(self.centers,metric='sqeuclidean')

        # TODO: Maybe move the squareform into the KernelRadial to save time.
        return self.KernelRadial(scidist.squareform(diffs)) # Convert distance list to matrix.