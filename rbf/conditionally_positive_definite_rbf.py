import rbf
from scipy.misc import comb as nchoosek
class Polynomial(object):
    def __init__(self, polynomial_degree, dimension=2):
        self.degree = polynomial_degree
        self.dimension = dimension
        self.num_coefficients = self.NumCoefficients()

    def NumCoefficients(self):
        """
        There are (n+d) choose n coefficients where n is the degree of the polynomial and d is the dimension
        :return: Return the number of coefficients corresponding to the polynomial given the degree and dimension
        """

        # TODO: Seems risky, is there a better way to handle this?
        return round(nchoosek(self.degree + self.dimension, self.degree))

class CPDRBF(object):
    """
    A conditionally positive definite (CPD) RBF. A CPD RBF generates a non-singular fit matrix only in the case
    that certain additional polynomial constraints and dimensional constraints are verified.
    """
    def __init__(self, centers, poly_degree):
        self.coefs = np.zeros((len(centers), 1), dtype=float)
        self.centers = centers  # An np.array of shape (N,d) where N is the number of data points and d is the dimension.
        self.poly_degree = poly_degree
        self.polynomial = Polynomial(poly_degree)
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
        return np.dot(self.coefs, self.kernel(x, self.centers)[0,:])

    def EvaluateCentersKernel(self):
        diffs = scidist.pdist(self.centers,metric='sqeuclidean')

        # TODO: Maybe move the squareform into the KernelRadial to save time.
        return self.KernelRadial(scidist.squareform(diffs)) # Convert distance list to matrix.