import rbf.rbf

import rbf.polynomials as poly
import numpy as np
import scipy.linalg as sl
import scipy.spatial.distance as scidist
import scipy.special as sspec

class CPDRBF(object):
    """
    A conditionally positive definite (CPD) RBF. A CPD RBF generates a non-singular fit matrix only in the case
    that certain additional polynomial constraints and dimensional constraints are verified.
    """
    def __init__(self, centers, poly_degree, dimension = 2):
        self.coefs = np.zeros((len(centers), 1), dtype=float)
        self.centers = centers  # An np.array of shape (N,d) where N is the number of data points and d is the dimension.
        self.poly_degree = poly_degree
        self.dimension = dimension
        self.polynomial = poly.Polynomial(poly_degree, dimension)
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
        print("The shape of Y is", np.shape(Y))
        """
        Generates the RBF coefficients to fit a set of given data values Y for centers self.centers
        :param Y: A set of dependent data values corresponding to self.centers
        :return: Void, sets the self.coefs values
        """
        kernel_matrix = self.EvaluateCentersKernel()
        kernel_matrix[np.isinf(kernel_matrix)] = 0 # TODO: Is there a better way to avoid the diagonal?
        monomial_basis = poly.GetMonomialBasis(self.dimension, self.poly_degree)
        poly_matrix = poly.BuildPolynomialMatrix(monomial_basis, self.centers.transpose()) # TODO: Probably remove transpose requirement
        poly_shape = np.shape(poly_matrix)
        # Get the number of columns, as we need to make an np.zeros((num_cols,num_cols))
        num_cols = poly_shape[1]
        zero_mat = np.zeros((num_cols,num_cols))
        upper_matrix = np.hstack((kernel_matrix, poly_matrix))
        lower_matrix = np.hstack((poly_matrix.transpose(),zero_mat))
        rbf_matrix = np.vstack((upper_matrix,lower_matrix))
        Y = np.concatenate((Y,np.zeros((num_cols)))) # Extend with zeros for the polynomial annihilation
        self.coefs = sl.solve(rbf_matrix, Y, sym_pos=False)

    def Evaluate(self, x):
        return np.dot(self.coefs, self.kernel(x, self.centers)[0,:]) + self.polynomial.Evaluate(x)

    def EvaluateCentersKernel(self):
        """
        Computes the pairwise distances between the centers used for the interpolation fit. We produce a square
        matrix from this by computing scidist.squareform(diffs) to convert it into a symmetric matrix

        :return: A symmetric matrix of pairwise differences evaluated with the function. Mathematically, given
         the kenrel K, computes K(x,y) for each x,y in the set of centers X.
        """
        diffs = scidist.pdist(self.centers,metric='sqeuclidean')

        # TODO: Maybe move the squareform into the KernelRadial to save time.
        return self.KernelRadial(scidist.squareform(diffs)) # Convert distance list to matrix.

class ThinPlateSpline(CPDRBF):

    """
    The thin plate spline is conditionally positive definite of order m for m > d/2 where d is the dimension of the
    space the centers live in. In even dimensions, the thin plate spline takes the form of r^k log(r). Our current
    implementation focuses on r^2 log(r).
    """
    #The Gaussian RBF is a positive definite kernel corresponding to exp(-gamma*r^2). The Gaussian kernel provides
    #spectral  accuracy.
    #
    def __init__(self,centers, poly_degree, dimension=2):
        CPDRBF.__init__(self,centers,poly_degree, dimension)

    def KernelRadial(self,r):
        kernels = .5 * r * np.log(r)  # Please note, r is truly distance squared here, so this is 1/2 x^2 log(x^2) = x^2 log(x)
        kernels[np.isnan(kernels)] = 0 # TODO: Is there a better way to avoid the diagonal?
        return kernels
