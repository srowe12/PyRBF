import numpy as np
from scipy.misc import comb as nchoosek

class Polynomial(object):
    def __init__(self, polynomial_degree, dimension=2):
        self.degree = polynomial_degree
        self.dimension = dimension
        self.num_coefficients = self.NumCoefficients()
        self.basis = GetMonomialBasis(dimension, polynomial_degree)

    def NumCoefficients(self):
        """
        There are (n+d) choose n coefficients where n is the degree of the polynomial and d is the dimension
        :return: Return the number of coefficients corresponding to the polynomial given the degree and dimension
        """
        return nchoosek(self.degree + self.dimension, self.degree, exact=True)

    def Evaluate(self, coefficients, x):
        # TODO: Use Horner's Method to evaluate the polynomial
        # See: https://en.wikipedia.org/wiki/Horner%27s_method
        # Also,
        result = 0
        for monomial_index, monomial in enumerate(self.basis):
            val = 1
            for index,power in enumerate(monomial):
                val *= x[index]**power
            result += coefficients[monomial_index]*val
        return result

def BuildPolynomialMatrix(monomial_basis, evaluation_list):
    """
    Produces a Vandermonde like matrix where each column corresponds to a monomial and every row corresponds to
    the evaluation of that monomial on a point in the evaluation_list. That is, P(row,col) is the evaluation of
    the monomial_basis[row].Evaluate(evaluation_list[col])
    :param monomial_basis: A list of lists containing the powers of corresponding monomials. See GetMonomialBasis
    :param evaluation_list: A numpy matrix of m columns, where each column represents a point to be evaluated, and each row
    represents the value of the point in that corresponding dimension. For example, [[3,1,2],[2,1,0]] represents three points
    where the first point has x value 3 and y value 1.
    :return: A Vandermonde like matrix evaluating the monomials at every center given.
    """
    shape = np.shape(evaluation_list)
    num_centers = shape[1] # Number of columns in evaluation_list determines number of input points. Amend documentation appropriately.
    P = np.zeros((num_centers, len(monomial_basis)))
    # Loop over Polynomial basis?
    for monomial_index, monomial in enumerate(monomial_basis):
        # monomial index determines the column we're in.
        P[:,monomial_index] = 1
        for power_index, power in enumerate(monomial):
            P[:,monomial_index] *= evaluation_list[power_index,:]**power
    return P


def GetMonomialBasis(dimension, degree):
    """
    Computes the monomial basis for space of 'dimension' dim polynomials of up to degree 'degree'.
    :param dimension: The number of independent variables for the polynomials to be comprised of. For example, dimension
    3 has x,y,z as independent variables.
    :param degree: The maximum of the sum of the powers of any given monomial in the space of polynomials
    :return: A list of all possible powers of polynomials in the given space
    """
    monomial_basis = []
    for deg in range(degree,0,-1):
        monomials_of_current_degree = GetMonomialsOfFixedDegree(dimension,deg)
        monomial_basis += monomials_of_current_degree
    # Now, let's append the "constant" monomial of zero powers
    monomial_basis += [dimension*[0]]
    return monomial_basis

def GetMonomialsOfFixedDegree(dimension,degree):
    """
    Finds a list of all monomials for a given degree and dimension

    For example, in degree 3, dimension 2 (two variables), the monomials are:
    x^3, x^2y, xy^2, y^3
    We can represent these monomials as tuples or lists. For example, x^3  is represented as (3,0), y^3 is (0,3) and
    (1,2) would be xy^2.

    :param dimension: The number of independent variables. If you want a polynomial in x,y,z, then the dimension is 3.
    :param degree: The degree of the polynomial is the max of the sum of the powers of the individual monomials
    :return: Returns a list of lists, where each entry in the returned list represents the powers of a monomial.
    """
    # Start by working on the far left, and move over
    monomial_set = []
    #Dimension is the number of positions we have to choose

    for position in range(0,dimension):
        for deg in range(degree,0,-1):
            #orders[position] = deg # Plug in my degree here. Do I have any options left?
            # What's left?
            leftover_power = degree - deg
            if leftover_power:
                polys = GetMonomialsOfFixedDegree(dimension-position-1, degree-deg)
                for poly in polys:
                    # Everything to my left is zero, and my current position has power "deg"
                    sub_monomial = (position+1)*[0]
                    sub_monomial[position] = deg
                    # Now, concatenate sub_monomial onto all of these polys which are to my right
                    monomial = sub_monomial + poly
                    monomial_set.append(monomial)
            else:
                monomial = dimension*[0]
                monomial[position] = degree
                monomial_set.append(monomial)

    return monomial_set