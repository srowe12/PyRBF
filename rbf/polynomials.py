import numpy as np
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
        return nchoosek(self.degree + self.dimension, self.degree, exact=True)

    def Evaluate(self, coefficients, x):
        # TODO: Use Horner's Method to evaluate the polynomial
        # See: https://en.wikipedia.org/wiki/Horner%27s_method
        # Also,

        pass

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