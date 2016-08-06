import numpy as np
import sys
sys.path.append("..")
import conditionally_positive_definite_rbf as cpdrbf

def test_PolynomialNumCoefficients():
    degree = 3
    dimension =1
    # Example: ax^3 + bx^2 + cx + d. Should have 4 coefficients

    poly = cpdrbf.Polynomial(degree, dimension)
    assert poly.num_coefficients== 4

    degree = 3
    dimension = 2
    # Example: ax^3 + bx^2y + cxy^2 + dy^3 + ex^2 + fxy + gy^2 + hx + ky + m
    # Should have 10 coefficients
    poly = cpdrbf.Polynomial(degree, dimension)
    assert poly.num_coefficients == 10

    degree = 4
    # Should choose default dimension =2
    poly = cpdrbf.Polynomial(degree)
    assert poly.num_coefficients == 15

def test_PolynomialEvaluate():
    degree = 3
    dimension = 1
    coefficients = np.array([1.0,2.0,3.0,4.0])

    poly = cpdrbf.Polynomial(degree,dimension)

    assert 10 == poly.Evaluate(coefficients, 1.0)

def GetPolynomialMultiset(dimension,degree):
    # TODO: Choose this number
    total_num_options = 6
    # Start by working on the far left, and move over
    multiset = []
    #Dimension is the number of positions we have to choose
    print("I'm getting a polynomial of degree ", degree, "and dimension", dimension)

    #if dimension == 1:
    #    for item in range(degree,0,-1):
    #        multiset.append([item])
    #else:

    for position in range(0,dimension):


        # Choose my highest value
        # Do I have dim - position entries left?

        # I'm at a fixed position, so now I want to know how many values I can put in my spot. So loop down from degree
        for deg in range(degree,0,-1):
            #orders[position] = deg # Plug in my degree here. Do I have any options left?
            # What's left?
            leftover_power = degree - deg

            if leftover_power:

                polys = GetPolynomialMultiset(dimension-position-1, degree-deg)
                print("The polys are", polys)
                for poly in polys:
                    print("The poly is", poly)
                    # You need a sub_monomial
                    sub_monomial = (position+1)*[0]
                    sub_monomial[position] = deg
                    monomial = sub_monomial + poly
                    print("The monomial produced is", monomial)
                    multiset.append(monomial)
            else:
                print("The dimension is", dimension)
                monomial = dimension*[0]
                monomial[position] = degree
                multiset.append(monomial)

    return multiset



def test_Polynomial3DimensionsDegree2():
    expected_solution = [[2,0,0],[1,1,0],[1,0,1], [0,2,0],[0,1,1],[0,0,2]]
    GetPolynomialMultiset(dimension=3,degree=2)
    assert solution == expected_solution