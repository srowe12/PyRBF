import rbf.polynomials as poly

import numpy as np


def test_PolynomialNumCoefficients():
    degree = 3
    dimension =1
    # Example: ax^3 + bx^2 + cx + d. Should have 4 coefficients

    p = poly.Polynomial(degree, dimension)
    assert p.num_coefficients== 4

    degree = 3
    dimension = 2
    # Example: ax^3 + bx^2y + cxy^2 + dy^3 + ex^2 + fxy + gy^2 + hx + ky + m
    # Should have 10 coefficients
    p = poly.Polynomial(degree, dimension)
    assert p.num_coefficients == 10

    degree = 4
    # Should choose default dimension =2
    p = poly.Polynomial(degree)
    assert p.num_coefficients == 15

def test_PolynomialEvaluate():
    degree = 3
    dimension = 1
    coefficients = np.array([1.0,2.0,3.0,4.0])

    p = poly.Polynomial(degree,dimension)

    assert 10 == p.Evaluate(coefficients, [1.0])

    assert 4 == p.Evaluate(coefficients, [0.0])

def test_PolynomialEvaluateDim2Deg3():
    degree = 3
    dimension = 2

    coefficients = [1,2,3,4,5,6,7,8,9,10]

    p = poly.Polynomial(degree, dimension)

    assert 366 == p.Evaluate(coefficients, [2.0,3.0])



def test_Polynomial3DimensionsDegree2():
    expected_solution = [[2,0,0],[1,1,0],[1,0,1], [0,2,0],[0,1,1],[0,0,2]]
    solution = poly.GetMonomialsOfFixedDegree(dimension=3,degree=2)
    assert solution == expected_solution

def test_Polynomial4DimensionsDegree3():
    expected_solution = [[3,0,0,0],
                         [2,1,0,0],
                         [2,0,1,0],
                         [2,0,0,1],
                         [1,2,0,0],
                         [1,1,1,0],
                         [1,1,0,1],
                         [1,0,2,0],
                         [1,0,1,1],
                         [1,0,0,2],
                         [0,3,0,0],
                         [0,2,1,0],
                         [0,2,0,1],
                         [0,1,2,0],
                         [0,1,1,1],
                         [0,1,0,2],
                         [0,0,3,0],
                         [0,0,2,1],
                         [0,0,1,2],
                         [0,0,0,3]]

    solution = poly.GetMonomialsOfFixedDegree(dimension=4,degree=3)
    assert solution == expected_solution

def test_MonomialBasisDegree2Dimension2():
    expected_monomial_basis = [[2,0],
                               [1,1],
                               [0,2],
                               [1,0],
                               [0,1],
                               [0,0]]
    dimension = 2
    degree = 2

    monomial_basis = poly.GetMonomialBasis(dimension, degree)
    assert monomial_basis == expected_monomial_basis


def test_BuildPolynomialMatrix():
    degree = 2
    dimension = 2
    monomial_basis = poly.GetMonomialBasis(dimension,degree)
    evaluation_list = np.array([[1,2,3],[4,5,6]])
    P = poly.BuildPolynomialMatrix(monomial_basis, evaluation_list)
    num_centers = np.shape(evaluation_list)[1]

    poly_matrix_shape = np.shape(P)
    assert poly_matrix_shape[0] == num_centers # Should be one row per evaluation point
    assert poly_matrix_shape[1] == len(monomial_basis)  # Should be one column per basis element

    expected_matrix = np.array([[1,4,9],[4,10,18],[16,25,36],[1,2,3],[4,5,6],[1,1,1]]).transpose()

    assert np.linalg.norm(expected_matrix - P) < 1e-12