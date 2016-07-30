import numpy as np
import sys
sys.path.append("..")
import conditionally_positive_definite_rbf as cpdrbf
import scipy.spatial.distance as dist
import os

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