import numpy as np
import sys
#sys.path.append("..")
import rbf.rbf as rbf
import scipy.spatial.distance as dist
import os

print("The current directory is ", os.curdir)
def test_GaussianKernel():
    gamma = 1.0
    centers = np.array([[1.0,2.0,3.0,4.0]])
    gaussian = rbf.Gaussian(centers,gamma)
    xvec = np.array([[1]])
    yvec = np.array([[1]])
    test_val = gaussian.kernel(xvec,yvec)

    assert test_val[0][0] == 1.0

    test_val = gaussian.kernel(xvec,2.0*yvec)
    assert test_val[0][0] == np.exp(-1)

def test_GaussianKernelEvaluate():
    gamma = .25
    centers = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float)
    gaussian = rbf.Gaussian(centers,gamma)
    xvec = np.array([[0,0],[1.0,1.0]], dtype=float)
    yvec = np.array([[1.0,1.0],[2.0,2.0],[3.0,3.0]])
    dists = gaussian.kernel(xvec,yvec)

    tol = 1e-15
    assert (dists[0][0] - np.exp(-.5) ) < tol
    assert (dists[0][1] - np.exp(-2) ) < tol
    assert (dists[1][0] - 1.0 ) < tol
    assert (dists[0][2] - np.exp(-18*gamma) ) < tol
    assert (dists[1][2] - np.exp(-8*gamma) ) < tol

def test_GaussianMatrix():
    gamma = .25
    centers = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float)
    gaussian = rbf.Gaussian(centers,gamma)
    data_values = np.array([1.0,2.0,3.0,4.0], dtype=float)
    gaussian_matrix = gaussian.EvaluateCentersKernel()
    assert np.shape(gaussian_matrix) == (4,4)
    expected_gaussian_matrix = np.array([[1.0, np.exp(-.25), np.exp(-.5), np.exp(-.25)],
                                         [np.exp(-.25), 1.0, np.exp(-.25), np.exp(-.5)],
                                        [np.exp(-.5), np.exp(-.25), 1.0, np.exp(-.25)],
                                         [np.exp(-.25), np.exp(-.5), np.exp(-.25), 1.0]])
    error = np.linalg.norm(gaussian_matrix-expected_gaussian_matrix)
    assert error < 1e-12
def test_GaussianFit():
    gamma = .25
    centers = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float)
    gaussian = rbf.Gaussian(centers,gamma)
    data_values = np.array([1.0,2.0,3.0,4.0], dtype=float)
    gaussian.fit(data_values)

    tol = 1e-14
    assert (1.0 - gaussian.Evaluate([centers[0]]) ) < tol
    assert (2.0 - gaussian.Evaluate([centers[1]]) ) < tol
    assert (3.0 - gaussian.Evaluate([centers[2]]) ) < tol
    assert (4.0 - gaussian.Evaluate([centers[3]]) ) < tol


def test_MultiquadricEvaluate():
    gamma = .25
    centers = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    inverse_multiquadric = rbf.InverseMultiquadric(centers,gamma)
    xvec = np.array([[0, 0], [1.0, 1.0]], dtype=float)
    yvec = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    dists = inverse_multiquadric.kernel(xvec, yvec)

    expected_distance = np.array([[    0.816496580927726 ,  0.577350269189626,   0.426401432711221],[
   1.000000000000000,   0.816496580927726,   0.577350269189626]])

    print("The dists computed is", dists)
    error = np.linalg.norm(expected_distance - dists)
    print("The error is", error)
    assert np.linalg.norm(dists-expected_distance) < 1e-8

def test_MultiquadricFit():
    gamma = .25
    centers = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float)
    inverse_multiquadric = rbf.InverseMultiquadric(centers,gamma)
    data_values = np.array([1.0,2.0,3.0,4.0], dtype=float)
    inverse_multiquadric.fit(data_values)

    tol = 1e-14
    assert (1.0 - inverse_multiquadric.Evaluate([centers[0]]) ) < tol
    assert (2.0 - inverse_multiquadric.Evaluate([centers[1]]) ) < tol
    assert (3.0 - inverse_multiquadric.Evaluate([centers[2]]) ) < tol
    assert (4.0 - inverse_multiquadric.Evaluate([centers[3]]) ) < tol

def test_InverseQuadraticEvaluate():
    gamma = .25
    centers = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    inverse_quadratic = rbf.InverseQuadratic(centers,gamma)
    xvec = np.array([[0, 0], [1.0, 1.0]], dtype=float)
    yvec = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    dists = inverse_quadratic.kernel(xvec, yvec)

    expected_distance = np.array([[   0.666666666666667,   0.333333333333333,   0.181818181818182],[
   1.000000000000000,   0.666666666666667,   0.333333333333333]])

    print("The dists computed is", dists)
    error = np.linalg.norm(expected_distance - dists)
    print("The error is", error)
    assert np.linalg.norm(dists-expected_distance) < 1e-8

def test_InverseQuadraticFit():
    gamma = .25
    centers = np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float)
    inverse_quadratic = rbf.InverseQuadratic(centers,gamma)
    data_values = np.array([1.0,2.0,3.0,4.0], dtype=float)
    inverse_quadratic.fit(data_values)

    tol = 1e-14
    assert (1.0 - inverse_quadratic.Evaluate([centers[0]]) ) < tol
    assert (2.0 - inverse_quadratic.Evaluate([centers[1]]) ) < tol
    assert (3.0 - inverse_quadratic.Evaluate([centers[2]]) ) < tol
    assert (4.0 - inverse_quadratic.Evaluate([centers[3]]) ) < tol