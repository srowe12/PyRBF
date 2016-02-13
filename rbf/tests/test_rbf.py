import numpy as np
import rbf
import scipy.spatial.distance as dist
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
    print("The distance matrix is", dists)
    print("The shape is", np.shape(dists))
    print("Dists", dists[0][0])

    tol = 1e-15
    assert (dists[0][0] - np.exp(-.5) ) < tol
    assert (dists[0][1] - np.exp(-2) ) < tol
    assert (dists[1][0] - 1.0 ) < tol
    assert (dists[0][2] - np.exp(-18*gamma) ) < tol
    assert (dists[1][2] - np.exp(-8*gamma) ) < tol



