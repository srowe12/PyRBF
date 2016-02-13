import numpy as np
import rbf
def test_Gaussian():
    gamma = 1.0
    centers = np.array([[1.0,2.0,3.0,4.0]])
    gaussian = rbf.Gaussian(centers,gamma)
    xvec = np.array([[1]])
    yvec = np.array([[1]])
    test_val = gaussian.kernel(xvec,yvec)

    assert test_val[0][0] == 1.0

    test_val = gaussian.kernel(xvec,2.0*yvec)
    assert test_val[0][0] == np.exp(-1)
