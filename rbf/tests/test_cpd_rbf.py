import numpy as np
import rbf.conditionally_positive_definite_rbf as cpdrbf


def test_ThinPlateSplineKernelRadial():
    centers = np.array([[1, 2], [3, 4], [5, 6]])
    poly_degree = 2
    dimension = 2

    tps = cpdrbf.ThinPlateSpline(centers, poly_degree, dimension)

    kernel_evals = tps.EvaluateCentersKernel()
    kernel_evals_expected = np.array([[0, 8.317766166719347, 55.451774444795639], [
        8.317766166719347, 0, 8.317766166719347], [55.451774444795639, 8.317766166719347, 0]])
    print("The kernels we compute are", kernel_evals)
    assert np.linalg.norm(kernel_evals - kernel_evals_expected) < 1e-12


def test_ThinPlatSplineFit():
    poly_degree = 1
    dimension = 2
    centers = np.array([[1, 2],
                        [3, 4],
                        [5, 6],
                        [1, 1],
                        [2, 2],
                        [3, 3],
                        [4, 4],
                        [5, 5]])
    tps = cpdrbf.ThinPlateSpline(centers, poly_degree, dimension)
    data_values = 2*centers[:,0] - 4*centers[:,1] + 3
    tps.fit(data_values)
    num_centers = np.shape(centers)[0] # Number of rows implies number of centers
    expected_data_values = np.zeros((num_centers+3))
    expected_data_values[-3] = 2
    expected_data_values[-2] = -4
    expected_data_values[-1] = 3
    coefficient_difference = tps.coefs - expected_data_values
    coefficient_fit_error = np.linalg.norm(coefficient_difference)
    assert coefficient_fit_error < 1e-12
