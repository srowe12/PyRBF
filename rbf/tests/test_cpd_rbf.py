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
    assert np.linalg.norm(kernel_evals - kernel_evals_expected) < 1e-12


def test_ThinPlatSplineFitPolynomialReproduction():
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
    data_values = 2 * centers[:, 0] - 4 * centers[:, 1] + 3
    tps.fit(data_values)
    num_centers = np.shape(centers)[0]  # Number of rows implies number of centers
    expected_data_values = np.zeros((num_centers + 3))
    # I expect the interpolant to perfectly reproduce the linear curve 2x - 4y + 3
    expected_data_values[-3] = 2
    expected_data_values[-2] = -4
    expected_data_values[-1] = 3
    coefficient_difference = tps.coefs - expected_data_values
    coefficient_fit_error = np.linalg.norm(coefficient_difference)
    assert coefficient_fit_error < 1e-12


def test_ThinPlateSplineFit():
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
    data_values = np.sin(np.pi / 7 * centers[:, 0]) * np.sin(np.pi / 7 * centers[:, 1])
    tps.fit(data_values)
    num_centers = np.shape(centers)[0]  # Number of rows implies number of centers
    expected_solution = np.array([-0.004137888322575,
                                  0.032611379911058,
                                  -0.028473491588483,
                                  -0.023023910682168,
                                  0.004698775237651,
                                  0.007120133848076,
                                  0.015087842787749,
                                  -0.003882841191308,
                                  0.089325673449582,
                                  -0.049623717034734,
                                  1.532831698163157])
    coefficient_difference = tps.coefs - expected_solution
    coefficient_fit_error = np.linalg.norm(coefficient_difference)
    assert coefficient_fit_error < 1e-12

def test_ThinPlateSplineEvaluate():
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
    data_values = np.sin(np.pi / 7 * centers[:, 0]) * np.sin(np.pi / 7 * centers[:, 1])
    tps.fit(data_values)
    evaluation_points = centers
    evaluations = tps.Evaluate(evaluation_points)

    assert np.max(np.abs(evaluations-data_values)) < 1e-12

    # Let's choose a subset and verify it works when number of evaluation points is not the same as the number of centers
    evaluation_points = centers[0:4,:]
    evaluations = tps.Evaluate(evaluation_points)
    assert np.max(np.abs(evaluations - data_values[0:4])) < 1e-12
