import rbf.conditionally_positive_definite_rbf as cpdrbf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def BuildCenters(a=0,b=1,num_points=20):
    """
    Generates uniformly spaced data points
    :return: a numpy array of centers of X and Y data
    """
    x = np.linspace(a,b,num_points)
    X,Y = np.meshgrid(x,x)
    X = X.ravel()
    Y = Y.ravel()
    centers = np.array([X,Y]).transpose()
    return centers

def GenerateData(centers):
    """
    Takes in 2D data and returns evaluation of a function z = f(x,y) on input centers
    :param centers:  2D data. First column is x data, second column is y data.
    :return: Evaluation of function on x,y data in
    """
    return np.sin(2*np.pi*centers[:,0]) * np.exp(-.25*centers[:,1])

def Plot3D(points,values):
    """
    :param points: An N x 2 array of N points in 2D space
    :param values: Corresponding function values f(points[:,0],points[:,1])
    :return: Plots data
    """

    # Reshape centers. They need to be meshgrid'ed
    shape = np.shape(points)
    x = points[:,0]
    y = points[:,1]
    new_shape = np.round(np.sqrt(len(x)))

    X = np.reshape(x,(new_shape,new_shape))
    Y = np.reshape(y,(new_shape,new_shape))
    Z = np.reshape(values,(new_shape,new_shape))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,Z)
    plt.show()

if __name__ == '__main__':
    def main():
        centers = BuildCenters()
        data_values = GenerateData(centers)

        # We now have our input data, so let's fit it
        # Generate our fitter using a thin plate spline in 2D with linear polynomial reproduction
        tps = cpdrbf.ThinPlateSpline(centers, poly_degree=1, dimension = 2)
        tps.fit(data_values)

        evaluations = tps.Evaluate(centers)

        # The error should be very small; interpolation should match on the centers exactly.
        print("The error in the fit is " , np.max(np.abs(evaluations - data_values)))

        # Let's evaluate it on a completely different set of points and plot the function and errors!
        eval_points = BuildCenters(0,1,30)
        evaluations = tps.Evaluate(eval_points)

        expected_evaluations = GenerateData(eval_points)

        errors = evaluations - expected_evaluations

        # 3D Plotting expects meshgrid types, so let's reshape centers.
        Plot3D(eval_points, evaluations)
        Plot3D(eval_points, errors)

if __name__ == "__main__":
    main()

