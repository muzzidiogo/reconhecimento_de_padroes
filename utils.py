import numpy as np

def gen_2D_gaussians(s1 = 0.3, s2 = 0.3, nc = 100, c1 = np.array([3, 3]), c2 = np.array([4, 4])):
    '''
    Generates two 2D Gaussian distributions with different means and variances.
    
    Parameters:

    s1 (float): Standard deviation of the first Gaussian.
    s2 (float): Standard deviation of the second Gaussian.
    nc (int): Number of samples in each Gaussian.
    c1 (ndarray): Center of the first Gaussian.
    c2 (ndarray): Center of the second Gaussian.
    
    Returns:
    X (ndarray): Combined data points from both Gaussian distributions.
    Y (ndarray): Labels for the data points, 1 for the first Gaussian and -1 for the second.
    '''

    xc1 = np.random.randn(nc, 2) * s1 + c1
    xc2 = np.random.randn(nc, 2) * s2 + c2
    yc1 = np.ones((nc,1))
    yc2 = -np.ones((nc,1))

    X = np.vstack((xc1, xc2))  
    Y = np.vstack((yc1, yc2))

    return X, Y