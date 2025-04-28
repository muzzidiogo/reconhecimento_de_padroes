import numpy as np
from matplotlib import pyplot as plt

def gen_2D_gaussians(s1 = 0.3, s2 = 0.3, nc = 100, c1 = np.array([1, 1]), c2 = np.array([2, 2])):
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
    yc1 = np.ones((nc))
    yc2 = -np.ones((nc))

    X = np.vstack((xc1, xc2))  
    Y = np.concatenate((yc1, yc2))
    return X, Y

def make_normal_data(s = 0.3, n = 100, c = np.array([0,0]), dim = 2, label = 1):
    """Generates a dataset from a Gaussian distribution with specified parameters.
    
    Parameters:
    s (float): Standard deviation of the Gaussian.
    n (int): Number of samples.
    c (ndarray): Center of the Gaussian.
    dim (int): Dimension of the Gaussian.
    label (int): Label for the generated samples.
    """
    assert len(c) == dim, f"Center point dimension mismatch: Expected {dim}, but got {len(c)}."
    
    X = np.random.randn(n, dim) * s + c
    if label == 1:
        Y = np.ones((n))
    elif label == -1:
        Y = -np.ones((n))
    return X, Y

def gen_1D_gaussians(x: np.ndarray, s=0.3, c=4):
    """
    Computes the Gaussian distribution values for an array of input values.
    
    Parameters:
    x (array-like): Input values.
    s (float): Standard deviation of the Gaussian distribution.
    c (float): Mean of the Gaussian distribution.
    
    Returns:
    list: Gaussian distribution values corresponding to x.
    """
    return list((1 / (s * np.sqrt(2 * np.pi))) * np.exp(-((x - c) ** 2) / (2 * s ** 2)))

def plot_1D_gaussians(X: np.ndarray, Y: np.ndarray):
    '''
    Plots the generated 1D Gaussian distributions.
    
    Parameters:
    X (ndarray): Data points.
    Y (ndarray): Labels for the data points.
    '''
    plt.figure(figsize=(8, 4))
    plt.hist(X[Y == 1], bins=100, alpha=0.6, label='Class 1')
    plt.hist(X[Y == -1], bins=100, alpha=0.6, label='Class -1')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('1D Gaussian Distributions')
    plt.legend()
    plt.show()

def pdfnvar(x, m, K, n):
        """Multivariate normal PDF calculator"""
        x_m = x - m
        return (1 / (np.sqrt((2 * np.pi) ** n * np.linalg.det(K)))) * np.exp(-0.5 * (x_m.T @ np.linalg.inv(K) @ x_m))

def pdfnormal(x, m, s):
    """Normal PDF calculator"""
    return (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-((x - m) ** 2) / (2 * s ** 2))

# def plot_superficie():
#     seqi = np.linspace(0, 6, 100)
#     seqj = np.linspace(0, 6, 100)
#     M1 = np.zeros((len(seqi), len(seqj)))
#     ci = 0
#     for i in seqi:
#         cj = 0
#         ci += 1
#         for j in seqj:
#             M1[ci][cj] = algumcalculo()

def mymix(x, inlist):
    """Calculates mixture model probability density.
    
    Parameters:
    x: array-like, input vector
    inlist: list of numpy arrays (data matrices)
    
    Returns:
    Probability density at point x
    """
    ng = len(inlist)
    klist = []
    mlist = []
    pglist = []
    nglist = []
    n = inlist[0].shape[1]  # Number of dimensions
    
    # Calculate covariance and means for each group
    for i in range(ng):
        klist.append(np.cov(inlist[i], rowvar=False))
        mlist.append(np.mean(inlist[i], axis=0))
        nglist.append(inlist[i].shape[0])
    
    # Calculate total number of observations
    N = sum(nglist)
    
    # Calculate prior probabilities
    for i in range(ng):
        pglist.append(nglist[i] / N)
    
    # Calculate mixture probability
    Px = 0
    for i in range(ng):
        Px += pglist[i] * pdfnvar(x, mlist[i], klist[i], n)
    
    return Px
