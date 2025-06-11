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

def pdfnormal(x: np.ndarray, m, s):
    """Normal PDF calculator
    Parameters:
    x: array-like, grid space
    m: mean
    s: standard deviation
    """
    return (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-((x - m) ** 2) / (2 * s ** 2))

# def plot_superficie():
# seqi = np.arange(0, 10, 0.5)
# seqj = np.arange(0, 10, 0.5)
# M1 = np.zeros((len(seqi), len(seqj)))
# for i in seqi:
#     for j in seqj:
#         M1[i][j] = algumcalculo()

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

class myPerceptron:
    def __init__(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        eta: float, 
        tol: float, 
        max_epochs: int, 
        par=1
        ) -> None:
        """
        Perceptron class. 
        Adjusts model weight based on training data.

        Parameters:
        X (ndarray): input data
        Y (ndarray): target labels
        eta (float): learning rate
        tol (float): error tolerance
        max_epochs (int): maximum number of epochs
        """
        if par == 1:
            # Add bias term to the input data
            w = np.random.randn(X.shape[1] + 1)
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            w = np.random.randn(X.shape[1])
        
        N = len(X)
        self.error_epoch = [tol + 1]
        self.n_epoch = [0]
        
        while self.n_epoch[-1] < max_epochs and \
            self.error_epoch[-1] > tol:
            xseq = np.random.permutation(N)
            ei2 = 0

            for i in range(N):
                i_rand = xseq[i]
                err = Y[i_rand] - np.sign(np.dot(w, X[i_rand, :]))
                w += eta * err * X[i_rand, :]
                ei2 += err ** 2
            self.error_epoch.append(ei2)
            self.n_epoch.append(self.n_epoch[-1] + 1)
        
        self.weights = w

    def predict(self, sample: np.ndarray, par=1) -> np.ndarray:
        """
        Predict sample class.

        Parameters:
        sample (ndarray): input data
        """
        if par == 1:
            # Add bias term to the input data
            sample = np.hstack(((1,), sample))
        output = np.dot(sample, self.weights)
        return 1 if output >= 0 else 0
