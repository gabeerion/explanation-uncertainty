# Generate data for our experiments
import numpy as np
import matplotlib.pyplot as plt

def linearRegression_normal(beta, cov, sigma, n):
    """
    Draw n observations from a linear regression
       Y  = X beta + epsilon
       epsilon ~ N(0, sigma)
       X ~ N(0, cov)
    for isotropic x's, call this function with cov=np.eye(len(beta))
    :param beta: in R^{d x 1}
    :param cov:  in R^{d x d}
    :param sigma: in R^+
    :param n: in R
    :return: X, Y  where X \in R^{n x d} and Y \in R^{n x 1}
    """
    d = len(beta)
    X = np.random.multivariate_normal(np.zeros(d), cov, size=n)
    eps = np.random.normal(loc=0, scale=sigma, size=n)

    Y = X @ beta + eps


    return X, Y


if __name__=="__main__":
    # An isotropic example
    X, Y = linearRegression_normal(beta=np.array([1, 1]).T,
                                  cov=np.eye(2),
                                  sigma=1,
                                  n=200)

    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.title("Isotropic X, beta = [1,1]")
    plt.show()

    # A correlated example
    X, Y = linearRegression_normal(beta=np.array([1, 1]).T,
                                   cov=np.array([[1, 0.5],
                                                 [0.5, 1]]),
                                   sigma=1,
                                   n=200)

    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.title("Correlated X, beta = [1,1]")
    plt.show()

