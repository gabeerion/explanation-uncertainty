# For the linear regression model, we have an analytical expression for the confidence intervals
# In particular,
#    betaHat +/- t_{n-p}^{alpha/2}sigmaHat sqrt{(X^T X)^{-1}_ii}
# where
#   sigmaHat = sqrt(1/(n-2) sum_i (y_i - yHat)^2)

import numpy as np
import scipy.stats

def analyticLinearCis(fittedModel, X, y, alpha=0.05):
    """
    Computes confidence intervals for linear regression
    :param fittedModel: An object with a function .predict(XPred) that returns yPred, and a
        field .coef_ that returns the coefficient vector
        An example would be modelFn = LinearRegression
    :param X: The data on which we have fit the model
    :param y: The observations on which we fit the model
    :param alpha: The nominal type I error for our confidence intervals
    :return: Lower and upper bounds for the alpha-confidence interval on the coefficients of the fitted model,
        as a tuple
    """
    n, d = X.shape
    yPred = fittedModel.predict(X)
    sigmaHat = np.sqrt(1/(n-2) * np.sum((y-yPred)**2))
    df = n - d - 1   # An extra df lost because of the intercept that I imagine we're fitting
    tStat = scipy.stats.t.ppf(1-alpha/2, df)
    intervalSizes = tStat * sigmaHat * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))

    fittedCoefs = fittedModel.coef_

    lcbs = fittedCoefs - intervalSizes
    ucbs = fittedCoefs + intervalSizes

    return lcbs, ucbs


def analyticLinearCis_GLRT(fittedModel, X, y, alpha=0.05):
    """
    Computes confidence intervals on betaHat using the analytic GLRT for the linear models case.
    Working from the notes in https://nowak.ece.wisc.edu/ece830/ece830_lecture10.pdf (example 3.1)
    AUUUGH, this is wrong, I think it's specifically for testing against 0, we have to redo for general theta
    observe that the log LR is
       1/sigma^2 (beta^T X^T y - 1/2 beta^T X^T X beta)  ~  ChiSq(d)
    (although it's probably ChiSq(d-1) since we'll have to estimate sigma^2)
    So we can get upper and lower 1-alpha confidence bounds for ChiSq(d), and write
       1/sigma^2 (beta^T X^T y - 1/2 beta^T X^T X beta) < u
    Solve for beta,
       beta^T X^T y - 1/2 beta^T X^T X beta < u sigma^2
    This is a quadratic form... it's solved when the derivative is zero:
       d/dbeta beta^T X^T y - 1/2 beta^T X^T X beta - u sigma^2 = 0
       d/dbeta beta^T X^T y - 1/2 beta^T X^T X beta = 0
       X^T y - X^T X beta = 0
       X^T y = X^T X beta
       (X^T X)^{-1} X^T y = beta
    lol, ok, so that's the MLE, and we have
       beta^T X^T y - 1/2 beta^T X^T X beta < u sigma^2
       y^T X (X^T X)^{-1} X^T y - 1/2 y^T X (X^T X)^{-1} X^T X (X^T X)^{-1} X^T y < u sigma^2
       y^T X (X^T X)^{-1} X^T y - 1/2 y^T X (X^T X)^{-1} X^T y < u sigma^2
       1/2 y^T X (X^T X)^{-1} X^T y < u sigma^2
    And, likewise, for the lower bound we have
       1/2 y^T X (X^T X)^{-1} X^T y > l sigma^2
    where l is the
    :param fittedModel:
    :param X:
    :param y:
    :param alpha:
    :return:
    """