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