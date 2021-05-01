# For the linear regression model, we have an analytical expression for the confidence intervals
# In particular,
#    betaHat +/- t_{n-p}^{alpha/2}sigmaHat sqrt{(X^T X)^{-1}_ii}
# where
#   sigmaHat = sqrt(1/(n-2) sum_i (y_i - yHat)^2)

import numpy as np
import scipy.stats

def analyticLinearCis(fittedModel, X, y, alpha=0.05, fitIntercept=True):
    """
    Computes confidence intervals for linear regression
    :param fittedModel: An object with a function .predict(XPred) that returns yPred, and a
        field .coef_ that returns the coefficient vector
    :param X: The data on which we have fit the model
    :param y: The observations on which we fit the model
    :param alpha: The nominal type I error for our confidence intervals
    :param fitIntercept: a boolean indicating whether the model fitted an intercept; if so, increment the degrees of freedom
    :return: Lower and upper bounds for the alpha-confidence interval on the coefficients of the fitted model,
        as a tuple
    """
    n, d = X.shape
    yPred = fittedModel.predict(X)
    sigmaHat = np.sqrt(1/(n-2) * np.sum((y-yPred)**2))
    df = n - d - (1 if fitIntercept else 0)   # An extra df lost because of the intercept if we fit it
    tStat = scipy.stats.t.ppf(1-alpha/2, df)
    intervalSizes = tStat * sigmaHat * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))

    fittedCoefs = fittedModel.coef_

    lcbs = fittedCoefs - intervalSizes
    ucbs = fittedCoefs + intervalSizes

    return lcbs, ucbs


def analyticLinearTest_GLRT_singleCoef(fittedModel, X, y):
    """
    Computes a test statistic determining whether the coefficient in the fitted model is significantly different from
    zero, using the analytical version of the chi-squared test (which is the GLRT for this specific instance).
    Working from the notes in https://nowak.ece.wisc.edu/ece830/ece830_lecture10.pdf (example 3.1)
    Right now, this only works on models with a single fitted coefficient, because I'm very certain of the test
    in that setting
    :param fittedModel: An object with a function .predict(XPred) that returns yPred
    :param X: The data on which we have fit the model
    :param y: The observations on which we fit the model
    :return: a p-value indicating whether we can reject the hypothesis that the coefficient is zero
    """
    # GLRT test statistic: 1/sigma^2 y^T X (X^T X)^-1 X^T y
    # and then you compare it to a chi-squared distribution with d (in this case 1) degrees of freedom
    # We don't have sigma; instead we have sigmaHat. However, as a first guess, let's use sigmaHat in place of sigma
    # and hope that we have enough data that this isn't a big issue.
    print("DEPRECATED! Please use analyticLinearTest_GLRT instead of analyticLinearTest_GLRT_singleCoef")
    n, d = X.shape
    if d != 1:
        print("analyticLinearTest_GLRT_singleCoef currently only works on 1 fitted coefficient;", d, "provided")
    yPred = fittedModel.predict(X)
    sigmaHat = np.sqrt(1 / (n - 2) * np.sum((y - yPred) ** 2))

    beta = np.linalg.solve(X.T @ X, X.T @ y)

    testStat = 1/sigmaHat**2 * y.T @ X @ beta

    chiSqPval = 1 - scipy.stats.chi2.cdf(x=testStat, df=d)

    return chiSqPval


def analyticLinearTest_GLRT(fittedModel, X, y):
    """
    Computes a test statistic determining whether any coefficient in the fitted model is significantly different from
    zero, using the analytical version of the chi-squared test (which is the GLRT for this specific instance).
    Working from the notes in https://nowak.ece.wisc.edu/ece830/ece830_lecture10.pdf (example 3.1)
    :param fittedModel: An object with a function .predict(XPred) that returns yPred
    :param X: The data on which we have fit the model
    :param y: The observations on which we fit the model
    :return: a p-value indicating whether we can reject the null hypothesis that ALL coefficients are zero
    """
    # GLRT test statistic: 1/sigma^2 y^T X (X^T X)^-1 X^T y
    # and then you compare it to a chi-squared distribution with d degrees of freedom
    # We don't have sigma; instead we have sigmaHat. However, as a first guess, let's use sigmaHat in place of sigma
    # and hope that we have enough data that this isn't a big issue.
    n, d = X.shape
    yPred = fittedModel.predict(X)
    sigmaHat = np.sqrt(1 / (n - 2) * np.sum((y - yPred) ** 2))

    beta = np.linalg.solve(X.T @ X, X.T @ y)

    testStat = 1/sigmaHat**2 * y.T @ X @ beta

    chiSqPval = 1 - scipy.stats.chi2.cdf(x=testStat, df=d)

    return chiSqPval

