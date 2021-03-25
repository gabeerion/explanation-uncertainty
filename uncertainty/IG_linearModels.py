# Compute integrated gradients for the special case of linear models

import numpy as np
from .analyticLinearRegressionCIs import analyticLinearCis

def integratedGradients_Linear(coefficients, X):
    """
    Computes integrated gradients for every X, using the linear model fittedModel. Integrated gradients will be computed
    relative to the baseline of the average X
    :param coefficients, as returned by, for example, fittedModel.coef_
    :param X: The features on which we have fit the model (n by d)
    :return: A d-element vector that averages the absolute value of the integrated gradient computed for each
    data point n
    """
    baseline = np.mean(X, axis=0)   # d dimensional

    # For integrated gradients, since the gradient is constant (the coefficient), the value is just the coefficient
    # times the difference between X and the baseline
    deltas = X - baseline
    igs = deltas * coefficients  #n by d dimensional

    # Each integrated gradient represents how important that feature is to that data point, but it can be positive or
    # negative. To get an importance measure over the entire data set, we average the absolute values of the integrated
    # gradients.
    avgIgs = np.mean(np.abs(igs), axis=0) # d dimensional

    return(avgIgs)


def bootstrapIntegratedGradients_Linear(modelFn, X, y, alpha=0.05, replicates=1000):
    """
    Bootstrap a two-sided confidence interval at level alpha on the integrated gradient feature importances for modelFn,
    which must be a linear model
    :param modelFn: A function that, when called with no arguments, returns an object  with a function .fit(X, y),
        a function .predict(XPred) that returns yPred, and a field .coef_ that returns the coefficient vector
        An example would be modelFn = LinearRegression. Must be a linear model.
    :param X: We will sample with replacement from (X, y) to generate bootstrap replicates.
    :param y: See X
    :param alpha: The desired Type 1 error rate for the confidence intervals
    :param replicates: The number of bootstrap replicates desired. For smaller alpha, this needs to increase.
    :return: Lower and upper bounds for the alpha-confidence interval on the average integrated gradients of the
        fitted model, as a tuple
    """

    rng = np.random.default_rng()
    n, d = X.shape
    avgIgs = np.zeros((d, replicates))
    for i in range(replicates):
        # Get the bootstrap samples
        idcs = rng.integers(low=0, high=n, size=n)
        Xsample = X[idcs, :]
        ySample = y[idcs]

        # Get a clean model object and fit it to the bootstrapped data
        model = modelFn()
        model.fit(Xsample, ySample)
        # Get the integrated gradients
        avgIgs[:, i] = integratedGradients_Linear(model.coef_, X)

    # Find the desired percentiles, on a per-coefficient basis
    avgIgs = np.sort(avgIgs, axis=1)
    lIndex = int(replicates * alpha / 2.0)
    uIndex = int(replicates * (1 - alpha / 2.0))

    return avgIgs[:, lIndex], avgIgs[:, uIndex]


# Analytical CI's on integrated gradients?
# Use analytical CI's on coefficients, then IG the extremes?
def integratedGradients_LinearAnalytical(fittedModel, X, y, alpha=0.05):
    """
    Computes confidence intervals on the (average absolute) integrated gradients for linear regression
    :param fittedModel: An object with a function .predict(XPred) that returns yPred, and a
        field .coef_ that returns the coefficient vector
        An example would be modelFn = LinearRegression
        MUST be a linear model
    :param X: The data on which we have fit the model
    :param y: The observations on which we fit the model
    :param alpha: The nominal type I error for our confidence intervals
    :return: Lower and upper bounds for the alpha-confidence interval on the coefficients of the fitted model,
        as a tuple
    """
    # Get confidence intervals on the coefficients
    lcbs, ucbs = analyticLinearCis(fittedModel, X, y, alpha)
    # Take the most and least extreme values of the CI's. The "most extreme" is always the maximum of the absolute
    # values. If the interval does not cover zero, then the "least extreme" is the minimum of the absolute values.
    # If the interval does cover zero, then the "least extreme" is zero.
    mostExtreme = np.max(np.vstack([np.abs(lcbs), np.abs(ucbs)]), axis=0)
    leastExtreme = np.array([min(np.abs(lcb), np.abs(ucb)) if np.sign(lcb) == np.sign(ucb) else 0 for (lcb, ucb) in zip(lcbs, ucbs)])

    # Get the most and least extreme feature importances
    return integratedGradients_Linear(leastExtreme.T, X), integratedGradients_Linear(mostExtreme.T, X)