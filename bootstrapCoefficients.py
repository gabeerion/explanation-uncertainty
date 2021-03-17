# Estimate confidence intervals on the coefficients using bootstrap resampling of the training data

import numpy as np

def bootstrapCis(modelFn, X, y, alpha=0.05, replicates=1000):
    """
    Bootstrap a two-sided confidence interval at level alpha on the coefficients in modelFn
    :param modelFn: A function that, when called with no arguments, returns an object  with a function .fit(X, y),
        a function .predict(XPred) that returns yPred, and a field .coef_ that returns the coefficient vector
        An example would be modelFn = LinearRegression
    :param X: We will sample with replacement from (X, y) to generate bootstrap replicates.
    :param y: See X
    :param alpha: The desired Type 1 error rate for the confidence intervals
    :param replicates: The number of bootstrap replicates desired. For smaller alpha, this needs to increase.
    :return: Lower and upper bounds for the alpha-confidence interval on the coefficients of the fitted model,
        as a tuple
    """
    rng = np.random.default_rng()
    n, d = X.shape
    coefs = np.zeros((d, replicates))
    for i in range(replicates):
        # Get the bootstrap samples
        idcs = rng.integers(low=0, high=n, size=n)
        Xsample = X[idcs, :]
        ySample = y[idcs]

        # Get a clean model object and fit it to the bootstrapped data
        model = modelFn()
        model.fit(Xsample, ySample)
        coefs[:, i] = model.coef_

    # Find the desired percentiles, on a per-coefficient basis
    coefs = np.sort(coefs, axis=1)
    lIndex = int(replicates * alpha / 2.0)
    uIndex = int(replicates * (1 - alpha / 2.0))

    return coefs[:, lIndex], coefs[:, uIndex]