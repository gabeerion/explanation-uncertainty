# The GLRT rejects any model as 'implausible' if its likelihood ratio wrt the unconstrained model falls
# outside the middle (1-alpha) percentile.
# Computing the GLRT statistic (cutoff value) is difficult for copmlicated models, and the chi-squared approximation
# suffers from both inaccuracy for small n, and reliance on a calculated degrees of freedom (which is not necessarily
# available for complicated models). Therefore, we propose to compute the alpha/2 and (1-alpha/2) percentiles of the
# GLRT statistic distribution by bootstrapping.
# We will bootstrap percentiles via repeated calls to the provided model's .fit and .predict functions; we expect
# all models to support such an interface.
# There remains the question of whether we can get statistically sound confidence intervals on the MLE's likelihood
# by bootstrapping from the data that we used to fit the MLE, or if we have to partition the data into train and test.
# We still need to figure that out; it's safest to use test data for now.

import numpy as np

def bootstrapGLRTcis(modelFn, X, y, logLikFn, alpha=0.05, replicates=1000):
    """
    Bootstrap a two-sided confidence interval at level alpha on the likelihood of the maximum likelihood solution from
    fitting model modelFn to data.
    :param modelFn: A function that, when called with no arguments, returns an object  with a function .fit(X, y) and
        a function .predict(XPred) that returns yPred.
        An example would be modelFn = LinearRegression
    :param X: We will sample with replacement from (X, y) to generate bootstrap replicates. This should probably be
        a validation set.
    :param y: See X
    :param logLikFn: a function that takes (y, yPred) and produces the log likelihood (eg, the MSE)
    :param alpha: The desired Type 1 error rate for the confidence intervals
    :param replicates: The number of bootstrap replicates desired. For smaller alpha, this needs to increase.
    :return: Lower and upper bounds for the alpha-confidence interval on the log likelihood of the fitted model,
        as a tuple
    """
    rng = np.random.default_rng()
    n, d = X.shape
    logLikValues = np.zeros(replicates)
    for i in range(replicates):
        # Get the bootstrap samples
        idcs = rng.integers(low=0, high=n, size=n)
        Xsample = X[idcs, :]
        ySample = y[idcs]

        # Get a clean model object and fit it to the bootstrapped data
        model = modelFn()
        model.fit(Xsample, ySample)
        yFit = model.predict(Xsample)
        logLikValues[i] = logLikFn(ySample, yFit)

    # Find the desired percentiles
    logLikValues = np.sort(logLikValues)
    lIndex = int(replicates*alpha/2.0)
    uIndex = int(replicates*(1 - alpha/2.0))

    return logLikValues[lIndex], logLikValues[uIndex]
