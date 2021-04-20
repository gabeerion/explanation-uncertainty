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

def bootstrapGLRTcis(modelFn, X, y, nllFn, alpha=0.05, replicates=1000):
    """
    Bootstrap a one-sided confidence interval at level alpha on the log likelihood of the maximum likelihood solution
    to the model fitted with modelFn on data from whatever distribution generated X and y
    Do this by first bootstrapping the negative log of the GLRT test statistic
        log max_theta L(theta) - log L(theta0) 
    where theta0 is the MLE on the original data set. After you have the alpha-percentile of that distribution, call it
    c_alpha, and we know that 
        P_{Y|X ~ H0} (log max_theta L(theta) - log L(theta0)  > c_alpha) < alpha.
    Finally, to report a floor on the likelihood with the data (X, y), return
        c_alpha + log L(theta0 | X, y)
    
    :param modelFn: A function that, when called with no arguments, returns an object  with a function .fit(X, y) and
        a function .predict(XPred) that returns yPred.
        An example would be modelFn = LinearRegression
    :param X: We will sample with replacement from (X, y) to generate bootstrap replicates. This should be the data set
              you're fitting to (if we change this in the future, no problem, but we'll need to return Lambda instead of
              the minimum plausible likelihood)
    :param y: See X
    :param nllFn: a function that takes (y, yPred) and produces the negative log likelihood (eg, the MSE)
    :param alpha: The desired Type 1 error rate for the confidence intervals
    :param replicates: The number of bootstrap replicates desired. For smaller alpha, this needs to increase.
    :return: An upper bound for the alpha-confidence interval on the negative log likelihood
    """
    rng = np.random.default_rng()
    n, d = X.shape
    negLogLambdas = np.zeros(replicates)
    
    # Get the original MLE solution
    model0 = modelFn()
    model0.fit(X, y)
    for i in range(replicates):
        # Get the bootstrap samples
        idcs = rng.integers(low=0, high=n, size=n)
        Xsample = X[idcs, :]
        ySample = y[idcs]

        # Get a clean model object and fit it to the bootstrapped data
        model = modelFn()
        model.fit(Xsample, ySample)
        negLogLambdas[i] = -1*(nllFn(ySample, model.predict(Xsample)) - nllFn(ySample, model0.predict(Xsample)))

    # Find the desired percentiles
    negLogLambdas = np.sort(negLogLambdas)
    lIndex = int(replicates*(1-alpha))
    cAlpha = negLogLambdas[lIndex]
    
    # As a convenience, return the upper confidence bound on the negative log likelihood
    # (which is offset from the nll of the MLE on this data X, y)
    nll0 = nllFn(y, model0.predict(X))

    return nll0 + cAlpha
