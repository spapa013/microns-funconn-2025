import statsmodels.api as sm
import numpy as np


def multipletests(p, method="fdr_bh"):
    not_nan = ~np.isnan(p)
    p_adj = np.full(p.shape, np.nan)
    p_adj[not_nan] = sm.stats.multipletests(p[not_nan], method=method)[1]
    return p_adj


def sample_cond_distribution(x, y, cond, rng=np.random.default_rng(), n_samples=1):
    """Samples from the conditional distribution of (x, y) given cond, assuming Gaussian.

    Parameters
    ----------
    x : array-like
        The conditioning variable
    y : array-like
        The variable to sample from
    cond : array-like
        The conditions
    rng : np.random.Generator, optional
        by default np.random.default_rng()
    n_samples : int, optional
        number of samples to sample per condition, by default 1

    Returns
    -------
    np.array
        Samples from the conditional distribution of (x, y) given cond, (cond, return) will have the same distribution as (x, y).
    """
    data = np.c_[x, y]
    cond = np.array(cond)

    # fit a 2D gaussian
    mean = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False)
    # Extract components from the covariance matrix
    sigma_xx = covariance[0, 0]  # Variance of x
    sigma_yy = covariance[1, 1]  # Variance of y
    sigma_xy = covariance[0, 1]  # Covariance between x and y

    samples = []
    for c in cond:
        # Calculate the conditional mean and variance of y given x'
        mu_y_given_x_prime = mean[1] + sigma_xy / sigma_xx * (c - mean[0])
        sigma_y_given_x_prime = np.sqrt(sigma_yy - sigma_xy**2 / sigma_xx)
        # Sample y' from the conditional distribution
        _y = rng.normal(mu_y_given_x_prime, sigma_y_given_x_prime, n_samples)
        samples.append(_y)
    return np.array(samples)
