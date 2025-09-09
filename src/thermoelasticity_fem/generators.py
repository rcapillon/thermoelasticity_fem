import numpy as np
from scipy.stats import gamma


def generator_gamma_rv(mean_value, dispersion_coefficient, n_samples):
    """
    Generates samples for a gamma random variable with prescribed mean and dispersion coefficient.
    """
    std = mean_value * dispersion_coefficient
    a = 1 / dispersion_coefficient ** 2
    b = (std ** 2) / mean_value
    samples = gamma.rvs(a, scale=b, size=n_samples)

    return samples
