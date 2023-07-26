import math
import numpy as np
import scipy.linalg as la

from scipy import stats
from scipy.stats import uniform
from scipy.stats import multinomial
import itertools

from tqdm.notebook import tqdm


class ConditionedPolytopeGaussianSampler:
    """This class describes a sampler that follows a sampling algorithm to provide
    Gaussian distribution is assumed to be be standard normal (see `preprocessing.standardtize` -  special function to ''rectification'' of Gaussian - making standard)
    Planes normal vectors are assumed to have unit euclidean norm
    1) A sample from a mixture
    2) A series of samples

    Samples for a mixture are obtained as follows
    1) Pick a plane `i` with probability `w_i`; The latter is propotional ...
    2) Generate Gaussian sample conditionel on plane `i` as follows
    2.1) Sample z ~ N(0, I_n)
    2.2) Sample u ~ U(0,1)
    2.3) Compute y = F^{-1}(u F(-b_i))
    2.4) Compute and yield x = - (a_i * y + (I - a_i.T * a_i) z)
    """

    def __init__(self, A, b):
        assert np.allclose(
            np.linalg.norm(A, axis=1), np.ones(A.shape[0])
        ), "Planes normal vectors must have unit lengths"

        # Initializing polytope data
        self.A = A
        self.b = b
        # Initializing mixture weights
        cdfs = stats.norm.cdf(-b)
        # self.w = np.ones(len(cdfs)) / len(cdfs)
        self.w = cdfs / np.sum(cdfs)
        # Multinomial sampler initilization
        self.rng = np.random.default_rng()

    def sample(self):
        """Generator for samples from a standard Gaussian conditioned on a plane `A[i]` which is chosen with probability `w[i]`
        Algorithm:
            1) Sample z ~ N(0, I_n)
            2) Sample u ~ U(0,1)
            3) Compute y = F^{-1}(u F(-b_i))
            4) Compute and yield x = - (a_i * y + (I - a_i.T * a_i) z)
        Returns:
            [type]: [description]
        """
        while True:
            # Pick a hyperplane
            multinom_rv = self.rng.multinomial(1, self.w, size=1)
            hplane = np.argmax(multinom_rv)
            A_i = self.A[hplane, :]
            b_i = self.b[hplane]

            # Sample conditionally on it
            J = len(self.b)
            n = self.A.shape[1]
            z = self.rng.multivariate_normal(np.zeros(n), np.eye(n))
            u = self.rng.uniform(0, 1)
            y = stats.norm.ppf(u * stats.norm.cdf(-b_i))
            x = A_i * y + (np.eye(n) - np.outer(A_i, A_i)).dot(z)
            x = -x
            if np.isnan(x).any():
                raise ValueError
            yield x
