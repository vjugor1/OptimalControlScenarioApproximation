import numpy as np
from scipy.linalg import sqrtm


def normalize_ineqs(A: np.ndarray, b: np.ndarray):
    """
    For each inequality `A[i]^T x <= b[i]` makes `A[i]` unit norm and rescales `b[i]` correspondingly

    Args:
        A (numpy.ndarray): 2d matrix, each row is a normal vector of a plane
        b (numpy.ndarray): 1d array, each element is a constant corresponding to normal vector

    Returns:
        tuple: normalized matrix and constant of given system of inequalities
    """
    norms = np.linalg.norm(A, axis=1)
    return A / norms.reshape(A.shape[0], 1), b / norms


def standartize(
    Gamma_: np.ndarray, Beta_: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, standartize_ineqs: bool = True,
):
    """Given linear inequalities gamma_i^T (x + xi) <= b_i with xi being arbitrary gaussian vector,
         reformulates inequality to have randomness represented as standard normal vector
         Gamma_i^T x + A_i^T chi <= beta_i

         xi = sqrt_Sigma @ chi + mu
         Gamma (x + sqrt_Sigma @ chi + mu) <= b
         Gamma x + (Gamma @ sqrt_Sigma) chi + Gamma mu <= b
         Gamma x + A chi <= b - Gamma mu

    Args:
        Gamma_ (np.ndarray): matrix that contains gamma_i - normals
        Beta_ (np.ndarray): vector that contains plane constants - b_i
        mu (np.ndarray): mean for distribution of xi
        Sigma (np.ndarray): covariance matrix
        standartize_ineqs (bool): if to normalize ineqs on Gamma_ @ sqrtm(Sigma)

    Returns:
        tuple: new matrix for x, chi - standard normal vector, beta - constants
    """
    sqrt_Sigma = sqrtm(Sigma)
    Gamma_sqrt_Sigma = Gamma_ @ sqrt_Sigma
    if standartize_ineqs:
        Gamma = Gamma_ / np.linalg.norm(Gamma_sqrt_Sigma, axis=1).reshape(
            Gamma_.shape[0], -1
        )
        A = (Gamma_ @ sqrt_Sigma) / np.linalg.norm(Gamma_sqrt_Sigma, axis=1).reshape(
            Gamma_.shape[0], -1
        )
        Beta = (Beta_ - (Gamma_ @ mu)) / np.linalg.norm(Gamma_sqrt_Sigma, axis=1)  # Beta =
    else:
        Gamma = Gamma_
        A = (Gamma_ @ sqrt_Sigma)
        Beta = (Beta_ - (Gamma_ @ mu))
    
    return Gamma, Beta, A
    # Gamma = []
    # A = []
    # Beta = []
    # sqrt_Sigma = sqrtm(Sigma)
    # for j in range(Gamma_.shape[0]):
    #     gamma_ = Gamma_[j]
    #     if np.linalg.norm(gamma_) > 0.0:
    #         # standartize one plane normal
    #         sqrt_S_gamma = sqrt_Sigma.dot(gamma_)
    #         gamma = gamma_ / np.linalg.norm(sqrt_S_gamma)
    #         Gamma.append(gamma)
    #         a = sqrt_S_gamma / np.linalg.norm(sqrt_S_gamma)
    #         A.append(a)
    #         # Beta after standartization
    #         beta = (Beta_[j] - gamma_.dot(mu)) / np.linalg.norm(sqrt_S_gamma)
    #         Beta.append(beta)

    # # casting type to numpy arrays
    # Gamma = np.array(Gamma)
    # A = np.array(A)
    # Beta = np.array(Beta)
    # return Gamma, Beta, A
