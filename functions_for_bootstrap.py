import numpy as np
from scipy import sparse
import warnings
from experiment_setup import *
from embedding_functions import *
import gc


def parametric_bootstrap(A, d, B, return_P_hat=False, sparse=False):
    """
    Generates B bootstrapped adjacency matrices from a single adjacency matrix A.

    input:
    A: (numpy array (n, n)) adjacency matrix
    d: (int) embedding dimension. In theory the rank of the noise-free A matrix.
    B: (int) number of bootstrap samples

    output:
    A_star: (numpy array (B, n, n)) bootstrapped adjacency matrices
    """

    # Compute the spectral embedding of A
    X_hat = single_spectral(A, d)

    # Compute the estimated probability matrix
    P_hat = X_hat @ X_hat.T

    # Check if P_hat is a valid probability matrix
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1]. Consider increasing n.")

    if not sparse:
        A_star = np.array([make_inhomogeneous_rg(P_hat) for _ in range(B)])
    else:
        A_star = np.array([make_inhomogeneous_rg_sparse(P_hat) for _ in range(B)])

    if return_P_hat:
        return A_star, P_hat
    else:
        return A_star
