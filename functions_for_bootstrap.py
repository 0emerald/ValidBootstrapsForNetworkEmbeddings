import numpy as np
from scipy import sparse
import warnings
from experiment_setup import *
from embedding_functions import *
import gc
from tqdm import tqdm
import numba as nb


def plot_power(p_hat_list, plot=True):
    # Plot the ROC curve
    roc = []
    alphas = []
    for alpha in np.linspace(0, 1, 100):
        alphas.append(alpha)
        num_below_alpha = sum(p_hat_list < alpha)
        roc_point = num_below_alpha / len(p_hat_list)
        roc.append(roc_point)

    # Get the power at the 5% significance level
    power_significance = 0.05
    power_idx = alphas.index(min(alphas, key=lambda x: abs(x - power_significance)))
    power = roc[power_idx]

    if plot:
        plt.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), linestyle="--", c="grey")
        _ = plt.plot(alphas, roc)
        plt.show()

    return power



def parametric_bootstrap(A, d, B, return_P_hat=False, sparse=False, verbose=False):
    """
    Generates B bootstrapped adjacency matrices from a single adjacency matrix A.

    input:
    A: (numpy array (n, n)) adjacency matrix
    d: (int) embedding dimension. In theory the rank of the noise-free A matrix.
    B: (int) number of bootstrap samples

    output:
    A_star: (numpy array (B, n, n)) bootstrapped adjacency matrices
    """

    if verbose:
        print("Generating bootstrapped adjacency matrices...")

    # Compute the spectral embedding of A
    X_hat = single_spectral(A, d)

    # Compute the estimated probability matrix
    P_hat = X_hat @ X_hat.T

    # Check if P_hat is a valid probability matrix
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1]. Consider increasing n.")

    if verbose:
        if not sparse:
            A_star = np.array([make_inhomogeneous_rg(P_hat) for _ in range(B)])
        else:
            A_star = np.array(
                [make_inhomogeneous_rg_sparse(P_hat) for _ in range(B)]
            )
    else:
        if not sparse:
            A_star = np.array([make_inhomogeneous_rg(P_hat) for _ in range(B)])
        else:
            A_star = np.array([make_inhomogeneous_rg_sparse(P_hat) for _ in range(B)])

    if return_P_hat:
        return A_star, P_hat
    else:
        return A_star


def row_sample_with_replacement(A, B):
    """
    Implemented for just a single bootstrap sample for now

    input:
    A: (numpy array (n, n)) adjacency matrix
    B: (int) number of bootstrap samples to take

    output:
    A_star: (numpy array (B, n, n)) bootstrapped adjacency matrices
    """

    # this is the number of nodes
    n = A.shape[0]

    # initialize an array to store the bootstrapped adjacency matrices
    A_row_jumbles = np.zeros((B,n,n))

    for i in range(B):
        # for each bootstrap sample, select which rows to put in the new matrix
        idx = np.random.choice(n, size=n, replace=True)
        for j in range(n):
            # put the rows in the new matrix
            A_row_jumbles[i][j,:] += A[idx[j],:]

    return A_row_jumbles


def edgelist_jackknife(A, B, num_times):
    """
    Pick a random entry and set it to zero
    """
    n = A.shape[0]
    A_star = np.zeros((B, n, n))

    for i in range(B):
        A_star[i] = A.copy()
        for j in range(num_times):
            idx = np.random.choice(n, size=2, replace=True)
            A_star[i][idx[0], idx[1]] = 0
            A_star[i][idx[1], idx[0]] = 0

    return A_star



# NOTE this is garbage. it don't work
# @nb.njit
# def bootstrap_testing_per_node(ya_node_set, normal_samples_for_each_node, n_sim=200):
#     # Hypothesis test to check whether the observed difference is significant with respect to the
#     #  bootstrap samples

#     assert len(ya_node_set.shape) == 3

#     n_node_set = ya_node_set.shape[1]

#     # Written slowly so I can jit
#     displacement = ya_node_set[0] - ya_node_set[1]
#     observed = np.array([np.linalg.norm(displacement[i]) for i in range(n_node_set)])

#     # TODO: don't forget bonferroni correction (is there a +1 or no??)
#     p_hat_per_node = np.zeros((n_node_set,))
#     for i in range(n_node_set):
#         tests_for_node = np.zeros((n_sim + 1,))
#         tests_for_node[0] = observed[i]
#         tests_for_node[1:] = normal_samples_for_each_node[i]

#         # for sim in range(n_sim):
#         # Draw a bunch of samples from a normal dist and use a hypothesis test to check if the
#         #  observed difference is significant
#         # This is because we expect the difference between the two embeddings to be normally
#         #  distributed with mean 0 and variance 4 sigma_hat

#         # 4 here because we do (Y1 - Ytrue) + (Y0 - Ytrue) (4 combos)
#         # normal_sample = np.random.multivariate_normal(
#         #     np.zeros(d), 4 * (sigma_hats[i]), size=1
#         # ).flatten()
#         # tests_for_node[sim] = np.linalg.norm(normal_sample)

#         p_hat = 1 / (n_sim + 1) * np.sum(tests_for_node >= observed[i])

#         # Apply bonferroni correction
#         p_hat_per_node[i] = p_hat
#         # p_hat_per_node[i] = p_hat / n_node_set

#     return p_hat_per_node
