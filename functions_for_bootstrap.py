import numpy as np
from scipy import sparse
import warnings
from experiment_setup import *
from embedding_functions import *
import gc
from tqdm import tqdm
import numba as nb#
import random


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



"""This is needed for the test_bootstrap function"""
@nb.njit
def P_est_from_A_obs(n, A_obs, n_neighbors, indices):
    P_est = np.zeros((n, n))
    for i in range(n):
        idx = indices[i]
        A_i = (1/n_neighbors) * np.sum(A_obs[:, idx], axis=1)
        P_est[:, i] = A_i
    return P_est

"""This takes in an adjacency matrix,
embeds the matrix via spectral embedding,
finds the k-nearest neighbors of each node, 
uses the A values of the k nearest neighbors to estimate the P matrix.
You are your own first neighbour, so k=1 just gives P_est as A that is input. 
It doesn't do a test!!!!!
It just gives you B bootstrapped matrices for a given observed matrix!
"""
def test_bootstrap(A, d, B=100, n_neighbors=5):
    n = A.shape[0]
    A_obs = A.copy()

    # Embed the graphs -------------------------------  

    yhat = UASE([A], d=d, flat=True)

    # run a k-NN on the embedding yhat
    # Here we use Minkowski distance, with p=2 (these are the defaults),
    # which corresponds to Euclidean distance
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(yhat)
    distances, indices = nbrs.kneighbors(yhat)

    # Estimate the P matrix -------------------------------
    P_est = P_est_from_A_obs(n, A_obs, n_neighbors=n_neighbors, indices=indices)

    # Bootstrap -----------------------------------------
    # B = 100
    random.seed(10)
    p_vals = []
    A_boots = []
    for i in range(B):
        A_est = make_inhomogeneous_rg(P_est)

        yhat_est = UASE([A_obs,A_est], d=d)
        p_val = test_temporal_displacement_two_times(yhat_est, n)
        p_vals.append(p_val)
        A_boots.append(A_est)

    return p_vals, A_boots
    
def create_single_kNN_bootstrap(A, d, Q=1000, n_neighbors=5):
	n = A.shape[0]
	A_obs = A.copy()

	# Embed the graphs -------------------------------  

	yhat = UASE([A], d=d, flat=True)

	# run a k-NN on the embedding yhat
	# Here we use Minkowski distance, with p=2 (these are the defaults),
	# which corresponds to Euclidean distance
	from sklearn.neighbors import NearestNeighbors
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(yhat)
	distances, indices = nbrs.kneighbors(yhat)

	# Estimate the P matrix -------------------------------
	P_est = P_est_from_A_obs(n, A_obs, n_neighbors=n_neighbors, indices=indices)

	# Bootstrap -----------------------------------------
	A_est = make_inhomogeneous_rg(P_est)

	# embed the observed and bootstrapped matrices together --------------------------------
	yhat_est = UASE([A_obs,A_est], d=d)

	# do a test between the obs and the bootstrap, get a p-value ---------------------------------
	p_val = test_temporal_displacement_two_times(yhat_est, n, n_sim=Q)

	return p_val, A_est


def check_matrix_range(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    if min_val < 0 or max_val > 1:
        raise ValueError("Matrix values must be within the range [0, 1].")
    else:
        print("Matrix values are within the range [0, 1].")

def multiply_with_weights(A, w1, w2):
    # Reshape w1 and w2 to have dimensions (n, 1)
    w1_reshaped = w1[:, np.newaxis]  # shape (n, 1)
    w2_reshaped = w2[np.newaxis, :]  # shape (1, n)
    
    # Perform element-wise multiplication
    M = w1_reshaped * w2_reshaped * A
    
    return M


"""Should work for estimating bootstraps of A that can be weighted and directed"""
def test_bootstrap_universal(A, d, B=100, n_neighbors=5):
    n = A.shape[0]
    A_obs = A.copy()

    # Embed the graphs -------------------------------  
    yhat = UASE([A], d=d, flat=True)

    # run a k-NN on the embedding yhat
    # Here we use Minkowski distance, with p=2 (these are the defaults), 
    # which corresponds to Euclidean distance
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='minkowski', p=2).fit(yhat)
    distances, indices = nbrs.kneighbors(yhat)

    # Estimate the "weighted" P matrix (idk how you'd sample from this) -------------------------------
    P_est = P_est_from_A_obs(n, A_obs, n_neighbors=n_neighbors, indices=indices)
    
    # Calculate the in and out weight vectors of A_obs -------------------------------  
    out_weights = np.sum(A_obs, axis=1) #row sum
    in_weights = np.sum(A_obs, axis=0) #column sum

    # Adjusted P_est matrix ( not sure if this will be in [0,1]) -------------------------------
    P_est_adj = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (out_weights[i]*in_weights[j]) == 0:
                P_est_adj[i,j] = 0
            else:
                P_est_adj[i,j] = (1/(out_weights[i]*in_weights[j])) * P_est[i,j]

    # Check matrix range
    check_matrix_range(P_est_adj)

    # Bootstrap -----------------------------------------
    # B = 100
    p_vals = []
    A_boots = []
    for i in range(B):
        A_est_adj = make_inhomogeneous_rg(P_est_adj)
        A_est = multiply_with_weights(out_weights, in_weights, A_est_adj)[0]

        yhat_est = UASE([A_obs,A_est], d=d)
        p_val = test_temporal_displacement_two_times(yhat_est, n)
        p_vals.append(p_val)
        A_boots.append(A_est)

    return p_vals, A_boots


