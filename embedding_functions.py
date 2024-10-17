# import networkx as nx
from scipy.spatial import distance
from scipy import stats
import numpy as np
import pandas as pd
from scipy import sparse
import plotly.express as px
import nodevectors
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numba as nb
import networkx as nx


def single_spectral(A, d, seed=None):
    if seed is not None:
        UA, SA, VAt = sparse.linalg.svds(A, d, random_state=seed)
    else:
        UA, SA, VAt = sparse.linalg.svds(A, d)

    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    XA = UA @ np.diag(np.sqrt(SA))
    return XA

def single_spectral_X(A, d, seed=None):
    if seed is not None:
        UA, SA, VAt = sparse.linalg.svds(A, d, random_state=seed)
    else:
        UA, SA, VAt = sparse.linalg.svds(A, d)

    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    # Output the left spectral embedding
    XA = UA @ np.diag(np.sqrt(SA))
    return XA
    
def single_spectral_Y(A, d, seed=None):
    if seed is not None:
        UA, SA, VAt = sparse.linalg.svds(A, d, random_state=seed)
    else:
        UA, SA, VAt = sparse.linalg.svds(A, d)

    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    # Output the right spectral embedding
    YA = VA @ np.diag(np.sqrt(SA))
    return YA

def sparse_stack(As):
    A = As[0]
    T = len(As)
    for t in range(1, T):
        A = sparse.hstack((A, As[t]))
    return A


def UASE(As, d, flat=True, sparse_matrix=False, return_left=False, verbose=False):
    """Computes the unfolded adjacency spectral embedding"""
    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    if verbose:
        print("Forming unfolded matrix...")

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

    if verbose:
        print("Computing spectral embedding...")

    # SVD spectral embedding
    UA, SA, VAt = sparse.linalg.svds(A, d)
    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    YA_flat = VA @ np.diag(np.sqrt(SA))
    XA = UA @ np.diag(np.sqrt(SA))
    
    if verbose:
        print("Formatting embedding...")

    if flat:
        YA = YA_flat
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t, :, :] = YA_flat[n * t : n * (t + 1), :]

    if not return_left:
        return YA
    else:
        return XA, YA



def unfolded_prone(As, d, p=1, q=1, flat=True, two_hop=False, sparse_matrix=False):
    """Computes the unfolded prone embedding"""

    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        if len(As.shape) == 2:
            As = np.array([As[:, :]])
        if len(As.shape) == 3:
            T = len(As)
            A = As[0, :, :]
            for t in range(1, T):
                A = np.block([A, As[t]])

    # Construct the dilated unfolded adjacency matrix
    DA = sparse.bmat([[None, A], [A.T, None]])
    DA = sparse.csr_matrix(DA)

    # Compute node2vec
    n2v_obj = nodevectors.ProNE(
        n_components=d,
        verbose=False,
    )
    if two_hop:
        DA = DA @ DA.T

    n2v = n2v_obj.fit_transform(DA)

    # Take the rows of the embedding corresponding to the right embedding
    # ([0:n] will be left embedding)
    right_n2v = n2v[n:, :]

    if flat:
        YA = right_n2v
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t] = right_n2v[t * n : (t + 1) * n, 0:d]

    return YA
    
    
    
@nb.njit()
def test_temporal_displacement_two_times(ya, n, n_sim=1000):
    """Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    This can be used in the case where you are comparing ONLY two time points.
    In this case it's much faster than the above function.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    changepoint (int > radius, <= T-radius)
    n_sim: (int) number of permuted test statistics computed.
    """
    # Number of time points must = 2 for this method
    T = 2

    # Select time point embedding just before and after the changepoint
    ya1 = ya[0:n, :]
    ya2 = ya[n : 2 * n, :]

    # Get observed value of the test
    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)
    t_obs = np.linalg.norm(sum_axis)

    # Permute the sets
    t_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        # Randomly swap the signs of each row of displacement
        # signs = np.random.choice([-1, 1], n)
        signs = np.random.randint(0, 2, size=displacement.shape) * 2 - 1
        displacement_permuted = displacement * signs
        sum_axis_permuted = displacement_permuted.sum(axis=0)
        t_star = np.linalg.norm(sum_axis_permuted)
        t_stars[sim_iter] = t_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_stars >= t_obs)
    return p_hat


def test_temporal_displacement_two_times_not_jit(ya, n, radius=1, n_sim=1000):
    """Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    radius: (int) number of time points to permute before/after the changepoint
    changepoint (int > radius, <= T-radius)
    n_sim: (int) number of permuted test statistics computed.
    """
    # Number of time points must = 2 for this method
    T = 2

    # Select time point embedding just before and after the changepoint
    ya1 = ya[0:n, :]
    ya2 = ya[n : 2 * n, :]

    # Get observed value of the test
    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)
    t_obs = np.linalg.norm(sum_axis)

    # Permute the sets
    t_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        # Randomly swap the signs of each row of displacement
        # signs = np.random.choice([-1, 1], n)
        signs = np.random.randint(0, 2, size=displacement.shape) * 2 - 1
        displacement_permuted = displacement * signs
        sum_axis_permuted = displacement_permuted.sum(axis=0)
        t_star = np.linalg.norm(sum_axis_permuted)
        t_stars[sim_iter] = t_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_stars >= t_obs)
    return p_hat





###########################################

def ainv(x):
    return np.linalg.inv(x.T @ x) @ x.T


def plot_embedding(ya, n, T, tau, return_df=False, title=None):
    yadf = pd.DataFrame(ya[:, 0:2])
    yadf.columns = ["Dimension {}".format(i + 1) for i in range(yadf.shape[1])]
    yadf["Time"] = np.repeat([t for t in range(T)], n)
    yadf["Community"] = list(tau) * T
    yadf["Community"] = yadf["Community"].astype(str)
    pad_x = (max(ya[:, 0]) - min(ya[:, 0])) / 50
    pad_y = (max(ya[:, 1]) - min(ya[:, 1])) / 50
    fig = px.scatter(
        yadf,
        x="Dimension 1",
        y="Dimension 2",
        color="Community",
        animation_frame="Time",
        range_x=[min(ya[:, 0]) - pad_x, max(ya[:, 0]) + pad_x],
        range_y=[min(ya[:, 1]) - pad_y, max(ya[:, 1]) + pad_y],
    )
    if title:
        fig.update_layout(title=title)

    fig.show()
    if return_df:
        return yadf


def test_mean_change(ya1, ya2, tau_permutation=None, n_sim=1000):
    """Mean change permutation test from two embedding sets.

    Use this function for spatial testing. If you require temporal testing - see vector displacement test.

    Null hypothesis: The two embedding sets come from the same distribution.
    Alternative: They two embedding sets come from different distributions

    ya1 & ya2: (numpy arrays) Two equal-shape embedding sets
    tau_permutation: (numpy array) Known community allocations - no need to use in spatial testing case.
    n_sim: (int) Number of permuted test statistics computed.
    """
    print("USING MEAN CHANGE")

    # Compute mean difference between observed sets
    obs1 = np.mean(ya1, axis=0)
    obs2 = np.mean(ya2, axis=0)
    t_obs = np.linalg.norm(obs1 - obs2)

    n_1 = ya1.shape[0]
    n_2 = ya2.shape[0]

    # Compute permuted test statistics
    if tau_permutation is not None:
        # Get the idx of the permutable groups (in this case groups of tau=0 or tau=1)
        ya_all = np.row_stack([ya1, ya2])
        idx_group_1 = np.where(np.tile(tau_permutation, 2) == 0)[0]
        idx_group_2 = np.where(np.tile(tau_permutation, 2) == 1)[0]

        # Permute the idx groups
        t_obs_stars = np.zeros((n_sim,))
        for i in range(n_sim):
            idx_group_1_shuffled = idx_group_1.copy()
            idx_group_2_shuffled = idx_group_2.copy()

            np.random.shuffle(idx_group_1_shuffled)
            np.random.shuffle(idx_group_2_shuffled)

            ya_all_perm = np.zeros((ya1.shape[0] * 2, ya1.shape[1]))
            ya_all_perm[idx_group_1_shuffled] = ya_all[idx_group_1]
            ya_all_perm[idx_group_2_shuffled] = ya_all[idx_group_2]

            # Split back into the two time sets, now including the temporal permutations
            ya_star_1 = ya_all_perm[0:n_1, :]
            ya_star_2 = ya_all_perm[n_2:, :]

            # Compute the permuted test statistic
            obs_star_1 = np.mean(ya_star_1, axis=0)
            obs_star_2 = np.mean(ya_star_2, axis=0)

            t_obs_star = np.linalg.norm(obs_star_1 - obs_star_2)
            # t_obs_star = test_statistic(ya_star_1, ya_star_2)
            t_obs_stars[i] = t_obs_star

    else:
        # Permute all nodes
        ya_all = np.row_stack([ya1, ya2])
        n_1 = ya1.shape[0]
        n_2 = ya2.shape[0]
        t_obs_stars = np.zeros((n_sim,))
        for i in range(n_sim):
            # Create two randomly sets from the observed sets
            idx_shuffled = np.arange(n_1 + n_2)
            np.random.shuffle(idx_shuffled)
            idx_1 = idx_shuffled[0:n_1]
            idx_2 = idx_shuffled[n_1 : n_1 + n_2]
            ya_star_1 = ya_all[idx_1]
            ya_star_2 = ya_all[idx_2]

            # Calculate test statistic for permuted vectors
            obs_star_1 = np.mean(ya_star_1, axis=0)
            obs_star_2 = np.mean(ya_star_2, axis=0)
            t_obs_star = np.linalg.norm(obs_star_1 - obs_star_2)

            # Add permuted test statistic to null distribution
            t_obs_stars[i] = t_obs_star

    # p-value is given by how extreme the observed test statistic is vs the null distribution
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)

    return p_hat


@nb.njit()
def vector_displacement_test(ya1, ya2):
    """Computes the vector displacement between two embedding sets

    ya1 & ya2 are numpy arrays of equal shape (n, d)
    """

    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)

    # Magnitude of average displacement vector
    t_obs = np.linalg.norm(sum_axis)

    return t_obs


@nb.njit()
def masked_vector_displacement_test(ya1, ya2, mask=None):
    """Computes the vector displacement between two embedding sets

    ya1 & ya2 are numpy arrays of equal shape (n, d)
    """

    displacement = ya2 - ya1
    if mask is not None:
        displacement = displacement[mask]

    sum_axis = displacement.sum(axis=0)

    # Magnitude of average displacement vector
    t_obs = np.linalg.norm(sum_axis)

    return t_obs


@nb.njit
def faster_inner_product_distance(ya1, ya2):
    """Computes the inner product distance between two embedding sets

    ya1 & ya2 are numpy arrays of equal shape (n, d)
    """

    inner_product_dists = 1 - np.sum(ya1 * ya2, axis=1)
    sum_axis = inner_product_dists.sum(axis=0)

    # Magnitude of average displacement vector
    # t_obs = np.linalg.norm(sum_axis)

    return sum_axis


@nb.njit
def cosine_distance(ya1, ya2):
    norms = np.linalg.norm(ya1, axis=1) * np.linalg.norm(ya2, axis=1)
    mask = np.where(norms != 0)
    dot_product = np.dot(ya1[mask], ya2[mask].T)
    distances = 1 - (dot_product / norms[mask])
    sum_axis = distances.sum(axis=0)
    t_obs = np.linalg.norm(sum_axis)
    return t_obs


@nb.njit()
def test_temporal_displacement(ya, n, T, changepoint, n_sim=1000):
    """Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    This function should only be used when comparing more than two time points across a single changepoint.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    T: (int) number of time points
    radius: (int) number of time points to permute before/after the changepoint
    changepoint (int > radius, <= T-radius)
    n_sim: (int) number of permuted test statistics computed.
    """

    if changepoint < 1:
        raise Exception("Changepoint must be at least 1.")
    elif changepoint > T:
        raise Exception("Changepoint must be less than or equal T")

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * changepoint : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = vector_displacement_test(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        # for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            possible_perms = ya[j::n, :]
            ya_star[j::n, :] = possible_perms[np.random.choice(T, T, replace=False), :]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * changepoint, :]
        ya_star_2 = ya_star[n * changepoint : n * (changepoint + 1), :]
        t_obs_star = vector_displacement_test(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat


@nb.njit()
def mean_change_test_stat(ya1, ya2):
    ya1_mean_cols = np.zeros((ya1.shape[1]))
    ya2_mean_cols = np.zeros((ya1.shape[1]))
    for colidx in range(ya1.shape[1]):
        ya1_mean_cols[colidx] = np.mean(ya1[:, colidx])
        ya2_mean_cols[colidx] = np.mean(ya2[:, colidx])

    ya_mean_col_diff = ya1_mean_cols - ya2_mean_cols
    t_obs = np.linalg.norm(ya_mean_col_diff)
    return t_obs


@nb.njit()
def test_graph_mean_change(ya, n, T, changepoint, n_sim=1000, perm_range=2):
    """
    Computes a mean change permutation test with temporal node permutations.

    changepoint: (int 1-T) the first time point of a change.
    perm_range: (int > 1) number of time points either side of the changepoint from which permutations can be taken.
        If changepoint = 5, then permuations will be taken from times 3,4,5,6.
    """

    # If the perm_range was 1, only 0 p-values would be returned as the test statistic will be the same, no matter the permutation
    if perm_range <= 1:
        raise Exception("The permutation range must be at least 2.")

    if changepoint + perm_range > T:
        raise Exception(
            "The permuation range goes above the number of available time points."
        )

    T_for_perms = perm_range * 2

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * (changepoint) : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = mean_change_test_stat(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            # Restrict the possible permutations to perm_range around the changepoint
            possible_perms = ya[
                np.array(
                    [
                        j + t * n
                        for t in range(
                            changepoint - perm_range, changepoint + perm_range
                        )
                    ]
                ),
                :,
            ]

            # times_for_perms = np.arange(
            #     changepoint - perm_range, changepoint + perm_range)
            # possible_perms = possible_perms[times_for_perms, :]

            ya_star[
                np.array(
                    [
                        j + t * n
                        for t in range(
                            changepoint - perm_range, changepoint + perm_range
                        )
                    ]
                ),
                :,
            ] = possible_perms[
                np.random.choice(T_for_perms, T_for_perms, replace=False), :
            ]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * (changepoint), :]
        ya_star_2 = ya_star[n * (changepoint) : n * (changepoint + 1), :]

        t_obs_star = mean_change_test_stat(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat


@nb.njit()
def test_graph_mean_change_asym(
    ya, n, T, changepoint, n_sim=1000, perm_range=1, before=True
):
    """
    Computes a mean change permutation test with temporal node permutations.

    changepoint: (int 1-T) the first time point of a change.
    perm_range: (int > 0) number of time points either side of the changepoint from which permutations can be taken.
    before: (bool) should the extra non-testing time point be before[True] or after[False] the change?

        If changepoint=5 and before=True, then permuations will be taken from times 3,4,5.
        If changepoint=5 and before=False, then permuations will be taken from times 4,5,6.
    """

    # If the perm_range was 1, only 0 p-values would be returned as the test statistic will be the same, no matter the permutation
    if perm_range < 1:
        raise Exception("The permutation range must be at least 1.")

    if changepoint + perm_range > T:
        raise Exception(
            "The permuation range goes above the number of available time points."
        )

    T_for_perms = perm_range * 2 + 1

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * (changepoint) : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = mean_change_test_stat(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            # Restrict the possible permutations to perm_range around the changepoint
            if before:
                # Allow the nodes to permute more times before the change
                possible_perms = ya[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - (perm_range + 1), changepoint + perm_range
                            )
                        ]
                    ),
                    :,
                ]

                ya_star[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - (perm_range + 1), changepoint + perm_range
                            )
                        ]
                    ),
                    :,
                ] = possible_perms[
                    np.random.choice(T_for_perms, T_for_perms, replace=False), :
                ]

            else:
                # Allow the nodes to permute more times after the change
                possible_perms = ya[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - perm_range, changepoint + (perm_range + 1)
                            )
                        ]
                    ),
                    :,
                ]

                ya_star[
                    np.array(
                        [
                            j + t * n
                            for t in range(
                                changepoint - perm_range, changepoint + (perm_range + 1)
                            )
                        ]
                    ),
                    :,
                ] = possible_perms[
                    np.random.choice(T_for_perms, T_for_perms, replace=False), :
                ]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * (changepoint), :]
        ya_star_2 = ya_star[n * (changepoint) : n * (changepoint + 1), :]

        t_obs_star = mean_change_test_stat(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat










# @nb.njit
def general_unfolded_matrix(As, sparse_matrix=False):
    """Forms the general unfolded matrix from an adjacency series"""
    T = len(As)
    n = As[0].shape[0]

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))

        # Construct the dilated unfolded adjacency matrix
        DA = sparse.bmat([[None, A], [A.T, None]])
        DA = sparse.csr_matrix(DA)
    else:
        A = As[0]
        for t in range(1, T):
            A = np.block([A, As[t]])

        DA = np.zeros((n + n * T, n + n * T))
        DA[0:n, n:] = A
        DA[n:, 0:n] = A.T

    return DA

