# %%
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embedding_functions import *
from experiment_setup import *
import gc
from functions_for_bootstrap import parametric_bootstrap
import re
import numba as nb


# %%
@nb.njit
def compute_sample_sigma(ya_boots, n, d):
    """
    Given a dynamic embedding (T, n, n), compute the sample covariance matrix for each node
    """
    sigma_hats = np.zeros((n, d, d))
    for i in range(n):
        sigma_hats[i] = np.cov(ya_boots[:, i, :].T)

    return sigma_hats


@nb.njit
def get_power_fast(p_hat_list):
    # Plot the ROC curve
    roc = np.zeros((100,))
    alphas = np.linspace(0, 1, 100)
    for i in range(100):
        alpha = alphas[i]
        num_below_alpha = np.sum(p_hat_list < alpha)
        roc[i] = num_below_alpha / len(p_hat_list)

    # Get the power at the 5% significance level
    power_significance = 0.05
    power_idx = np.argmin(np.abs(alphas - power_significance))
    power = roc[power_idx]

    return power


def plot_power(p_hat_list):
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

    plt.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), linestyle="--", c="grey")
    _ = plt.plot(alphas, roc)
    plt.show()

    return power


# %%
T = 100  # Number of true resamples to estimate sigma from
d = 2
d_bootstrap = 2
# d_to_try = [2, 3, 5, 10, 20, 100, 150]
# d_to_try = [1, 1, 1, 1, 2, 2, 2, 2, 100, 100, 100, 100]

n_to_try = [500]

iid_prob_to_try = np.linspace(0.5, 0.9, 10)

n = n_to_try[0]  # TODO for debugging

power_list = []

B = 1  # Number of bootstrapped resamples
# num_p_vals = 200
n_sim = 1000

community_of_interest = 0  # Set of nodes to perform testing on

# power_per_n = np.zeros((len(d_to_try),))
# power_per_n = np.zeros((len(n_to_try),))
power_per_n = np.zeros((len(iid_prob_to_try),))
# for ni, d_bootstrap in enumerate(d_to_try):
# for ni, n in enumerate(n_to_try):
for ni, iid_prob in tqdm(enumerate(iid_prob_to_try)):
    ##################################################
    # # REALITY CHECK (make sure A_boots comes from bootstrap of A_obs after removing)
    # A_boots_with_true, tau = make_iid(
    #     n, T + B, iid_prob=0.50
    # )  # True resamples (many to estimate sigma)
    # A_true = A_boots_with_true[:T].copy()  # True resamples (many to estimate sigma)
    # A_obs = A_true[0].copy()  # Observed graph (to bootstrap from)
    # A_boots = A_boots_with_true[T:].copy()  # True resamples (many to estimate sigma)
    # del A_boots_with_true
    # print("WARNING: Reality check in place")
    ##################################################

    ##################################################
    # TODO: will have to split up communities when they're distinguishable
    A_true, tau = make_iid(
        n, T, iid_prob=iid_prob
    )  # True resamples (many to estimate sigma)
    A_boots = parametric_bootstrap(
        A_true[0], d_bootstrap, B=B, sparse=False, verbose=False
    )
    ##################################################

    A_boots_with_true = np.concatenate((A_true, A_boots), axis=0)

    # Embed observed and bootstrapped graphs
    # print("Embedding...")
    # ya_boots_with_true = UASE(A_boots_with_true, d, flat=False, sparse_matrix=True)
    ya_boots_with_true = UASE(A_boots_with_true, d, flat=False, sparse_matrix=False)
    del A_boots_with_true

    ####################################################################
    n_test = n
    # OPTIONAL: Select the community of interest
    # ------------------------------------------------------------------
    # node_set = np.where(tau == community_of_interest)[0]
    # ya_boots_with_true = ya_boots_with_true[:, node_set, :].copy()
    # n_test = len(node_set)
    ####################################################################

    ya_boots = ya_boots_with_true[T:].copy()  # Bootstrap embedding
    ya_true = ya_boots_with_true[:T].copy()  # Embedding of observed T graphs
    ya_obs = ya_true[0].copy()  # Embedding of observed graph
    del ya_boots_with_true

    # print("Embedding complete.")
    # print("Testing...")

    # Estimate sigma from the true resamples for each node
    sigma_true = compute_sample_sigma(ya_true, n_test, d)
    assert np.sum(np.isnan(sigma_true)) == 0
    del ya_true

    # For each node, we expect that |Xobs_i - Xtrue_i| ~ N(0, 2 sigma_i)
    # H0: |Xobs_i - Xboots_i| ~ N(0, 2 sigma_i)
    # H1: |Xobs_i - Xboots_i| !~ N(0, 2 sigma_i)
    # We get a p-value for each node (as we are not assuming constant variance in general)
    # As we're working with a random graph where we DO ahve constant variance, I'll just treat these as
    #  n p-values for the same node
    p_vals = np.zeros((n_test,))
    for i in range(n_test):
        all_tests = np.zeros((n_sim + 1,))
        t_obs = np.linalg.norm(ya_obs[i, :] - ya_boots[0, i, :])
        all_tests[0] = t_obs

        # we do (Yobs - Yboots) - if the bootstrap is perfect, Yboots_i has covariance sigma_true
        # therefore the covariance of this quantity will be 2 sigma_true.
        normal_samples = np.random.multivariate_normal(
            np.zeros(d), 2 * (sigma_true[i]), size=n_sim
        )
        all_tests[1:] = np.linalg.norm(normal_samples, axis=1)

        # Compute p-value
        p_vals[i] = 1 / (n_sim + 1) * np.sum(all_tests >= t_obs)

    # power = plot_power(p_vals)
    power = get_power_fast(p_vals)
    # print("Power:", power)
    power_per_n[ni] = power
# %%
###############################################
## Bootstrap embedding as a whole tends to overdispurse (when communities are distinguishable)

## This overdispursion leads to invalid p-values
###############################################
plt.figure()
plt.title("Whole bootstrap embedding")
plt.scatter(ya_obs[:, 0], ya_obs[:, 1], label="True")
plt.scatter(ya_boots[0, :, 0], ya_boots[0, :, 1], label="Bootstrap")

# %%
###############################################
## The bootstrap node distributions does not line up with the true

## The variance of the bootstrap node distributions does look like the true
###############################################
plt.figure()
plt.title("Bootstraps of a single node")

B_plot = 100
As_boots_plot = parametric_bootstrap(
    A_true[0], d, B=B_plot, sparse=False, verbose=False
)
As_boots_and_obs_plot = np.concatenate((A_true, As_boots_plot), axis=0)
ya_boots_and_obs_plot = UASE(As_boots_and_obs_plot, d, flat=False, sparse_matrix=False)
ya_true_plot = ya_boots_and_obs_plot[:T].copy()
ya_boots_plot = ya_boots_and_obs_plot[T : B_plot + T].copy()
ya_obs_plot = ya_boots_and_obs_plot[0].copy()

node_to_plot = 0
plt.scatter(
    ya_true_plot[:, node_to_plot, 0],
    ya_true_plot[:, node_to_plot, 1],
    label="True",
)
plt.scatter(
    ya_boots_plot[:, node_to_plot, 0],
    ya_boots_plot[:, node_to_plot, 1],
    label="Bootstrap",
)
plt.scatter(
    ya_obs_plot[node_to_plot, 0], ya_obs_plot[node_to_plot, 1], label="Observed"
)
plt.legend()

# %%
