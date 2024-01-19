# %%
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embedding_functions import *
from experiment_setup import *
import gc
from functions_for_bootstrap import parametric_bootstrap

# %%
"""
TO DO
1. Embed two time points where a change may or may not occur
2. Bootstrap the first graph B times and unfold embed to get bootstrapped embeddings
3. Compute sigma hat as the sample variance of the bootstrapped embeddings for node i
4. Check if the distance between the two embeddings is greater than 2 sigma hat (get p-value)

REALITY CHECKS
1. Check that you get uniform p-values when no change is present after many runs of the procedure on 
    iid data
2. Make sure that it works on systems beyond SBM

SUMMARY OF STUFF I'VE DONE
- Got to the point where we can look at p-values to see if the embedding at the next time point has 
    changed significantly from the previous one relative to bootstrapped embeddings
- For easier problems (iid_prob=0.85, closer to 1 = easier), p-values are uniformly distributed (with  
    a high enough B).
- P-values are non-uniform when a change occurs (great!)
- However, in iid examples where the communities are closer (and therefore more difficult to bootstrap),
    the p-values appear slightly super-uniform. So this procedure is not generally valid.
- May need to look at some theory to make it generally valid.

"""

# %%
####################
# This plot looks really weird.
# In most of the examples the original sample seems to be on the edge of the bootstrap samples
# I'd expect it to be in the middle
####################
i = 15
plt.scatter(
    ya_both[1:100, i, 0], ya_both[1:100, i, 1], color="green", label="True Resample"
)
plt.scatter(ya_both[100:, i, 0], ya_both[100:, i, 1], color="blue", label="Bootstrap")
plt.scatter(ya_both[0, i, 0], ya_both[0, i, 1], color="red", label="Original")

plt.legend()
# %%
T = 2
d = 2
# n_to_try = [100, 200, 500, 1500, 2000]
n_to_try = [2500]

power_list = []

B = 50

for n in n_to_try:
    ####################################################################
    # ## USING ONLINE EMBEDDING FOR MEMORY EFFICIENCY ##

    # SBM
    # As, tau = make_iid(n, T, iid_prob=0.55)  # Easier to bootstrap
    # As, tau = make_iid(n, T, iid_prob=0.9)  # Harder to bootstrap (often conservative)

    # # Embed the first time point
    # ya = UASE(As[:2], d, flat=False, sparse_matrix=False)
    # xa = ya[0].copy()
    # xa_inv = xa @ np.linalg.inv(xa.T @ xa)

    # # Bootstrap first time point B times using parametric bootstrap
    # ya_star = np.zeros((B, n, d))
    # for b in range(B):
    #     A_star = parametric_bootstrap(As[0], d, B=1, sparse=False)
    #     ya_star[b] = A_star[0].T @ xa_inv

    #     del A_star
    ####################################################################

    # ####################################################################
    # ## NORMAL WAY OF PARAMETRIC BOOTSTRAP ##

    # # SBM
    # As, tau = make_iid(n, T, iid_prob=0.55)  # Easier to bootstrap
    # # As, tau = make_iid(n, T, iid_prob=0.9)  # Harder to bootstrap (often conservative)

    # A_star = parametric_bootstrap(As[0], d, B=B, sparse=False)
    # A_star_with_obs = np.concatenate((As, A_star), axis=0)

    # # Embed all graphs
    # ya_star_with_obs = UASE(A_star_with_obs, d, flat=False, sparse_matrix=False)
    # ya_star = ya_star_with_obs[2:].copy()  # Bootstrap embedding
    # ya = ya_star_with_obs[:2].copy()  # Embedding of observed T graphs

    # ####################################################################

    #################
    ## SPARSE WAY OF PARAMETRIC BOOTSTRAP ##
    As, tau = make_iid_sparse(n, T, iid_prob=0.55)
    A_star = parametric_bootstrap(As[0], d, B=B, sparse=True)

    A_star_with_obs = []
    A_star_with_obs.extend(As)
    A_star_with_obs.extend(A_star)

    # Embed all graphs
    ya_star_with_obs = UASE(A_star_with_obs, d, flat=False, sparse_matrix=True)
    ya_star = ya_star_with_obs[2:].copy()  # Bootstrap embedding
    ya = ya_star_with_obs[:2].copy()  # Embedding of observed T graphs

    #################

    # Estimate sigma hat for each node using the embeddings of the bootstrapped graphs
    sigma_hats = np.zeros((n, d, d))
    for i in range(n):
        sigma_hats[i] = np.cov(ya_star[:, i, :].T)

    # Make sure sigma_hats are not nan (likely B=1, make it at least 2)
    assert np.sum(np.isnan(sigma_hats)) == 0

    # Hypothesis test to check whether the observed difference is significant with respect to the
    #  bootstrap samples
    p_hat_list = []
    observed_values = []
    community_of_interest = 0
    for i in np.where(tau == community_of_interest)[0]:
        observed = np.linalg.norm(ya[0, i, :] - ya[1, i, :])
        observed_values.append(observed)

        all_tests = []
        all_tests.append(observed)

        sigma_hat = sigma_hats[i]
        for b in range(B):
            # Draw a bunch of samples from a normal dist and use a hypothesis test to check if the
            #  observed difference is significant
            # This is because we expect the difference between the two embeddings to be normally
            #  distributed with mean 0 and variance 4 sigma_hat

            # 4 here because we do (Y1 - Ytrue) + (Y0 - Ytrue) (4 combos)
            normal_sample = np.random.multivariate_normal(
                np.zeros(d), 4 * (sigma_hat), size=1
            ).flatten()
            all_tests.append(np.linalg.norm(normal_sample))

        # Are new_point and bootstrap_points from the same distribution?
        p_hat = 1 / (B + 1) * np.sum(all_tests >= observed)
        p_hat_list.append(p_hat)

    # Plot the ROC curve
    alphas_list = []
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
    power_list.append(power)
    print("n: {}, power: {}".format(n, power))

    plt.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), linestyle="--", c="grey")
    _ = plt.plot(alphas, roc)
    _ = plt.title("P-values for community {}".format(community_of_interest))

# %%
# Plot the embedding of the first time point with one of its boostrapped versions
plt.figure()

# plt.figure()
plt.scatter(
    ya_star[2, tau == 0, 0],
    ya_star[2, tau == 0, 1],
    color="blue",
    alpha=0.4,
    label=r"Bootstrap $\tau=0$",
)
plt.scatter(
    ya_star[2, tau == 1, 0],
    ya_star[2, tau == 1, 1],
    color="red",
    alpha=0.4,
    label=r"Bootstrap $\tau=1$",
)
plt.scatter(
    ya[0, tau == 0, 0],
    ya[0, tau == 0, 1],
    color="C0",
    alpha=0.4,
    label=r"Original $\tau=0$",
)
plt.scatter(
    ya[0, tau == 1, 0],
    ya[0, tau == 1, 1],
    color="C1",
    alpha=0.4,
    label=r"Original $\tau=1$",
)


plt.legend()
