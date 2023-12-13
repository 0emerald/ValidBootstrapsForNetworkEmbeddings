# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embedding_functions import *
from experiment_setup import *

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
# Generate an SBM of two communities
n = 500
T = 2
d = 2
As, tau, _ = make_iid(n, T, iid_prob=0.85)
# As, tau, _ = make_temporal_simple(n, T, move_prob=0.9)

# ya = UASE(As, d, flat=False)
# %%
# Bootstrap first time point B times using parametric bootstrap
B = 100
X_hat = single_spectral(As[0], d)
P_hat = X_hat @ X_hat.T

if np.min(P_hat) < 0 or np.max(P_hat) > 1:
    print("Warning: P_hat is not a valid probability matrix")

A_star = [make_inhomogeneous_rg(P_hat) for _ in range(B)]

A_star_with_obs = np.array(list(As) + A_star)

ya_star_with_obs = UASE(A_star_with_obs, d, flat=False)
ya_star = ya_star_with_obs[2:].copy()
ya = ya_star_with_obs[:2].copy()


# %%
# Estimate sigma hat for each node
sigma_hats = np.zeros((n, d, d))
for i in range(n):
    sigma_hats[i] = np.cov(ya_star[:, i, :].T)


# # %%
# # Plot the differences to make sure that they look sensible
# i = 0
# new_point = ya[0, i, :] - ya[1, i, :]

# bootstrap_points = np.zeros((B, d))
# for b in range(B):
#     bootstrap_points[b] = ya[0, i, :] - ya_star[b, i, :]

# plt.figure()
# plt.scatter(bootstrap_points[:, 0], bootstrap_points[:, 1], color="C0")
# plt.scatter(new_point[0], new_point[1], color="red")


# %%
# Hypothesis test to check whether the observed difference is significant with respect to the
#  bootstrap samples

p_hat_list = []
community_of_interest = 0
for i in np.where(tau == community_of_interest)[0]:
    observed = np.linalg.norm(ya[0, i, :] - ya[1, i, :])

    all_tests = []
    all_tests.append(observed)
    # for b in range(B):
    #     all_tests.append(np.linalg.norm(ya[0, i, :] - ya_star[b, i, :]))

    sigma_hat = sigma_hats[i]
    for b in range(B):
        # Draw a bunch of samples from a normal dist and use a hypothesis test to check if the
        #  observed difference is significant
        normal_sample = np.random.multivariate_normal(
            np.zeros(d), d * (sigma_hat), size=1
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

plt.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), linestyle="--", c="grey")
_ = plt.plot(alphas, roc)
_ = plt.title("P-values for community {}".format(community_of_interest))


# %%
# Plot the embedding of the first time point with one of its boostrapped versions
plt.figure()

plt.scatter(ya[0, tau == 0, 0], ya[0, tau == 0, 1], color="C0", alpha=0.4)
plt.scatter(ya[0, tau == 1, 0], ya[0, tau == 1, 1], color="C1", alpha=0.4)

plt.scatter(ya_star[2, tau == 0, 0], ya_star[2, tau == 0, 1], color="blue", alpha=0.4)
plt.scatter(ya_star[2, tau == 1, 0], ya_star[2, tau == 1, 1], color="red", alpha=0.4)


# %%
plt.figure()

plt.scatter(ya_star[:, 0, 0], ya_star[:, 0, 1], color="black")
plt.scatter(ya[0, 0, 0], ya[0, 0, 1], color="red")

# %%
# Compare to true
As_true, tau, _ = make_iid(n, 100, iid_prob=0.9)

B = 100
X_hat = single_spectral(As_true[0], d)
P_hat = X_hat @ X_hat.T
As_star = [make_inhomogeneous_rg(P_hat) for _ in range(B)]

As_both = np.array(list(As_true) + As_star)

ya_both = UASE(As_both, d, flat=False)
# %%
plt.scatter(ya_both[0, i, 0], ya_both[0, i, 1], color="red")
plt.scatter(ya_both[1:100, i, 0], ya_both[1:100, i, 1], color="green")
plt.scatter(ya_both[100:, i, 0], ya_both[100:, i, 1], color="blue")

# %%
