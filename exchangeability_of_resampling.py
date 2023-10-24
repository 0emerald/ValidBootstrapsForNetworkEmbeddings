# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embedding_functions import *
from experiment_setup import *
import numba as nb
from tqdm import tqdm


# %%
@nb.njit
def unfolded_edge_resample(obs_A, T):
    """First snapshot is the observed A, the rest are resampled

    Currently just resampling edges with replacement
    """

    # Get a list of edges from the adjacency matrix
    edges = np.argwhere(obs_A == 1)
    num_edges = edges.shape[0]

    n = obs_A.shape[0]
    As_tilde = np.zeros((T, n, n))
    As_tilde[0] = obs_A.copy()
    for i in range(1, T):
        # Sample num_edges with replacement from edges
        # This is the bootstrap step
        bootstrap_edges_idx = np.random.choice(num_edges, num_edges, replace=True)
        new_edges = edges[bootstrap_edges_idx].copy()

        # Form adjacency from new edge
        new_A = np.zeros((n, n))
        for edge in new_edges:
            new_A[edge[0], edge[1]] += 1
            # new_A[edge[1], edge[0]] = 1

        As_tilde[i] = new_A

    return As_tilde


def unfolded_p_matrix_resample(obs_A, T, d):
    # TODO

    X_hat = single_spectral(obs_A, d=d)
    P_hat = X_hat @ X_hat.T

    # align with observed
    P_hat_rot = procrust_align(obs_A, P_hat)
    P_hat = P_hat_rot.copy()

    As_tilde = np.zeros((T, n, n))
    for t in range(T):
        A_tilde = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_hat
        As_tilde[t] = A_tilde

    return As_tilde


# %%
n = 400
T = 3
d = 2

As, tau, _ = make_iid(n=n, T=1, iid_prob=0.5)

# As, tau, _ = make_temporal_simple(n=n, T=T, move_prob=0.9)


# ya = UASE(As, d, flat=False)
# ya = regularised_ULSE(As, d, flat=False, regulariser=0)

# %%
# Check that we keep exchangeability

obs_A = As[0].copy()
p_hat_list = []

for _ in range(200):
    # Select resampling method
    # As_tilde = unfolded_edge_resample(obs_A, T)
    As_tilde = unfolded_p_matrix_resample(obs_A, T, d=100)

    # Set the first snapshot to be the observed A as a sanity check
    As_tilde[0] = obs_A.copy()

    ya_tilde = UASE(As_tilde, d, flat=False)

    ya1 = ya_tilde[0]  # embedding of observed
    ya2 = ya_tilde[2]  # embedding of resampled

    # Paired displacement testing
    p_hat = test_temporal_displacement_two_times(
        np.row_stack([ya1, ya2]), n=ya1.shape[0]
    )
    p_hat_list.append(p_hat)

# Plot the p-value dist (or ROC)
roc = []
alphas = []
for alpha in np.linspace(0, 1, len(p_hat_list)):
    alphas.append(alpha)
    num_below_alpha = sum(p_hat_list < alpha)
    roc_point = num_below_alpha / len(p_hat_list)
    roc.append(roc_point)

# Get the power at the 5% significance level
power_significance = 0.05
power_idx = alphas.index(min(alphas, key=lambda x: abs(x - power_significance)))
power = roc[power_idx]
print("Power: {}".format(power))

plt.figure()
plt.plot(alphas, roc)
plt.plot([0, 1], [0, 1], "--", color="black")

# %%
As_tilde = unfolded_p_matrix_resample(obs_A, T, d=d)
As_tilde[0] = obs_A.copy()
ya_tilde_flat = UASE(As_tilde, d, flat=True)
# ya_tilde_flat = regularised_ULSE(As_tilde, d, flat=True, regulariser=0)
# ya_tilde_flat = unfolded_n2v(As_tilde, d, flat=True)
plot_embedding(ya_tilde_flat, n, T, tau)

# %% [markdown]
### Exploring the affect of embedding dimension on the P_hat estimation
# %%

As, tau, _ = make_temporal_simple(n=n, T=T, move_prob=0.9, K=8)

x_hat_d_list = []
power_list = []
d_to_try = [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 19]
for x_hat_d in tqdm(d_to_try):
    obs_A = As[0].copy()
    p_hat_list = []

    for _ in range(200):
        # Select resampling method
        # As_tilde = unfolded_edge_resample(obs_A, T)
        As_tilde = unfolded_p_matrix_resample(obs_A, T, d=x_hat_d)

        # Set the first snapshot to be the observed A as a sanity check
        As_tilde[0] = obs_A.copy()

        ya_tilde = UASE(As_tilde, d, flat=False)

        ya1 = ya_tilde[0]  # embedding of observed
        ya2 = ya_tilde[2]  # embedding of resampled

        # Paired displacement testing
        p_hat = test_temporal_displacement_two_times(
            np.row_stack([ya1, ya2]), n=ya1.shape[0]
        )
        p_hat_list.append(p_hat)

    # Plot the p-value dist (or ROC)
    roc = []
    alphas = []
    for alpha in np.linspace(0, 1, len(p_hat_list)):
        alphas.append(alpha)
        num_below_alpha = sum(p_hat_list < alpha)
        roc_point = num_below_alpha / len(p_hat_list)
        roc.append(roc_point)

    # Get the power at the 5% significance level
    power_significance = 0.05
    power_idx = alphas.index(min(alphas, key=lambda x: abs(x - power_significance)))
    power = roc[power_idx]
    print("x_hat_d: {} | Power: {}".format(x_hat_d, power))

    x_hat_d_list.append(x_hat_d)
    power_list.append(power)

# %%
plt.figure(figsize=(10, 5))
plt.plot(x_hat_d_list, power_list, marker="o")
plt.axhline(0.05, linestyle="--", color="black")
plt.axvline(8, linestyle="--", color="red")

# %%
