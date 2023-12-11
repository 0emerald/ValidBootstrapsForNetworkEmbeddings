# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embedding_functions import *
from experiment_setup import *


# %%
def gaussian_ellipse(mean, cov):
    """Stolen from Ian"""
    if mean.shape == (1, 2):
        mean = np.array(mean)[0]
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    rtheta = np.arctan2(u[1], u[0])
    v = 3.0 * np.sqrt(2.0) * np.sqrt(v)
    width = v[0]
    height = v[1]

    R = np.array(
        [
            [np.cos(rtheta), -np.sin(rtheta)],
            [np.sin(rtheta), np.cos(rtheta)],
        ]
    )
    theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)
    x, y = np.dot(R, np.array([x, y]))
    x += mean[0]
    y += mean[1]

    return [x, y]


def unfolded_resample(obs_A, T):
    """First snapshot is the observed A, the rest are resampled"""
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

    As_tilde = np.zeros((T, n, n))
    for t in range(T):
        A_tilde = np.random.uniform(0, 1, n**2).reshape((n, n)) < P_hat
        As_tilde[t] = A_tilde

    return As_tilde


# %%
n = 200
T = 3
d = 2

# As, tau, _ = make_temporal_simple(n=n, T=T, move_prob=0.9)
As, tau, _ = make_iid(n=n, T=T, iid_prob=0.9)
ya = UASE(As, d, flat=False)

node_of_interest = 0
node_positions = ya[:, node_of_interest, :]

# %%
# Resampling
obs_A = As[0].copy()


# As_tilde = unfolded_resample(obs_A, T)
As_tilde = unfolded_p_matrix_resample(obs_A, T, d)

ya_tilde = UASE(As_tilde, d, flat=True)


# %%
degrees_obs = np.sum(obs_A, axis=0)
degrees_tilde = np.sum(As_tilde[-1], axis=0)
degrees_tilde_2 = np.sum(As_tilde[-2], axis=0)

# plot the sorted degrees (descending)
plt.figure()
plt.title("Sorted Degrees")
plt.plot(np.sort(degrees_obs)[::-1], label="obs")
plt.plot(np.sort(degrees_tilde)[::-1], label="tilde")
plt.plot(np.sort(degrees_tilde_2)[::-1], label="tilde 2")
plt.legend()

# %%
# Procrustes align ya_tilde with ya

# [This is fine because in theory ya_tilde is a resample of an estimated ya? I.e. up to noise they the same?]
ya_flat = ya.reshape((T * n, d))
ya_tilde_flat = ya_tilde.reshape((T * n, d))

ya_tilde_rot_flat = procrust_align(ya_flat, ya_tilde_flat)

ya_tilde_new = ya_tilde_rot_flat.reshape((T, n, d))
ya_tilde = ya_tilde_new.copy()
ya_tilde_flat = ya_tilde.reshape((T * n, d))

# node_of_interest = 0
# node_positions_tilde = ya_tilde[:, node_of_interest, :]


# %%
# Plot ya_tilde_flat with its mean and gaussian ellipse and ya_flat

# obs_ya = ya_tilde[0]
# resample_ya = ya_tilde[1]

obs_ya = ya_flat
resample_ya = ya_tilde_flat
# resample_ya = ya_tilde[1]

mean_tilde_flat_1 = np.mean(resample_ya[np.where(tau == 0)], axis=0)
mean_tilde_flat_2 = np.mean(resample_ya[np.where(tau == 1)], axis=0)

mean_flat_1 = np.mean(obs_ya[np.where(tau == 0)], axis=0)
mean_flat_2 = np.mean(obs_ya[np.where(tau == 1)], axis=0)

cov_tilde_1 = np.cov(resample_ya[np.where(tau == 0)].T)
ellipse_tilde_1 = gaussian_ellipse(mean_tilde_flat_1[0:2], cov_tilde_1[0:2, 0:2])
cov_tilde_2 = np.cov(resample_ya[np.where(tau == 1)].T)
ellipse_tilde_2 = gaussian_ellipse(mean_tilde_flat_2[0:2], cov_tilde_2[0:2, 0:2])

cov_1 = np.cov(obs_ya[np.where(tau == 0)].T)
ellipse_1 = gaussian_ellipse(mean_flat_1[0:2], cov_1[0:2, 0:2])
cov_2 = np.cov(obs_ya[np.where(tau == 1)].T)
ellipse_2 = gaussian_ellipse(mean_flat_2[0:2], cov_2[0:2, 0:2])

plt.figure()
plt.scatter(obs_ya[:, 0], obs_ya[:, 1], c="blue", s=5, alpha=1)
plt.scatter(resample_ya[:, 0], resample_ya[:, 1], c="orchid", s=5, alpha=1)

plt.scatter(mean_flat_1[0], mean_flat_1[1], marker="o", s=40, c="green")
plt.scatter(mean_flat_2[0], mean_flat_2[1], marker="o", s=40, c="green")


plt.scatter(mean_tilde_flat_1[0], mean_tilde_flat_1[1], marker="o", s=40, c="red")
plt.scatter(mean_tilde_flat_2[0], mean_tilde_flat_2[1], marker="o", s=40, c="red")

plt.plot(ellipse_tilde_1[0], ellipse_tilde_1[1], "--", color="red")
plt.plot(ellipse_tilde_2[0], ellipse_tilde_2[1], "--", color="red")

plt.plot(ellipse_1[0], ellipse_1[1], "--", color="green")
plt.plot(ellipse_2[0], ellipse_2[1], "--", color="green")

# legend
plt.scatter([], [], c="blue", s=5, alpha=1, label="Obs embedding")
plt.scatter([], [], c="orchid", s=5, alpha=1, label="Resampled embedding")
plt.scatter([], [], marker="o", s=40, c="green", label="Obs mean")
plt.scatter([], [], marker="o", s=40, c="red", label="Resampled mean")
plt.plot([], [], "--", color="green", label="Obs variance")
plt.plot([], [], "--", color="red", label="Resampled variance")

# place legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

# %%
# Given, mean preservence, can we use this to get a tighter variance?

T = 200
As_tilde = unfolded_resample(obs_A, T)

xa_normal = single_spectral(obs_A, d)

xa_tilde, ya_tilde = UASE(As_tilde, d, flat=False, return_left=True)


# ya_avg = np.zeros((n, d))
# for i in range(n):
#     ya_avg[i] = np.mean(ya_tilde[:, i, :], axis=0)

xa_tilde = procrust_align(xa_normal, xa_tilde)
# %%
