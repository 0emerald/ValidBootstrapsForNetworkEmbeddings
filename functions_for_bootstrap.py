import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import warnings
from experiment_setup import *
from embedding_functions import *
import gc
from tqdm import tqdm
import numba as nb  #
import random
from numba.typed import List, Dict
from numba import types
from numba.types import ListType
from numpy.linalg import LinAlgError
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


int_list_type = ListType(types.int32)


def plot_power(p_hat_list, plot=True):
    """Plot the QQ-plot curve"""
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
    
    
    
def compute_roc_and_areas(p_hat_list, significance_level=0.05):
    """Plot the QQ-plot and compute the area between the x=y line and the curve, the 'Bootstrap Validity Score'"""
    roc = []
    alphas = []

    for alpha in np.linspace(0, 1, 100):
        alphas.append(alpha)
        num_below_alpha = sum(p_hat_list < alpha)
        roc_point = num_below_alpha / len(p_hat_list)
        roc.append(roc_point)

    # Get the power at the significance level
    power_idx = alphas.index(min(alphas, key=lambda x: abs(x - significance_level)))
    power = roc[power_idx]

    # Calculate the area between ROC and y=x line
    def compute_area_above_below_curve(x, y):
        area_above = 0.0
        area_below = 0.0

        for i in range(1, len(x)):
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[i - 1], y[i]
            line0, line1 = x0, x1  # Since line y = x

            if y1 == y0:  # Vertical segment
                if y0 > x0:
                    area_above += (y0 - x0) * (x1 - x0)
                else:
                    area_below += (x0 - y0) * (x1 - x0)
                continue

            # Find intersection with y = x
            if (y0 >= x0 and y1 >= x1) or (y0 <= x0 and y1 <= x1):
                if y0 >= x0 and y1 >= x1:
                    area_above += 0.5 * (y0 + y1 - x0 - x1) * (x1 - x0)
                else:
                    area_below += 0.5 * (x0 + x1 - y0 - y1) * (x1 - x0)
            else:
                x_intersect = x0 + (x0 - y0) * (x1 - x0) / (y1 - y0)
                if y0 < x0:
                    area_below += 0.5 * (x0 - y0) * (x_intersect - x0)
                    area_above += 0.5 * (y1 - x1) * (x1 - x_intersect)
                else:
                    area_above += 0.5 * (y0 - x0) * (x_intersect - x0)
                    area_below += 0.5 * (x1 - y1) * (x1 - x_intersect)

        return area_above, area_below

    x = np.linspace(0, 1, 100)
    roc_interpolated = np.interp(x, alphas, roc)

    # Compute areas
    area_above, area_below = compute_area_above_below_curve(x, roc_interpolated)
    total_area = area_above + area_below

    return {
        "area_above": area_above,
        "area_below": area_below,
        "total_area": total_area
    }



# Calculate the area between ROC and y=x line
def compute_area_above_below_curve(x, y):
    area_above = 0.0
    area_below = 0.0

    for i in range(1, len(x)):
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        line0, line1 = x0, x1  # Since line y = x

        if y1 == y0:  # Vertical segment
            if y0 > x0:
                area_above += (y0 - x0) * (x1 - x0)
            else:
                area_below += (x0 - y0) * (x1 - x0)
            continue

        # Find intersection with y = x
        if (y0 >= x0 and y1 >= x1) or (y0 <= x0 and y1 <= x1):
            if y0 >= x0 and y1 >= x1:
                area_above += 0.5 * (y0 + y1 - x0 - x1) * (x1 - x0)
            else:
                area_below += 0.5 * (x0 + x1 - y0 - y1) * (x1 - x0)
        else:
            x_intersect = x0 + (x0 - y0) * (x1 - x0) / (y1 - y0)
            if y0 < x0:
                area_below += 0.5 * (x0 - y0) * (x_intersect - x0)
                area_above += 0.5 * (y1 - x1) * (x1 - x_intersect)
            else:
                area_above += 0.5 * (y0 - x0) * (x_intersect - x0)
                area_below += 0.5 * (x1 - y1) * (x1 - x_intersect)

    return area_above, area_below




def edgelist_sample_with_replacement_addRandomEdges_v2(A):
    """Samples edges with replacement and ensures that the bootstrapped matrix has the same number of edges.
    In a binary setting, any edge selected more than once will be set to 1.
    Random edges will be populated so that the observed and the bootstrapped matrix have the same number of edges.
    """
    number_edges = np.count_nonzero(A)
    # find the edge locations
    edges = np.transpose(np.nonzero(A))
    n = A.shape[0]
    
    # sample the edges with replacement
    sampled_edges = edges[random.choices(range(len(edges)), k=number_edges)]
    
    # create A_new and set sampled edges to 1
    A_new = np.zeros((n, n), dtype=int)
    A_new[sampled_edges[:, 0], sampled_edges[:, 1]] = 1
    
    # add in random edges
    missing_edges = number_edges - np.count_nonzero(A_new)
    while missing_edges > 0:
        i, j = np.random.randint(0, n, size=2)
        if A_new[i, j] == 0:
            A_new[i, j] = 1
            missing_edges -= 1
    
    return A_new
    
    
    
def create_single_kNN_bootstrap(A, d, Q=1000, n_neighbors=5):
    """Create a single ASE-kNN bootstrap, where ASE is into d dimensions, uses Q permutations to find p-val, and k=n_neighbors in kNN"""
    n = A.shape[0]
    A_obs = A.copy()

    # Embed the graphs -------------------------------

    yhat = UASE([A], d=d, flat=True)

    # run a k-NN on the embedding yhat
    # Here we use Minkowski distance, with p=2 (these are the defaults),
    # which corresponds to Euclidean distance
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric="minkowski", p=2
    ).fit(yhat)
    distances, indices = nbrs.kneighbors(yhat)

    # Estimate the P matrix -------------------------------
    P_est = P_est_from_A_obs(n, A_obs, n_neighbors=n_neighbors, indices=indices)

    # Bootstrap -----------------------------------------
    A_est = make_inhomogeneous_rg(P_est)

    # embed the observed and bootstrapped matrices together --------------------------------
    yhat_est = UASE([A_obs, A_est], d=d)

    # do a test between the obs and the bootstrap, get a p-value ---------------------------------
    p_val = test_temporal_displacement_two_times(yhat_est, n, n_sim=Q)

    return p_val, A_est
    
    
    
def create_single_YYT_bootstrap_cropPto0_1range(A, d, Q=1000):
    """
    Generates a single bootstrapped adjacency matrix from a single adjacency matrix A.

    input:
    A: (numpy array (n, n)) adjacency matrix
    d: (int) embedding dimension. In theory the rank of the noise-free A matrix.
    Q: (int) number of simulations in the paired exchangeability test.

    output:
    A_star: (numpy array (n, n)) bootstrapped adjacency matrix
    p_val: (float) p-value from the exch test between the obs and the bootstrapped matrix
    """

    # Compute ONLY the left spectral embeddings of A
    Y_hat = single_spectral_Y(A, d)  # right

    # Compute the estimated probability matrix
    P_hat = Y_hat @ Y_hat.T

    # Check if P_hat is a valid probability matrix
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1]. The values outside this range will be clipped to lie in the range.")
        
    # Clip values in P_hat to be between 0 and 1
    P_hat = np.clip(P_hat, 0, 1)
    
    # Check the values in P_hat have been clipped 
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1] after the clipping code, please check the function.")

    A_star = np.array([make_inhomogeneous_rg(P_hat)])

    # embed the observed and bootstrapped matrix together
    yhat_est = UASE([A, A_star[0]], d=d)
    # do a test between the obs and the bootstrap, get a p-value ---------------------------------
    p_val = test_temporal_displacement_two_times(yhat_est, n=A.shape[0], n_sim=Q) 

    return p_val, A_star[0]

    
def create_single_XXT_bootstrap_cropPto0_1range(A, d, Q=1000):
    """
    Generates a single bootstrapped adjacency matrix from a single adjacency matrix A.

    input:
    A: (numpy array (n, n)) adjacency matrix
    d: (int) embedding dimension. In theory the rank of the noise-free A matrix.
    Q: (int) number of simulations in the paired exchangeability test.

    output:
    A_star: (numpy array (n, n)) bootstrapped adjacency matrix
    p_val: (float) p-value from the exch test between the obs and the bootstrapped matrix
    """

    # Compute ONLY the left spectral embeddings of A
    X_hat = single_spectral_X(A, d)  # left

    # Compute the estimated probability matrix
    P_hat = X_hat @ X_hat.T

    # Check if P_hat is a valid probability matrix
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1]. The values outside this range will be clipped to lie in the range.")
        
    # Clip values in P_hat to be between 0 and 1
    P_hat = np.clip(P_hat, 0, 1)
    
    # Check the values in P_hat have been clipped 
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1] after the clipping code, please check the function.")

    A_star = np.array([make_inhomogeneous_rg(P_hat)])

    # embed the observed and bootstrapped matrix together
    yhat_est = UASE([A, A_star[0]], d=d)
    # do a test between the obs and the bootstrap, get a p-value ---------------------------------
    p_val = test_temporal_displacement_two_times(yhat_est, n=A.shape[0], n_sim=Q) 

    return p_val, A_star[0]
        
        
        
def edgelist_sample_with_replacement_addRandomEdges(A):
    """Actually just samples with replacement. As a binary setting, any edge selected more than once will be set to 1
    Random edges will be populated so that the observed and the bootstrapped matrix have the same number of edges
    """
    number_edges = np.count_nonzero(A)
    # find the edge locations
    edges = np.transpose(np.nonzero(A))
    # sample the edges with replacement
    edge_idx = random.choices(range(0,number_edges), k=number_edges)
    # remove duplicates
    edge_idx = list(set(edge_idx))
    # create an array from edges with the sampled indices
    edge_sample = edges[edge_idx]
    n = A.shape[0]
    A_new = np.zeros((n,n))
    # populate A_new with a 1 where edge_sample is the index
    for e in range(len(edge_sample)):
        edge_id = edge_sample[e]
        A_new[edge_id[0], edge_id[1]] = 1
        
    # add in random edges
    num_to_add = number_edges - len(edge_sample)
    while num_to_add > 0:
        i, j = np.random.randint(low=0, high=n, size=2)
        # check not in edges
        if A_new[i, j] == 0:
            A_new[i, j] = 1
            # if not in edge, add to edge_sample, else repeat loop
            num_to_add -= 1
    
    return A_new
    
    
def edgelist_sample_with_replacement(A):
    """Actually just samples with replacement. As a binary setting, any edge selected more than once will be set to 1"""
    number_edges = np.count_nonzero(A)
    # find the edge locations
    edges = np.transpose(np.nonzero(A))
    # sample the edges with replacement
    edge_idx = random.choices(range(0,number_edges), k=number_edges)
    # remove duplicates
    edge_idx = list(set(edge_idx))
    # create an array from edges with the sampled indices
    edge_sample = edges[edge_idx]
    n = A.shape[0]
    A_new = np.zeros((n,n))
    # populate A_new with a 1 where edge_sample is the index
    for e in range(len(edge_sample)):
        edge_id = edge_sample[e]
        A_new[edge_id[0], edge_id[1]] = 1
    
    return(A_new)       
        
    
"""This is needed for the test_bootstrap function"""
@nb.njit
def P_est_from_A_obs(n, A_obs, n_neighbors, indices):
    P_est = np.zeros((n, n))
    for i in range(n):
        idx = indices[i]
        A_i = (1 / n_neighbors) * np.sum(A_obs[:, idx], axis=1)
        P_est[:, i] = A_i
    return P_est
    
    
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
    A_row_jumbles = np.zeros((B, n, n))

    for i in range(B):
        # for each bootstrap sample, select which rows to put in the new matrix
        idx = np.random.choice(n, size=n, replace=True)
        for j in range(n):
            # put the rows in the new matrix
            A_row_jumbles[i][j, :] += A[idx[j], :]

    return A_row_jumbles   
    
    
@nb.njit
def P_est_from_edge_list(n, node_to_edges, n_neighbors, indices):
    P_est = np.zeros((n, n))
    for i in range(n):
        idx_set = set(indices[i])
        A_i = np.zeros(n)
        for node in idx_set:
            for edge in node_to_edges[node]:
                A_i[edge] += 1
        A_i = (1 / n_neighbors) * A_i
        P_est[:, i] = A_i
    return P_est
    
def create_single_kNN_prone_bootstrap(A, d, Q=1000, n_neighbors=5):
    n = A.shape[0]
    A_obs = A.copy()

    # Embed the graphs -------------------------------
    yhat = unfolded_prone(A, d=d, flat=True)

    # run a k-NN on the embedding yhat
    # Here we use Minkowski distance, with p=2 (these are the defaults),
    # which corresponds to Euclidean distance

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric="minkowski", p=2
    ).fit(yhat)
    distances, indices = nbrs.kneighbors(yhat)

    # Estimate the P matrix -------------------------------
    P_est = P_est_from_A_obs(n, A_obs, n_neighbors=n_neighbors, indices=indices)

    # Bootstrap -----------------------------------------
    A_est = make_inhomogeneous_rg(P_est)

    # embed the observed and bootstrapped matrices together --------------------------------
    yhat_est = UASE([A_obs, A_est], d=d)

    # do a test between the obs and the bootstrap, get a p-value ---------------------------------
    p_val = test_temporal_displacement_two_times(yhat_est, n, n_sim=Q)


    return p_val, A_est
    
# Function to plot ROC curves
def plot_roc(ax, indices, title):
    for i in indices:
        p_hat_list = p_list_allnames[i]
        roc = []
        alphas = []
        for alpha in np.linspace(0, 1, 100):
            alphas.append(alpha)
            num_below_alpha = sum(p_hat_list < alpha)
            roc_point = num_below_alpha / len(p_hat_list)
            roc.append(roc_point)

        score = np.round(compute_roc_and_areas(p_hat_list, significance_level=0.05)['total_area'], 3)

        ax.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), linestyle="--", c="grey")
        ax.plot(alphas, roc, color=colors[i], label=(labels[i]) + ",  $S=$" + str(score))

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=10)
    ax.set_title(title, fontsize=16)
    
    
def plot_ellipse(ax, mean, cov, color, lw=2):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues[:2])
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', lw=lw, label=f'Covariance Ellipse ({color})')
    ax.add_patch(ellipse)



    
    
# TO AVOID SINGULAR MATRIX ERROR
def points_within_ellipse(points, mean, cov, regularization=1e-32, threshold=3):
    try:
        # Attempt to calculate the inverse of the covariance matrix
        inv_cov = np.linalg.inv(cov)
    except LinAlgError:
        # If the matrix is singular, regularize and retry
        cov += np.eye(cov.shape[0]) * regularization
        inv_cov = np.linalg.inv(cov)
    
    # Calculate the Mahalanobis distance from the mean
    diff = points - mean
    mahalanobis_distances = np.sum(diff @ inv_cov * diff, axis=1)/cov.shape[0]
    
    # Points within the ellipse have a Mahalanobis distance <= threshold
    return mahalanobis_distances <= threshold

    
    
    
def plot_ellipse_3mahals(ax, mean, cov, color='blue', lw=1):
    """
    Plot an ellipse representing the covariance matrix.
    
    Parameters:
    - ax: matplotlib axis to plot on.
    - mean: 2D array for the center of the ellipse.
    - cov: 2x2 covariance matrix.
    - color: Color of the ellipse.
    - lw: Line width.
    """
    # Eigenvalues and eigenvectors for the covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort the eigenvalues and corresponding eigenvectors in descending order
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # Calculate angle of ellipse based on largest eigenvector
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    # Define standard deviations to plot (1, 2, and 3 SDs)
    std_devs = [1, 2, 3]
    
    for std_dev in std_devs:
        # Width and height of ellipse correspond to 2*sqrt(eigenvalue)
        width, height = 2 * std_dev * np.sqrt(vals)
        
        # Create and add ellipse patch
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      color=color, lw=lw, fill=False, alpha=0.8)
        ax.add_patch(ell)

def plot_ellipse_3rd_mahals(ax, mean, cov, color='blue', lw=1):
    """
    Plot an ellipse representing the covariance matrix at 3 SDs
    
    Parameters:
    - ax: matplotlib axis to plot on.
    - mean: 2D array for the center of the ellipse.
    - cov: 2x2 covariance matrix.
    - color: Color of the ellipse.
    - lw: Line width.
    """
    # Eigenvalues and eigenvectors for the covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    
    # Sort the eigenvalues and corresponding eigenvectors in descending order
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # Calculate angle of ellipse based on largest eigenvector
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
    std_dev = 3
    # Width and height of ellipse correspond to 2*sqrt(eigenvalue)
    width, height = 2 * std_dev * np.sqrt(vals)
    
    # Create and add ellipse patch
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                    color=color, lw=lw, fill=False, alpha=0.8)
    ax.add_patch(ell)
  
  
    
def get_score(A,d,B=100,Q=1000,seed=None,f=create_single_kNN_bootstrap,  *args, **kwargs):
    if(seed):
        random.seed(seed)
        np.random.seed(100)
    p_vals = []
    A_boots_list = []

    for b in tqdm(range(B)):
        p_val, A_boots = f(A, d=d, Q=1000, *args, **kwargs)
        p_vals.append(p_val)
        A_boots_list.append(A_boots)

    # Provided code
    p_hat_list = p_vals
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
    
    # Calculate the area between ROC and y=x line
    x = np.linspace(0, 1, 100)
    roc_interpolated = np.interp(x, alphas, roc)

    # Compute areas
    area_above, area_below = compute_area_above_below_curve(x, roc_interpolated)
    total_area = area_above + area_below

    return total_area, power, alphas, roc
    
    
    
def create_fuzziness_matrix(yadf, d, n, threshold=3):
    """
    Creates a fuzziness matrix by analyzing node data and identifying points within a certain ellipse
    based on the mean and covariance of the data.

    Parameters:
    yadf (DataFrame): Input data frame containing node and matrix data.
    d (int): Number of dimensions to consider.
    n (int): Number of nodes to process.

    Returns:
    in_cov_friends_symm (2D array): The symmetric fuzziness matrix.
    """
    # Initialize the fuzziness matrix
    in_cov_friends = np.zeros((n, n))
    
    for i in range(n):
        # Filter data for the specific node number and select the relevant dimensions
        node_number = i
        data_d_dim = yadf[yadf["NodeNumber"] == node_number].iloc[:, 0:d].to_numpy()

        # Calculate the mean and covariance for all d dimensions
        mean_d_dim = np.mean(data_d_dim, axis=0)
        cov_d_dim = np.cov(data_d_dim, rowvar=False)

        # Use the first point corresponding to the node_number in matrix 0 as the center
        point = data_d_dim[0]
        obs_points = yadf[yadf["Matrix"] == 0].iloc[:, 0:d].to_numpy()

        # Find points within the ellipse
        inside_ellipse = points_within_ellipse(obs_points, point, cov_d_dim, threshold=threshold)

        # Extract node numbers for points inside the ellipse
        node_numbers_inside_ellipse = yadf[yadf["Matrix"] == 0].iloc[inside_ellipse].index.tolist()

        # Update the fuzziness matrix for this node
        in_cov_friends[i, node_numbers_inside_ellipse] = 1

    # Symmetrize the fuzziness matrix
    in_cov_friends_symm = np.minimum(in_cov_friends, in_cov_friends.T)
    
    return in_cov_friends_symm




def uncertainty_score(E,A):
    ## Takes an adjacency matrix A and an embedding E and averages the length of all edges of A in E
    n = A.shape[0]
    score = 0
    meandist = 0
    edgecount = 0
    for i in range(n):
        for j in range(n):
            edgecount += A[i,j]
            meandist += A[i,j] * np.sqrt((np.linalg.norm(E[i,:] - E[j,:])))
    meandist = meandist / edgecount
    for i in range(n):
        for j in range(n):
            score += A[i,j] * (np.sqrt((np.linalg.norm(E[i,:] - E[j,:]))) - meandist)**2 /edgecount
    return(np.sqrt(score))



def get_friends(yadf,threshold=3):
    """"create an adjacency matrix of nodes that overlap in uncertainty"""
    n=np.int64(yadf.shape[0]/(1+np.max(yadf["Matrix"])))
    in_cov_friends = np.zeros((n,n))
    
    for i in range(n):
        # Filter data for node number and select the relevant dimensions
        node_number = i
        data_d_dim = yadf[yadf["NodeNumber"] == node_number].iloc[:, 0:d].to_numpy()

        # Calculate the mean and covariance considering all d dimensions
        mean_d_dim = np.mean(data_d_dim, axis=0)
        cov_d_dim = np.cov(data_d_dim, rowvar=False)

        # Use the point corresponding to the specific node_number in matrix 0 as the center
        point = data_d_dim[0]
        obs_points = yadf[yadf["Matrix"] == 0].iloc[:, 0:d].to_numpy()

        # Filter points within the ellipse
        inside_ellipse = points_within_ellipse(obs_points, point, cov_d_dim, threshold=threshold)

        # Extract node numbers for points inside the ellipse
        node_numbers_inside_ellipse = yadf[yadf["Matrix"] == 0].iloc[inside_ellipse].index.tolist()

        # set in_cov_friends[i, node_numbers_inside_ellipse] = 1
        in_cov_friends[i, node_numbers_inside_ellipse] = 1

    # symmetrize the matrix in_cov_friends - 
    # minimum means both must be 1
    # maximum means at least one must be 1
    in_cov_friends_symm = np.minimum(in_cov_friends, in_cov_friends.T)
    return(in_cov_friends_symm)   
    
    
    
def create_single_parametric_bootstrap_cropPto0_1range(A, d, Q=1000):
    """
    Generates a single bootstrapped adjacency matrix from a single adjacency matrix A.

    input:
    A: (numpy array (n, n)) adjacency matrix
    d: (int) embedding dimension. In theory the rank of the noise-free A matrix.
    Q: (int) number of simulations in the paired exchangeability test.

    output:
    A_star: (numpy array (n, n)) bootstrapped adjacency matrix
    p_val: (float) p-value from the exch test between the obs and the bootstrapped matrix
    """

    # Compute the left and right spectral embeddings of A
    X_hat = single_spectral_X(A, d)  # left

    # Compute the estimated probability matrix
    P_hat = X_hat @ X_hat.T

    # Check if P_hat is a valid probability matrix
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1]. The values outside this range will be clipped to lie in the range.")
        
    # Clip values in P_hat to be between 0 and 1
    P_hat = np.clip(P_hat, 0, 1)
    
    # Check the values in P_hat have been clipped 
    if np.min(P_hat) < 0 or np.max(P_hat) > 1:
        warnings.warn("P_hat contains values outside of [0,1] after the clipping code, please check the function.")

    A_star = np.array([make_inhomogeneous_rg(P_hat)])

    # embed the observed and bootstrapped matrix together
    yhat_est = UASE([A, A_star[0]], d=d)
    # do a test between the obs and the bootstrap, get a p-value ---------------------------------
    p_val = test_temporal_displacement_two_times(yhat_est, n=A.shape[0], n_sim=Q) 

    return p_val, A_star[0]
    
    
    
    
    
    
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%% all above are defo in the code

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
            A_star = np.array([make_inhomogeneous_rg_sparse(P_hat) for _ in range(B)])
    else:
        if not sparse:
            A_star = np.array([make_inhomogeneous_rg(P_hat) for _ in range(B)])
        else:
            A_star = np.array([make_inhomogeneous_rg_sparse(P_hat) for _ in range(B)])

    if return_P_hat:
        return A_star, P_hat
    else:
        return A_star




        
        

        
        
        




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

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric="minkowski", p=2
    ).fit(yhat)
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

        yhat_est = UASE([A_obs, A_est], d=d)
        p_val = test_temporal_displacement_two_times(yhat_est, n)
        p_vals.append(p_val)
        A_boots.append(A_est)

    return p_vals, A_boots


def get_node_to_edges(edge_list, n):
    node_to_edges = Dict.empty(key_type=types.int32, value_type=int_list_type)
    for i in range(n):
        node_to_edges[i] = List(lsttype=int_list_type)
    for edge in edge_list:
        node_to_edges[edge[1]].append(edge[0])
    return node_to_edges



def create_single_kNN_n2v_bootstrap(A, d, Q=1000, n_neighbors=5):
    n = A.shape[0]
    A_obs = A.copy()

    # Embed the graphs -------------------------------

    yhat = unfolded_n2v([A], d=d, flat=True)

    # run a k-NN on the embedding yhat
    # Here we use Minkowski distance, with p=2 (these are the defaults),
    # which corresponds to Euclidean distance
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="ball_tree", metric="minkowski", p=2
    ).fit(yhat)
    distances, indices = nbrs.kneighbors(yhat)

    # Estimate the P matrix -------------------------------
    P_est = P_est_from_A_obs(n, A_obs, n_neighbors=n_neighbors, indices=indices)

    # Bootstrap -----------------------------------------
    A_est = make_inhomogeneous_rg(P_est)

    # embed the observed and bootstrapped matrices together --------------------------------
    yhat_est = UASE([A_obs, A_est], d=d)

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









